#!/usr/bin/env python3
"""Which of CODI's latents carry anything on RuleChain (depth2)? Cosine geometry + per-site ablation, with all
latent positions kept (unlike the latent-count sweep, which truncates the chain and so changes the sequence layout).

Sites z_0..z_{K-1}: z_s = the projected vector fed INTO latent iteration s+1 (z_0 = prj(hidden at <bot>)), the same
numbering as necessity_codi.py / patch_minimal_pair_codi.py. Replacing z_s changes that input; everything after it
is recomputed from the changed state.

Conditions (each one forward per item, answer read teacher-forced at the ':' of "<eot> The answer is:", which
matches generated accuracy within +-0.004 per level, see diagnose_codi_latents.py):
  clean              no change
  mean_s             z_s := population mean of clean z_s over --mean_json (valid, so no test statistics enter)
  resample_s         z_s := clean z_s of another test item (seeded permutation, donor level drawn uniformly)
  mean_tail_ge_s     z_t := mean_t for every t >= s ("do the later latents add anything?")
  mean_all           every site mean-ablated (positions still run)
Geometry (clean test latents): K x K cosine between sites (raw, and after subtracting each site's population mean),
cross-item cosine at each site, and the item-specific share of each site's energy ||z - mean||^2 / ||z||^2.

Run inside the CODI venv from the CODI checkout (like eval_synthetic_codi.py):
  cd ../codi && HASH_FILE=... .venv/bin/python ../ai-capstone/scripts/ablate_latents_rulechain_codi.py \\
      --ckpt_dir ../codi_runs_l1to7w_codi/checkpoint-17820 --test_json ../synthetic_data_v2_l1to7w/depth2/test.json \\
      --mean_json ../synthetic_data_v2_l1to7w/depth2/valid.json --slug v2-l1to7w-depth2-codi-latent-ablation \\
      --hardware "MacBook CPU" --checkpoint_label checkpoint-17820 --output_dir /tmp/u \\
      --model_name_or_path gpt2 --seed 11 --model_max_length 512 --lora_r 128 --lora_alpha 32 --lora_init \\
      --batch_size 64 --greedy True --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \\
      --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \\
      --fix_attn_mask True --pad_invariant True --include_last_cot False
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers
from torch.nn import functional as F

sys.path.insert(0, os.getcwd())  # the CODI checkout: `src.model`
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.model import DataArguments, ModelArguments, TrainingArguments  # noqa: E402

from diagnose_codi_latents import pick_device, summarise  # noqa: E402  (also patches torch.load for CPU)
from eval_codi import PadInvariant, build_model, question_position_ids, tokenize_questions  # noqa: E402
from eval_synthetic_codi import answer_token, check_flags  # noqa: E402
from synthetic_common import load_synthetic  # noqa: E402

from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402


@dataclass
class AblArguments:
    test_json: str = field(default="")
    mean_json: str = field(default="")
    device: str = field(default="cpu")
    perm_seed: int = field(default=0)
    slug: str = field(default="v2-depth2-codi-latent-ablation")
    stage: str = field(default="full_run")
    hardware: str = field(default="local")
    checkpoint_label: Optional[str] = field(default=None)
    smoke_n: int = field(default=0, metadata={"help": "first n test items (smoke test, not logged)"})
    no_log: bool = field(default=False)


@torch.no_grad()
def run(model, tokenizer, rows, data_args, training_args, override=None):
    """One forward per item: question, K latent passes, <eot> "The answer is:". `override(s, idx) -> [B, 1, d] or
    None` replaces z_s for the batch items with global indices idx. Returns (clean-or-used z [N, K, d], argmax ids)."""
    device = next(model.parameters()).device
    body = model.codi.get_base_model().transformer
    K = training_args.inf_latent_iterations
    dec = [model.eot_id] + tokenizer.encode("The answer is:")
    embed = model.get_embd(model.codi, model.model_name)
    questions = [r["question"].strip() + "\n" for r in rows]
    zs, preds, start = [], [], 0
    for batch in tokenize_questions(model, tokenizer, questions, data_args.batch_size, training_args, device):
        B = batch["input_ids"].size(0)
        idx = torch.arange(start, start + B)
        start += B
        q_pos = question_position_ids(batch["attention_mask"]) if training_args.pad_invariant else None
        out = body(input_ids=batch["input_ids"], use_cache=True, attention_mask=batch["attention_mask"], position_ids=q_pos)
        past, z = out.past_key_values, model.prj(out.last_hidden_state[:, -1:, :])
        pi = PadInvariant(batch["attention_mask"]) if training_args.pad_invariant else None
        used = []
        for s in range(K):
            if override is not None:
                rep = override(s, idx)
                if rep is not None:
                    z = rep.to(device, z.dtype)
            used.append(z[:, 0].float().cpu())
            out = body(inputs_embeds=z, use_cache=True, past_key_values=past, **(pi.next(1) if pi else {}))
            past, z = out.past_key_values, model.prj(out.last_hidden_state[:, -1:, :])
        emb = embed(torch.tensor(dec, dtype=torch.long, device=device))[None].expand(B, -1, -1)
        out = model.codi(inputs_embeds=emb, use_cache=True, past_key_values=past,
                         **(pi.next(len(dec)) if pi else {"attention_mask": None}))
        preds += out.logits[:, -1, :model.codi.config.vocab_size - 1].argmax(-1).tolist()
        zs.append(torch.stack(used, 1))
    return torch.cat(zs), preds


def geometry(z: torch.Tensor, mu: torch.Tensor) -> dict:
    """z [N, K, d] clean test latents, mu [K, d] population means (from mean_json)."""
    def kxk(x):
        x = F.normalize(x, dim=-1)
        return torch.einsum("nkd,njd->kj", x, x) / x.size(0)
    zc = z - mu[None]
    g = torch.Generator().manual_seed(0)
    a, b = torch.randint(0, z.size(0), (4000,), generator=g), torch.randint(0, z.size(0), (4000,), generator=g)
    keep = a != b
    cross = F.cosine_similarity(z[a[keep]], z[b[keep]], dim=-1).mean(0)
    r = lambda t: [[round(v, 3) for v in row] for row in t.tolist()]  # noqa: E731
    return {"cos_sites_raw": r(kxk(z)), "cos_sites_centered": r(kxk(zc)),
            "cross_item_cos_per_site": [round(v, 3) for v in cross.tolist()],
            "item_specific_energy_per_site": [round(v, 3) for v in (zc.pow(2).sum(-1).mean(0) / z.pow(2).sum(-1).mean(0)).tolist()],
            "norm_per_site": [round(v, 2) for v in z.norm(dim=-1).mean(0).tolist()]}


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, AblArguments))
    model_args, data_args, training_args, aa = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    flags = check_flags(model_args.ckpt_dir, training_args)
    device = pick_device(aa.device)
    model, tokenizer, load_result = build_model(model_args, training_args, device=device, dtype=torch.float32)
    if any(not k.startswith("codi.base_model.model.lm_head") for k in load_result.missing_keys):
        raise SystemExit(f"checkpoint is missing weights: {load_result.missing_keys[:5]}")
    model.eval()
    K = training_args.inf_latent_iterations
    test = load_synthetic(aa.test_json).rows
    if aa.smoke_n:
        test = test[:aa.smoke_n]
    gold = [answer_token(tokenizer, r["answer"]) for r in test]
    score = lambda preds: [None if g is None else p == g for p, g in zip(preds, gold)]  # noqa: E731

    t0 = time.perf_counter()
    mean_rows = load_synthetic(aa.mean_json).rows
    z_mean_src, _ = run(model, tokenizer, mean_rows[:aa.smoke_n] if aa.smoke_n else mean_rows, data_args, training_args)
    mu = z_mean_src.mean(0)  # [K, d]
    z_clean, p_clean = run(model, tokenizer, test, data_args, training_args)
    conds = {"clean": score(p_clean)}
    print(f"clean {summarise(test, conds['clean'], None)['accuracy']} ({time.perf_counter() - t0:.0f}s)", flush=True)

    perm = torch.randperm(len(test), generator=torch.Generator().manual_seed(aa.perm_seed))
    same = perm == torch.arange(len(test))
    perm[same] = (perm[same] + 1) % len(test)
    mean_at = lambda s, idx: mu[s][None, None].expand(len(idx), 1, -1)  # noqa: E731
    specs = {}
    for s in range(K):
        specs[f"mean_{s}"] = (lambda s0: lambda s, idx: mean_at(s, idx) if s == s0 else None)(s)
        specs[f"resample_{s}"] = (lambda s0: lambda s, idx: z_clean[perm[idx], s][:, None] if s == s0 else None)(s)
    for s in range(1, K):
        specs[f"mean_tail_ge_{s}"] = (lambda s0: lambda s, idx: mean_at(s, idx) if s >= s0 else None)(s)
    specs["mean_all"] = mean_at
    for name, fn in specs.items():
        t1 = time.perf_counter()
        _, p = run(model, tokenizer, test, data_args, training_args, override=fn)
        conds[name] = score(p)
        print(f"{name}: {summarise(test, conds[name], None)['accuracy']} ({time.perf_counter() - t1:.0f}s)", flush=True)

    summary = {k: summarise(test, v, None) for k, v in conds.items()}
    geo = geometry(z_clean, mu)
    print(json.dumps(geo, indent=1))
    if aa.smoke_n or aa.no_log:
        print("not logging (smoke_n / no_log)")
        return

    records = [{"idx": i, "id": r["id"], "level": int(r["level"]), "gold_answer": r["answer"], "question": r["question"],
                "predicted_answer": tokenizer.decode([p_clean[i]]).strip(), "correct": conds["clean"][i],
                **{f"correct_{k}": v[i] for k, v in conds.items() if k != "clean"}} for i, r in enumerate(test)]
    elapsed = time.perf_counter() - t0
    metrics = {"final_answer_accuracy": summary["clean"]["accuracy"], "unparseable_rate": 0.0, "compute_steps": K,
               "sec_per_example": elapsed / len(test),
               "extra": {"accuracy_kind": "teacher-forced answer token at the distillation position (no generation)",
                         "conditions": summary, "geometry": geo, "mean_json": aa.mean_json,
                         "data_sha256": hashlib.sha256(Path(aa.test_json).read_bytes()).hexdigest(),
                         "matched_flags": flags}}
    record = RunRecord(
        run_id=new_run_id("codi", aa.slug), mechanism="codi", stage=aa.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=aa.checkpoint_label or model_args.ckpt_dir,
                        n_params=sum(p.numel() for p in model.parameters())),
        dataset=DatasetInfo(name="synthetic-depth2", split="test", n_examples=len(test), seed=None),
        metrics=metrics,
        hyperparams={"inference": {"latent_iterations": K, "batch_size": data_args.batch_size, "dtype": "float32",
                                   "device": device, "perm_seed": aa.perm_seed}, "test_json": aa.test_json},
        seed=None, hardware=f"{aa.hardware} / {device}",
        notes="RuleChain CODI latent geometry + per-site mean/resample ablation, all positions kept.")
    manifest = record.save(predictions=records)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
