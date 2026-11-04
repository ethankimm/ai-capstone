#!/usr/bin/env python3
"""What do CODI's latent positions encode on RuleChain (depth2)? Site x step probes for the running value v_j, plus
LM-head / embedding decodes of each latent. Companion to ablate_latents_rulechain_codi.py, which found z_1..z_5 to be
near-copies (centered cosine >= 0.97) with z_1 carrying the item.

Sites (last-layer hidden states, ln_f applied, one clean forward per item):
  h_0      at <bot> (end of the question pass); z_0 = prj(h_0) is fed into latent pass 1
  h_s      output of latent pass s (s = 1..K); z_s = prj(h_s) is fed into pass s+1 (h_K is fed nowhere)
  h_ans    the ':' of "<eot> The answer is:" (CODI's distillation / readout position)
Targets: v_j = value after rule j (steps "<<a op b=v_j>>"), j = 1..max level, on items with level >= j; and the final
answer. Values 0-99 as two 10-way digit heads (tens, ones), "exact" = both right, linear softmax probes with L2 chosen
on a validation slice (fit_probe from probe_rulechain_branch_codi.py), fit on a train.json sample, scored on test.
Baseline: majority class per target on test.

Decodes: top token of lm_head(h_s) (logit lens of that position) and of z_s . wte^T (nearest input embedding, the
space z_s is fed in), parsed as an integer; reported as the share equal to v_j for each j and to the answer.

  cd ../codi && HASH_FILE=... .venv/bin/python ../ai-capstone/scripts/probe_latents_rulechain_codi.py \\
      --ckpt_dir ../codi_runs_l1to7w_codi/checkpoint-17820 --test_json ../synthetic_data_v2_l1to7w/depth2/test.json \\
      --train_json ../synthetic_data_v2_l1to7w/depth2/train.json --train_per_level 2000 \\
      --slug v2-l1to7w-depth2-codi-latent-probes --hardware "MacBook CPU" --checkpoint_label checkpoint-17820 \\
      <model flags as in ablate_latents_rulechain_codi.py>
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers

sys.path.insert(0, os.getcwd())  # the CODI checkout: `src.model`
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.model import DataArguments, ModelArguments, TrainingArguments  # noqa: E402

from diagnose_codi_latents import per_level, pick_device  # noqa: E402  (also patches torch.load for CPU)
from eval_codi import PadInvariant, build_model, question_position_ids, tokenize_questions  # noqa: E402
from eval_synthetic_codi import check_flags  # noqa: E402
from probe_rulechain_branch_codi import fit_probe, standardise  # noqa: E402
from synthetic_common import load_synthetic  # noqa: E402

from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402


@dataclass
class LatProbeArguments:
    test_json: str = field(default="")
    train_json: str = field(default="")
    train_per_level: int = field(default=2000)
    device: str = field(default="cpu")
    probe_iters: int = field(default=150)
    slug: str = field(default="v2-depth2-codi-latent-probes")
    stage: str = field(default="full_run")
    hardware: str = field(default="local")
    checkpoint_label: Optional[str] = field(default=None)
    smoke_n: int = field(default=0, metadata={"help": "items per level (smoke test, not logged)"})


def values(row) -> list[int]:
    return [int(re.search(r"=(-?\d+)>>", s).group(1)) for s in row["steps"]]


def number_tokens(tokenizer, V: int) -> dict[int, Optional[int]]:
    out = {}
    for i in range(V):
        try:
            t = tokenizer.decode([i]).strip()
        except TypeError:  # ids without a token string
            t = ""
        out[i] = int(t) if re.fullmatch(r"-?\d+", t) else None
    return out


@torch.no_grad()
def collect(model, tokenizer, rows, data_args, training_args):
    """[N, K+2, d] sites (h_0..h_K, h_ans) and [N, K+1] decoded ints from lm_head(h_s) and from z_s . wte^T."""
    device = next(model.parameters()).device
    body = model.codi.get_base_model().transformer
    lm_head = model.codi.get_base_model().lm_head
    wte = body.wte.weight
    K = training_args.inf_latent_iterations
    dec = [model.eot_id] + tokenizer.encode("The answer is:")
    embed = model.get_embd(model.codi, model.model_name)
    V = model.codi.config.vocab_size - 1
    num = number_tokens(tokenizer, V)
    questions = [r["question"].strip() + "\n" for r in rows]
    H, D_lm, D_emb = [], [], []
    for batch in tokenize_questions(model, tokenizer, questions, data_args.batch_size, training_args, device):
        B = batch["input_ids"].size(0)
        q_pos = question_position_ids(batch["attention_mask"]) if training_args.pad_invariant else None
        out = body(input_ids=batch["input_ids"], use_cache=True, attention_mask=batch["attention_mask"], position_ids=q_pos)
        past, h = out.past_key_values, out.last_hidden_state[:, -1]
        pi = PadInvariant(batch["attention_mask"]) if training_args.pad_invariant else None
        hs, zs = [h], []
        for _ in range(K):
            z = model.prj(h[:, None])
            zs.append(z[:, 0])
            out = body(inputs_embeds=z, use_cache=True, past_key_values=past, **(pi.next(1) if pi else {}))
            past, h = out.past_key_values, out.last_hidden_state[:, -1]
            hs.append(h)
        emb = embed(torch.tensor(dec, dtype=torch.long, device=device))[None].expand(B, -1, -1)
        out = body(inputs_embeds=emb, use_cache=True, past_key_values=past, **(pi.next(len(dec)) if pi else {}))
        hs.append(out.last_hidden_state[:, -1])
        Hb = torch.stack(hs, 1)
        H.append(Hb.float().cpu())
        lm_top = lm_head(Hb[:, :K + 1])[..., :V].argmax(-1).cpu()
        emb_top = (torch.stack(zs + [model.prj(hs[K][:, None])[:, 0]], 1) @ wte[:V].T).argmax(-1).cpu()
        D_lm.append(lm_top.apply_(lambda i: -1 if num[i] is None else num[i]))
        D_emb.append(emb_top.apply_(lambda i: -1 if num[i] is None else num[i]))
    return torch.cat(H), torch.cat(D_lm), torch.cat(D_emb)


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LatProbeArguments))
    model_args, data_args, training_args, pa = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    flags = check_flags(model_args.ckpt_dir, training_args)
    device = pick_device(pa.device)
    model, tokenizer, load_result = build_model(model_args, training_args, device=device, dtype=torch.float32)
    if any(not k.startswith("codi.base_model.model.lm_head") for k in load_result.missing_keys):
        raise SystemExit(f"checkpoint is missing weights: {load_result.missing_keys[:5]}")
    model.eval()
    K = training_args.inf_latent_iterations
    test = per_level(load_synthetic(pa.test_json).rows, pa.smoke_n)
    train = per_level(load_synthetic(pa.train_json).rows, pa.smoke_n or pa.train_per_level)
    t0 = time.perf_counter()
    Htr, _, _ = collect(model, tokenizer, train, data_args, training_args)
    Hte, Dlm, Demb = collect(model, tokenizer, test, data_args, training_args)
    print(f"collected train {tuple(Htr.shape)} test {tuple(Hte.shape)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    site_names = [f"h_{s}" for s in range(K + 1)] + ["h_ans"]
    Xtr, Xte = standardise(Htr, Hte)

    vtr, vte = [values(r) for r in train], [values(r) for r in test]
    max_j = max(len(v) for v in vte)
    targets = {f"v_{j}": j - 1 for j in range(1, max_j + 1)}
    targets["answer"] = -1
    probes, majority, decodes = {}, {}, {}
    for name, j in targets.items():
        itr = [i for i, v in enumerate(vtr) if j == -1 or len(v) > j]
        ite = [i for i, v in enumerate(vte) if j == -1 or len(v) > j]
        ytr = torch.tensor([vtr[i][j] for i in itr])
        yte = torch.tensor([vte[i][j] for i in ite])
        correct = torch.ones(len(ite), Xte.size(1), dtype=torch.bool)
        for head in (lambda y: y // 10, lambda y: y % 10):
            logits, _ = fit_probe(Xtr[itr], head(ytr), Xte[ite], 10, pa.probe_iters)
            correct &= logits.argmax(-1) == head(yte)[:, None]
        probes[name] = {s: round(c, 3) for s, c in zip(site_names, correct.float().mean(0).tolist())}
        majority[name] = round((yte == yte.mode().values).float().mean().item(), 3)
        decodes[name] = {"lm_head": [round((Dlm[ite, s] == yte).float().mean().item(), 3) for s in range(K + 1)],
                         "embedding": [round((Demb[ite, s] == yte).float().mean().item(), 3) for s in range(K + 1)],
                         "n_test": len(ite)}
        print(f"{name} (n={len(ite)}, majority {majority[name]}): " + " ".join(f"{s} {v:.3f}" for s, v in probes[name].items())
              + f" | lm-dec {decodes[name]['lm_head']} | emb-dec {decodes[name]['embedding']}", flush=True)
    numeric = {"lm_head": [round((Dlm[:, s] >= 0).float().mean().item(), 3) for s in range(K + 1)],
               "embedding": [round((Demb[:, s] >= 0).float().mean().item(), 3) for s in range(K + 1)]}
    print("share of decodes that are a number:", json.dumps(numeric), flush=True)
    if pa.smoke_n:
        print("not logging (smoke_n)")
        return

    elapsed = time.perf_counter() - t0
    records = [{"idx": i, "id": r["id"], "level": int(r["level"]), "gold_answer": r["answer"], "question": r["question"],
                "predicted_answer": None, "correct": None,
                "lm_head_decode": Dlm[i].tolist(), "embedding_decode": Demb[i].tolist()} for i, r in enumerate(test)]
    metrics = {"final_answer_accuracy": None, "unparseable_rate": None, "compute_steps": K,
               "sec_per_example": elapsed / (len(test) + len(train)),
               "extra": {"sites": site_names, "probe_exact": probes, "majority_baseline": majority, "decodes": decodes,
                         "decode_is_number": numeric, "probe": "two 10-way digit heads, linear softmax, L-BFGS, L2 on val slice",
                         "train_items": len(train), "train_json": pa.train_json,
                         "data_sha256": hashlib.sha256(Path(pa.test_json).read_bytes()).hexdigest(),
                         "matched_flags": flags}}
    record = RunRecord(
        run_id=new_run_id("codi", pa.slug), mechanism="codi", stage=pa.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=pa.checkpoint_label or model_args.ckpt_dir,
                        n_params=sum(p.numel() for p in model.parameters())),
        dataset=DatasetInfo(name="synthetic-depth2", split="test", n_examples=len(test), seed=None),
        metrics=metrics,
        hyperparams={"inference": {"latent_iterations": K, "batch_size": data_args.batch_size, "dtype": "float32",
                                   "device": device}, "train_per_level": pa.train_per_level, "probe_iters": pa.probe_iters},
        seed=None, hardware=f"{pa.hardware} / {device}",
        notes="RuleChain CODI latent-site x step value probes and LM-head / embedding decodes.")
    manifest = record.save(predictions=records)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
