#!/usr/bin/env python3
"""Necessity of CODI's latents under every replacement type (Q2 "necessary?" cell):
early termination k=0..6, ablate-all and ablate-single with zero / population-mean /
norm-matched Gaussian noise replacement, plus sampled pass@k on the ablate-all
conditions. Extends `early_termination_codi.py` (20260919-192228: truncation, ablate-all
zero/mean, single mean on the 200-slice) to noise, single zero/noise, the full test set
for every condition, PWC + flip split, and pass@k. Scoring lives in `necessity_common.py`.

Sites are z_0..z_5, z_s = the latent fed into loop iteration s+1 (z_0 = latent-0, the
bot-position latent), same numbering as Coconut passes 0..5 and as
`patch_minimal_pair_codi.py`. Replacements for site s:
  zero   all-zeros vector
  mean   population mean of z_s over `mean_sample_n` test examples (seeded; identical to
         early_termination_codi.population_mean_feeds)
  noise  N(0, I) direction scaled to the norm of THIS example's clean z_s (seeded per
         example and site) -- matches scale, removes content
Ablate-all replaces z_0..z_5 at once (positions still run); ablate-single one site.

pass@k: `k_samples` answers sampled at `temperature` from the same (deterministic)
latent state -- only the answer tokens are stochastic. Conditions: baseline (trunc_6),
trunc_0, ablate_all_{zero,mean,noise}. Does the ablation remove the capability, or only
move the greedy mode?

Run inside the CODI venv, from the CODI checkout (4 shards in parallel on one GPU):

  cd /workspace/codi && for i in 0 1 2 3; do nohup .venv/bin/python /workspace/ai-capstone/scripts/necessity_codi.py \\
      --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \\
      --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \\
      --lora_r 128 --lora_alpha 32 --lora_init --greedy True \\
      --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \\
      --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \\
      --num_shards 4 --shard_index $i --shard_dir /workspace/necessity_codi > nec_codi_$i.log 2>&1 & done
  # then, same flags plus:  --merge_dir /workspace/necessity_codi --slug necessity-all-replacements \\
  #     --stage full_run --hardware "RunPod RTX A5000 (secure)"
"""
from __future__ import annotations

import copy
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())  # the CODI checkout: `src.model`

from src.model import DataArguments, ModelArguments, TrainingArguments  # noqa: E402
from eval_codi import build_model  # noqa: E402
from decode_patch_codi import decode_answer, run_thoughts  # noqa: E402
from early_termination_codi import clone_cache, population_mean_feeds, run_chain_with_snapshots  # noqa: E402
import necessity_common as nc  # noqa: E402

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.eval.metrics import extract_final_number, is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402


@dataclass
class NecArguments:
    slug: str = field(default="necessity-all-replacements")
    stage: str = field(default="full_run")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: Optional[str] = field(default=None)
    smoke_n: int = field(default=0, metadata={"help": ">0: first n examples, print, don't save"})
    mean_sample_n: int = field(default=300)
    max_new_tokens: int = field(default=64)
    k_samples: int = field(default=10)
    temperature: float = field(default=0.7)
    num_shards: int = field(default=1)
    shard_index: int = field(default=0)
    shard_dir: str = field(default="/workspace/necessity_codi")
    merge_dir: Optional[str] = field(default=None)


def expand_cache(pkv, k: int):
    pkv = copy.deepcopy(pkv)
    if hasattr(pkv, "batch_repeat_interleave"):
        pkv.batch_repeat_interleave(k)
        return pkv
    return tuple((a.repeat_interleave(k, dim=0), b.repeat_interleave(k, dim=0)) for a, b in pkv)


@torch.no_grad()
def sample_answers(model, tokenizer, pkv, device, k: int, temperature: float, max_new_tokens: int,
                   gen: torch.Generator) -> list[str]:
    """`decode_answer`'s tail, batched over k temperature samples from one KV cache."""
    pkv = expand_cache(pkv, k)
    embed = model.get_embd(model.codi, model.model_name)
    output = embed(torch.tensor([model.eot_id], dtype=torch.long, device=device)).unsqueeze(0).expand(k, -1, -1)
    done = torch.zeros(k, dtype=torch.bool, device=device)
    toks: list[list[int]] = [[] for _ in range(k)]
    for _ in range(max_new_tokens):
        out = model.codi(inputs_embeds=output, output_hidden_states=False, attention_mask=None,
                          use_cache=True, past_key_values=pkv)
        pkv = out.past_key_values
        logits = out.logits[:, -1, :model.codi.config.vocab_size - 1].float() / temperature
        ids = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=gen)  # [k, 1]
        for r in range(k):
            if not done[r]:
                if ids[r, 0].item() == tokenizer.eos_token_id:
                    done[r] = True
                else:
                    toks[r].append(ids[r, 0].item())
        if bool(done.all()):
            break
        output = embed(ids).to(device)  # [k, 1, D]
    return [tokenizer.decode(t, skip_special_tokens=True) for t in toks]


def finalize(merged: dict, na: NecArguments, argv: str) -> None:
    meta = merged["meta"]
    n_sites = meta["n_sites"]
    records, samples = merged["records"], merged["samples"]
    baseline = f"trunc_{n_sites}"
    summary = nc.summarize(records, samples, baseline, meta["k_samples"])
    nc.print_table(summary, meta["k_samples"])
    curve = {k: summary[f"trunc_{k}"]["accuracy"] for k in range(n_sites + 1)}
    acc = summary[baseline]["accuracy"]
    n = summary[baseline]["n"]
    print(f"early_termination_necessity = {acc:.4f} - {curve[0]:.4f} = {acc - curve[0]:.4f}")
    if na.smoke_n:
        return
    all_test = {ex.idx: ex.answer for ex in load_gsm8k_aug(split="test", n=None, seed=None)}
    metrics = {
        "final_answer_accuracy": acc,
        "unparseable_rate": summary[baseline]["unparseable_rate"],
        "compute_steps": n_sites,
        "sec_per_example": sum(merged["elapsed_sec"]) / n,
        "early_termination_necessity": acc - curve[0],
        "extra": {
            "early_termination_necessity_definition": f"acc(trunc_{n_sites}) - acc(trunc_0), greedy, full test set",
            "site_definition": "z_s = latent fed into loop iteration s+1; z_0 = latent-0 (bot position)",
            "replacements": {"zero": "all-zeros", "mean": f"population mean of z_s over {meta['mean_sample_n']} test examples",
                             "noise": "Gaussian direction scaled to this example's clean ||z_s||, seed idx*1000+s"},
            "pwc_definition": "P(condition correct | baseline trunc_6 correct)",
            "sampling": {"k": meta["k_samples"], "temperature": meta["temperature"],
                         "what_is_sampled": "answer tokens only; latents are deterministic",
                         "conditions": nc.sampled_condition_names(n_sites)},
            "truncation_curve_accuracy_by_k": curve,
            "conditions": summary,
            "mean_norms": meta["mean_norms"],
            "num_shards": meta["num_shards"],
            "prior_run_ids": ["20260919-192228_codi_early-termination-ablate-all"],
        },
    }
    record = RunRecord(
        run_id=new_run_id("codi", na.slug),
        mechanism="codi",
        stage=na.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=meta["checkpoint"], n_params=meta["n_params"]),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=n, seed=None),
        metrics=metrics,
        hyperparams={"inf_latent_iterations": n_sites, "greedy": True, "mean_sample_n": meta["mean_sample_n"],
                     "max_new_tokens": meta["max_new_tokens"], "k_samples": meta["k_samples"],
                     "temperature": meta["temperature"]},
        seed=0,
        hardware=f"{na.hardware} / {meta['gpu']}",
        notes="Necessity, CODI: early termination + ablate-all/single x {zero, mean, norm-matched noise}, "
              "full test set, PWC + flip split, pass@k on ablate-all.",
    )
    path = record.save(predictions=nc.predictions_rows(records, samples, all_test))
    (path.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {path.parent / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, NecArguments))
    model_args, data_args, training_args, na = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    if na.merge_dir:
        finalize(nc.load_shards(na.merge_dir), na, argv)
        return

    device = "cuda"
    model, tokenizer, _ = build_model(model_args, training_args)
    assert not model.training and not any(m.training for m in model.modules()), "model must be in eval mode"
    n_sites = training_args.inf_latent_iterations
    all_test = load_gsm8k_aug(split="test", n=None, seed=None)
    examples = all_test[:na.smoke_n] if na.smoke_n else all_test[na.shard_index::na.num_shards]
    print(f"shard {na.shard_index}/{na.num_shards}: {len(examples)} examples")

    mean_feed, mean_n = population_mean_feeds(model, tokenizer, all_test, device, n_sites, na.mean_sample_n)
    mean_z = {s: mean_feed[s + 1] for s in range(n_sites)}  # feed i = z_{i-1}
    zero = torch.zeros_like(mean_z[0])
    mean_norms = [mean_z[s].float().norm().item() for s in range(n_sites)]
    print("mean z norms: " + ", ".join(f"z{s}={v:.2f}" for s, v in enumerate(mean_norms)))

    def greedy(pkv) -> dict:
        pred = extract_final_number(decode_answer(model, tokenizer, pkv, None, device, na.max_new_tokens))
        return pred

    def sampled(pkv, ex, cond_seed: int) -> list[bool]:
        gen = torch.Generator(device=device).manual_seed(cond_seed)
        raws = sample_answers(model, tokenizer, pkv, device, na.k_samples, na.temperature, na.max_new_tokens, gen)
        return [is_correct(extract_final_number(r), ex.answer) for r in raws]

    conds = nc.condition_names(n_sites)
    sampled_conds = nc.sampled_condition_names(n_sites)
    records = {c: {} for c in conds}
    samples = {c: {} for c in sampled_conds}
    t0 = time.perf_counter()
    for n_done, ex in enumerate(examples, 1):
        q = ex.question.strip().replace("  ", " ")

        def put(cond, pred):
            records[cond][ex.idx] = {"pred": pred, "correct": is_correct(pred, ex.answer)}

        snaps = run_chain_with_snapshots(model, tokenizer, q, device, n_sites)
        clean_z = {s: snaps[s][1] for s in range(n_sites)}
        for k in range(n_sites + 1):
            kv = snaps[k][0]
            if f"trunc_{k}" in samples:
                samples[f"trunc_{k}"][ex.idx] = sampled(kv, ex, ex.idx * 100 + k)
            put(f"trunc_{k}", greedy(clone_cache(kv)))

        noise_z = {}
        for s in range(n_sites):
            g = torch.Generator().manual_seed(ex.idx * 1000 + s)
            d = torch.randn(clean_z[s].numel(), generator=g)
            d = d / d.norm() * clean_z[s].float().norm().cpu()
            noise_z[s] = d.view_as(clean_z[s]).to(device=device, dtype=clean_z[s].dtype)
        repl = {"zero": lambda s: zero, "mean": lambda s: mean_z[s], "noise": lambda s: noise_z[s]}

        for ri, (r, vec) in enumerate(repl.items()):
            pkv, _, _ = run_thoughts(model, tokenizer, q, device, n_sites,
                                     override_input_at={s + 1: vec(s) for s in range(n_sites)})
            samples[f"ablate_all_{r}"][ex.idx] = sampled(pkv, ex, ex.idx * 100 + 50 + ri)
            put(f"ablate_all_{r}", greedy(pkv))
            for s in range(n_sites):
                pkv, _, _ = run_thoughts(model, tokenizer, q, device, n_sites, override_input_at={s + 1: vec(s)})
                put(f"ablate_single_{r}_{s}", greedy(pkv))

        if n_done % 25 == 0 or n_done == len(examples):
            el = time.perf_counter() - t0
            acc = sum(v["correct"] for v in records[f"trunc_{n_sites}"].values()) / n_done
            print(f"{n_done}/{len(examples)} ({el:.0f}s, {el / n_done:.2f}s/ex, baseline acc {acc:.3f})", flush=True)
    elapsed = time.perf_counter() - t0

    meta = {"n_sites": n_sites, "num_shards": na.num_shards, "mean_sample_n": mean_n, "mean_norms": mean_norms,
            "k_samples": na.k_samples, "temperature": na.temperature, "max_new_tokens": na.max_new_tokens,
            "checkpoint": na.checkpoint_label or model_args.ckpt_dir,
            "n_params": sum(p.numel() for p in model.parameters()),
            "gpu": torch.cuda.get_device_name(0), "elapsed_sec": elapsed}
    payload = {"meta": meta, "records": records, "samples": samples}
    if na.smoke_n:
        finalize({"meta": meta, "records": records, "samples": samples, "elapsed_sec": [elapsed]}, na, argv)
        return
    path = nc.write_shard(na.shard_dir, na.shard_index, payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
