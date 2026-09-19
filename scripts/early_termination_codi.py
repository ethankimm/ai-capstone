#!/usr/bin/env python3
"""What are CODI's six continuous thoughts FOR, if wiping the best-decoding one costs
nothing (results/20260919-090132_codi_ablate-attn)? Three questions on the released
checkpoint, in eval mode, greedy, batch 1:

  (i)  early termination: run only k in 0..6 thoughts, then eot -> answer. k=6 is the
       honest baseline. `early_termination_necessity` (PROPOSAL.md / manifest.py) is
       computed here as acc(k=6) - acc(k=0): the accuracy the full-budget model loses
       when its scratchpad is truncated to nothing at inference. The full curve is in
       metrics.extra.
  (ii) ablate-all: all six thought INPUTS replaced (zero vector / per-iteration
       population mean of the latent that normally feeds that iteration). The six
       positions still run, so the sequence layout is unchanged; only the content
       carried along the thought chain is removed.
  (iii) ablate-single: one iteration's input replaced by its population mean, for each
       iteration 1..6 -- does any single position matter on its own.

Reuses `run_thoughts` / `decode_answer` / `encode_question` from decode_patch_codi.py
(so the intervention mechanic is the exact one used by the patch/ablation runs:
`override_input_at={i: vec}` replaces the post-projection latent fed INTO iteration i).
Truncation reuses the honest chain: the KV cache is snapshotted after each iteration and
the answer decoded from every snapshot, which is byte-identical to running k iterations
from scratch (checked at startup against the plain path).

Run inside the CODI venv, from the CODI checkout (same convention as decode_patch_codi.py):

  cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/early_termination_codi.py \\
      --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \\
      --slug early-termination-ablate-all --stage full_run --hardware "RunPod RTX A5000 (secure)" \\
      --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \\
      --lora_r 128 --lora_alpha 32 --lora_init --greedy True \\
      --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \\
      --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \\
      --single_n 200 --mean_sample_n 300
"""
from __future__ import annotations

import copy
import json
import os
import random
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
from decode_patch_codi import (  # noqa: E402
    encode_question, run_thoughts, decode_answer, wilson_ci, mcnemar_exact_p,
)

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.eval.metrics import extract_final_number, is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402


@dataclass
class EtArguments:
    slug: str = field(default="early-termination-ablate-all")
    stage: str = field(default="full_run")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: Optional[str] = field(default=None)
    smoke_n: int = field(default=0, metadata={"help": ">0: run everything on the first n examples only, don't log"})
    single_n: int = field(default=200, metadata={"help": "ablate-single on the shared 200-slice (200) or the full test set (0)"})
    mean_sample_n: int = field(default=300)
    max_new_tokens: int = field(default=64)


def clone_cache(pkv):
    """Snapshot a KV cache so several continuations can branch from it."""
    return copy.deepcopy(pkv)


@torch.no_grad()
def initial_latent(model, tokenizer, question: str, device: str):
    """The post-projection latent that feeds iteration 1 (the bot-position hidden state),
    plus the KV cache after the question+bot pass. First lines of run_thoughts, verbatim."""
    input_ids, attn = encode_question(model, tokenizer, question, device)
    outputs = model.codi(input_ids=input_ids, use_cache=True, output_hidden_states=True,
                          past_key_values=None, attention_mask=attn)
    latent = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
    if model.use_prj:
        latent = model.prj(latent)
    return outputs.past_key_values, latent


@torch.no_grad()
def run_chain_with_snapshots(model, tokenizer, question: str, device: str, n_latents: int):
    """Honest chain; returns {k: (kv_after_k_iterations, latent_feeding_iteration_k+1)}
    for k = 0..n_latents. Same forward passes as run_thoughts(n_iters=k) for every k."""
    input_ids, attn = encode_question(model, tokenizer, question, device)
    outputs = model.codi(input_ids=input_ids, use_cache=True, output_hidden_states=True,
                          past_key_values=None, attention_mask=attn)
    pkv = outputs.past_key_values
    latent = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
    if model.use_prj:
        latent = model.prj(latent)
    snaps = {0: (clone_cache(pkv), latent)}
    for it in range(1, n_latents + 1):
        outputs = model.codi(inputs_embeds=latent, use_cache=True, output_hidden_states=True, past_key_values=pkv)
        pkv = outputs.past_key_values
        pre_hidden = outputs.hidden_states[-1][:, -1, :]
        latent = model.prj(pre_hidden.unsqueeze(1)) if model.use_prj else pre_hidden.unsqueeze(1)
        snaps[it] = (clone_cache(pkv) if it < n_latents else pkv, latent)
    return snaps


@torch.no_grad()
def population_mean_feeds(model, tokenizer, examples, device, n_latents, sample_n, seed=0):
    """mean_feed[i] = mean over sample_n examples of the latent that normally feeds
    iteration i (i=1: the bot-position latent; i>=2: iteration i-1's post-proj output)."""
    rng = random.Random(seed)
    pool = list(examples)
    rng.shuffle(pool)
    pool = pool[:sample_n]
    acc = {i: None for i in range(1, n_latents + 1)}
    for ex in pool:
        q = ex.question.strip().replace("  ", " ")
        _, lat0 = initial_latent(model, tokenizer, q, device)
        _, recs, _ = run_thoughts(model, tokenizer, q, device, n_latents)
        feeds = {1: lat0.detach().float()}
        for i in range(2, n_latents + 1):
            feeds[i] = recs[i - 2]["post"].detach().float()
        for i, v in feeds.items():
            acc[i] = v.clone() if acc[i] is None else acc[i] + v
    dtype = next(model.parameters()).dtype
    return {i: (acc[i] / len(pool)).to(device=device, dtype=dtype) for i in acc}, len(pool)


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, EtArguments))
    model_args, data_args, training_args, et_args = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = "cuda"

    model, tokenizer, load_result = build_model(model_args, training_args)
    assert not model.training and not any(m.training for m in model.modules()), \
        "model (or a submodule) is in training mode -- build_model must call model.eval()"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {model_args.ckpt_dir}: missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}; eval mode OK")
    n_latents = training_args.inf_latent_iterations

    all_test = load_gsm8k_aug(split="test", n=None, seed=None)
    slice_idx = {ex.idx for ex in load_gsm8k_aug(split="test", n=200, seed=0)}
    examples = all_test[:et_args.smoke_n] if et_args.smoke_n else all_test
    if et_args.single_n == 0 or et_args.smoke_n:
        single_examples = examples
    else:
        single_examples = [ex for ex in all_test if ex.idx in slice_idx]
    print(f"n_examples={len(examples)} (trunc + ablate-all), n_single={len(single_examples)} (ablate-single), 200-slice n={len(slice_idx)}")

    def q_of(ex):
        return ex.question.strip().replace("  ", " ")

    # ---- determinism + snapshot-equivalence check on 5 examples --------------------------
    t0 = time.perf_counter()
    det_ok, snap_ok = True, True
    for ex in examples[:5]:
        q = q_of(ex)
        outs = []
        for _ in range(2):
            pkv, _, lat = run_thoughts(model, tokenizer, q, device, n_latents)
            outs.append(decode_answer(model, tokenizer, pkv, lat, device, et_args.max_new_tokens))
        det_ok &= outs[0] == outs[1]
        snaps = run_chain_with_snapshots(model, tokenizer, q, device, n_latents)
        for k in (0, 3, n_latents):
            pkv_k, _, lat_k = run_thoughts(model, tokenizer, q, device, k)
            plain = decode_answer(model, tokenizer, pkv_k, lat_k, device, et_args.max_new_tokens)
            kv_s, lat_s = snaps[k]
            via_snap = decode_answer(model, tokenizer, clone_cache(kv_s), lat_s, device, et_args.max_new_tokens)
            snap_ok &= plain == via_snap
    print(f"determinism (same input decoded twice, identical): {det_ok}; "
          f"snapshot path == plain run_thoughts(k) path at k in (0,3,{n_latents}): {snap_ok} "
          f"({time.perf_counter() - t0:.1f}s)")
    assert det_ok, "greedy decode is not deterministic -- refusing to run"
    assert snap_ok, "KV-snapshot truncation differs from the plain path -- refusing to run"

    # ---- population means of each iteration's input ---------------------------------------
    t0 = time.perf_counter()
    mean_feed, mean_n = population_mean_feeds(model, tokenizer, all_test, device, n_latents, et_args.mean_sample_n)
    zero_vec = torch.zeros_like(mean_feed[1])
    print(f"per-iteration mean input vectors from {mean_n} examples ({time.perf_counter() - t0:.1f}s); "
          f"norms: " + ", ".join(f"i{i}={mean_feed[i].float().norm().item():.2f}" for i in mean_feed))

    trunc_conds = [f"trunc_{k}" for k in range(0, n_latents + 1)]
    all_conds = ["ablate_all_zero", "ablate_all_mean"]
    single_conds = [f"ablate_single_mean_{i}" for i in range(1, n_latents + 1)]
    baseline_cond = f"trunc_{n_latents}"

    # ---- pass 1: truncation sweep + ablate-all on `examples` ------------------------------
    records: dict[str, dict[int, dict]] = {c: {} for c in trunc_conds + all_conds + single_conds}
    t1 = time.perf_counter()
    for n_done, ex in enumerate(examples, 1):
        q = q_of(ex)
        snaps = run_chain_with_snapshots(model, tokenizer, q, device, n_latents)
        for k in range(0, n_latents + 1):
            kv_k, lat_k = snaps[k]
            raw = decode_answer(model, tokenizer, kv_k if k == n_latents else clone_cache(kv_k), lat_k, device, et_args.max_new_tokens)
            pred = extract_final_number(raw)
            records[f"trunc_{k}"][ex.idx] = {"raw_output": raw, "predicted_answer": pred, "correct": is_correct(pred, ex.answer)}
        for cond, vec_for in (("ablate_all_zero", lambda i: zero_vec), ("ablate_all_mean", lambda i: mean_feed[i])):
            pkv, _, lat = run_thoughts(model, tokenizer, q, device, n_latents,
                                       override_input_at={i: vec_for(i) for i in range(1, n_latents + 1)})
            raw = decode_answer(model, tokenizer, pkv, lat, device, et_args.max_new_tokens)
            pred = extract_final_number(raw)
            records[cond][ex.idx] = {"raw_output": raw, "predicted_answer": pred, "correct": is_correct(pred, ex.answer)}
        if n_done % 100 == 0 or n_done == len(examples):
            el = time.perf_counter() - t1
            print(f"pass1 {n_done}/{len(examples)} ({el:.0f}s, {el / n_done:.2f}s/ex, "
                  f"acc@{baseline_cond}={sum(r['correct'] for r in records[baseline_cond].values()) / n_done:.3f})", flush=True)
    pass1_elapsed = time.perf_counter() - t1

    # ---- pass 2: ablate-single (mean) at each iteration on `single_examples` ---------------
    t2 = time.perf_counter()
    for n_done, ex in enumerate(single_examples, 1):
        q = q_of(ex)
        for i in range(1, n_latents + 1):
            pkv, _, lat = run_thoughts(model, tokenizer, q, device, n_latents, override_input_at={i: mean_feed[i]})
            raw = decode_answer(model, tokenizer, pkv, lat, device, et_args.max_new_tokens)
            pred = extract_final_number(raw)
            records[f"ablate_single_mean_{i}"][ex.idx] = {"raw_output": raw, "predicted_answer": pred, "correct": is_correct(pred, ex.answer)}
        if n_done % 100 == 0 or n_done == len(single_examples):
            el = time.perf_counter() - t2
            print(f"pass2 {n_done}/{len(single_examples)} ({el:.0f}s, {el / n_done:.2f}s/ex)", flush=True)
    pass2_elapsed = time.perf_counter() - t2

    # ---- summarise -------------------------------------------------------------------------
    gold = {ex.idx: ex.answer for ex in all_test}
    base = records[baseline_cond]

    def summarise(cond: str, idx_filter=None) -> dict:
        rows = {i: r for i, r in records[cond].items() if idx_filter is None or i in idx_filter}
        n = len(rows)
        if n == 0:
            return {"n": 0}
        correct = sum(r["correct"] for r in rows.values())
        unparse = sum(r["predicted_answer"] is None for r in rows.values())
        common = [i for i in rows if i in base]
        b = sum(1 for i in common if rows[i]["correct"] and not base[i]["correct"])
        c = sum(1 for i in common if base[i]["correct"] and not rows[i]["correct"])
        changed = sum(1 for i in common if rows[i]["predicted_answer"] != base[i]["predicted_answer"])
        return {
            "n": n, "accuracy": correct / n, "accuracy_wilson_ci": list(wilson_ci(correct, n)),
            "unparseable_rate": unparse / n,
            "answer_changed_vs_baseline": changed / len(common) if common else None,
            "mcnemar_vs_baseline": {"b_cond_right_base_wrong": b, "c_base_right_cond_wrong": c, "p_exact": mcnemar_exact_p(b, c)},
        }

    conds = trunc_conds + all_conds + single_conds
    summary_full = {c: summarise(c) for c in conds}
    summary_slice = {c: summarise(c, slice_idx) for c in conds}
    acc6 = summary_full[baseline_cond]["accuracy"]
    curve = {k: summary_full[f"trunc_{k}"]["accuracy"] for k in range(0, n_latents + 1)}
    print("\ncondition                 n     acc    CI              unparse  changed  McNemar(b,c,p)     | 200-slice acc")
    for c in conds:
        s, sl = summary_full[c], summary_slice[c]
        m = s["mcnemar_vs_baseline"]
        print(f"{c:24s} {s['n']:5d}  {s['accuracy']:.3f}  [{s['accuracy_wilson_ci'][0]:.3f},{s['accuracy_wilson_ci'][1]:.3f}]  "
              f"{s['unparseable_rate']:.3f}    {s['answer_changed_vs_baseline']:.3f}    ({m['b_cond_right_base_wrong']},{m['c_base_right_cond_wrong']},{m['p_exact']:.3g})"
              f"   | {sl.get('accuracy', float('nan')):.3f} (n={sl['n']})")
    print(f"\nearly_termination_necessity = acc(k={n_latents}) - acc(k=0) = {acc6:.4f} - {curve[0]:.4f} = {acc6 - curve[0]:.4f}")

    if et_args.smoke_n:
        print("smoke run -- not logging")
        return

    metrics = {
        "final_answer_accuracy": acc6,
        "unparseable_rate": summary_full[baseline_cond]["unparseable_rate"],
        "compute_steps": n_latents,
        "sec_per_example": pass1_elapsed / len(examples),
        "early_termination_necessity": acc6 - curve[0],
        "extra": {
            "early_termination_necessity_definition": f"acc(trunc_{n_latents}) - acc(trunc_0): accuracy lost when the full-budget model's thoughts are truncated to zero at inference (greedy, batch 1, full test set)",
            "truncation_curve_accuracy_by_k": curve,
            "truncation_drop_by_k": {k: acc6 - curve[k] for k in curve},
            "conditions_full": summary_full,
            "conditions_200_slice": summary_slice,
            "n_examples_trunc_and_ablate_all": len(examples),
            "n_examples_ablate_single": len(single_examples),
            "mean_sample_n": mean_n,
            "mean_feed_norms": {i: mean_feed[i].float().norm().item() for i in mean_feed},
            "determinism_check_passed": det_ok,
            "snapshot_equivalence_check_passed": snap_ok,
            "pass1_elapsed_sec": pass1_elapsed,
            "pass2_elapsed_sec": pass2_elapsed,
            "prior_run_ids": ["20260919-090132_codi_ablate-attn", "20260919-080349_codi_decode-patch-full",
                              "20260918-021217_codi_released-weights-6lat"],
        },
    }
    # one predictions.jsonl row per (example, condition), so the whitelisted file holds everything
    predictions = []
    for c in conds:
        for i, r in records[c].items():
            predictions.append({"idx": i, "condition": c, "gold_answer": gold[i], **r, "in_200_slice": i in slice_idx})

    record = RunRecord(
        run_id=new_run_id("codi", et_args.slug),
        mechanism="codi",
        stage=et_args.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=et_args.checkpoint_label or model_args.ckpt_dir, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(examples), seed=None),
        metrics=metrics,
        hyperparams={"inf_latent_iterations": n_latents, "greedy": training_args.greedy,
                     "num_latent": training_args.num_latent, "single_n": len(single_examples),
                     "mean_sample_n": mean_n, "max_new_tokens": et_args.max_new_tokens},
        seed=None,
        hardware=f"{et_args.hardware} / {torch.cuda.get_device_name(0)}",
        notes="Early termination (k=0..6 thoughts at inference) + ablate-all (zero / per-iteration mean inputs) "
              "+ ablate-single (mean) per iteration, released CODI checkpoint, eval mode, greedy batch 1.",
    )
    manifest_out = record.save(predictions=predictions)
    out_dir = manifest_out.parent
    (out_dir / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
