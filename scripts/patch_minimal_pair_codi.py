#!/usr/bin/env python3
"""E3 (`steered_to_donor_audit.md` §5): same-problem minimal-pair donor-interchange
patching for CODI. Donor = the recipient's OWN question with one eligible number
perturbed and the gold chain re-executed (`latentreasoning.data.minimal_pairs`), so the
twin's final answer literally IS the counterfactual value (Metric A == Metric B by
construction) and E2's cross-problem confound is gone entirely.

Site indexing (fixed 2026-09-26). CODI feeds six latents into its loop: z_0 (latent-0,
the post-projection hidden state at the bot position after encoding the question) into
iteration 1, then z_s (iteration s's post-projection output) into iteration s+1, for
s = 1..5. z_6 is computed but never consumed -- `decode_answer` continues from the KV
cache with eot. `run_thoughts(..., override_input_at={i: v})` replaces the vector fed
INTO iteration i, so the aligned patch of site s is `{s + 1: twin z_s}`, s = 0..5. This
is the same site numbering as Coconut's passes 0..5 (`coconut_common.run_passes`: pass p
fills the p-th <|latent|> slot, pass 0 = the hidden state at <|start-latent|>).

The committed E3 run (20260920-190420) and E2/E4 used `{i: twin z_i}` for i = 1..6, i.e.
the twin's z_i placed where the recipient's z_{i-1} normally goes: one position early,
never transplanting z_0, and feeding z_6 (normally unused) into iteration 6. Those
conditions are kept here as `*_legacy` so the old numbers can be reproduced on the same
pairs.

Conditions per qualified pair (recipient and twin both answered correctly, greedy):
  all_slot            z_0..z_5 all from the twin (aligned)
  single_slot[s]      only z_s from the twin, s = 0..5
  prefix[s]           z_0..z_s from the twin (prefix[5] == all_slot)
  leave_one_out[s]    all sites from the twin except z_s -- is s NECESSARY for steering
  all_slot_legacy     {i: twin z_i}, i = 1..6 (old E3 all-slot)
  single_slot_legacy[i]  {i: twin z_i} (old E3 single-slot "iter i")
Non-responsive pairs (recipient correct, twin answered WRONG): all_slot only, scored
against the twin's own (wrong) prediction -- do the latents carry the twin's actual
computation, not just its correct answer?

Every candidate twin decoded is logged (`candidates.jsonl`) with the original's and the
twin's correctness, so counterfactual responsiveness can be stratified directly.

Run inside the CODI venv, from the CODI checkout:

  cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/patch_minimal_pair_codi.py \\
      --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \\
      --slug minimal-pair-patch-aligned --stage full_run --hardware "RunPod RTX A5000 (secure)" \\
      --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \\
      --lora_r 128 --lora_alpha 32 --lora_init --greedy True \\
      --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \\
      --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \\
      --eval_n 0 --n_pairs 1000 --n_nonresponsive 300
"""
from __future__ import annotations

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
from decode_patch_codi import decode_answer, run_thoughts, wilson_ci  # noqa: E402

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.data.minimal_pairs import generate_minimal_pair  # noqa: E402
from latentreasoning.eval.counterfactual import parse_steps, qualifying_steps  # noqa: E402
from latentreasoning.eval.metrics import extract_final_number, is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402

DELTAS = (1, -1, 2, -2, 3, -3)


@dataclass
class MPArguments:
    slug: str = field(default="minimal-pair-patch-aligned")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: Optional[str] = field(default=None)
    eval_n: int = field(default=0, metadata={"help": "0 = full test set"})
    eval_seed: int = field(default=0)
    n_pairs: int = field(default=1000, metadata={"help": "cap on qualified (both-correct) pairs patched"})
    n_nonresponsive: int = field(default=300, metadata={"help": "cap on twin-wrong pairs patched (all_slot only)"})
    max_new_tokens: int = field(default=64)
    pair_seed: int = field(default=0)
    smoke_n: int = field(default=0, metadata={"help": ">0: first n examples only, don't log"})


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, MPArguments))
    model_args, data_args, training_args, mpa = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = "cuda"

    model, tokenizer, load_result = build_model(model_args, training_args)
    assert not model.training and not any(m.training for m in model.modules()), "model must be in eval mode"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {model_args.ckpt_dir}: missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}")
    n_latents = training_args.inf_latent_iterations
    sites = list(range(n_latents))  # z_0 .. z_{n-1}; z_s feeds iteration s+1

    if mpa.eval_n:
        examples = load_gsm8k_aug(split="test", n=mpa.eval_n, seed=mpa.eval_seed)
    else:
        examples = load_gsm8k_aug(split="test", n=None, seed=None)
    if mpa.smoke_n:
        examples = examples[:mpa.smoke_n]
    print(f"n_examples={len(examples)}, n_latents={n_latents}")

    def decode_one(question: str):
        """Greedy answer + {iter: post-proj latent}, iter 0..n_latents (0 = latent-0)."""
        q = question.strip().replace("  ", " ")
        pkv, thoughts, latent = run_thoughts(model, tokenizer, q, device, n_latents, include_latent0=True)
        raw = decode_answer(model, tokenizer, pkv, latent, device, mpa.max_new_tokens)
        return extract_final_number(raw), {rec["iter"]: rec["post"] for rec in thoughts}

    def run_override(question: str, override: dict):
        q = question.strip().replace("  ", " ")
        pkv, _, latent = run_thoughts(model, tokenizer, q, device, n_latents, override_input_at=override)
        return extract_final_number(decode_answer(model, tokenizer, pkv, latent, device, mpa.max_new_tokens))

    def aligned(z: dict, site_set) -> dict:
        return {s + 1: z[s] for s in site_set}

    # ---- decode pass: base answer + correctness for every example ---------------------------
    t0 = time.perf_counter()
    base_info: dict[int, dict] = {}
    for ex in examples:
        pred, _ = decode_one(ex.question)
        base_info[ex.idx] = {"pred": pred, "correct": is_correct(pred, ex.answer),
                              "qualifying": qualifying_steps(parse_steps(ex.rationale))}
        if len(base_info) % 200 == 0:
            print(f"decoded {len(base_info)}/{len(examples)}", flush=True)
    decode_elapsed = time.perf_counter() - t0
    accuracy = sum(v["correct"] for v in base_info.values()) / len(base_info)
    print(f"decode pass {decode_elapsed:.0f}s, base accuracy={accuracy:.3f}")

    # ---- candidates from EVERY example (CFR needs base-wrong originals too) ------------------
    rng = random.Random(mpa.pair_seed)
    candidates = []
    for ex in examples:
        for k in base_info[ex.idx]["qualifying"]:
            for delta in DELTAS:
                mp = generate_minimal_pair(ex, k, delta, rng=rng)
                if mp is not None:
                    candidates.append(mp)
                    break
    rng.shuffle(candidates)
    print(f"minimal-pair candidates (propagation-qualified step, any original): {len(candidates)}")

    # ---- decode every twin; classify ---------------------------------------------------------
    t1 = time.perf_counter()
    cand_log, responsive, nonresponsive = [], [], []
    for ci, mp in enumerate(candidates):
        twin_pred, twin_z = decode_one(mp.twin.question)
        orig_ok = base_info[mp.original.idx]["correct"]
        twin_ok = is_correct(twin_pred, mp.twin.answer)
        cand_log.append({"recipient_idx": mp.original.idx, "step": mp.step, "delta": mp.delta,
                         "perturbed_number": mp.perturbed_number,
                         "chain_len": len(parse_steps(mp.original.rationale)),
                         "original_pred": base_info[mp.original.idx]["pred"], "original_correct": orig_ok,
                         "twin_answer": mp.twin.answer, "twin_pred": twin_pred, "twin_correct": twin_ok})
        if orig_ok and twin_ok and len(responsive) < mpa.n_pairs:
            responsive.append((mp, twin_pred, twin_z))
        elif orig_ok and not twin_ok and twin_pred is not None and len(nonresponsive) < mpa.n_nonresponsive:
            nonresponsive.append((mp, twin_pred, twin_z))
        if (ci + 1) % 200 == 0:
            print(f"twins decoded {ci + 1}/{len(candidates)} ({time.perf_counter() - t1:.0f}s): "
                  f"{len(responsive)} responsive, {len(nonresponsive)} non-responsive", flush=True)
    n_orig_ok = sum(c["original_correct"] for c in cand_log)
    print(f"CFR: {len(responsive)}/{n_orig_ok} twins correct given original correct "
          f"({sum(c['original_correct'] and c['twin_correct'] for c in cand_log)} before caps)")

    # ---- patch qualified pairs -----------------------------------------------------------------
    records = []
    t2 = time.perf_counter()
    for pi, (mp, twin_pred, z) in enumerate(responsive):
        ex = mp.original
        pred_base = base_info[ex.idx]["pred"]
        twin_answer = mp.twin.answer

        def cond(pred, pred_base=pred_base, twin_answer=twin_answer):
            return {"pred": pred, "answer_changed": pred != pred_base,
                    "matches_twin": is_correct(pred, twin_answer) and not is_correct(pred_base, twin_answer)}

        all_slot = cond(run_override(ex.question, aligned(z, sites)))
        single = {s: cond(run_override(ex.question, aligned(z, [s]))) for s in sites}
        prefix = {s: cond(run_override(ex.question, aligned(z, range(s + 1)))) for s in sites[:-1]}
        prefix[sites[-1]] = all_slot
        loo = {s: cond(run_override(ex.question, aligned(z, [t for t in sites if t != s]))) for s in sites}
        all_legacy = cond(run_override(ex.question, {i: z[i] for i in range(1, n_latents + 1)}))
        single_legacy = {i: cond(run_override(ex.question, {i: z[i]})) for i in range(1, n_latents + 1)}

        records.append({
            "recipient_idx": ex.idx, "step": mp.step, "perturbed_number": mp.perturbed_number,
            "delta": mp.delta, "answer_base": pred_base, "recipient_gold": ex.answer,
            "twin_question": mp.twin.question, "twin_answer": twin_answer, "twin_pred": twin_pred,
            "all_slot": all_slot, "single_slot": single, "prefix": prefix, "leave_one_out": loo,
            "all_slot_legacy": all_legacy, "single_slot_legacy": single_legacy,
        })
        if (pi + 1) % 25 == 0:
            print(f"patched {pi + 1}/{len(responsive)} pairs ({time.perf_counter() - t2:.0f}s)", flush=True)

    # ---- non-responsive pairs: does the patch carry the twin's WRONG answer? -------------------
    nr_records = []
    for mp, twin_pred, z in nonresponsive:
        ex = mp.original
        pred_base = base_info[ex.idx]["pred"]
        pred = run_override(ex.question, aligned(z, sites))
        nr_records.append({
            "recipient_idx": ex.idx, "step": mp.step, "delta": mp.delta, "answer_base": pred_base,
            "twin_answer": mp.twin.answer, "twin_pred": twin_pred, "pred": pred,
            "answer_changed": pred != pred_base,
            "matches_twin_pred": is_correct(pred, twin_pred) and not is_correct(pred_base, twin_pred),
            "matches_twin_gold": is_correct(pred, mp.twin.answer),
        })
    patch_elapsed = time.perf_counter() - t2

    # ---- aggregate -------------------------------------------------------------------------------
    def summarize(key_fn, rows=records) -> dict:
        n = len(rows)
        changed = sum(key_fn(r)["answer_changed"] for r in rows)
        matched = sum(key_fn(r)["matches_twin"] for r in rows)
        return {"n": n, "answer_changed_rate": changed / n if n else 0.0,
                "answer_changed_wilson_ci": list(wilson_ci(changed, n)),
                "matches_twin_rate": matched / n if n else 0.0,
                "matches_twin_wilson_ci": list(wilson_ci(matched, n))}

    summary = {
        "all_slot": summarize(lambda r: r["all_slot"]),
        "single_slot": {s: summarize(lambda r, s=s: r["single_slot"][s]) for s in sites},
        "prefix": {s: summarize(lambda r, s=s: r["prefix"][s]) for s in sites},
        "leave_one_out": {s: summarize(lambda r, s=s: r["leave_one_out"][s]) for s in sites},
        "all_slot_legacy": summarize(lambda r: r["all_slot_legacy"]),
        "single_slot_legacy": {i: summarize(lambda r, i=i: r["single_slot_legacy"][i]) for i in range(1, n_latents + 1)},
    }
    n_nr = len(nr_records)
    nr_matched = sum(r["matches_twin_pred"] for r in nr_records)
    nr_changed = sum(r["answer_changed"] for r in nr_records)
    nonresp_summary = {"n": n_nr, "matches_twin_pred_rate": nr_matched / n_nr if n_nr else None,
                       "matches_twin_pred_wilson_ci": list(wilson_ci(nr_matched, n_nr)),
                       "answer_changed_rate": nr_changed / n_nr if n_nr else None,
                       "matches_twin_gold_rate": sum(r["matches_twin_gold"] for r in nr_records) / n_nr if n_nr else None}
    n_both = sum(c["original_correct"] and c["twin_correct"] for c in cand_log)
    n_orig_wrong = len(cand_log) - n_orig_ok
    cfr = {"n_candidates": len(cand_log), "n_original_correct": n_orig_ok,
           "twin_correct_given_original_correct": n_both / n_orig_ok if n_orig_ok else None,
           "twin_correct_given_original_correct_wilson_ci": list(wilson_ci(n_both, n_orig_ok)),
           "twin_correct_given_original_wrong": (sum(c["twin_correct"] for c in cand_log if not c["original_correct"])
                                                 / n_orig_wrong) if n_orig_wrong else None,
           "both_correct_rate": n_both / len(cand_log) if cand_log else None}

    def line(name, s):
        return f"{name:22s} n={s['n']:4d} matches_twin={s['matches_twin_rate']:.3f} changed={s['answer_changed_rate']:.3f}"
    print(line("ALL-SLOT (aligned)", summary["all_slot"]))
    print(line("ALL-SLOT (legacy)", summary["all_slot_legacy"]))
    for s in sites:
        print(line(f"single z{s}", summary["single_slot"][s]) + " | " + line(f"leave-out z{s}", summary["leave_one_out"][s]))
    for s in sites:
        print(line(f"prefix z0..z{s}", summary["prefix"][s]))
    for i in range(1, n_latents + 1):
        print(line(f"single legacy iter{i}", summary["single_slot_legacy"][i]))
    print(f"non-responsive pairs n={n_nr}: matches_twin_pred={nonresp_summary['matches_twin_pred_rate']} "
          f"changed={nonresp_summary['answer_changed_rate']}")
    print(f"CFR: {json.dumps(cfr)}")

    if mpa.smoke_n:
        print("smoke run -- not logging")
        return

    metrics = {
        "final_answer_accuracy": accuracy,
        "compute_steps": n_latents,
        "sec_per_example": decode_elapsed / len(examples),
        "intervention_accuracy": summary["all_slot"]["matches_twin_rate"],
        "extra": {
            "design": "E3 rerun with aligned sites z_0..z_5 (z_s -> input of iteration s+1), leave-one-out, "
                      "legacy (pre-fix) conditions on the same pairs, non-responsive pairs, full CFR log",
            "site_definition": "site s = z_s = latent fed into loop iteration s+1; z_0 = latent-0 (bot position); "
                               "legacy iter i = twin z_i fed into iteration i",
            "n_examples_decoded": len(examples), "n_qualified_pairs": len(records),
            **summary, "nonresponsive": nonresp_summary, "cfr": cfr,
            "decode_elapsed_sec": decode_elapsed, "patch_elapsed_sec": patch_elapsed,
            "related_runs": ["20260920-190420_codi_minimal-pair-patch", "20260926-235221_codi_cfr-minimal-pairs"],
        },
    }
    record = RunRecord(
        run_id=new_run_id("codi", mpa.slug),
        mechanism="codi",
        stage=mpa.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=mpa.checkpoint_label or model_args.ckpt_dir, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(examples),
                            seed=mpa.eval_seed if mpa.eval_n else None),
        metrics=metrics,
        hyperparams={"inf_latent_iterations": n_latents, "greedy": training_args.greedy,
                     "n_pairs_cap": mpa.n_pairs, "n_nonresponsive_cap": mpa.n_nonresponsive,
                     "deltas": list(DELTAS), "max_new_tokens": mpa.max_new_tokens},
        seed=mpa.pair_seed,
        hardware=f"{mpa.hardware} / {torch.cuda.get_device_name(0)}",
        notes="E3 rerun, CODI: aligned-site same-problem minimal-pair patch (all/single/prefix/leave-one-out) "
              "+ legacy conditions + non-responsive pairs + CFR log.",
    )
    manifest = record.save(predictions=records)
    out_dir = manifest.parent
    with (out_dir / "candidates.jsonl").open("w") as f:
        for c in cand_log:
            f.write(json.dumps(c, default=str) + "\n")
    with (out_dir / "nonresponsive.jsonl").open("w") as f:
        for r in nr_records:
            f.write(json.dumps(r, default=str) + "\n")
    (out_dir / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
