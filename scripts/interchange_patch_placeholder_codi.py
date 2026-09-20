#!/usr/bin/env python3
"""CODI donor-interchange patching at the non-decodable "placeholder" positions
(z0, z3 -- the inputs to iterations 1 and 4) vs the decodable ones (z2, z4 -- inputs
to iterations 3 and 5), single-slot and jointly-grouped.

Follow-up to `results/20260919-184323_codi_decode-patch-full-eval` (single-slot
interchange patching at the *decodable* focus iteration only -- null result, p in
[0.27, 0.91]) and `results/20260919-192228_codi_early-termination-ablate-all`
(mean-ablation showed z0/z3 are load-bearing, p=6e-5 / p=0.0025, while z2/z4 are not).
That run answered "does content at z0/z3 matter" with a content-free mean vector; this
one asks the sharper question with a content-*bearing* donor vector: does swapping in
ANOTHER example's z0/z3 steer the answer toward that donor's answer?

Two outcomes distinguish the paper's central claim:
  A) donor-swap at z0/z3 steers the answer toward the donor -> the placeholder slots
     carry transferable, semantically structured state; the logit lens is just looking
     at the wrong basis / position.
  B) donor-swap at z0/z3 does NOT steer the answer (despite z0/z3 being load-bearing
     under ablation) -> the computation there is non-representational / distributed /
     non-linearly entangled -- necessary but not portable the way the positive control
     (explicit CoT, `20260919-185332_explicit_cot_patch-positive-control`) is.

Unlike `decode_patch_codi.py` (which only ever patches at whichever iteration wins a
step's logit-lens accuracy -- by construction one of the *decodable* iterations),
z0/z3 never win any step, so this script drives `override_input_at` directly by
iteration index and selects donor/recipient pairs by DIFFERENT FINAL ANSWERS (there is
no legible "value" at these positions to match on).

Reuses `run_thoughts` / `decode_answer` / `topk_token_strings` / `wilson_ci` /
`mcnemar_exact_p` from `decode_patch_codi.py` -- same intervention mechanic as every
other patch/ablation run in this repo.

Run inside the CODI venv, from the CODI checkout (same convention as decode_patch_codi.py):

  cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/interchange_patch_placeholder_codi.py \\
      --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \\
      --slug interchange-placeholder-pilot --stage pilot --hardware "RunPod RTX A5000 (secure)" \\
      --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \\
      --lora_r 128 --lora_alpha 32 --lora_init --greedy True \\
      --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \\
      --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \\
      --full_test False --eval_n 200 --n_pairs_per_site 60
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())  # the CODI checkout: `src.model`

from src.model import DataArguments, ModelArguments, TrainingArguments  # noqa: E402
from eval_codi import build_model  # noqa: E402
from decode_patch_codi import (  # noqa: E402
    run_thoughts, decode_answer, topk_token_strings, wilson_ci, mcnemar_exact_p,
)

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.eval.metrics import extract_final_number, is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402

# iter index (1..6) whose INPUT is z_{iter-1}: iter 1 <- z0 (bot latent), iter 4 <- z3.
NONDECODABLE_ITERS = (1, 4)  # z0, z3 -- odd/placeholder positions, load-bearing under ablation
DECODABLE_ITERS = (3, 5)     # z2, z4 -- best logit-lens decode positions, null under single patch
ALL_SINGLE_ITERS = (1, 2, 3, 4, 5, 6)


@dataclass
class InterchangeArguments:
    slug: str = field(default="interchange-placeholder")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: Optional[str] = field(default=None)
    full_test: bool = field(default=False)
    eval_n: int = field(default=200)
    eval_seed: int = field(default=0)
    n_pairs_per_site: int = field(default=60, metadata={"help": "recipient/donor pairs for each single-iter site"})
    n_grouped_pairs: int = field(default=100, metadata={"help": "recipient/donor pairs for the grouped {z0,z3} vs {z2,z4} comparison"})
    max_new_tokens: int = field(default=64)
    pair_seed: int = field(default=0)


def answer_matches(pred: Optional[str], gold: str) -> bool:
    return is_correct(pred, gold)


def sample_pairs(examples: list, n: int, rng: random.Random) -> list[tuple]:
    """(recipient, donor) pairs with DIFFERENT gold final answers, sampled without
    requiring recipient == a fixed pool -- donors are drawn from the same pool."""
    pool = list(examples)
    pairs = []
    attempts = 0
    max_attempts = n * 20
    while len(pairs) < n and attempts < max_attempts:
        attempts += 1
        r, d = rng.sample(pool, 2)
        if r.answer.strip() != d.answer.strip():
            pairs.append((r, d))
    return pairs


@torch.no_grad()
def get_thought_cache(model, tokenizer, ex, device, n_latents, cache: dict):
    if ex.idx not in cache:
        q = ex.question.strip().replace("  ", " ")
        pkv, recs, latent = run_thoughts(model, tokenizer, q, device, n_latents)
        cache[ex.idx] = recs
    return cache[ex.idx]


@torch.no_grad()
def patched_answer(model, tokenizer, recipient, device, n_latents, override: dict, max_new_tokens: int) -> str:
    q = recipient.question.strip().replace("  ", " ")
    pkv, thoughts, latent = run_thoughts(model, tokenizer, q, device, n_latents, override_input_at=override)
    return decode_answer(model, tokenizer, pkv, latent, device, max_new_tokens)


def summarize(records: list[dict], label: str) -> dict:
    n = len(records)
    x_changed = sum(r["answer_changed"] for r in records)
    x_steered = sum(r["steered_to_donor"] for r in records if r["steered_to_donor"] is not None)
    n_steered_eligible = sum(1 for r in records if r["steered_to_donor"] is not None)
    return {
        "label": label, "n": n,
        "answer_changed_rate": x_changed / n if n else 0.0,
        "answer_changed_wilson_ci": list(wilson_ci(x_changed, n)),
        "steered_to_donor_rate": x_steered / n_steered_eligible if n_steered_eligible else 0.0,
        "steered_to_donor_wilson_ci": list(wilson_ci(x_steered, n_steered_eligible)),
        "n_steered_eligible": n_steered_eligible,
    }


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, InterchangeArguments))
    model_args, data_args, training_args, ia = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = "cuda"

    model, tokenizer, load_result = build_model(model_args, training_args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {model_args.ckpt_dir}: missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}")

    if ia.full_test:
        examples = load_gsm8k_aug(split="test", n=None, seed=None)
    else:
        examples = load_gsm8k_aug(split="test", n=ia.eval_n, seed=ia.eval_seed)
    n_latents = training_args.inf_latent_iterations
    print(f"n_examples={len(examples)}")

    thought_cache: dict = {}
    rng = random.Random(ia.pair_seed)

    def donor_vecs(donor, iters: tuple[int, ...]) -> dict:
        recs = get_thought_cache(model, tokenizer, donor, device, n_latents, thought_cache)
        return {it: recs[it - 1]["post"] for it in iters}

    def base_answer(recipient) -> str:
        q = recipient.question.strip().replace("  ", " ")
        pkv, thoughts, latent = run_thoughts(model, tokenizer, q, device, n_latents)
        return decode_answer(model, tokenizer, pkv, latent, device, ia.max_new_tokens)

    # ---- Task 1: single-slot donor-interchange patching at every iteration -----------------
    single_results: dict[int, list[dict]] = {it: [] for it in ALL_SINGLE_ITERS}
    t0 = time.perf_counter()
    n_done = 0
    for it in ALL_SINGLE_ITERS:
        pairs = sample_pairs(examples, ia.n_pairs_per_site, random.Random(ia.pair_seed * 1000 + it))
        for recipient, donor in pairs:
            out_base = base_answer(recipient)
            pred_base = extract_final_number(out_base)
            override = donor_vecs(donor, (it,))
            out_patched = patched_answer(model, tokenizer, recipient, device, n_latents, override, ia.max_new_tokens)
            pred_patched = extract_final_number(out_patched)
            steered = None
            if pred_base is not None:
                steered = answer_matches(pred_patched, donor.answer) and not answer_matches(pred_base, donor.answer)
            single_results[it].append({
                "recipient_idx": recipient.idx, "donor_idx": donor.idx,
                "recipient_gold": recipient.answer, "donor_gold": donor.answer,
                "answer_base": pred_base, "answer_patched": pred_patched,
                "answer_changed": pred_patched != pred_base,
                "steered_to_donor": steered,
            })
            n_done += 1
            if n_done % 50 == 0:
                print(f"single-slot: {n_done}/{len(ALL_SINGLE_ITERS) * ia.n_pairs_per_site} done "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)

    single_summaries = {it: summarize(single_results[it], f"iter{it}") for it in ALL_SINGLE_ITERS}
    for it, s in single_summaries.items():
        tag = "NONDECODABLE" if it in NONDECODABLE_ITERS else ("decodable" if it in DECODABLE_ITERS else "other")
        print(f"iter={it} ({tag}): answer_changed={s['answer_changed_rate']:.3f} "
              f"steered_to_donor={s['steered_to_donor_rate']:.3f} (n={s['n']}, "
              f"n_steered_eligible={s['n_steered_eligible']})")

    # McNemar: nondecodable (pooled iters 1,4) vs decodable (pooled iters 3,5) single-slot patch
    nondec_pooled = single_results[NONDECODABLE_ITERS[0]] + single_results[NONDECODABLE_ITERS[1]]
    dec_pooled = single_results[DECODABLE_ITERS[0]] + single_results[DECODABLE_ITERS[1]]
    # unpaired (different recipient/donor draws per site) -- report as independent-samples
    # contrast via Wilson CIs; McNemar needs the SAME pairs, which Task 2 provides.

    # ---- Task 2: grouped multi-slot patching -- same donor, same recipient, two conditions --
    grouped_pairs = sample_pairs(examples, ia.n_grouped_pairs, random.Random(ia.pair_seed + 999))
    grouped_records = []
    t1 = time.perf_counter()
    for pi, (recipient, donor) in enumerate(grouped_pairs):
        out_base = base_answer(recipient)
        pred_base = extract_final_number(out_base)

        override_nondec = donor_vecs(donor, NONDECODABLE_ITERS)
        out_nondec = patched_answer(model, tokenizer, recipient, device, n_latents, override_nondec, ia.max_new_tokens)
        pred_nondec = extract_final_number(out_nondec)

        override_dec = donor_vecs(donor, DECODABLE_ITERS)
        out_dec = patched_answer(model, tokenizer, recipient, device, n_latents, override_dec, ia.max_new_tokens)
        pred_dec = extract_final_number(out_dec)

        steered_nondec = steered_dec = None
        if pred_base is not None:
            steered_nondec = answer_matches(pred_nondec, donor.answer) and not answer_matches(pred_base, donor.answer)
            steered_dec = answer_matches(pred_dec, donor.answer) and not answer_matches(pred_base, donor.answer)
        grouped_records.append({
            "recipient_idx": recipient.idx, "donor_idx": donor.idx,
            "recipient_gold": recipient.answer, "donor_gold": donor.answer,
            "answer_base": pred_base,
            "answer_patched_nondecodable_group": pred_nondec,
            "answer_patched_decodable_group": pred_dec,
            "answer_changed_nondecodable_group": pred_nondec != pred_base,
            "answer_changed_decodable_group": pred_dec != pred_base,
            "steered_to_donor_nondecodable_group": steered_nondec,
            "steered_to_donor_decodable_group": steered_dec,
        })
        if (pi + 1) % 25 == 0:
            with open("/tmp/codi_interchange_grouped_progress.jsonl", "w") as f:
                for r in grouped_records:
                    f.write(json.dumps(r, default=str) + "\n")
            print(f"grouped: {pi + 1}/{len(grouped_pairs)} ({time.perf_counter() - t1:.0f}s)", flush=True)

    n_g = len(grouped_records)
    changed_nondec = sum(r["answer_changed_nondecodable_group"] for r in grouped_records)
    changed_dec = sum(r["answer_changed_decodable_group"] for r in grouped_records)
    steered_nondec_vals = [r["steered_to_donor_nondecodable_group"] for r in grouped_records if r["steered_to_donor_nondecodable_group"] is not None]
    steered_dec_vals = [r["steered_to_donor_decodable_group"] for r in grouped_records if r["steered_to_donor_decodable_group"] is not None]
    steered_nondec_rate = sum(steered_nondec_vals) / len(steered_nondec_vals) if steered_nondec_vals else 0.0
    steered_dec_rate = sum(steered_dec_vals) / len(steered_dec_vals) if steered_dec_vals else 0.0

    b_changed = sum(1 for r in grouped_records if r["answer_changed_nondecodable_group"] and not r["answer_changed_decodable_group"])
    c_changed = sum(1 for r in grouped_records if r["answer_changed_decodable_group"] and not r["answer_changed_nondecodable_group"])
    p_changed = mcnemar_exact_p(b_changed, c_changed)

    b_steer = sum(1 for r in grouped_records if r["steered_to_donor_nondecodable_group"] and not r["steered_to_donor_decodable_group"])
    c_steer = sum(1 for r in grouped_records if r["steered_to_donor_decodable_group"] and not r["steered_to_donor_nondecodable_group"])
    p_steer = mcnemar_exact_p(b_steer, c_steer)

    print(f"GROUPED n={n_g}: answer_changed nondecodable={changed_nondec / n_g:.3f} "
          f"decodable={changed_dec / n_g:.3f}  McNemar b={b_changed} c={c_changed} p={p_changed:.4f}")
    print(f"GROUPED steered_to_donor nondecodable={steered_nondec_rate:.3f} decodable={steered_dec_rate:.3f} "
          f"McNemar b={b_steer} c={c_steer} p={p_steer:.4f}")

    # ---- Log the run ------------------------------------------------------------------------
    metrics = {
        "compute_steps": n_latents,
        "intervention_accuracy": steered_nondec_rate,  # headline: z0+z3 grouped steer-to-donor rate
        "extra": {
            "single_slot_summaries": single_summaries,
            "single_slot_nondecodable_pooled_n": len(nondec_pooled),
            "single_slot_decodable_pooled_n": len(dec_pooled),
            "grouped_n": n_g,
            "grouped_answer_changed_nondecodable": changed_nondec / n_g if n_g else None,
            "grouped_answer_changed_decodable": changed_dec / n_g if n_g else None,
            "grouped_answer_changed_wilson_ci_nondecodable": list(wilson_ci(changed_nondec, n_g)),
            "grouped_answer_changed_wilson_ci_decodable": list(wilson_ci(changed_dec, n_g)),
            "grouped_mcnemar_answer_changed": {"b": b_changed, "c": c_changed, "p_exact": p_changed},
            "grouped_steered_to_donor_nondecodable": steered_nondec_rate,
            "grouped_steered_to_donor_decodable": steered_dec_rate,
            "grouped_mcnemar_steered_to_donor": {"b": b_steer, "c": c_steer, "p_exact": p_steer},
            "n_trainable_params": None,
            "related_runs": [
                "20260919-184323_codi_decode-patch-full-eval",
                "20260919-192228_codi_early-termination-ablate-all",
                "20260919-185332_explicit_cot_patch-positive-control",
            ],
        },
    }

    record = RunRecord(
        run_id=new_run_id("codi", ia.slug),
        mechanism="codi",
        stage=ia.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=ia.checkpoint_label or model_args.ckpt_dir, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(examples), seed=ia.eval_seed),
        metrics=metrics,
        hyperparams={
            "inf_latent_iterations": n_latents, "num_latent": training_args.num_latent,
            "n_pairs_per_site": ia.n_pairs_per_site, "n_grouped_pairs": ia.n_grouped_pairs,
            "nondecodable_iters": list(NONDECODABLE_ITERS), "decodable_iters": list(DECODABLE_ITERS),
        },
        seed=ia.pair_seed,
        hardware=f"{ia.hardware} / {torch.cuda.get_device_name(0)}",
        notes="Donor-interchange patching at CODI's non-decodable placeholder positions "
              "(z0, z3) vs its best-decoding positions (z2, z4): single-slot at every "
              "iteration, then jointly-grouped {z0,z3} vs {z2,z4} paired on the same "
              "donor/recipient.",
    )
    all_predictions = (
        [{"condition": "single", "iter": it, **r} for it, recs in single_results.items() for r in recs]
        + [{"condition": "grouped", **r} for r in grouped_records]
    )
    manifest = record.save(predictions=all_predictions)
    out_dir = manifest.parent
    (out_dir / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
