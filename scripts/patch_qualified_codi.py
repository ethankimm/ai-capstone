#!/usr/bin/env python3
"""E2 (`steered_to_donor_audit.md` §5): step-aligned, base-correct, controlled raw
single-slot patch for CODI -- the latent analogue of `patch_explicit_cot.py`'s positive-
control table, testing EVERY latent iteration (not only the empirically best-decoding
one, so the known load-bearing-but-non-decodable z0/z3 slots get tested too).

Site -> step assignment is POSITIONAL (`latentreasoning.eval.counterfactual.site_to_step`),
not decoding-accuracy-fit: iteration i is assigned a step by splitting the n_latents
iterations into `max_step` contiguous groups. Deliberately simpler than
`decode_patch_codi.py`'s `best_iter_for_step` (which only ever tests whichever iteration
decodes a given step best) -- E2 wants every site tested, including ones a
decoding-accuracy fit would never select.

Qualified pairs (§4.2): recipient AND donor both base-correct (greedy final answer
matches gold); recipient's step-k gold value is an operand of step k+1's expression, so
injecting a different value there CAN propagate (`qualifying_steps`); donor supplies a
distinct step-k gold value. Three conditions per pair, same site: real donor, an
independently-drawn random donor (distinct value too), mean ablation (mean live
activation at that site over a sample of base-correct examples). Scored with
`score_patch` (Metric B `matches_cf` primary, Metric A `matches_donor_final` secondary,
full outcome taxonomy) + paired exact McNemar (real vs random, real vs mean).

Reuses `decode_patch_codi.py`'s model-loading and generation primitives (same released
checkpoint, same paper protocol) -- run inside the CODI venv/checkout, same convention:

  cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/patch_qualified_codi.py \\
      --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \\
      --slug qualified-patch --stage pilot --hardware "RunPod RTX A5000 (secure)" \\
      --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \\
      --lora_r 128 --lora_alpha 32 --lora_init --greedy True \\
      --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \\
      --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \\
      --eval_n 600 --n_pairs_per_site 200
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter
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
    decode_answer, mcnemar_exact_p, run_thoughts, wilson_ci,
)

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.eval.counterfactual import parse_steps, qualifying_steps, score_patch, site_to_step  # noqa: E402
from latentreasoning.eval.metrics import extract_final_number, is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402


@dataclass
class QualArguments:
    slug: str = field(default="qualified-patch")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: Optional[str] = field(default=None)
    eval_n: int = field(default=600)
    eval_seed: int = field(default=0)
    n_pairs_per_site: int = field(default=200)
    mean_sample_n: int = field(default=200)
    max_new_tokens: int = field(default=64)


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, QualArguments))
    model_args, data_args, training_args, qa = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = "cuda"

    model, tokenizer, load_result = build_model(model_args, training_args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {model_args.ckpt_dir}: missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}")

    n_latents = training_args.inf_latent_iterations
    examples = load_gsm8k_aug(split="test", n=qa.eval_n, seed=qa.eval_seed)
    by_idx = {ex.idx: ex for ex in examples}
    print(f"n_examples={len(examples)}, n_latents={n_latents}")

    # ---- decode pass: base answer + correctness + cached per-iter latents -----------------
    t0 = time.perf_counter()
    thought_cache: dict[int, list] = {}
    base_info: dict[int, dict] = {}
    for ex in examples:
        q = ex.question.strip().replace("  ", " ")
        pkv, thoughts, latent = run_thoughts(model, tokenizer, q, device, n_latents)
        raw = decode_answer(model, tokenizer, pkv, latent, device, qa.max_new_tokens)
        pred = extract_final_number(raw)
        correct = is_correct(pred, ex.answer)
        thought_cache[ex.idx] = thoughts
        steps = parse_steps(ex.rationale)
        base_info[ex.idx] = {"pred": pred, "correct": correct, "steps": steps,
                              "qualifying": set(qualifying_steps(steps))}
        if len(base_info) % 100 == 0:
            print(f"decoded {len(base_info)}/{len(examples)}", flush=True)
    decode_elapsed = time.perf_counter() - t0
    accuracy = sum(v["correct"] for v in base_info.values()) / len(base_info)
    print(f"decode pass done in {decode_elapsed:.1f}s, base accuracy={accuracy:.3f}")

    max_step = max((len(v["steps"]) for v in base_info.values()), default=0)
    rng = random.Random(0)
    correct_idxs = [idx for idx, v in base_info.items() if v["correct"]]
    print(f"base-correct pool: {len(correct_idxs)}/{len(examples)}, max_step={max_step}")

    def val_at(idx: int, k: int) -> Optional[str]:
        steps = base_info[idx]["steps"]
        return steps[k]["val"] if k < len(steps) else None

    # ---- mean activation per site, sampled from base-correct examples ---------------------
    mean_sample = rng.sample(correct_idxs, min(qa.mean_sample_n, len(correct_idxs)))
    mean_vec = {}
    for it in range(1, n_latents + 1):
        vecs = [thought_cache[idx][it - 1]["post"] for idx in mean_sample]
        mean_vec[it] = torch.stack(vecs, dim=0).mean(dim=0)
    print(f"mean activation computed from {len(mean_sample)} base-correct examples per site")

    # ---- build qualified pairs + run three conditions, per site ---------------------------
    site_records: dict[int, list[dict]] = {it: [] for it in range(1, n_latents + 1)}
    t1 = time.perf_counter()
    n_done = 0
    for it in range(1, n_latents + 1):
        k = site_to_step(it - 1, n_latents, max_step)
        recipient_pool = [idx for idx in correct_idxs if k in base_info[idx]["qualifying"]]
        donor_pool_all = [idx for idx in correct_idxs if val_at(idx, k) is not None]
        rng.shuffle(recipient_pool)

        pairs = []
        for ridx in recipient_pool:
            rv = val_at(ridx, k)
            cands = [d for d in donor_pool_all if d != ridx and val_at(d, k) != rv]
            if not cands:
                continue
            donor_idx = rng.choice(cands)
            rand_cands = [d for d in donor_pool_all if d != ridx and d != donor_idx and val_at(d, k) != rv]
            if not rand_cands:
                continue
            rand_idx = rng.choice(rand_cands)
            pairs.append((ridx, donor_idx, rand_idx))
            if len(pairs) >= qa.n_pairs_per_site:
                break
        print(f"site iter={it} step={k}: {len(recipient_pool)} qualifying recipients, {len(pairs)} pairs selected")

        for ridx, donor_idx, rand_idx in pairs:
            ex = by_idx[ridx]
            q = ex.question.strip().replace("  ", " ")
            donor_post = thought_cache[donor_idx][it - 1]["post"]
            rand_post = thought_cache[rand_idx][it - 1]["post"]

            pkv_b, _, latent_b = run_thoughts(model, tokenizer, q, device, n_latents)
            pred_base = extract_final_number(decode_answer(model, tokenizer, pkv_b, latent_b, device, qa.max_new_tokens))

            pkv_r, _, latent_r = run_thoughts(model, tokenizer, q, device, n_latents, override_input_at={it: donor_post})
            pred_real = extract_final_number(decode_answer(model, tokenizer, pkv_r, latent_r, device, qa.max_new_tokens))

            pkv_n, _, latent_n = run_thoughts(model, tokenizer, q, device, n_latents, override_input_at={it: rand_post})
            pred_rand = extract_final_number(decode_answer(model, tokenizer, pkv_n, latent_n, device, qa.max_new_tokens))

            pkv_m, _, latent_m = run_thoughts(model, tokenizer, q, device, n_latents, override_input_at={it: mean_vec[it]})
            pred_mean = extract_final_number(decode_answer(model, tokenizer, pkv_m, latent_m, device, qa.max_new_tokens))

            recipient_chain = base_info[ridx]["steps"]
            recipient_values = [s["val"] for s in recipient_chain]
            donor_values = [s["val"] for s in base_info[donor_idx]["steps"]]
            rand_values = [s["val"] for s in base_info[rand_idx]["steps"]]
            donor_val, rand_val = val_at(donor_idx, k), val_at(rand_idx, k)

            scored_real = score_patch(answer_base=pred_base, answer_patched=pred_real,
                                       recipient_gold=ex.answer, donor_final=by_idx[donor_idx].answer,
                                       recipient_chain=recipient_chain, donor_value=donor_val, step=k,
                                       recipient_values=recipient_values, donor_values=donor_values)
            scored_rand = score_patch(answer_base=pred_base, answer_patched=pred_rand,
                                       recipient_gold=ex.answer, donor_final=by_idx[rand_idx].answer,
                                       recipient_chain=recipient_chain, donor_value=rand_val, step=k,
                                       recipient_values=recipient_values, donor_values=rand_values)
            scored_mean = score_patch(answer_base=pred_base, answer_patched=pred_mean,
                                       recipient_gold=ex.answer, donor_final=None,
                                       recipient_chain=recipient_chain, donor_value=None, step=k,
                                       recipient_values=recipient_values, donor_values=[])

            site_records[it].append({
                "recipient_idx": ridx, "donor_idx": donor_idx, "random_donor_idx": rand_idx,
                "step": k, "iter": it,
                "answer_base": pred_base, "answer_real": pred_real, "answer_random": pred_rand, "answer_mean": pred_mean,
                "recipient_value_at_step": val_at(ridx, k), "donor_value_at_step": donor_val,
                "random_donor_value_at_step": rand_val,
                "real": scored_real, "random": scored_rand, "mean": scored_mean,
            })
            n_done += 1
            if n_done % 50 == 0:
                with open("/tmp/codi_qualified_patch_progress.jsonl", "w") as f:
                    for recs in site_records.values():
                        for r in recs:
                            f.write(json.dumps(r, default=str) + "\n")
                print(f"patched {n_done} ({time.perf_counter() - t1:.0f}s elapsed)", flush=True)

    # ---- aggregate --------------------------------------------------------------------------
    def summarize(records: list[dict], cond: str) -> dict:
        n = len(records)
        changed = sum(r[cond]["answer_changed"] for r in records)
        elig_cf = [r for r in records if r[cond]["matches_cf"] is not None]
        cf_hits = sum(r[cond]["matches_cf"] for r in elig_cf)
        elig_a = [r for r in records if r[cond]["matches_donor_final"] is not None]
        a_hits = sum(r[cond]["matches_donor_final"] for r in elig_a)
        taxonomy = Counter(r[cond]["outcome"] for r in records)
        return {
            "n": n, "answer_changed_rate": changed / n if n else 0.0,
            "answer_changed_wilson_ci": list(wilson_ci(changed, n)),
            "matches_cf_rate": (cf_hits / len(elig_cf)) if elig_cf else None,
            "matches_cf_n": len(elig_cf),
            "matches_cf_wilson_ci": list(wilson_ci(cf_hits, len(elig_cf))) if elig_cf else None,
            "matches_donor_final_rate": (a_hits / len(elig_a)) if elig_a else None,
            "matches_donor_final_n": len(elig_a),
            "outcome_taxonomy": dict(taxonomy),
        }

    def mcnemar_key(records: list[dict], cond_a: str, cond_b: str, key: str) -> dict:
        b = c = 0
        for r in records:
            va, vb = r[cond_a][key], r[cond_b][key]
            if va is None or vb is None:
                continue
            b += bool(va) and not vb
            c += bool(vb) and not va
        return {"b": b, "c": c, "p_exact": mcnemar_exact_p(b, c)}

    site_summary = {}
    for it in range(1, n_latents + 1):
        records = site_records[it]
        k = site_to_step(it - 1, n_latents, max_step)
        site_summary[it] = {
            "step_target": k, "n_pairs": len(records),
            "real": summarize(records, "real"), "random": summarize(records, "random"), "mean": summarize(records, "mean"),
            "mcnemar_real_vs_random_answer_changed": mcnemar_key(records, "real", "random", "answer_changed"),
            "mcnemar_real_vs_mean_answer_changed": mcnemar_key(records, "real", "mean", "answer_changed"),
            "mcnemar_real_vs_random_matches_cf": mcnemar_key(records, "real", "random", "matches_cf"),
            "mcnemar_real_vs_mean_matches_cf": mcnemar_key(records, "real", "mean", "matches_cf"),
        }
        s = site_summary[it]
        print(f"site iter={it} step={k} n={s['n_pairs']}: "
              f"matches_cf real={s['real']['matches_cf_rate']} random={s['random']['matches_cf_rate']} mean={s['mean']['matches_cf_rate']}")

    total_pairs = sum(len(v) for v in site_records.values())
    overall = {}
    for cond in ("real", "random", "mean"):
        all_records = [r for recs in site_records.values() for r in recs]
        overall[cond] = summarize(all_records, cond)
    print(f"OVERALL n={total_pairs}: matches_cf real={overall['real']['matches_cf_rate']} "
          f"random={overall['random']['matches_cf_rate']} mean={overall['mean']['matches_cf_rate']}")

    # ---- Log the run --------------------------------------------------------------------------
    metrics = {
        "final_answer_accuracy": accuracy,
        "unparseable_rate": sum(v["pred"] is None for v in base_info.values()) / len(base_info),
        "compute_steps": n_latents,
        "sec_per_example": decode_elapsed / len(examples),
        "decoding_accuracy": None,
        "intervention_accuracy": overall["real"]["matches_cf_rate"],
        "extra": {
            "design": "E2: step-aligned base-correct controlled raw single-slot patch (steered_to_donor_audit.md)",
            "n_examples_decoded": len(examples), "n_base_correct": len(correct_idxs), "max_step": max_step,
            "n_pairs_per_site_requested": qa.n_pairs_per_site, "mean_sample_n": len(mean_sample),
            "total_pairs": total_pairs,
            "site_summary": site_summary,
            "overall": overall,
            "n_trainable_params": n_params,
        },
    }

    record = RunRecord(
        run_id=new_run_id("codi", qa.slug),
        mechanism="codi",
        stage=qa.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=qa.checkpoint_label or model_args.ckpt_dir, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(examples), seed=qa.eval_seed),
        metrics=metrics,
        hyperparams={"inf_latent_iterations": n_latents, "greedy": training_args.greedy,
                     "num_latent": training_args.num_latent, "n_pairs_per_site": qa.n_pairs_per_site,
                     "mean_sample_n": qa.mean_sample_n},
        seed=None,
        hardware=f"{qa.hardware} / {torch.cuda.get_device_name(0)}",
        notes="E2: step-aligned, base-correct, controlled raw single-slot patch at EVERY CODI "
              "iteration (positional site->step assignment, not decoding-accuracy-fit). Real "
              "donor / random donor / mean-ablation conditions, scored with the shared "
              "counterfactual module (Metric B matches_cf primary).",
    )
    predictions = [{"idx": idx, **{k: v for k, v in info.items() if k != "qualifying"},
                    "qualifying_steps": sorted(info["qualifying"])} for idx, info in base_info.items()]
    manifest = record.save(predictions=predictions)
    out_dir = manifest.parent
    with (out_dir / "patch_pairs.jsonl").open("w") as f:
        for recs in site_records.values():
            for r in recs:
                f.write(json.dumps(r, default=str) + "\n")
    (out_dir / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
