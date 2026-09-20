#!/usr/bin/env python3
"""E3 (`steered_to_donor_audit.md` §5): same-problem minimal-pair donor-interchange
patching for CODI. Donor = the recipient's OWN question with one eligible number
perturbed and the gold chain re-executed (`latentreasoning.data.minimal_pairs`), so the
twin's final answer literally IS the counterfactual value (Metric A == Metric B by
construction) and E2's cross-problem confound -- the donor's remaining program living
in question text the recipient never saw -- is gone entirely.

(a) ALL-SLOT: override every latent iteration's input with the twin's own latent at
    that iteration. Upper bound on what the thought chain carries at all -- if the
    recipient's answer does not move to the twin's answer even here, the perturbed
    value isn't in the thoughts and everything finer-grained (single-slot, subspace)
    is moot.
(b) SINGLE-SLOT sweep (every iteration alone) and CUMULATIVE PREFIX {1..i} to localize
    which iteration(s), if any, carry it.

Qualified pairs: recipient base-correct AND twin base-correct (greedy decode), the
perturbed step a `qualifying_steps` position (so a change there CAN propagate to the
final answer). One pair per (example, qualifying step) that succeeds under
`generate_minimal_pair`, trying deltas (1,-1,2,-2,3,-3) until one produces a valid
non-negative-integer twin whose target step actually changes.

Scoring: `matches_twin` = patched answer equals the twin's own (ground-truth,
re-executed) final answer, and the base answer did not already equal it -- valid here
specifically because of the minimal-pair construction (see module docstring on
`latentreasoning/data/minimal_pairs.py`); this is the same shape as `steered_to_donor`
but not the invalid version audited in `steered_to_donor_audit.md` #2.1, since the twin
IS the correct counterfactual by construction.

Reuses `decode_patch_codi.py`'s model-loading and generation primitives, run inside the
CODI venv/checkout, same convention as `patch_qualified_codi.py`:

  cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/patch_minimal_pair_codi.py \\
      --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \\
      --slug minimal-pair-patch --stage pilot --hardware "RunPod RTX A5000 (secure)" \\
      --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \\
      --lora_r 128 --lora_alpha 32 --lora_init --greedy True \\
      --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \\
      --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \\
      --eval_n 600 --n_pairs 150
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
from decode_patch_codi import decode_answer, run_thoughts, wilson_ci  # noqa: E402

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.data.minimal_pairs import generate_minimal_pair  # noqa: E402
from latentreasoning.eval.counterfactual import parse_steps, qualifying_steps  # noqa: E402
from latentreasoning.eval.metrics import extract_final_number, is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402

DELTAS = (1, -1, 2, -2, 3, -3)


@dataclass
class MPArguments:
    slug: str = field(default="minimal-pair-patch")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: Optional[str] = field(default=None)
    eval_n: int = field(default=600)
    eval_seed: int = field(default=0)
    n_pairs: int = field(default=150)
    max_candidate_checks: int = field(default=800, metadata={"help": "cap on twin-decode checks so a low hit-rate can't blow the budget"})
    max_new_tokens: int = field(default=64)
    pair_seed: int = field(default=0)


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, MPArguments))
    model_args, data_args, training_args, mpa = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = "cuda"

    model, tokenizer, load_result = build_model(model_args, training_args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {model_args.ckpt_dir}: missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}")

    n_latents = training_args.inf_latent_iterations
    examples = load_gsm8k_aug(split="test", n=mpa.eval_n, seed=mpa.eval_seed)
    print(f"n_examples={len(examples)}, n_latents={n_latents}")

    def decode_one(question: str):
        q = question.strip().replace("  ", " ")
        pkv, thoughts, latent = run_thoughts(model, tokenizer, q, device, n_latents)
        raw = decode_answer(model, tokenizer, pkv, latent, device, mpa.max_new_tokens)
        return extract_final_number(raw), thoughts

    def run_override(question: str, override: dict):
        q = question.strip().replace("  ", " ")
        pkv, _, latent = run_thoughts(model, tokenizer, q, device, n_latents, override_input_at=override)
        return extract_final_number(decode_answer(model, tokenizer, pkv, latent, device, mpa.max_new_tokens))

    # ---- decode pass over base examples: correctness + qualifying steps -------------------
    t0 = time.perf_counter()
    base_info: dict[int, dict] = {}
    for ex in examples:
        pred, _ = decode_one(ex.question)
        steps = parse_steps(ex.rationale)
        base_info[ex.idx] = {"pred": pred, "correct": is_correct(pred, ex.answer),
                              "qualifying": qualifying_steps(steps)}
        if len(base_info) % 100 == 0:
            print(f"decoded {len(base_info)}/{len(examples)}", flush=True)
    decode_elapsed = time.perf_counter() - t0
    accuracy = sum(v["correct"] for v in base_info.values()) / len(base_info)
    correct_examples = [ex for ex in examples if base_info[ex.idx]["correct"]]
    print(f"decode pass done in {decode_elapsed:.1f}s, base accuracy={accuracy:.3f}, "
          f"base-correct pool={len(correct_examples)}/{len(examples)}")

    # ---- build minimal-pair candidates from base-correct, propagation-qualified steps -----
    rng = random.Random(mpa.pair_seed)
    candidates = []
    for ex in correct_examples:
        for k in base_info[ex.idx]["qualifying"]:
            for delta in DELTAS:
                mp = generate_minimal_pair(ex, k, delta, rng=rng)
                if mp is not None:
                    candidates.append(mp)
                    break
    rng.shuffle(candidates)
    print(f"minimal-pair candidates (original base-correct, propagation-qualified): {len(candidates)}")

    # ---- keep only candidates whose twin is ALSO base-correct ------------------------------
    pairs = []
    twin_cache: dict[int, dict] = {}
    n_checked = 0
    for mp in candidates:
        if len(pairs) >= mpa.n_pairs or n_checked >= mpa.max_candidate_checks:
            break
        n_checked += 1
        twin_pred, twin_thoughts = decode_one(mp.twin.question)
        if is_correct(twin_pred, mp.twin.answer):
            pairs.append(mp)
            twin_cache[id(mp)] = {"pred": twin_pred, "thoughts": twin_thoughts}
        if n_checked % 50 == 0:
            print(f"checked {n_checked} candidates, {len(pairs)} qualified so far", flush=True)
    print(f"qualified pairs (twin also base-correct): {len(pairs)}/{n_checked} candidates checked")

    # ---- all-slot / single-slot / cumulative-prefix patches, per qualified pair -----------
    records = []
    t1 = time.perf_counter()
    for pi, mp in enumerate(pairs):
        ex = mp.original
        twin_thoughts = twin_cache[id(mp)]["thoughts"]
        twin_vecs = {it: twin_thoughts[it - 1]["post"] for it in range(1, n_latents + 1)}
        pred_base = base_info[ex.idx]["pred"]
        twin_answer = mp.twin.answer

        def matches_twin(pred: Optional[str], pred_base=pred_base, twin_answer=twin_answer) -> bool:
            return is_correct(pred, twin_answer) and not is_correct(pred_base, twin_answer)

        pred_all = run_override(ex.question, twin_vecs)
        all_slot = {"pred": pred_all, "answer_changed": pred_all != pred_base, "matches_twin": matches_twin(pred_all)}

        single_slot = {}
        for it in range(1, n_latents + 1):
            pred_i = run_override(ex.question, {it: twin_vecs[it]})
            single_slot[it] = {"pred": pred_i, "answer_changed": pred_i != pred_base, "matches_twin": matches_twin(pred_i)}

        prefix = {}
        for i in range(1, n_latents + 1):
            pred_p = run_override(ex.question, {it: twin_vecs[it] for it in range(1, i + 1)})
            prefix[i] = {"pred": pred_p, "answer_changed": pred_p != pred_base, "matches_twin": matches_twin(pred_p)}

        records.append({
            "recipient_idx": ex.idx, "step": mp.step, "perturbed_number": mp.perturbed_number,
            "delta": mp.delta, "answer_base": pred_base, "recipient_gold": ex.answer,
            "twin_question": mp.twin.question, "twin_answer": twin_answer,
            "all_slot": all_slot, "single_slot": single_slot, "prefix": prefix,
        })
        if (pi + 1) % 10 == 0:
            with open("/tmp/codi_minimal_pair_progress.jsonl", "w") as f:
                for r in records:
                    f.write(json.dumps(r, default=str) + "\n")
            print(f"patched {pi + 1}/{len(pairs)} pairs ({time.perf_counter() - t1:.0f}s)", flush=True)

    # ---- aggregate --------------------------------------------------------------------------
    def summarize(key_fn) -> dict:
        n = len(records)
        changed = sum(key_fn(r)["answer_changed"] for r in records)
        matched = sum(key_fn(r)["matches_twin"] for r in records)
        return {"n": n, "answer_changed_rate": changed / n if n else 0.0,
                "answer_changed_wilson_ci": list(wilson_ci(changed, n)),
                "matches_twin_rate": matched / n if n else 0.0,
                "matches_twin_wilson_ci": list(wilson_ci(matched, n))}

    all_slot_summary = summarize(lambda r: r["all_slot"])
    single_slot_summary = {it: summarize(lambda r, it=it: r["single_slot"][it]) for it in range(1, n_latents + 1)}
    prefix_summary = {i: summarize(lambda r, i=i: r["prefix"][i]) for i in range(1, n_latents + 1)}

    print(f"ALL-SLOT n={all_slot_summary['n']}: matches_twin={all_slot_summary['matches_twin_rate']:.3f} "
          f"answer_changed={all_slot_summary['answer_changed_rate']:.3f}")
    for it in range(1, n_latents + 1):
        s = single_slot_summary[it]
        print(f"single-slot iter={it}: matches_twin={s['matches_twin_rate']:.3f} answer_changed={s['answer_changed_rate']:.3f}")
    for i in range(1, n_latents + 1):
        s = prefix_summary[i]
        print(f"prefix 1..{i}: matches_twin={s['matches_twin_rate']:.3f} answer_changed={s['answer_changed_rate']:.3f}")

    metrics = {
        "final_answer_accuracy": accuracy,
        "compute_steps": n_latents,
        "sec_per_example": decode_elapsed / len(examples),
        "intervention_accuracy": all_slot_summary["matches_twin_rate"],
        "extra": {
            "design": "E3: same-problem minimal-pair donor-interchange patch (steered_to_donor_audit.md)",
            "n_examples_decoded": len(examples), "n_base_correct": len(correct_examples),
            "n_candidates_checked": n_checked, "n_qualified_pairs": len(pairs),
            "all_slot": all_slot_summary, "single_slot": single_slot_summary, "prefix": prefix_summary,
            "n_trainable_params": n_params,
            "related_runs": ["20260920-085206_codi_qualified-patch"],
        },
    }

    record = RunRecord(
        run_id=new_run_id("codi", mpa.slug),
        mechanism="codi",
        stage=mpa.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=mpa.checkpoint_label or model_args.ckpt_dir, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(examples), seed=mpa.eval_seed),
        metrics=metrics,
        hyperparams={"inf_latent_iterations": n_latents, "greedy": training_args.greedy,
                     "num_latent": training_args.num_latent, "n_pairs_requested": mpa.n_pairs,
                     "max_candidate_checks": mpa.max_candidate_checks},
        seed=mpa.pair_seed,
        hardware=f"{mpa.hardware} / {torch.cuda.get_device_name(0)}",
        notes="E3: same-problem minimal-pair donor-interchange patch -- all-slot, single-slot, "
              "and cumulative-prefix conditions, scored against the twin's own ground-truth "
              "final answer (Metric A == Metric B by minimal-pair construction).",
    )
    manifest = record.save(predictions=records)
    out_dir = manifest.parent
    (out_dir / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
