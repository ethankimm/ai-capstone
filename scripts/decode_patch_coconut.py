#!/usr/bin/env python3
"""Coconut decodability + donor-interchange causal-patching -- the Coconut counterpart
to `decode_patch_codi.py` / `interchange_patch_placeholder_codi.py`, extending the
same "decodability != causal faithfulness" sweep to the OTHER width-based LRM this
project studies, on the released checkpoint from the paper we're comparing against
(Dilgren & Wiegreffe, COLM 2026, "Are Latent Reasoning Models Easily Interpretable?"):
`connordilgren/gpt2-gsm8k-coconut` (checkpoint_33).

Unlike CODI (odd iterations near-zero logit-lens accuracy, even ones near the strongest
decode position, by construction of the paper's own ablation table), the D&W paper's
own backtracking-search finding is that Coconut+GPT2 DOES encode gold reasoning traces
in most of its latent positions when correct (54-93%, depending on what counts as a
match) -- so this script does NOT assume a particular decodable/non-decodable split
the way `interchange_patch_placeholder_codi.py` targets CODI's known z0/z3. Instead it:

1. Decodes every latent pass (vocabulary projection on the LIVE hidden state -- the
   paper's own method, no extra forward pass) against the gold step values, with a
   held-out mapping split (fit best-pass-per-step on half A, report on half B) --
   exactly `decode_patch_codi.py`'s design.
2. Donor-interchange patches EVERY pass, single-slot, with donor/recipient pairs that
   differ at the matched gold step (same design as CODI's focused single-slot patch).
3. From the empirical top1 decode-accuracy ranking (half A), groups the passes into
   the top-half "most decodable" vs bottom-half "least decodable" and jointly patches
   each group (same donor, same recipient) -- the Coconut analogue of CODI's
   {z0,z3} vs {z2,z4} grouped comparison, but data-driven instead of assumed.

Uses `scripts/coconut_common.py` (this repo) + the VANILLA `facebookresearch/coconut`
checkpoint format (see that module's docstring for why the Dilgren & Wiegreffe fork's
checkpoint loads unchanged there). Data: `gsm_original_test.json` (1319, matches
CODI's/our own gsm8k-aug test split -- same underlying corpus, same questions) for
accuracy, `gsm_valid-gold-reasoning-trace_test.json` (1194, filtered + MultiChain-
enriched) for gold step values -- both generated locally via
`~/Projects/are-lrms-easily-interpretable/preprocessing/prepare_gsm8k.py`, then
copied onto the pod (see `--data_dir`).

Run inside a venv pinned to Coconut's own `requirements.txt` (torch==2.5.1,
transformers==4.46.2 -- newer transformers return a `Cache` object instead of legacy
`past_key_values` tuples, which `coconut_common._to_legacy_cache` normalizes, but
`create_causal_mask` in transformers>=4.5x REQUIRES a real Cache object again, so the
pinned version is the one actually verified end-to-end, not just made to import):

  cd /workspace/ai-capstone && .venv_coconut/bin/python scripts/decode_patch_coconut.py \\
      --checkpoint_path /workspace/coconut_checkpoints/gsm-coconut/checkpoint_33 \\
      --data_dir /workspace/coconut_data \\
      --slug decode-patch-pilot --stage pilot --hardware "RunPod RTX A5000 (secure)" \\
      --num_latents 6 --full_test False --eval_n 200 --n_patch_pairs 60
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
from coconut_common import (  # noqa: E402
    load_coconut, encode_question, run_passes, finish_and_decode,
    topk_token_strings, extract_answer_after_delimiter, parse_step_value,
    num_match, wilson_ci, mcnemar_exact_p,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from latentreasoning.eval.counterfactual import parse_steps, score_patch  # noqa: E402
from latentreasoning.eval.metrics import is_correct  # noqa: E402
from latentreasoning.mechanisms.coconut import name as mechanism_name  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402


@dataclass
class CoconutArguments:
    checkpoint_path: str = field(metadata={"help": "path to the downloaded checkpoint_33 file"})
    data_dir: str = field(metadata={"help": "dir with gsm_original_test.json / gsm_valid-gold-reasoning-trace_test.json"})
    model_id: str = field(default="openai-community/gpt2")
    slug: str = field(default="decode-patch")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: str = field(default="hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33")
    num_latents: int = field(default=6)
    full_test: bool = field(default=False)
    eval_n: int = field(default=200)
    eval_seed: int = field(default=0)
    mapping_split: bool = field(default=True)
    n_patch_pairs: int = field(default=60, metadata={"help": "per-pass single-slot interchange patch pairs"})
    n_grouped_pairs: int = field(default=100)
    max_new_tokens: int = field(default=48)
    device: str = field(default="cuda")


class Example:
    __slots__ = ("idx", "question", "answer", "steps")

    def __init__(self, idx, question, answer, steps):
        self.idx, self.question, self.answer, self.steps = idx, question, answer, steps


def load_examples(data_dir: str, full_test: bool, eval_n: int, eval_seed: int) -> list[Example]:
    """gold-reasoning-trace file (subset of the 1319 test set with steps verified to
    reach the gold answer) -- both accuracy-scoring and decode/patch targets come from
    here so every example has usable gold step values."""
    rows = json.loads((Path(data_dir) / "gsm_valid-gold-reasoning-trace_test.json").read_text())
    examples = [Example(i, r["question"], r["answer"].strip(), r["steps"]) for i, r in enumerate(rows)]
    if not full_test:
        random.Random(eval_seed).shuffle(examples)
        examples = examples[:eval_n]
    return examples


def build_matrix(decode_records: list[dict], n_passes: int, max_step: int):
    hits1 = {p: {s: [0, 0] for s in range(1, max_step + 1)} for p in range(n_passes)}
    hits5 = {p: {s: [0, 0] for s in range(1, max_step + 1)} for p in range(n_passes)}
    for r in decode_records:
        for s_idx, gold_val in enumerate(r["step_values"], start=1):
            if gold_val is None:
                continue
            for pr in r["per_pass"]:
                p = pr["pass"]
                hits1[p][s_idx][1] += 1
                hits5[p][s_idx][1] += 1
                if num_match([pr["top1"]], gold_val):
                    hits1[p][s_idx][0] += 1
                if num_match(pr["top5"], gold_val):
                    hits5[p][s_idx][0] += 1
    return hits1, hits5


def rate(hits, p, s):
    c, n = hits[p][s]
    return c / n if n else 0.0


def main() -> None:
    parser = transformers.HfArgumentParser((CoconutArguments,))
    (ca,) = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = ca.device

    tokenizer, base_model, embedding, special_ids = load_coconut(ca.model_id, ca.checkpoint_path, device)
    n_params = sum(p.numel() for p in base_model.parameters())
    eos_id = tokenizer.eos_token_id

    examples = load_examples(ca.data_dir, ca.full_test, ca.eval_n, ca.eval_seed)
    print(f"n_examples={len(examples)}")

    # ---- Task 1: decode every example --------------------------------------------------
    t0 = time.perf_counter()
    decode_records = []
    for ex in examples:
        input_ids, attn = encode_question(tokenizer, special_ids, ex.question, ca.num_latents, device)
        pass_records, inputs_embeds, kv_cache, ncr = run_passes(
            base_model, embedding, input_ids, attn, device, ca.num_latents, latent_token_id=special_ids["latent"])
        raw_output = finish_and_decode(base_model, embedding, tokenizer, inputs_embeds, kv_cache, ncr, attn,
                                        device, ca.max_new_tokens, eos_id)
        pred = extract_answer_after_delimiter(raw_output)
        correct = is_correct(pred, ex.answer)
        step_values = [parse_step_value(s) for s in ex.steps]
        per_pass = []
        for pr in pass_records:
            top1 = topk_token_strings(tokenizer, pr["logits"], 1)
            top5 = topk_token_strings(tokenizer, pr["logits"], 5)
            per_pass.append({"pass": pr["pass"], "top1": top1[0], "top5": top5})
        decode_records.append({
            "idx": ex.idx, "n_steps": len(step_values), "step_values": step_values,
            "gold_answer": ex.answer, "raw_output": raw_output, "predicted_answer": pred,
            "correct": correct, "per_pass": per_pass,
        })
        if len(decode_records) % 50 == 0:
            print(f"decoded {len(decode_records)}/{len(examples)}", flush=True)
    decode_elapsed = time.perf_counter() - t0
    accuracy = sum(r["correct"] for r in decode_records) / len(decode_records)
    print(f"decode pass done in {decode_elapsed:.1f}s, final_answer_accuracy={accuracy:.3f} "
          f"(paper Table 1: GPT-2 Coconut GSM8k-Aug = 33.1%, our repro Hao et al. col = 34.1%)")

    if ca.mapping_split:
        half_a = [r for r in decode_records if r["idx"] % 2 == 0]
        half_b = [r for r in decode_records if r["idx"] % 2 == 1]
    else:
        half_a = half_b = decode_records
    max_step = max((r["n_steps"] for r in decode_records), default=0)

    hits_a_1, hits_a_5 = build_matrix(half_a, ca.num_latents, max_step)
    best_pass_for_step = {}
    for s in range(1, max_step + 1):
        best_pass_for_step[s] = max(range(ca.num_latents), key=lambda p: rate(hits_a_1, p, s))
    print(f"best_pass_for_step (fit on half A): {best_pass_for_step}")

    hits_b_1, hits_b_5 = build_matrix(half_b, ca.num_latents, max_step)
    matched_c1 = sum(hits_b_1[best_pass_for_step[s]][s][0] for s in best_pass_for_step)
    matched_n1 = sum(hits_b_1[best_pass_for_step[s]][s][1] for s in best_pass_for_step)
    matched_c5 = sum(hits_b_5[best_pass_for_step[s]][s][0] for s in best_pass_for_step)
    overall_matched_top1 = matched_c1 / matched_n1 if matched_n1 else 0.0
    overall_matched_top5 = matched_c5 / matched_n1 if matched_n1 else 0.0
    print(f"HELD-OUT matched-pass top1={overall_matched_top1:.4f} top5={overall_matched_top5:.4f} n={matched_n1}")

    # rank passes by their OWN unconditional top1 rate (avg over steps they were ever best at,
    # or simplest: avg top1 rate at step 1 -- the pass that's live earliest for every example)
    # -- use avg top1 rate across all steps each pass was evaluated on, from half A (fit set).
    pass_avg_top1 = {}
    for p in range(ca.num_latents):
        rates = [rate(hits_a_1, p, s) for s in range(1, max_step + 1) if hits_a_1[p][s][1] > 0]
        pass_avg_top1[p] = sum(rates) / len(rates) if rates else 0.0
    ranked_passes = sorted(range(ca.num_latents), key=lambda p: -pass_avg_top1[p])
    half = ca.num_latents // 2
    most_decodable = sorted(ranked_passes[:half])
    least_decodable = sorted(ranked_passes[half:])
    print(f"pass_avg_top1 (fit on half A): {pass_avg_top1}")
    print(f"most_decodable={most_decodable}  least_decodable={least_decodable}")

    # ---- Task 2: single-slot donor-interchange patching, every pass --------------------
    def answer_for(question: str, override_at_pass=None):
        input_ids, attn = encode_question(tokenizer, special_ids, question, ca.num_latents, device)
        pass_records, inputs_embeds, kv_cache, ncr = run_passes(
            base_model, embedding, input_ids, attn, device, ca.num_latents,
            override_at_pass=override_at_pass, latent_token_id=special_ids["latent"])
        raw = finish_and_decode(base_model, embedding, tokenizer, inputs_embeds, kv_cache, ncr, attn,
                                 device, ca.max_new_tokens, eos_id)
        return extract_answer_after_delimiter(raw), pass_records

    pass_cache: dict[int, list] = {}

    def get_pass_records(ex: Example):
        if ex.idx not in pass_cache:
            _, recs = answer_for(ex.question)
            pass_cache[ex.idx] = recs
        return pass_cache[ex.idx]

    half_b_examples = {r["idx"] for r in half_b}
    examples_by_idx = {ex.idx: ex for ex in examples}
    by_steps: dict[int, list] = {}
    for ex in examples:
        if ex.idx not in half_b_examples:
            continue
        by_steps.setdefault(len(ex.steps), []).append(ex)

    rng = random.Random(0)
    single_records: dict[int, list[dict]] = {p: [] for p in range(ca.num_latents)}
    t1 = time.perf_counter()
    n_done = 0
    for target_pass in range(ca.num_latents):
        pool = []
        for n_steps, exs in by_steps.items():
            for s in range(1, n_steps + 1):
                if best_pass_for_step.get(s) != target_pass:
                    continue
                for a in range(len(exs)):
                    for b in range(len(exs)):
                        if a == b:
                            continue
                        ea, eb = exs[a], exs[b]
                        va, vb = parse_step_value(ea.steps[s - 1]), parse_step_value(eb.steps[s - 1])
                        if va is not None and vb is not None and va != vb:
                            pool.append((ea, eb, s))
        rng.shuffle(pool)
        pairs = pool[:ca.n_patch_pairs]
        for recipient, donor, s in pairs:
            pred_base, thoughts_base = answer_for(recipient.question)
            donor_recs = get_pass_records(donor)
            donor_vec = donor_recs[target_pass]["live_hidden"]
            pred_patched, thoughts_patched = answer_for(recipient.question, override_at_pass={target_pass: donor_vec})
            # `donor_val` (donor's INTERMEDIATE value at step s) was previously compared
            # directly against the patched answer -- invalid: it asks the model to abandon
            # its remaining steps and echo a scratchpad variable, not its own final answer
            # (see steered_to_donor_audit.md #2.1). `score_patch`'s `matches_cf` is the
            # correct question: does the patched answer equal what a faithful continuation
            # of the RECIPIENT's own chain would give with step s's value replaced by the
            # donor's.
            donor_val = parse_step_value(donor.steps[s - 1])
            recipient_chain = parse_steps(" ".join(recipient.steps))
            recipient_values = [parse_step_value(st) for st in recipient.steps]
            recipient_values = [v for v in recipient_values if v is not None]
            donor_values = [v for v in (parse_step_value(st) for st in donor.steps) if v is not None]
            scored = score_patch(
                answer_base=pred_base, answer_patched=pred_patched,
                recipient_gold=recipient.answer, donor_final=donor.answer,
                recipient_chain=recipient_chain, donor_value=donor_val, step=s - 1,
                recipient_values=recipient_values, donor_values=donor_values,
            )
            single_records[target_pass].append({
                "recipient_idx": recipient.idx, "donor_idx": donor.idx, "step": s,
                "answer_base": pred_base, "answer_patched": pred_patched,
                "donor_value_at_step": donor_val,
                **scored,
                "steered_to_donor": scored["matches_donor_final"],
            })
            n_done += 1
            if n_done % 50 == 0:
                print(f"single-slot: {n_done} done ({time.perf_counter() - t1:.0f}s)", flush=True)

    def summarize(records):
        n = len(records)
        x_changed = sum(r["answer_changed"] for r in records)
        elig = [r for r in records if r["steered_to_donor"] is not None]
        x_steer = sum(r["steered_to_donor"] for r in elig)
        return {
            "n": n, "answer_changed_rate": x_changed / n if n else 0.0,
            "answer_changed_wilson_ci": list(wilson_ci(x_changed, n)),
            "steered_to_donor_rate": x_steer / len(elig) if elig else 0.0,
            "n_steered_eligible": len(elig),
        }

    single_summaries = {p: summarize(single_records[p]) for p in range(ca.num_latents)}
    for p, s in single_summaries.items():
        tag = "most-decodable" if p in most_decodable else "least-decodable"
        print(f"pass={p} ({tag}, avg_top1={pass_avg_top1[p]:.3f}): "
              f"answer_changed={s['answer_changed_rate']:.3f} steered_to_donor={s['steered_to_donor_rate']:.3f} "
              f"(n={s['n']}, n_steered_eligible={s['n_steered_eligible']})")

    # ---- Task 3: grouped patching -- most_decodable vs least_decodable, same donor -----
    all_examples_flat = [ex for exs in by_steps.values() for ex in exs]
    grouped_records = []
    t2 = time.perf_counter()
    for pi in range(ca.n_grouped_pairs):
        recipient, donor = rng.sample(all_examples_flat, 2)
        if recipient.answer.strip() == donor.answer.strip():
            continue
        pred_base, _ = answer_for(recipient.question)
        donor_recs = get_pass_records(donor)

        override_most = {p: donor_recs[p]["live_hidden"] for p in most_decodable}
        pred_most, _ = answer_for(recipient.question, override_at_pass=override_most)
        override_least = {p: donor_recs[p]["live_hidden"] for p in least_decodable}
        pred_least, _ = answer_for(recipient.question, override_at_pass=override_least)

        steered_most = steered_least = None
        if pred_base is not None:
            steered_most = num_match([pred_most] if pred_most else [], donor.answer) and not num_match([pred_base], donor.answer)
            steered_least = num_match([pred_least] if pred_least else [], donor.answer) and not num_match([pred_base], donor.answer)
        grouped_records.append({
            "recipient_idx": recipient.idx, "donor_idx": donor.idx,
            "answer_base": pred_base, "answer_patched_most_decodable": pred_most, "answer_patched_least_decodable": pred_least,
            "answer_changed_most_decodable": pred_most != pred_base,
            "answer_changed_least_decodable": pred_least != pred_base,
            "steered_to_donor_most_decodable": steered_most,
            "steered_to_donor_least_decodable": steered_least,
        })
        if (pi + 1) % 25 == 0:
            print(f"grouped: {pi + 1}/{ca.n_grouped_pairs} ({time.perf_counter() - t2:.0f}s)", flush=True)

    n_g = len(grouped_records)
    changed_most = sum(r["answer_changed_most_decodable"] for r in grouped_records)
    changed_least = sum(r["answer_changed_least_decodable"] for r in grouped_records)
    b = sum(1 for r in grouped_records if r["answer_changed_least_decodable"] and not r["answer_changed_most_decodable"])
    c = sum(1 for r in grouped_records if r["answer_changed_most_decodable"] and not r["answer_changed_least_decodable"])
    p_val = mcnemar_exact_p(b, c)
    print(f"GROUPED n={n_g}: answer_changed most_decodable={changed_most / n_g:.3f} "
          f"least_decodable={changed_least / n_g:.3f}  McNemar b={b} c={c} p={p_val:.4f}")

    # ---- Log the run --------------------------------------------------------------------
    metrics = {
        "final_answer_accuracy": accuracy,
        "unparseable_rate": sum(r["predicted_answer"] is None for r in decode_records) / len(decode_records),
        "compute_steps": ca.num_latents,
        "sec_per_example": decode_elapsed / len(decode_records),
        "decoding_accuracy": overall_matched_top1,
        "intervention_accuracy": summarize(single_records[ranked_passes[0]])["steered_to_donor_rate"],
        "extra": {
            "held_out_split": ca.mapping_split, "half_a_n": len(half_a), "half_b_n": len(half_b),
            "decoding_accuracy_matched_top1": overall_matched_top1,
            "decoding_accuracy_matched_top5": overall_matched_top5,
            "best_pass_for_step_fit_on_half_A": best_pass_for_step,
            "pass_avg_top1_fit_on_half_A": pass_avg_top1,
            "most_decodable_passes": most_decodable, "least_decodable_passes": least_decodable,
            "single_slot_summaries": single_summaries,
            "grouped_n": n_g,
            "grouped_answer_changed_most_decodable": changed_most / n_g if n_g else None,
            "grouped_answer_changed_least_decodable": changed_least / n_g if n_g else None,
            "grouped_mcnemar_answer_changed": {"b": b, "c": c, "p_exact": p_val},
            "n_trainable_params": n_params,
            "paper_table1_reference": {"gpt2_coconut_gsm8k_aug_accuracy": 0.331, "hao_et_al_repro": 0.341},
        },
    }

    record = RunRecord(
        run_id=new_run_id("coconut", ca.slug),
        mechanism=mechanism_name,
        stage=ca.stage,
        model=ModelInfo(backbone=ca.model_id, checkpoint=ca.checkpoint_label, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(examples), seed=ca.eval_seed),
        metrics=metrics,
        hyperparams={"num_latents": ca.num_latents, "n_patch_pairs": ca.n_patch_pairs,
                     "n_grouped_pairs": ca.n_grouped_pairs, "mapping_split": ca.mapping_split},
        seed=ca.eval_seed,
        hardware=f"{ca.hardware} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        notes="Coconut (released connordilgren/gpt2-gsm8k-coconut checkpoint) decoding "
              "+ donor-interchange patching, mirroring decode_patch_codi.py / "
              "interchange_patch_placeholder_codi.py but with a DATA-DRIVEN "
              "decodable/non-decodable pass split instead of CODI's known z0/z3.",
    )
    manifest = record.save(predictions=decode_records + [{"condition": "single", "pass": p, **r} for p, recs in single_records.items() for r in recs] + [{"condition": "grouped", **r} for r in grouped_records])
    out_dir = manifest.parent
    (out_dir / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
