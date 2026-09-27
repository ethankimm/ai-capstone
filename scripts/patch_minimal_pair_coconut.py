#!/usr/bin/env python3
"""E3 (`steered_to_donor_audit.md` §5): same-problem minimal-pair donor-interchange
patching for Coconut -- the Coconut counterpart to `patch_minimal_pair_codi.py`. Donor
= the recipient's OWN question with one eligible number perturbed and the gold chain
re-executed (`latentreasoning.data.minimal_pairs`), so the twin's final answer
literally IS the counterfactual value (Metric A == Metric B by construction).

Sites: pass p = the vector spliced into the p-th <|latent|> slot (pass 0 = the hidden
state at <|start-latent|>), p = 0..5 -- the same numbering as CODI's aligned z_0..z_5.
`run_passes(override_at_pass={p: twin live_hidden[p]})` is already aligned (the twin's
pass-p vector goes where the recipient's pass-p vector goes).

Conditions per qualified pair (recipient and twin both answered correctly, greedy):
  all_slot, single_slot[p], prefix[p] (passes 0..p), leave_one_out[p] (all but p).
Non-responsive pairs (recipient correct, twin answered WRONG): all_slot only, scored
against the twin's own prediction. Every candidate twin decoded is logged
(`candidates.jsonl`) for counterfactual responsiveness.

Run inside the venv pinned to Coconut's own requirements.txt:

  cd /workspace/ai-capstone && .venv_coconut/bin/python scripts/patch_minimal_pair_coconut.py \\
      --checkpoint_path /workspace/coconut_checkpoints/checkpoint_33 \\
      --data_dir /workspace/coconut_data \\
      --slug minimal-pair-patch-full --stage full_run --hardware "RunPod RTX A5000 (secure)" \\
      --num_latents 6 --eval_n 0 --n_pairs 1000 --n_nonresponsive 300
"""
from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coconut_common import (  # noqa: E402
    encode_question, extract_answer_after_delimiter, finish_and_decode, load_coconut,
    num_match, run_passes, wilson_ci,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from latentreasoning.data.gsm8k_aug import Example  # noqa: E402
from latentreasoning.data.minimal_pairs import generate_minimal_pair  # noqa: E402
from latentreasoning.eval.counterfactual import parse_steps, qualifying_steps  # noqa: E402
from latentreasoning.eval.metrics import is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402

DELTAS = (1, -1, 2, -2, 3, -3)


@dataclass
class MPArguments:
    checkpoint_path: str = field(metadata={"help": "path to the downloaded checkpoint_33 file"})
    data_dir: str = field(metadata={"help": "dir with gsm_valid-gold-reasoning-trace_test.json"})
    model_id: str = field(default="openai-community/gpt2")
    slug: str = field(default="minimal-pair-patch-full")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: str = field(default="hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33")
    num_latents: int = field(default=6)
    eval_n: int = field(default=0, metadata={"help": "0 = every gold-trace example (1194)"})
    eval_seed: int = field(default=0)
    n_pairs: int = field(default=1000)
    n_nonresponsive: int = field(default=300)
    max_new_tokens: int = field(default=48)
    pair_seed: int = field(default=0)
    smoke_n: int = field(default=0, metadata={"help": ">0: first n examples only, don't log"})
    device: str = field(default="cuda")


def load_examples(data_dir: str, eval_n: int, eval_seed: int) -> list[Example]:
    rows = json.loads((Path(data_dir) / "gsm_valid-gold-reasoning-trace_test.json").read_text())
    examples = [Example(question=r["question"], rationale=" ".join(r["steps"]),
                         answer=r["answer"].strip(), idx=i) for i, r in enumerate(rows)]
    random.Random(eval_seed).shuffle(examples)
    return examples[:eval_n] if eval_n else examples


def main() -> None:
    parser = transformers.HfArgumentParser((MPArguments,))
    (mpa,) = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = mpa.device

    tokenizer, base_model, embedding, special_ids = load_coconut(mpa.model_id, mpa.checkpoint_path, device)
    n_params = sum(p.numel() for p in base_model.parameters())
    eos_id = tokenizer.eos_token_id
    sites = list(range(mpa.num_latents))

    examples = load_examples(mpa.data_dir, mpa.eval_n, mpa.eval_seed)
    if mpa.smoke_n:
        examples = examples[:mpa.smoke_n]
    print(f"n_examples={len(examples)}, num_latents={mpa.num_latents}")

    def answer_for(question: str, override_at_pass=None):
        input_ids, attn = encode_question(tokenizer, special_ids, question, mpa.num_latents, device)
        pass_records, inputs_embeds, kv_cache, ncr = run_passes(
            base_model, embedding, input_ids, attn, device, mpa.num_latents,
            override_at_pass=override_at_pass, latent_token_id=special_ids["latent"])
        raw = finish_and_decode(base_model, embedding, tokenizer, inputs_embeds, kv_cache, ncr, attn,
                                 device, mpa.max_new_tokens, eos_id)
        return extract_answer_after_delimiter(raw), pass_records

    t0 = time.perf_counter()
    base_info: dict[int, dict] = {}
    for ex in examples:
        pred, _ = answer_for(ex.question)
        base_info[ex.idx] = {"pred": pred, "correct": is_correct(pred, ex.answer),
                              "qualifying": qualifying_steps(parse_steps(ex.rationale))}
    decode_elapsed = time.perf_counter() - t0
    accuracy = sum(v["correct"] for v in base_info.values()) / len(base_info)
    print(f"decode pass {decode_elapsed:.0f}s, base accuracy={accuracy:.3f}")

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
    print(f"minimal-pair candidates (any original): {len(candidates)}")

    cand_log, responsive, nonresponsive = [], [], []
    for mp in candidates:
        twin_pred, twin_pass_records = answer_for(mp.twin.question)
        z = {p: twin_pass_records[p]["live_hidden"] for p in sites}
        orig_ok = base_info[mp.original.idx]["correct"]
        twin_ok = is_correct(twin_pred, mp.twin.answer)
        cand_log.append({"recipient_idx": mp.original.idx, "step": mp.step, "delta": mp.delta,
                         "perturbed_number": mp.perturbed_number,
                         "chain_len": len(parse_steps(mp.original.rationale)),
                         "original_pred": base_info[mp.original.idx]["pred"], "original_correct": orig_ok,
                         "twin_answer": mp.twin.answer, "twin_pred": twin_pred, "twin_correct": twin_ok})
        if orig_ok and twin_ok and len(responsive) < mpa.n_pairs:
            responsive.append((mp, twin_pred, z))
        elif orig_ok and not twin_ok and twin_pred is not None and len(nonresponsive) < mpa.n_nonresponsive:
            nonresponsive.append((mp, twin_pred, z))
    n_orig_ok = sum(c["original_correct"] for c in cand_log)
    print(f"CFR: {len(responsive)}/{n_orig_ok} twins correct given original correct")

    records = []
    t2 = time.perf_counter()
    for pi, (mp, twin_pred, z) in enumerate(responsive):
        ex = mp.original
        pred_base = base_info[ex.idx]["pred"]
        twin_answer = mp.twin.answer

        def cond(pred, pred_base=pred_base, twin_answer=twin_answer):
            base_matched = num_match([pred_base] if pred_base else [], twin_answer)
            return {"pred": pred, "answer_changed": pred != pred_base,
                    "matches_twin": num_match([pred] if pred else [], twin_answer) and not base_matched}

        def run(site_set):
            return cond(answer_for(ex.question, override_at_pass={p: z[p] for p in site_set})[0])

        all_slot = run(sites)
        single = {p: run([p]) for p in sites}
        prefix = {p: run(range(p + 1)) for p in sites[:-1]}
        prefix[sites[-1]] = all_slot
        loo = {p: run([t for t in sites if t != p]) for p in sites}
        records.append({
            "recipient_idx": ex.idx, "step": mp.step, "perturbed_number": mp.perturbed_number,
            "delta": mp.delta, "answer_base": pred_base, "recipient_gold": ex.answer,
            "twin_question": mp.twin.question, "twin_answer": twin_answer, "twin_pred": twin_pred,
            "all_slot": all_slot, "single_slot": single, "prefix": prefix, "leave_one_out": loo,
        })
        if (pi + 1) % 25 == 0:
            print(f"patched {pi + 1}/{len(responsive)} pairs ({time.perf_counter() - t2:.0f}s)", flush=True)

    nr_records = []
    for mp, twin_pred, z in nonresponsive:
        ex = mp.original
        pred_base = base_info[ex.idx]["pred"]
        pred = answer_for(ex.question, override_at_pass={p: z[p] for p in sites})[0]
        nr_records.append({
            "recipient_idx": ex.idx, "step": mp.step, "delta": mp.delta, "answer_base": pred_base,
            "twin_answer": mp.twin.answer, "twin_pred": twin_pred, "pred": pred,
            "answer_changed": pred != pred_base,
            "matches_twin_pred": num_match([pred] if pred else [], twin_pred)
                                 and not num_match([pred_base] if pred_base else [], twin_pred),
            "matches_twin_gold": num_match([pred] if pred else [], mp.twin.answer),
        })
    patch_elapsed = time.perf_counter() - t2

    def summarize(key_fn) -> dict:
        n = len(records)
        changed = sum(key_fn(r)["answer_changed"] for r in records)
        matched = sum(key_fn(r)["matches_twin"] for r in records)
        return {"n": n, "answer_changed_rate": changed / n if n else 0.0,
                "answer_changed_wilson_ci": list(wilson_ci(changed, n)),
                "matches_twin_rate": matched / n if n else 0.0,
                "matches_twin_wilson_ci": list(wilson_ci(matched, n))}

    summary = {
        "all_slot": summarize(lambda r: r["all_slot"]),
        "single_slot": {p: summarize(lambda r, p=p: r["single_slot"][p]) for p in sites},
        "prefix": {p: summarize(lambda r, p=p: r["prefix"][p]) for p in sites},
        "leave_one_out": {p: summarize(lambda r, p=p: r["leave_one_out"][p]) for p in sites},
    }
    n_nr = len(nr_records)
    nr_matched = sum(r["matches_twin_pred"] for r in nr_records)
    nonresp_summary = {"n": n_nr, "matches_twin_pred_rate": nr_matched / n_nr if n_nr else None,
                       "matches_twin_pred_wilson_ci": list(wilson_ci(nr_matched, n_nr)),
                       "answer_changed_rate": sum(r["answer_changed"] for r in nr_records) / n_nr if n_nr else None,
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
    print(line("ALL-SLOT", summary["all_slot"]))
    for p in sites:
        print(line(f"single pass{p}", summary["single_slot"][p]) + " | " + line(f"leave-out pass{p}", summary["leave_one_out"][p]))
    for p in sites:
        print(line(f"prefix 0..{p}", summary["prefix"][p]))
    print(f"non-responsive: {json.dumps(nonresp_summary)}")
    print(f"CFR: {json.dumps(cfr)}")

    if mpa.smoke_n:
        print("smoke run -- not logging")
        return

    metrics = {
        "final_answer_accuracy": accuracy,
        "compute_steps": mpa.num_latents,
        "sec_per_example": decode_elapsed / len(examples),
        "intervention_accuracy": summary["all_slot"]["matches_twin_rate"],
        "extra": {
            "design": "E3 rerun at full n: same-problem minimal-pair patch (all/single/prefix/leave-one-out), "
                      "non-responsive pairs, full CFR log",
            "site_definition": "pass p = vector spliced into the p-th <|latent|> slot (pass 0 = hidden at <|start-latent|>)",
            "n_examples_decoded": len(examples), "n_qualified_pairs": len(records),
            **summary, "nonresponsive": nonresp_summary, "cfr": cfr,
            "decode_elapsed_sec": decode_elapsed, "patch_elapsed_sec": patch_elapsed,
            "n_trainable_params": n_params,
            "related_runs": ["20260920-195725_coconut_minimal-pair-patch", "20260926-235221_coconut_cfr-minimal-pairs"],
        },
    }
    record = RunRecord(
        run_id=new_run_id("coconut", mpa.slug),
        mechanism="coconut",
        stage=mpa.stage,
        model=ModelInfo(backbone=mpa.model_id, checkpoint=mpa.checkpoint_label, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(examples), seed=mpa.eval_seed),
        metrics=metrics,
        hyperparams={"num_latents": mpa.num_latents, "n_pairs_cap": mpa.n_pairs,
                     "n_nonresponsive_cap": mpa.n_nonresponsive, "deltas": list(DELTAS)},
        seed=mpa.pair_seed,
        hardware=f"{mpa.hardware} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        notes="E3 rerun, Coconut, full gold-trace set: minimal-pair patch (all/single/prefix/leave-one-out) "
              "+ non-responsive pairs + CFR log.",
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
