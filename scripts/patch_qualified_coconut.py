#!/usr/bin/env python3
"""E2 (`steered_to_donor_audit.md` §5): step-aligned, base-correct, controlled raw
single-slot patch for Coconut -- the Coconut counterpart to `patch_qualified_codi.py`,
testing EVERY latent pass (not just the empirically most/least decodable half used by
`decode_patch_coconut.py`'s grouped task).

Site -> step assignment is POSITIONAL (`latentreasoning.eval.counterfactual.site_to_step`),
splitting the `num_latents` passes into `max_step` contiguous groups -- deliberately not
a decoding-accuracy fit, so every pass gets tested regardless of how well it happens to
decode.

Qualified pairs (§4.2): recipient AND donor both base-correct (greedy final answer
matches gold); recipient's step-k gold value is an operand of step k+1's expression, so
injecting a different value there CAN propagate (`qualifying_steps`); donor supplies a
distinct step-k gold value. Three conditions per pair, same pass: real donor, an
independently-drawn random donor (distinct value too), mean ablation (mean live hidden
at that pass over a sample of base-correct examples). Scored with `score_patch` (Metric
B `matches_cf` primary, Metric A `matches_donor_final` secondary, full outcome
taxonomy) + paired exact McNemar (real vs random, real vs mean).

Run inside a venv pinned to Coconut's own requirements.txt, same convention as
`decode_patch_coconut.py`:

  cd /workspace/ai-capstone && .venv_coconut/bin/python scripts/patch_qualified_coconut.py \\
      --checkpoint_path /workspace/coconut_checkpoints/gsm-coconut/checkpoint_33 \\
      --data_dir /workspace/coconut_data \\
      --slug qualified-patch --stage pilot --hardware "RunPod RTX A5000 (secure)" \\
      --num_latents 6 --eval_n 600 --n_pairs_per_site 200
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coconut_common import (  # noqa: E402
    encode_question, extract_answer_after_delimiter, finish_and_decode, load_coconut,
    mcnemar_exact_p, parse_step_value, run_passes, wilson_ci,
)

from latentreasoning.eval.counterfactual import parse_steps, qualifying_steps, score_patch, site_to_step  # noqa: E402
from latentreasoning.eval.metrics import is_correct  # noqa: E402
from latentreasoning.mechanisms.coconut import name as mechanism_name  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402


@dataclass
class QualArguments:
    checkpoint_path: str = field(metadata={"help": "path to the downloaded checkpoint_33 file"})
    data_dir: str = field(metadata={"help": "dir with gsm_valid-gold-reasoning-trace_test.json"})
    model_id: str = field(default="openai-community/gpt2")
    slug: str = field(default="qualified-patch")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: str = field(default="hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33")
    num_latents: int = field(default=6)
    eval_n: int = field(default=600)
    eval_seed: int = field(default=0)
    n_pairs_per_site: int = field(default=200)
    mean_sample_n: int = field(default=200)
    max_new_tokens: int = field(default=48)
    device: str = field(default="cuda")


class Example:
    __slots__ = ("idx", "question", "answer", "steps")

    def __init__(self, idx, question, answer, steps):
        self.idx, self.question, self.answer, self.steps = idx, question, answer, steps


def load_examples(data_dir: str, eval_n: int, eval_seed: int) -> list[Example]:
    rows = json.loads((Path(data_dir) / "gsm_valid-gold-reasoning-trace_test.json").read_text())
    examples = [Example(i, r["question"], r["answer"].strip(), r["steps"]) for i, r in enumerate(rows)]
    random.Random(eval_seed).shuffle(examples)
    return examples[:eval_n]


def main() -> None:
    parser = transformers.HfArgumentParser((QualArguments,))
    (qa,) = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = qa.device

    tokenizer, base_model, embedding, special_ids = load_coconut(qa.model_id, qa.checkpoint_path, device)
    n_params = sum(p.numel() for p in base_model.parameters())
    eos_id = tokenizer.eos_token_id

    examples = load_examples(qa.data_dir, qa.eval_n, qa.eval_seed)
    by_idx = {ex.idx: ex for ex in examples}
    print(f"n_examples={len(examples)}, num_latents={qa.num_latents}")

    def answer_for(question: str, override_at_pass=None):
        input_ids, attn = encode_question(tokenizer, special_ids, question, qa.num_latents, device)
        pass_records, inputs_embeds, kv_cache, ncr = run_passes(
            base_model, embedding, input_ids, attn, device, qa.num_latents,
            override_at_pass=override_at_pass, latent_token_id=special_ids["latent"])
        raw = finish_and_decode(base_model, embedding, tokenizer, inputs_embeds, kv_cache, ncr, attn,
                                 device, qa.max_new_tokens, eos_id)
        return extract_answer_after_delimiter(raw), pass_records

    # ---- decode pass: base answer + correctness + cached per-pass live activations --------
    t0 = time.perf_counter()
    pass_cache: dict[int, list] = {}
    base_info: dict[int, dict] = {}
    for ex in examples:
        pred, pass_records = answer_for(ex.question)
        correct = is_correct(pred, ex.answer)
        pass_cache[ex.idx] = pass_records
        steps = parse_steps(" ".join(ex.steps))
        base_info[ex.idx] = {"pred": pred, "correct": correct, "steps": steps,
                              "qualifying": set(qualifying_steps(steps))}
        if len(base_info) % 50 == 0:
            print(f"decoded {len(base_info)}/{len(examples)}", flush=True)
    decode_elapsed = time.perf_counter() - t0
    accuracy = sum(v["correct"] for v in base_info.values()) / len(base_info)
    print(f"decode pass done in {decode_elapsed:.1f}s, base accuracy={accuracy:.3f} "
          f"(paper Table 1: GPT-2 Coconut GSM8k-Aug = 33.1%)")

    max_step = max((len(v["steps"]) for v in base_info.values()), default=0)
    rng = random.Random(0)
    correct_idxs = [idx for idx, v in base_info.items() if v["correct"]]
    print(f"base-correct pool: {len(correct_idxs)}/{len(examples)}, max_step={max_step}")

    def val_at(idx: int, k: int) -> Optional[str]:
        steps = base_info[idx]["steps"]
        return steps[k]["val"] if k < len(steps) else None

    # ---- mean live activation per pass, sampled from base-correct examples ----------------
    mean_sample = rng.sample(correct_idxs, min(qa.mean_sample_n, len(correct_idxs)))
    mean_vec = {}
    for p in range(qa.num_latents):
        vecs = [pass_cache[idx][p]["live_hidden"] for idx in mean_sample]
        mean_vec[p] = torch.stack(vecs, dim=0).mean(dim=0)
    print(f"mean activation computed from {len(mean_sample)} base-correct examples per site")

    # ---- build qualified pairs + run three conditions, per site ---------------------------
    site_records: dict[int, list[dict]] = {p: [] for p in range(qa.num_latents)}
    t1 = time.perf_counter()
    n_done = 0
    for p in range(qa.num_latents):
        k = site_to_step(p, qa.num_latents, max_step)
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
        print(f"site pass={p} step={k}: {len(recipient_pool)} qualifying recipients, {len(pairs)} pairs selected")

        for ridx, donor_idx, rand_idx in pairs:
            ex = by_idx[ridx]
            donor_vec = pass_cache[donor_idx][p]["live_hidden"]
            rand_vec = pass_cache[rand_idx][p]["live_hidden"]

            pred_base, _ = answer_for(ex.question)
            pred_real, _ = answer_for(ex.question, override_at_pass={p: donor_vec})
            pred_rand, _ = answer_for(ex.question, override_at_pass={p: rand_vec})
            pred_mean, _ = answer_for(ex.question, override_at_pass={p: mean_vec[p]})

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

            site_records[p].append({
                "recipient_idx": ridx, "donor_idx": donor_idx, "random_donor_idx": rand_idx,
                "step": k, "pass": p,
                "answer_base": pred_base, "answer_real": pred_real, "answer_random": pred_rand, "answer_mean": pred_mean,
                "recipient_value_at_step": val_at(ridx, k), "donor_value_at_step": donor_val,
                "random_donor_value_at_step": rand_val,
                "real": scored_real, "random": scored_rand, "mean": scored_mean,
            })
            n_done += 1
            if n_done % 50 == 0:
                with open("/tmp/coconut_qualified_patch_progress.jsonl", "w") as f:
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
    for p in range(qa.num_latents):
        records = site_records[p]
        k = site_to_step(p, qa.num_latents, max_step)
        site_summary[p] = {
            "step_target": k, "n_pairs": len(records),
            "real": summarize(records, "real"), "random": summarize(records, "random"), "mean": summarize(records, "mean"),
            "mcnemar_real_vs_random_answer_changed": mcnemar_key(records, "real", "random", "answer_changed"),
            "mcnemar_real_vs_mean_answer_changed": mcnemar_key(records, "real", "mean", "answer_changed"),
            "mcnemar_real_vs_random_matches_cf": mcnemar_key(records, "real", "random", "matches_cf"),
            "mcnemar_real_vs_mean_matches_cf": mcnemar_key(records, "real", "mean", "matches_cf"),
        }
        s = site_summary[p]
        print(f"site pass={p} step={k} n={s['n_pairs']}: "
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
        "compute_steps": qa.num_latents,
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
            "paper_table1_reference": {"gpt2_coconut_gsm8k_aug_accuracy": 0.331},
        },
    }

    record = RunRecord(
        run_id=new_run_id("coconut", qa.slug),
        mechanism=mechanism_name,
        stage=qa.stage,
        model=ModelInfo(backbone=qa.model_id, checkpoint=qa.checkpoint_label, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(examples), seed=qa.eval_seed),
        metrics=metrics,
        hyperparams={"num_latents": qa.num_latents, "n_pairs_per_site": qa.n_pairs_per_site,
                     "mean_sample_n": qa.mean_sample_n},
        seed=qa.eval_seed,
        hardware=f"{qa.hardware} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        notes="E2: step-aligned, base-correct, controlled raw single-slot patch at EVERY Coconut "
              "pass (positional site->step assignment, not decoding-accuracy-fit). Real donor / "
              "random donor / mean-ablation conditions, scored with the shared counterfactual "
              "module (Metric B matches_cf primary).",
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
