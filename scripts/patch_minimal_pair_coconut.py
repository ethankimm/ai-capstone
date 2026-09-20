#!/usr/bin/env python3
"""E3 (`steered_to_donor_audit.md` §5): same-problem minimal-pair donor-interchange
patching for Coconut -- the Coconut counterpart to `patch_minimal_pair_codi.py`. Donor
= the recipient's OWN question with one eligible number perturbed and the gold chain
re-executed (`latentreasoning.data.minimal_pairs`), so the twin's final answer
literally IS the counterfactual value (Metric A == Metric B by construction) and E2's
cross-problem confound is gone entirely.

(a) ALL-SLOT: override every latent pass's live hidden with the twin's own live hidden
    at that pass. Upper bound on what the thought chain carries at all.
(b) SINGLE-SLOT sweep (every pass alone) and CUMULATIVE PREFIX {0..i} to localize which
    pass(es), if any, carry it.

Qualified pairs: recipient base-correct AND twin base-correct (greedy decode), the
perturbed step a `qualifying_steps` position. `matches_twin` = patched answer equals
the twin's own ground-truth final answer and the base answer did not already equal it
-- valid here specifically because of the minimal-pair construction (see
`latentreasoning/data/minimal_pairs.py`'s module docstring).

Run inside the venv pinned to Coconut's own requirements.txt, same convention as
`patch_qualified_coconut.py`:

  cd /workspace/ai-capstone && .venv_coconut/bin/python scripts/patch_minimal_pair_coconut.py \\
      --checkpoint_path /workspace/coconut_checkpoints/gsm-coconut/checkpoint_33 \\
      --data_dir /workspace/coconut_data \\
      --slug minimal-pair-patch --stage pilot --hardware "RunPod RTX A5000 (secure)" \\
      --num_latents 6 --eval_n 600 --n_pairs 150
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
    slug: str = field(default="minimal-pair-patch")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: str = field(default="hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33")
    num_latents: int = field(default=6)
    eval_n: int = field(default=600)
    eval_seed: int = field(default=0)
    n_pairs: int = field(default=150)
    max_candidate_checks: int = field(default=800, metadata={"help": "cap on twin-decode checks so a low hit-rate can't blow the budget"})
    max_new_tokens: int = field(default=48)
    pair_seed: int = field(default=0)
    device: str = field(default="cuda")


def load_examples(data_dir: str, eval_n: int, eval_seed: int) -> list[Example]:
    rows = json.loads((Path(data_dir) / "gsm_valid-gold-reasoning-trace_test.json").read_text())
    examples = [Example(question=r["question"], rationale=" ".join(r["steps"]),
                         answer=r["answer"].strip(), idx=i) for i, r in enumerate(rows)]
    random.Random(eval_seed).shuffle(examples)
    return examples[:eval_n]


def main() -> None:
    parser = transformers.HfArgumentParser((MPArguments,))
    (mpa,) = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = mpa.device

    tokenizer, base_model, embedding, special_ids = load_coconut(mpa.model_id, mpa.checkpoint_path, device)
    n_params = sum(p.numel() for p in base_model.parameters())
    eos_id = tokenizer.eos_token_id

    examples = load_examples(mpa.data_dir, mpa.eval_n, mpa.eval_seed)
    print(f"n_examples={len(examples)}, num_latents={mpa.num_latents}")

    def answer_for(question: str, override_at_pass=None):
        input_ids, attn = encode_question(tokenizer, special_ids, question, mpa.num_latents, device)
        pass_records, inputs_embeds, kv_cache, ncr = run_passes(
            base_model, embedding, input_ids, attn, device, mpa.num_latents,
            override_at_pass=override_at_pass, latent_token_id=special_ids["latent"])
        raw = finish_and_decode(base_model, embedding, tokenizer, inputs_embeds, kv_cache, ncr, attn,
                                 device, mpa.max_new_tokens, eos_id)
        return extract_answer_after_delimiter(raw), pass_records

    def decode_one(question: str):
        pred, pass_records = answer_for(question)
        return pred, pass_records

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
        twin_pred, twin_pass_records = decode_one(mp.twin.question)
        if is_correct(twin_pred, mp.twin.answer):
            pairs.append(mp)
            twin_cache[id(mp)] = {"pred": twin_pred, "pass_records": twin_pass_records}
        if n_checked % 50 == 0:
            print(f"checked {n_checked} candidates, {len(pairs)} qualified so far", flush=True)
    print(f"qualified pairs (twin also base-correct): {len(pairs)}/{n_checked} candidates checked")

    # ---- all-slot / single-slot / cumulative-prefix patches, per qualified pair -----------
    records = []
    t1 = time.perf_counter()
    for pi, mp in enumerate(pairs):
        ex = mp.original
        twin_pass_records = twin_cache[id(mp)]["pass_records"]
        twin_vecs = {p: twin_pass_records[p]["live_hidden"] for p in range(mpa.num_latents)}
        pred_base = base_info[ex.idx]["pred"]
        twin_answer = mp.twin.answer

        def matches_twin(pred: Optional[str], pred_base=pred_base, twin_answer=twin_answer) -> bool:
            base_matched = num_match([pred_base] if pred_base else [], twin_answer)
            return num_match([pred] if pred else [], twin_answer) and not base_matched

        pred_all, _ = answer_for(ex.question, override_at_pass=twin_vecs)
        all_slot = {"pred": pred_all, "answer_changed": pred_all != pred_base, "matches_twin": matches_twin(pred_all)}

        single_slot = {}
        for p in range(mpa.num_latents):
            pred_p, _ = answer_for(ex.question, override_at_pass={p: twin_vecs[p]})
            single_slot[p] = {"pred": pred_p, "answer_changed": pred_p != pred_base, "matches_twin": matches_twin(pred_p)}

        prefix = {}
        for i in range(mpa.num_latents):
            pred_pre, _ = answer_for(ex.question, override_at_pass={p: twin_vecs[p] for p in range(i + 1)})
            prefix[i] = {"pred": pred_pre, "answer_changed": pred_pre != pred_base, "matches_twin": matches_twin(pred_pre)}

        records.append({
            "recipient_idx": ex.idx, "step": mp.step, "perturbed_number": mp.perturbed_number,
            "delta": mp.delta, "answer_base": pred_base, "recipient_gold": ex.answer,
            "twin_question": mp.twin.question, "twin_answer": twin_answer,
            "all_slot": all_slot, "single_slot": single_slot, "prefix": prefix,
        })
        if (pi + 1) % 10 == 0:
            with open("/tmp/coconut_minimal_pair_progress.jsonl", "w") as f:
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
    single_slot_summary = {p: summarize(lambda r, p=p: r["single_slot"][p]) for p in range(mpa.num_latents)}
    prefix_summary = {i: summarize(lambda r, i=i: r["prefix"][i]) for i in range(mpa.num_latents)}

    print(f"ALL-SLOT n={all_slot_summary['n']}: matches_twin={all_slot_summary['matches_twin_rate']:.3f} "
          f"answer_changed={all_slot_summary['answer_changed_rate']:.3f}")
    for p in range(mpa.num_latents):
        s = single_slot_summary[p]
        print(f"single-slot pass={p}: matches_twin={s['matches_twin_rate']:.3f} answer_changed={s['answer_changed_rate']:.3f}")
    for i in range(mpa.num_latents):
        s = prefix_summary[i]
        print(f"prefix 0..{i}: matches_twin={s['matches_twin_rate']:.3f} answer_changed={s['answer_changed_rate']:.3f}")

    metrics = {
        "final_answer_accuracy": accuracy,
        "compute_steps": mpa.num_latents,
        "sec_per_example": decode_elapsed / len(examples),
        "intervention_accuracy": all_slot_summary["matches_twin_rate"],
        "extra": {
            "design": "E3: same-problem minimal-pair donor-interchange patch (steered_to_donor_audit.md)",
            "n_examples_decoded": len(examples), "n_base_correct": len(correct_examples),
            "n_candidates_checked": n_checked, "n_qualified_pairs": len(pairs),
            "all_slot": all_slot_summary, "single_slot": single_slot_summary, "prefix": prefix_summary,
            "n_trainable_params": n_params,
            "related_runs": ["20260920-085317_coconut_qualified-patch"],
        },
    }

    record = RunRecord(
        run_id=new_run_id("coconut", mpa.slug),
        mechanism="coconut",
        stage=mpa.stage,
        model=ModelInfo(backbone=mpa.model_id, checkpoint=mpa.checkpoint_label, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(examples), seed=mpa.eval_seed),
        metrics=metrics,
        hyperparams={"num_latents": mpa.num_latents, "n_pairs_requested": mpa.n_pairs,
                     "max_candidate_checks": mpa.max_candidate_checks},
        seed=mpa.pair_seed,
        hardware=f"{mpa.hardware} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
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
