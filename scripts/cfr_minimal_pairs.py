#!/usr/bin/env python3
"""Counterfactual responsiveness (CFR) of the UNPATCHED model on the E3 minimal pairs --
the behavioral reference the E3 steering numbers (CODI 70.1%, Coconut 77.1%, all-slot)
need beside them. No GPU: everything comes from saved run records.

E3 (`patch_minimal_pair_{codi,coconut}.py`) only scored pairs where BOTH the original
and its perturbed twin are answered correctly, so "patched answer moves to the twin's
answer" is measured on a behaviorally responsive subset by construction. How big is that
subset? CFR here = P(twin answered correctly | original answered correctly), over every
candidate twin E3 actually decoded -- how often a one-number change to the input moves
the model's own answer to the right new value.

Aggregate CFR is exact from E3's manifest (`n_qualified_pairs / n_candidates_checked`).
The breakdown by step / delta / chain length needs the rejected candidates, which E3 did
not save, so it is reconstructed: E3 ran on a fixed 600-example slice (seed 0), and an
earlier run on the same examples holds greedy base correctness for each of them
(`SOURCES[...]["base"]`). Replaying E3's candidate construction on that (same
`generate_minimal_pair` calls, same `random.Random(pair_seed)` consumption order, same
shuffle) regenerates the candidate list; the first `n_candidates_checked` are the twins
E3 decoded, and a twin passed iff its question is among E3's saved qualified records.
The breakdown is reported only if the replay reproduces the logged counts exactly
(candidates, qualified). That holds for Coconut. For CODI it does not: no saved run
reproduces E3's own base-correct set (bf16 greedy flips across pods; one flipped example
early in the slice shifts every later `rng.choice`), so CODI gets the aggregate only --
its stratified CFR comes from the E3 rerun, which logs every candidate.

  uv run python scripts/cfr_minimal_pairs.py [--coconut-data-dir DIR] [--no-save]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from latentreasoning.data.gsm8k_aug import Example, load_gsm8k_aug
from latentreasoning.data.minimal_pairs import generate_minimal_pair
from latentreasoning.eval.counterfactual import parse_steps, qualifying_steps
from latentreasoning.runlog.manifest import RESULTS_DIR, DatasetInfo, ModelInfo, RunRecord, new_run_id

DELTAS = (1, -1, 2, -2, 3, -3)
DEFAULT_COCONUT_DATA_DIR = Path.home() / "Projects/are-lrms-easily-interpretable/data"

SOURCES = {
    # "base": a run with greedy base correctness for every example of the E3 slice
    "codi": {"e3": "20260920-190420_codi_minimal-pair-patch", "base": "20260919-184323_codi_decode-patch-full-eval"},
    "coconut": {"e3": "20260920-195725_coconut_minimal-pair-patch", "base": "20260920-085317_coconut_qualified-patch"},
}


def wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = x / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def rate(x: int, n: int) -> dict:
    return {"n": n, "k": x, "rate": x / n if n else None, "wilson_ci": list(wilson_ci(x, n))}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_slice(mechanism: str, coconut_data_dir: Path) -> list[Example]:
    """The exact 600-example slice E3 ran on, in the order it iterated it."""
    if mechanism == "codi":
        return load_gsm8k_aug(split="test", n=600, seed=0)
    rows = json.loads((coconut_data_dir / "gsm_valid-gold-reasoning-trace_test.json").read_text())
    examples = [Example(question=r["question"], rationale=" ".join(r["steps"]),
                        answer=r["answer"].strip(), idx=i) for i, r in enumerate(rows)]
    random.Random(0).shuffle(examples)
    return examples[:600]


def replay_candidates(examples: list[Example], correct: dict[int, bool], pair_seed: int) -> list:
    """E3's candidate construction, verbatim."""
    rng = random.Random(pair_seed)
    candidates = []
    for ex in examples:
        if not correct[ex.idx]:
            continue
        for k in qualifying_steps(parse_steps(ex.rationale)):
            for delta in DELTAS:
                mp = generate_minimal_pair(ex, k, delta, rng=rng)
                if mp is not None:
                    candidates.append(mp)
                    break
    rng.shuffle(candidates)
    return candidates


def reconstruct(mechanism: str, e3_manifest: dict, e3_records: list[dict], coconut_data_dir: Path):
    """Labelled checked candidates [(MinimalPair, twin_correct)] if the replay reproduces
    E3's logged counts exactly, else (None, reason)."""
    extra = e3_manifest["metrics"]["extra"]
    examples = load_slice(mechanism, coconut_data_dir)
    slice_idx = {ex.idx for ex in examples}
    correct = {r["idx"]: r["correct"] for r in read_jsonl(RESULTS_DIR / SOURCES[mechanism]["base"] / "predictions.jsonl")
               if r["idx"] in slice_idx}
    if set(correct) != slice_idx:
        return None, "base-correctness run does not cover the slice"
    candidates = replay_candidates(examples, correct, e3_manifest["seed"])
    checked = candidates[:extra["n_candidates_checked"]]
    qualified = {(r["recipient_idx"], r["step"], r["twin_question"]) for r in e3_records}
    labelled = [(mp, (mp.original.idx, mp.step, mp.twin.question) in qualified) for mp in checked]
    n_q = sum(ok for _, ok in labelled)
    if len(candidates) != extra["n_candidates_checked"] and len(checked) < extra["n_candidates_checked"]:
        return None, f"replayed {len(candidates)} candidates < {extra['n_candidates_checked']} checked"
    if n_q != extra["n_qualified_pairs"]:
        return None, (f"replay recovers {n_q}/{extra['n_qualified_pairs']} qualified pairs "
                      f"({len(candidates)} candidates replayed)")
    return labelled, None


def stratified(labelled: list) -> dict:
    def by(key_fn) -> dict:
        groups: dict = defaultdict(list)
        for mp, ok in labelled:
            groups[key_fn(mp)].append(ok)
        return {str(k): rate(sum(v), len(v)) for k, v in sorted(groups.items())}

    return {
        "by_step": by(lambda mp: mp.step),
        "by_abs_delta": by(lambda mp: abs(mp.delta)),
        "by_delta_sign": by(lambda mp: "+" if mp.delta > 0 else "-"),
        "by_chain_len": by(lambda mp: min(len(parse_steps(mp.original.rationale)), 5)),  # 5 = "5+"
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coconut-data-dir", type=Path, default=DEFAULT_COCONUT_DATA_DIR)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    for mechanism in ("codi", "coconut"):
        src = SOURCES[mechanism]
        e3_manifest = json.loads((RESULTS_DIR / src["e3"] / "manifest.json").read_text())
        e3_records = read_jsonl(RESULTS_DIR / src["e3"] / "predictions.jsonl")
        extra = e3_manifest["metrics"]["extra"]

        cfr = rate(extra["n_qualified_pairs"], extra["n_candidates_checked"])
        p_orig = extra["n_base_correct"] / extra["n_examples_decoded"]
        steer = extra["all_slot"]["matches_twin_rate"]
        context = {
            "p_original_correct": p_orig,
            "cfr_twin_correct_given_original_correct": cfr["rate"],
            "all_slot_matches_twin_on_responsive_pairs": steer,
            # of all checked (original-correct) candidates: the input change moves the answer
            # correctly AND the all-slot latent patch reproduces that move
            "all_slot_matches_twin_times_cfr": steer * cfr["rate"],
        }
        labelled, why_not = reconstruct(mechanism, e3_manifest, e3_records, args.coconut_data_dir)
        strat = stratified(labelled) if labelled else None

        print(f"\n=== {mechanism}: CFR = {cfr['k']}/{cfr['n']} = {cfr['rate']:.3f} "
              f"[{cfr['wilson_ci'][0]:.3f}, {cfr['wilson_ci'][1]:.3f}]")
        print("  context: " + ", ".join(f"{k}={v:.3f}" for k, v in context.items()))
        if strat:
            for name, table in strat.items():
                print(f"  {name}: " + ", ".join(f"{k}: {v['k']}/{v['n']}={v['rate']:.2f}" for k, v in table.items()))
        else:
            print(f"  stratified breakdown skipped: {why_not}")

        if args.no_save:
            continue
        record = RunRecord(
            run_id=new_run_id(mechanism, "cfr-minimal-pairs"),
            mechanism=mechanism,
            stage="full_run",
            model=ModelInfo(**e3_manifest["model"]),
            dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=extra["n_examples_decoded"], seed=0),
            metrics={
                "compute_steps": e3_manifest["metrics"]["compute_steps"],
                "extra": {
                    "design": "offline: counterfactual responsiveness of the unpatched model on E3's "
                              "minimal-pair candidates",
                    "cfr_definition": "P(twin answered correctly | original answered correctly), greedy, "
                                      "over every E3 candidate twin that was decoded",
                    "cfr": cfr, "context": context,
                    "stratified": strat, "stratified_skipped_reason": why_not,
                    "source_runs": [src["e3"], src["base"]],
                },
            },
            hyperparams={"deltas": list(DELTAS), "pair_seed": e3_manifest["seed"]},
            seed=e3_manifest["seed"],
            hardware="local-cpu (offline, no model)",
            notes="Offline CFR on E3 minimal pairs: how often the unpatched model gets the perturbed twin "
                  "right given it gets the original right -- the behavioral reference for E3 steering.",
        )
        predictions = ([{"recipient_idx": mp.original.idx, "step": mp.step, "delta": mp.delta,
                         "perturbed_number": mp.perturbed_number,
                         "chain_len": len(parse_steps(mp.original.rationale)),
                         "twin_answer": mp.twin.answer, "twin_question": mp.twin.question, "twin_correct": ok}
                        for mp, ok in labelled] if labelled else [])
        path = record.save(predictions=predictions)
        print(f"  saved {path.parent}")


if __name__ == "__main__":
    main()
