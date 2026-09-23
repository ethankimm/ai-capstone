#!/usr/bin/env python3
"""Local, no-GPU rescore: ANY-iteration/pass top-1/top-5 decodability, vs. the already-
logged matched-iteration number. Reads a run's `predictions.jsonl` (per-slot top-1/top-5
was cached at decode time by `decode_patch_codi.py` / `decode_patch_coconut.py`) -- no
model, no network. Same "rescore already-logged predictions instead of paying for a new
run" precedent as `steered_to_donor_audit.md` E0.

Tests the hypothesis flagged in `20260919-184323_codi_decode-patch-full-eval`'s notes:
the paper's own Table-3-style metric doesn't commit to one "correct" iteration per step
the way `best_iter_for_step` does; a value appearing in ANY of the 6 iterations'/passes'
top-5 might close most of the gap to the paper's reported 97.1%/83.9%/75.0%.

Usage:
    uv run python scripts/rescore_any_iter_top5.py results/<run_id>/predictions.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def num_match(candidates: list[str], gold: str) -> bool:
    gold = gold.strip().lstrip("+")
    for c in candidates:
        c = c.strip()
        if c == gold:
            return True
        try:
            if float(c.replace(",", "")) == float(gold.replace(",", "")):
                return True
        except ValueError:
            continue
    return False


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def any_slot_hit(rec: dict, slots_key: str, gold_val: str, k: int) -> bool:
    field = "top1" if k == 1 else "top5"
    return any(
        num_match(slot[field] if k == 5 else [slot[field]], gold_val)
        for slot in rec[slots_key]
    )


def main() -> None:
    path = Path(sys.argv[1])
    all_records = load_records(path)
    # decode_patch_*.py's predictions.jsonl can interleave patch records (no per_iter/per_pass
    # key, "recipient_idx"/"donor_idx" instead of "idx") in the same file -- keep decode records
    # only.
    slots_key = "per_iter" if any("per_iter" in r for r in all_records) else "per_pass"
    records = [r for r in all_records if slots_key in r]

    # mirrors decode_patch_codi.py/decode_patch_coconut.py's own idx-parity held-out split --
    # the ANY-slot check doesn't need the half-A mapping fit at all (no circularity), but
    # scoring on the same half-B population keeps it directly comparable to the logged
    # matched-iteration/pass headline number.
    half_b = [r for r in records if r["idx"] % 2 == 1]

    any_hit1 = any_hit5 = any_n = 0
    for r in half_b:
        for gold_val in r["step_values"]:
            any_n += 1
            any_hit1 += any_slot_hit(r, slots_key, gold_val, 1)
            any_hit5 += any_slot_hit(r, slots_key, gold_val, 5)
    any_top1 = any_hit1 / any_n if any_n else 0.0
    any_top5 = any_hit5 / any_n if any_n else 0.0

    print(f"{path.parent.name}")
    print(f"  slots_key={slots_key}  half_b n_examples={len(half_b)}  n(example,step) pairs={any_n}")
    print(f"  ANY-slot top1={any_top1:.4f} ({any_hit1}/{any_n})  top5={any_top5:.4f} ({any_hit5}/{any_n})")

    by_step_count: dict[int, list[int]] = {}
    for r in half_b:
        if not r["correct"] or r["n_steps"] < 1:
            continue
        n = r["n_steps"]
        all_hit = all(any_slot_hit(r, slots_key, v, 5) for v in r["step_values"])
        c, tot = by_step_count.setdefault(n, [0, 0])
        by_step_count[n][1] += 1
        if all_hit:
            by_step_count[n][0] += 1

    print("  paper-style metric, ANY-slot top5, correct-only, by step count:")
    for n in sorted(by_step_count):
        c, tot = by_step_count[n]
        rate = c / tot if tot else 0.0
        print(f"    {n}-step: {c}/{tot} = {rate:.3f}")


if __name__ == "__main__":
    main()
