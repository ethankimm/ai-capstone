"""Transfer-ladder levels 2-4 (RESEARCH_PLAN §4 P1): donor construction and scoring shared
by `patch_ladder_codi.py` and `patch_ladder_coconut.py`. Pure Python, no model.

Ladder (level 1 = same-problem minimal-pair twin, E3, `patch_minimal_pair_*.py`):
  L2  different problem, same number of steps, same operator sequence
  L3  different problem, same number of steps, different operator sequence
  L4  any other problem (the random-donor control; E2's cross-problem donor)
Recipients are base-correct and must have a base-correct L2 AND L3 donor, so all three
levels are measured on the same recipients. Donors are base-correct, with a different
final answer from the recipient.

Scoring, per patched answer (first match wins):
  unparseable     no number extracted
  unchanged       == recipient's own (unpatched) answer
  donor_final     == donor's final answer (the donor's whole program transferred)
  cf_joint        == the recipient's program re-run with EVERY intermediate step value
                  replaced by the donor's value at the same step index (latents carry the
                  intermediate values; the final operation's question operands are read
                  from the recipient's own context). Only when distinct from donor_final.
  cf_single       == the recipient's chain re-run with ONE step's value replaced by the
                  donor's value at that step index (any step)
  recipient_intermediate / donor_intermediate   == a non-final step value of either chain
  other           anything else
Null: each patched answer is re-bucketed against a random other pair's donor at the same
level (`reps` permutations) -- a bucket means something only above its null.
"""
from __future__ import annotations

import random
from collections import Counter

from latentreasoning.eval.counterfactual import (
    counterfactual_answer, fmt_num, num_equal, parse_steps, safe_eval, substitute,
)

LEVELS = ("L2", "L3", "L4")
BUCKETS = ("unparseable", "unchanged", "donor_final", "cf_joint", "cf_single",
           "recipient_intermediate", "donor_intermediate", "other")
SIGNAL_BUCKETS = ("donor_final", "cf_joint", "cf_single", "recipient_intermediate", "donor_intermediate")


def op_signature(steps: list[dict]) -> tuple[str, ...]:
    return tuple("".join(c for c in st["expr"] if c in "+-*/") for st in steps)


def chain_ok(steps: list[dict], answer: str) -> bool:
    """Every step evaluates to its written value and the last step is the gold answer."""
    if len(steps) < 2 or not num_equal(steps[-1]["val"], answer):
        return False
    for st in steps:
        v = safe_eval(st["expr"])
        if v is None or not num_equal(fmt_num(v), st["val"]):
            return False
    return True


def joint_counterfactual(r_steps: list[dict], d_steps: list[dict]) -> str | None:
    """Recipient's program with each intermediate result j < n-1 forced to the donor's
    step-j value; later steps re-evaluated with the substituted operands."""
    mapping: dict[str, str] = {}
    val = None
    for j, st in enumerate(r_steps):
        if j < len(r_steps) - 1 and j < len(d_steps):
            val = d_steps[j]["val"]
        else:
            v = safe_eval(substitute(st["expr"], mapping))
            if v is None:
                return None
            val = fmt_num(v)
        if st["val"] not in mapping:
            mapping[st["val"]] = val
    return val


def build_ladder_pairs(examples: list, correct: dict[int, bool], rng: random.Random, n_max: int) -> list[dict]:
    """[{recipient, L2, L3, L4}] Example objects, up to `n_max` recipients."""
    info = {}
    for ex in examples:
        if not correct.get(ex.idx):
            continue
        steps = parse_steps(ex.rationale)
        if chain_ok(steps, ex.answer):
            info[ex.idx] = (ex, steps, op_signature(steps))
    idxs = sorted(info)
    by_len: dict[int, list[int]] = {}
    for i in idxs:
        by_len.setdefault(len(info[i][1]), []).append(i)

    order = list(idxs)
    rng.shuffle(order)
    pairs = []
    for r in order:
        ex, steps, sig = info[r]
        same_len = [d for d in by_len[len(steps)] if d != r and not num_equal(info[d][0].answer, ex.answer)]
        l2 = [d for d in same_len if info[d][2] == sig]
        l3 = [d for d in same_len if info[d][2] != sig]
        l4 = [d for d in idxs if d != r and not num_equal(info[d][0].answer, ex.answer)]
        if not (l2 and l3 and l4):
            continue
        pairs.append({"recipient": ex, "L2": info[rng.choice(l2)][0], "L3": info[rng.choice(l3)][0],
                      "L4": info[rng.choice(l4)][0]})
        if len(pairs) >= n_max:
            break
    return pairs


def targets(recipient, donor) -> dict:
    r_steps, d_steps = parse_steps(recipient.rationale), parse_steps(donor.rationale)
    singles = set()
    for k in range(min(len(r_steps), len(d_steps)) - 1):
        cf = counterfactual_answer(r_steps, k, d_steps[k]["val"], recipient.answer)
        if cf is not None:
            singles.add(cf)
    return {"donor_final": donor.answer, "cf_joint": joint_counterfactual(r_steps, d_steps),
            "cf_single": sorted(singles),
            "recipient_inter": [s["val"] for s in r_steps[:-1]], "donor_inter": [s["val"] for s in d_steps[:-1]]}


def bucket(pred: str | None, base: str | None, t: dict) -> str:
    if pred is None:
        return "unparseable"
    if num_equal(pred, base):
        return "unchanged"
    if num_equal(pred, t["donor_final"]):
        return "donor_final"
    if t["cf_joint"] is not None and num_equal(pred, t["cf_joint"]):
        return "cf_joint"
    if any(num_equal(pred, c) for c in t["cf_single"]):
        return "cf_single"
    if any(num_equal(pred, v) for v in t["recipient_inter"]):
        return "recipient_intermediate"
    if any(num_equal(pred, v) for v in t["donor_inter"]):
        return "donor_intermediate"
    return "other"


def wilson(x: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [max(0.0, c - m), min(1.0, c + m)]


def summarize(rows: list[dict], cond: str, level: str, reps: int, rng: random.Random) -> dict:
    """rows: records with row[level]["preds"][cond], row["base"], row[level]["targets"]."""
    n = len(rows)
    counts = Counter(bucket(r[level]["preds"][cond], r["base"], r[level]["targets"]) for r in rows)
    null = Counter()
    for _ in range(reps):
        perm = list(range(n))
        rng.shuffle(perm)
        for i, j in enumerate(perm):
            if i == j:
                continue
            null[bucket(rows[i][level]["preds"][cond], rows[i]["base"], rows[j][level]["targets"])] += 1
    null_n = sum(null.values()) or 1
    return {
        "n": n,
        "rates": {b: counts[b] / n if n else 0.0 for b in BUCKETS},
        "counts": {b: counts[b] for b in BUCKETS},
        "null_rates": {b: null[b] / null_n for b in BUCKETS},
        "donor_final_wilson_ci": wilson(counts["donor_final"], n),
        "cf_joint_wilson_ci": wilson(counts["cf_joint"], n),
        "answer_changed_rate": 1 - counts["unchanged"] / n if n else 0.0,
        "n_cf_joint_distinct": sum(r[level]["targets"]["cf_joint"] is not None
                                   and not num_equal(r[level]["targets"]["cf_joint"], r[level]["targets"]["donor_final"])
                                   for r in rows),
    }


def format_row(name: str, s: dict) -> str:
    parts = [f"{b[:10]}={s['rates'][b]:.3f}" + (f"(null {s['null_rates'][b]:.3f})" if b in SIGNAL_BUCKETS else "")
             for b in BUCKETS if s["counts"][b] or b in SIGNAL_BUCKETS]
    return f"{name:28s} n={s['n']:4d} " + " ".join(parts)
