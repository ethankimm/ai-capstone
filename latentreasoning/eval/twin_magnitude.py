"""Change-size scoring for twin (minimal-pair) transplants, Phase A step 1.

Pure Python on purpose: the patch scripts run inside the CODI and Coconut venvs, which
don't share this repo's dependencies. The offline analysis with the logistic fit is
`scripts/analyze_twin_magnitude.py`.

Per qualified pair (original and twin both answered correctly):
  abs_answer_change  |twin answer - original answer|
  rel_answer_change  abs_answer_change / |original answer| (None if the original is 0)
  fraction_moved     (patched - original) / (twin - original): 1 = fully steered to the
                     twin, 0 = unchanged, None if the patched answer is unparseable
"""
from __future__ import annotations

# (label, lower exclusive, upper inclusive); the first bin includes 0.
REL_BINS = (
    ("<=5%", 0.0, 0.05),
    ("5-15%", 0.05, 0.15),
    ("15-40%", 0.15, 0.40),
    ("40-100%", 0.40, 1.00),
    (">100%", 1.00, float("inf")),
)


def _to_float(x) -> float | None:
    if x is None:
        return None
    try:
        return float(str(x).replace(",", "").strip())
    except ValueError:
        return None


def magnitude_fields(original_answer, twin_answer, patched_answer) -> dict:
    o, t, p = _to_float(original_answer), _to_float(twin_answer), _to_float(patched_answer)
    out = {"abs_answer_change": None, "rel_answer_change": None, "fraction_moved": None}
    if o is None or t is None:
        return out
    out["abs_answer_change"] = abs(t - o)
    out["rel_answer_change"] = abs(t - o) / abs(o) if o != 0 else None
    if p is not None and t != o:
        out["fraction_moved"] = (p - o) / (t - o)
    return out


def rel_bin(rel: float | None) -> str | None:
    if rel is None:
        return None
    for label, lo, hi in REL_BINS:
        if (lo == 0.0 and rel <= hi) or (lo < rel <= hi):
            return label
    return None


def wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = x / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def bin_summary(rows: list[dict], steer_key: str = "matches_twin") -> dict:
    """rows: dicts with `rel_answer_change`, `fraction_moved` and a boolean `steer_key`.
    Returns per-bin n, steer rate + Wilson CI, mean fraction moved (over parseable rows)
    and mean clipped to [0, 1]."""
    out = {}
    for label, _, _ in REL_BINS:
        sel = [r for r in rows if rel_bin(r.get("rel_answer_change")) == label]
        n = len(sel)
        k = sum(bool(r[steer_key]) for r in sel)
        fm = [r["fraction_moved"] for r in sel if r.get("fraction_moved") is not None]
        out[label] = {
            "n": n, "steer_rate": k / n if n else None, "steer_wilson_ci": list(wilson_ci(k, n)),
            "mean_fraction_moved": sum(fm) / len(fm) if fm else None,
            "mean_fraction_moved_clipped": sum(min(1.0, max(0.0, f)) for f in fm) / len(fm) if fm else None,
            "median_fraction_moved": _median(fm),
            "n_fraction_parseable": len(fm),
        }
    return out
