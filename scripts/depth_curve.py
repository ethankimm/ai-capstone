#!/usr/bin/env python3
"""Accuracy-vs-level curves for the synthetic depth / width tasks, plus paired McNemar.

Inputs are run dirs (`results/<run_id>/`, reads `predictions.jsonl`) or predictions
files directly. Each record needs a `level` (int) and a correctness field (`correct`,
bool). The McNemar mode also needs a per-example key to pair on (`id`, falling back to
`idx`).

Per model:
  * per-level accuracy with Wilson 95% intervals;
  * a four-parameter sigmoid fit, weighted by per-level n:
        acc(l) = floor + (ceiling - floor) / (1 + exp(slope * (l - midpoint_param)))
    canonicalized to slope >= 0, so `floor` is the high-level asymptote and
    `ceiling` the low-level one. scipy isn't a dependency of this repo, so the fit
    is a grid search over (midpoint_param, slope) with closed-form weighted least
    squares for (floor, ceiling) at each grid point, then a finer grid around the
    best point. floor/ceiling are clipped to [0, 1]. Deterministic, no init needed;
  * **midpoint** (the plan's "50% point") = the level where the FITTED curve crosses
    halfway between the model's fitted floor and its best OBSERVED per-level
    accuracy. If the fitted curve doesn't cross that value inside
    [min level, max level], the midpoint is reported as "beyond range"
    (`midpoint=None`, `midpoint_status="beyond_range"`, plus the extrapolated value
    when one exists). Note that `midpoint` != the `midpoint_param` of the sigmoid
    unless the best observed accuracy equals the fitted ceiling;
  * **AUC** = trapezoidal area under the per-level accuracy curve divided by the
    level span (max - min), i.e. in [0, 1]; a single level gives its accuracy.

Paired comparison (`--mcnemar runA runB --min-level 6`): pairs examples by id, pooled
over levels >= min-level; reports acc_A, acc_B, diff = acc_A - acc_B with a 95% CI by
Newcombe's (1998) hybrid-score method for paired proportions (his method 10, no
continuity correction; Wilson intervals for each marginal combined via the phi
correlation of the 2x2 table), and the exact two-sided McNemar p (binomial on the
discordant pairs, b vs c).

Collapsed depth (`--by collapsed`, depth task only): the depth ops are + - * on integers with
no wrap, so consecutive additive steps fold into one offset and consecutive multiplications
into one product. collapsed depth = 1 + number of switches between additive (+ -) and
multiplicative (*) ops -- a tighter bound on the serial depth than the nominal level (the
chain is still affine in x0, so even this overstates it). `--by collapsed` recomputes it
from each record's `question` and uses it in place of `level` for the curves and McNemar.

Usage:
  uv run python scripts/depth_curve.py results/<run1> results/<run2> [--json out.json] [--md out.md]
  uv run python scripts/depth_curve.py results/<run> --by collapsed
  uv run python scripts/depth_curve.py --mcnemar results/<runA> results/<runB> --min-level 6

Library (no deps beyond numpy + stdlib, so it also runs from the Coconut/CODI venvs):
  sys.path.insert(0, "<repo>/scripts"); from depth_curve import curve_metrics
  metrics["extra"].update(curve_metrics(records))
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

Z95 = 1.959963984540054


# ---------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------
def load_records(path: str | Path) -> list[dict]:
    """A run dir (reads predictions.jsonl) or a .jsonl / .json predictions file."""
    p = Path(path)
    if p.is_dir():
        p = p / "predictions.jsonl"
    text = p.read_text()
    if p.suffix == ".json":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def label_for(path: str | Path) -> str:
    """Run dir -> its name; results/<run_id>/predictions*.jsonl -> run_id; other files -> stem."""
    p = Path(path)
    if p.is_dir():
        return p.name
    return p.parent.name if p.stem.startswith("predictions") and p.parent.name else p.stem


# ---------------------------------------------------------------------------------------
# per-level accuracy
# ---------------------------------------------------------------------------------------
def wilson_ci(x: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = x / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def per_level(records: Iterable[dict], level_key: str = "level", correct_key: str = "correct") -> list[dict]:
    """Sorted list of {level, n, n_correct, accuracy, ci95}."""
    counts: dict[int, list[int]] = {}
    for r in records:
        if r.get(level_key) is None:
            raise KeyError(f"record without {level_key!r}: {str(r)[:200]}")
        c = counts.setdefault(int(r[level_key]), [0, 0])
        c[0] += 1
        c[1] += bool(r[correct_key])
    return [
        {"level": lvl, "n": n, "n_correct": k, "accuracy": k / n, "ci95": list(wilson_ci(k, n))}
        for lvl, (n, k) in sorted(counts.items())
    ]


# ---------------------------------------------------------------------------------------
# four-parameter sigmoid
# ---------------------------------------------------------------------------------------
def sigmoid4(level, floor: float, ceiling: float, midpoint: float, slope: float):
    x = np.clip(slope * (np.asarray(level, dtype=float) - midpoint), -60, 60)
    return floor + (ceiling - floor) / (1.0 + np.exp(x))


def _grid_fit(levels: np.ndarray, y: np.ndarray, w: np.ndarray, mids: np.ndarray, slopes: np.ndarray):
    """For every (mid, slope) pair: closed-form weighted LS of y ~ floor + delta * s,
    s = 1/(1+exp(slope*(l-mid))); floor/ceiling clipped to [0,1]; returns best."""
    M, S = np.meshgrid(mids, slopes, indexing="ij")
    M, S = M.ravel(), S.ravel()
    sig = 1.0 / (1.0 + np.exp(np.clip(S[:, None] * (levels[None, :] - M[:, None]), -60, 60)))
    sw = w.sum()
    ss = (sig * w).sum(1)
    sss = (sig * sig * w).sum(1)
    sy = (w * y).sum()
    ssy = (sig * w * y).sum(1)
    det = sw * sss - ss * ss
    ok = det > 1e-12
    delta = np.where(ok, (sw * ssy - ss * sy) / np.where(ok, det, 1.0), 0.0)
    floor = (sy - delta * ss) / sw
    ceiling = floor + delta
    floor, ceiling = np.clip(floor, 0, 1), np.clip(ceiling, 0, 1)
    pred = floor[:, None] + (ceiling - floor)[:, None] * sig
    sse = ((pred - y[None, :]) ** 2 * w[None, :]).sum(1)
    i = int(np.argmin(sse))
    return float(floor[i]), float(ceiling[i]), float(M[i]), float(S[i]), float(sse[i])


def fit_sigmoid4(levels: Sequence[float], accs: Sequence[float], weights: Sequence[float] | None = None) -> dict:
    """Weighted 4PL fit. Returns {floor, ceiling, midpoint_param, slope, sse, rmse}.
    Canonical form: slope >= 0, floor = asymptote as level -> +inf."""
    lv = np.asarray(levels, dtype=float)
    y = np.asarray(accs, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    if len(lv) == 0:
        raise ValueError("no levels to fit")
    lo, hi = float(lv.min()), float(lv.max())
    span = max(hi - lo, 1.0)
    mags = np.logspace(-2, 1.5, 36)
    slopes = np.concatenate([-mags[::-1], mags])
    mids = np.linspace(lo - span, hi + span, 121)
    f, c, m, s, sse = _grid_fit(lv, y, w, mids, slopes)
    # refine around the coarse optimum (two rounds)
    dm, fac = mids[1] - mids[0], 1.25
    for _ in range(2):
        mids2 = np.linspace(m - 2 * dm, m + 2 * dm, 41)
        mag = abs(s)
        slopes2 = np.sign(s) * np.geomspace(mag / fac ** 2, mag * fac ** 2, 41)
        f2, c2, m2, s2, sse2 = _grid_fit(lv, y, w, mids2, slopes2)
        if sse2 <= sse:
            f, c, m, s, sse = f2, c2, m2, s2, sse2
        dm, fac = dm / 10, fac ** 0.25
    if s < 0:  # canonicalize: same curve with floor/ceiling swapped and slope negated
        f, c, s = c, f, -s
    return {"floor": f, "ceiling": c, "midpoint_param": m, "slope": s,
            "sse": sse, "rmse": math.sqrt(sse / w.sum())}


def halfway_level(fit: dict, best_observed: float, lo: float, hi: float) -> tuple[float | None, str, float | None]:
    """Level where the fitted curve equals (floor + best_observed) / 2.
    Returns (midpoint or None, status, extrapolated crossing or None)."""
    f, c, m, s = fit["floor"], fit["ceiling"], fit["midpoint_param"], fit["slope"]
    target = (f + best_observed) / 2
    crossing = None
    if s > 0 and abs(c - f) > 1e-9:
        r = (target - f) / (c - f)
        if 0 < r < 1:
            crossing = m + math.log(1 / r - 1) / s
    y_lo, y_hi = float(sigmoid4(lo, f, c, m, s)), float(sigmoid4(hi, f, c, m, s))
    if crossing is not None and min(y_lo, y_hi) <= target <= max(y_lo, y_hi) and lo <= crossing <= hi:
        return crossing, "in_range", crossing
    return None, "beyond_range", crossing


def normalized_auc(levels: Sequence[float], accs: Sequence[float]) -> float:
    lv = np.asarray(levels, dtype=float)
    y = np.asarray(accs, dtype=float)
    order = np.argsort(lv)
    lv, y = lv[order], y[order]
    if len(lv) == 1:
        return float(y[0])
    area = float(np.sum((lv[1:] - lv[:-1]) * (y[1:] + y[:-1]) / 2))
    return area / float(lv[-1] - lv[0])


FLAT_RANGE = 0.10  # best - worst per-level accuracy below this -> no midpoint ("flat": True)


def curve_metrics(records: Iterable[dict], level_key: str = "level", correct_key: str = "correct") -> dict[str, Any]:
    """Everything for `metrics["extra"]`: per-level accuracy / n / Wilson CI (keys are
    the level as a string, JSON-friendly), the 4PL fit, midpoint (+status), AUC."""
    rows = per_level(records, level_key, correct_key)
    levels = [r["level"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    fit = fit_sigmoid4(levels, accs, [r["n"] for r in rows])
    best = max(accs)
    mid, status, extrap = halfway_level(fit, best, min(levels), max(levels))
    flat = best - min(accs) < FLAT_RANGE
    if flat:
        # no drop inside the range: a floor/best halfway crossing would be fit noise
        # (e.g. CoT at 0.996-1.0), so report it like a curve that never drops
        mid, status = None, "beyond_range"
    return {
        "per_level_accuracy": {str(r["level"]): r["accuracy"] for r in rows},
        "per_level_n": {str(r["level"]): r["n"] for r in rows},
        "per_level_ci95": {str(r["level"]): r["ci95"] for r in rows},
        "curve_fit": {**fit, "best_observed": best, "halfway_target": (fit["floor"] + best) / 2,
                      "model": "floor + (ceiling-floor)/(1+exp(slope*(level-midpoint_param)))"},
        "midpoint": mid,
        "midpoint_status": status,
        "flat": flat,
        "midpoint_extrapolated": extrap,
        "auc": normalized_auc(levels, accs),
    }


# ---------------------------------------------------------------------------------------
# paired McNemar
# ---------------------------------------------------------------------------------------
def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar: 2 * P(X <= min(b, c)), X ~ Bin(b + c, 1/2), capped at 1."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def newcombe_paired_ci(a: int, b: int, c: int, d: int, z: float = Z95) -> tuple[float, float]:
    """95% CI for p_A - p_B from a paired 2x2 table (a both right, b A-only, c B-only,
    d both wrong). Newcombe 1998, method 10 (Wilson score, no continuity correction)."""
    n = a + b + c + d
    if n == 0:
        return (0.0, 0.0)
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson_ci(a + b, n, z)
    l2, u2 = wilson_ci(a + c, n, z)
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(denom) if denom > 0 else 0.0
    diff = p1 - p2
    dl = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return (max(-1.0, diff - dl), min(1.0, diff + du))


def collapsed_depth(question: str) -> int:
    """1 + number of additive<->multiplicative switches in a depth-task chain (module docstring)."""
    from latentreasoning.tasks.synthetic import parse_depth_question  # lazy: CLI-only dependency
    kinds = ["M" if op == "*" else "A" for op in parse_depth_question(question).ops]
    return 1 + sum(kinds[i] != kinds[i - 1] for i in range(1, len(kinds)))


def with_collapsed_level(records: Iterable[dict]) -> list[dict]:
    """Copies of `records` with `level` replaced by the collapsed depth (original kept as `nominal_level`)."""
    return [{**r, "nominal_level": r["level"], "level": collapsed_depth(r["question"])} for r in records]


def _key(r: dict, id_key: str | None):
    if id_key:
        return r[id_key]
    return r["id"] if "id" in r else r["idx"]


def paired_mcnemar(records_a: Iterable[dict], records_b: Iterable[dict], min_level: int | None = None,
                   max_level: int | None = None, id_key: str | None = None,
                   level_key: str = "level", correct_key: str = "correct") -> dict[str, Any]:
    def index(recs):
        out = {}
        for r in recs:
            if min_level is not None or max_level is not None:
                lvl = int(r[level_key])
                if (min_level is not None and lvl < min_level) or (max_level is not None and lvl > max_level):
                    continue
            k = _key(r, id_key)
            if k in out:
                raise ValueError(f"duplicate example key {k!r}")
            out[k] = bool(r[correct_key])
        return out

    ia, ib = index(records_a), index(records_b)
    keys = sorted(set(ia) & set(ib), key=str)
    a = sum(ia[k] and ib[k] for k in keys)
    b = sum(ia[k] and not ib[k] for k in keys)
    c = sum(ib[k] and not ia[k] for k in keys)
    d = len(keys) - a - b - c
    n = len(keys)
    lo, hi = newcombe_paired_ci(a, b, c, d)
    return {
        "n_pairs": n, "n_only_in_a": len(set(ia) - set(ib)), "n_only_in_b": len(set(ib) - set(ia)),
        "min_level": min_level, "max_level": max_level,
        "acc_a": (a + b) / n if n else None, "acc_b": (a + c) / n if n else None,
        "diff": (b - c) / n if n else None, "diff_ci95": [lo, hi],
        "ci_method": "Newcombe 1998 method 10 (paired, Wilson-score hybrid, no CC)",
        "both_correct": a, "a_only": b, "b_only": c, "both_wrong": d,
        "mcnemar_exact_p": mcnemar_exact_p(b, c),
    }


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------
def _fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def markdown_table(results: dict[str, dict]) -> str:
    levels = sorted({int(l) for r in results.values() for l in r["per_level_accuracy"]})
    head = "| run | " + " | ".join(f"L{l}" for l in levels) + " | floor | ceiling | midpoint | AUC |"
    sep = "|" + "---|" * (len(levels) + 5)
    lines = [head, sep]
    for name, r in results.items():
        cells = []
        for l in levels:
            acc = r["per_level_accuracy"].get(str(l))
            if acc is None:
                cells.append("-")
            else:
                lo, hi = r["per_level_ci95"][str(l)]
                cells.append(f"{acc:.2f} [{lo:.2f},{hi:.2f}]")
        fit = r["curve_fit"]
        mid = _fmt(r["midpoint"], 2) if r["midpoint"] is not None else "beyond range"
        lines.append(f"| {name} | " + " | ".join(cells)
                     + f" | {fit['floor']:.2f} | {fit['ceiling']:.2f} | {mid} | {r['auc']:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="*", help="run dirs or predictions files")
    ap.add_argument("--mcnemar", nargs=2, metavar=("RUN_A", "RUN_B"))
    ap.add_argument("--min-level", type=int, default=None)
    ap.add_argument("--max-level", type=int, default=None)
    ap.add_argument("--id-key", default=None, help="pairing key (default: 'id', else 'idx')")
    ap.add_argument("--json", default=None, help="write results JSON here")
    ap.add_argument("--md", default=None, help="write markdown table here")
    ap.add_argument("--by", choices=["level", "collapsed"], default="level",
                    help="x axis: nominal level (default) or collapsed depth (depth task)")
    args = ap.parse_args(argv)
    load = load_records if args.by == "level" else (lambda p: with_collapsed_level(load_records(p)))

    if args.mcnemar:
        ra, rb = (load(p) for p in args.mcnemar)
        res = paired_mcnemar(ra, rb, args.min_level, args.max_level, args.id_key)
        res = {"a": str(args.mcnemar[0]), "b": str(args.mcnemar[1]), **res}
        print(json.dumps(res, indent=2))
        if args.json:
            Path(args.json).write_text(json.dumps(res, indent=2) + "\n")
        return

    if not args.inputs:
        ap.error("give run dirs / predictions files, or --mcnemar A B")
    results = {}
    for p in args.inputs:
        recs = load(p)
        if args.min_level is not None or args.max_level is not None:
            recs = [r for r in recs if (args.min_level is None or int(r["level"]) >= args.min_level)
                    and (args.max_level is None or int(r["level"]) <= args.max_level)]
        name = label_for(p)
        while name in results:  # never silently overwrite a same-named input
            name += "'"
        results[name] = curve_metrics(recs)
    md = markdown_table(results)
    print(md)
    for name, r in results.items():
        f = r["curve_fit"]
        print(f"{name}: floor={f['floor']:.3f} ceiling={f['ceiling']:.3f} slope={f['slope']:.3f} "
              f"midpoint={_fmt(r['midpoint'], 2)} ({r['midpoint_status']}, extrapolated "
              f"{_fmt(r['midpoint_extrapolated'], 2)}) auc={r['auc']:.3f} rmse={f['rmse']:.3f}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n")
    if args.md:
        Path(args.md).write_text(md)


if __name__ == "__main__":
    main()
