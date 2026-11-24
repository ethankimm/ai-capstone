#!/usr/bin/env python3
"""EXPLORATORY (COCONUT_CODI_PLAN.md Step 0b, not preregistered): latent use vs serial depth on real GSM8K, from the logged
per-example necessity records (idx -> `load_gsm8k_aug(split="test")` order) joined with the number of calculator steps in the gold
rationale (`Example.intermediate_values`). Free: no model is run.

For each run (CODI / Coconut x GPT-2 / Llama-1B) and each replacement (zero / mean / noise on all latents, plus trunc_0):
  per step-count bucket: n, intact accuracy (trunc_<N>), accuracy under the intervention, PWC = P(correct | intact correct),
  use = 1 - PWC (share of intact-correct items that break), and the same Wilson CIs.
H3-style test among intact-correct items: logistic regression of (ablated correct) on step count, one-sided LRT for a NEGATIVE
slope (more steps -> lower PWC -> more reliance). Caveat: GSM8K step count is confounded with problem difficulty, wording and
magnitude of numbers; this is a lead, not a test of the depth2 hypotheses.

  uv run python scripts/gsm8k_use_vs_depth.py [--out /tmp/gsm8k_use_vs_depth.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug

RESULTS = Path(__file__).resolve().parents[1] / "results"
RUNS = {
    "CODI gpt2": "20260927-005051_codi_necessity-all-replacements",
    "Coconut gpt2": "20260927-005834_coconut_necessity-all-replacements",
    "CODI llama1b": "20261009-231032_codi_llama1b-necessity-all-replacements",
    "Coconut llama1b": "20261009-231035_coconut_llama1b-necessity-all-replacements",
}
CONDS = ("ablate_all_zero", "ablate_all_mean", "ablate_all_noise", "trunc_0")
BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 99)]


def wilson(x, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - m, c + m)


def logistic_lrt_neg_slope(x, y, iters=50):
    """One-sided LRT that the slope of y~x is negative. Returns (slope, p)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    X = np.c_[np.ones_like(x), x - x.mean()]

    def fit(Xm):
        b = np.zeros(Xm.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-Xm @ b))
            g = Xm.T @ (y - p)
            H = (Xm * (p * (1 - p))[:, None]).T @ Xm + 1e-9 * np.eye(Xm.shape[1])
            step = np.linalg.solve(H, g)
            b += step
            if np.abs(step).max() < 1e-8:
                break
        p = np.clip(1 / (1 + np.exp(-Xm @ b)), 1e-12, 1 - 1e-12)
        return b, float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())

    b1, ll1 = fit(X)
    _, ll0 = fit(X[:, :1])
    lrt = max(0.0, 2 * (ll1 - ll0))
    p_two = math.erfc(math.sqrt(lrt / 2))
    return float(b1[1]), (p_two / 2 if b1[1] < 0 else 1 - p_two / 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ex = load_gsm8k_aug(split="test", n=None, seed=None)
    steps = {e.idx: len(e.intermediate_values) for e in ex}
    out = {}
    for label, run in RUNS.items():
        rows = [json.loads(l) for l in (RESULTS / run / "predictions.jsonl").read_text().splitlines()]
        sites = sorted(c for c in rows[0]["greedy"] if c.startswith("trunc_"))
        intact_name = sites[-1]  # trunc_<N>: all latents
        out[label] = {"intact_condition": intact_name, "conditions": {}}
        print(f"\n=== {label}  ({run})  intact = {intact_name}")
        for cond in CONDS:
            print(f"-- {cond}")
            print(f"{'steps':>6} {'n':>5} {'intact':>7} {'cond':>6} {'PWC':>6} {'use=1-PWC':>10}  95% CI(use)")
            per, xs, ys = {}, [], []
            for lo, hi in BUCKETS:
                items = [r for r in rows if lo <= steps[r["idx"]] <= hi]
                ok = [r for r in items if r["greedy"][intact_name]["correct"]]
                kept = sum(r["greedy"][cond]["correct"] for r in ok)
                n, n_ok = len(items), len(ok)
                pwc = kept / n_ok if n_ok else float("nan")
                lo_ci, hi_ci = wilson(n_ok - kept, n_ok)
                acc_i = n_ok / n if n else float("nan")
                acc_c = sum(r["greedy"][cond]["correct"] for r in items) / n if n else float("nan")
                tag = f"{lo}" if lo == hi else f"{lo}+"
                print(f"{tag:>6} {n:5d} {acc_i:7.3f} {acc_c:6.3f} {pwc:6.3f} {1 - pwc:10.3f}  [{lo_ci:.3f}, {hi_ci:.3f}]")
                per[tag] = {"n": n, "intact_acc": acc_i, "cond_acc": acc_c, "pwc": pwc, "use": 1 - pwc, "use_ci": [lo_ci, hi_ci], "n_intact_correct": n_ok}
            for r in rows:
                if r["greedy"][intact_name]["correct"] and steps[r["idx"]] >= 1:
                    xs.append(min(steps[r["idx"]], 6))
                    ys.append(float(r["greedy"][cond]["correct"]))
            slope, p = logistic_lrt_neg_slope(xs, ys)
            print(f"   logit slope of ablated-correct on step count (intact-correct items, steps capped at 6): {slope:+.3f}, one-sided p(negative) = {p:.3g}, n={len(xs)}")
            out[label]["conditions"][cond] = {"buckets": per, "slope": slope, "p_negative_one_sided": p, "n": len(xs)}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
