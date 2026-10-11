#!/usr/bin/env python3
"""Phase A step 1: does twin-transplant steering depend on how much the answer changes?

Reads the `predictions.jsonl` of twin-magnitude runs (`patch_minimal_pair_{codi,coconut}.py
--change_set ... --conditions all_slot`), one record per qualified pair (original and twin
both answered correctly), and reports per run:

- steer rate (all-slot transplant lands on the twin's answer) and fraction moved,
  (patched - original) / (twin - original), by relative answer change, in the bins of
  `latentreasoning.eval.twin_magnitude.REL_BINS`, with Wilson 95% intervals;
- the same split by answer size (original answer < 100 vs >= 100) and by change kind
  (additive vs multiplicative);
- a logistic fit  steer ~ 1 + log(rel answer change) + log(|original answer|), Newton/IRLS,
  Wald standard errors.

Primary population: pairs passing the hidden-operand filter
(`minimal_pairs.passes_hidden_operand_filter`); all qualified pairs as secondary.

  uv run python scripts/analyze_twin_magnitude.py results/<codi_run> results/<coconut_run> \\
      --out-json /tmp/mag.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from latentreasoning.eval.twin_magnitude import REL_BINS, bin_summary, wilson_ci


def load_rows(run_dir: Path) -> list[dict]:
    rows = []
    for line in (run_dir / "predictions.jsonl").read_text().splitlines():
        r = json.loads(line)
        if not r.get("rel_answer_change"):
            # None: original answer 0, relative change undefined. 0: the changed step does not
            # reach the final answer (e.g. the answer is an earlier step), so steering is undefined
            # (matches_twin is False by construction). Both excluded; counted in `n_excluded`.
            continue
        rows.append({
            "matches_twin": bool(r["all_slot"]["matches_twin"]),
            "answer_changed": bool(r["all_slot"]["answer_changed"]),
            "rel_answer_change": float(r["rel_answer_change"]),
            "abs_answer_change": float(r["abs_answer_change"]),
            "fraction_moved": r.get("fraction_moved"),
            "orig_answer": abs(float(str(r["answer_base"]).replace(",", ""))),
            "change": r.get("change") or "",
            "hidden_operand_ok": bool(r.get("hidden_operand_ok", True)),
            "step": r.get("step"), "chain_len": r.get("chain_len"),
        })
    return rows


def logistic_fit(rows: list[dict]) -> dict | None:
    if len(rows) < 20:
        return None
    y = np.array([r["matches_twin"] for r in rows], dtype=float)
    X = np.column_stack([
        np.ones(len(rows)),
        np.log([r["rel_answer_change"] for r in rows]),
        np.log([max(r["orig_answer"], 1.0) for r in rows]),
    ])
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ beta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    se = np.sqrt(np.diag(cov))
    names = ["intercept", "log_rel_answer_change", "log_abs_original_answer"]
    out = {"n": len(rows)}
    for i, nm in enumerate(names):
        z = beta[i] / se[i]
        out[nm] = {"coef": float(beta[i]), "se": float(se[i]), "z": float(z),
                   "p": float(math.erfc(abs(z) / math.sqrt(2)))}
    return out


def rate(rows: list[dict]) -> dict:
    n, k = len(rows), sum(r["matches_twin"] for r in rows)
    return {"n": n, "steer_rate": k / n if n else None, "steer_wilson_ci": list(wilson_ci(k, n))}


def analyze(rows: list[dict]) -> dict:
    return {
        "overall": rate(rows),
        "by_rel_change": bin_summary(rows),
        "by_rel_change_answer_lt100": bin_summary([r for r in rows if r["orig_answer"] < 100]),
        "by_rel_change_answer_ge100": bin_summary([r for r in rows if r["orig_answer"] >= 100]),
        "by_change_kind": {"additive": rate([r for r in rows if not r["change"].startswith("x")]),
                           "multiplicative": rate([r for r in rows if r["change"].startswith("x")])},
        "logistic": logistic_fit(rows),
    }


def fmt_bins(b: dict) -> list[str]:
    lines = ["| rel. answer change | n | steer [95% CI] | mean frac. moved | clipped mean | median |",
             "|---|---|---|---|---|---|"]
    for label, _, _ in REL_BINS:
        v = b[label]
        if not v["n"]:
            lines.append(f"| {label} | 0 | | | | |")
            continue
        lo, hi = v["steer_wilson_ci"]
        f = lambda x: "" if x is None else f"{x:.2f}"  # noqa: E731
        lines.append(f"| {label} | {v['n']} | {v['steer_rate']:.3f} [{lo:.3f}, {hi:.3f}] | "
                     f"{f(v['mean_fraction_moved'])} | {f(v['mean_fraction_moved_clipped'])} | "
                     f"{f(v['median_fraction_moved'])} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--out-json", type=Path)
    args = ap.parse_args()
    report, md = {}, []
    for d in args.run_dirs:
        rows = load_rows(d)
        n_total = sum(1 for _ in (d / "predictions.jsonl").open())
        res = {"n_excluded_undefined_change": n_total - len(rows),"filtered": analyze([r for r in rows if r["hidden_operand_ok"]]), "all": analyze(rows)}
        report[d.name] = res
        md.append(f"_{n_total - len(rows)} of {n_total} pairs excluded: twin answer equal to the original, "
                  "or original answer 0._\n")
        for pop in ("filtered", "all"):
            r = res[pop]
            md.append(f"### {d.name}: {pop} (n={r['overall']['n']}, steer {r['overall']['steer_rate']:.3f})")
            md += fmt_bins(r["by_rel_change"])
            lg = r["logistic"]
            if lg:
                md.append("")
                md.append("logistic steer ~ log rel change + log |answer|: " + ", ".join(
                    f"{k} {lg[k]['coef']:+.3f} (se {lg[k]['se']:.3f}, p={lg[k]['p']:.2g})"
                    for k in ("log_rel_answer_change", "log_abs_original_answer")))
            md.append("")
    print("\n".join(md))
    if args.out_json:
        args.out_json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
