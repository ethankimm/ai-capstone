#!/usr/bin/env python3
"""Continuous-metric follow-up to `probe_codi.py`'s binary tol=0.01 hit-rate
(`20260920-032053_codi_probe-pilot`: ridge/MLP both at floor, "Next" note flagged that
a hard 1% threshold can't distinguish "no signal" from "signal, not to 1% precision").

Reads the `raw_predictions.jsonl` cache that `probe_codi.py` now writes alongside its
manifest (per-example, real-scale `y_gold`/`ridge_pred`/`mlp_pred` for every (iteration,
step) cell -- targets are already un-transformed out of signed-log1p by `probe_codi.py`
before being cached, so this script does no re-transformation itself). Computes R^2,
Pearson r, MAE, normalized MAE (MAE / std(y_gold)), and a tolerance-sweep hit-rate at
tau in {1%, 5%, 10%, 20%} per (iteration, step, predictor) cell, and renders diagnostic
scatter plots (predicted vs. gold) contrasting the CODI non-decodable placeholder
iterations {1, 4} (z0, z3) against the decodable comparison group {3, 5} (z2, z4) --
same grouping `probe_codi.py` uses, defined in `next_experiments.md` / the ablation run
`20260919-192228_codi_early-termination-ablate-all`.

Pure numpy/matplotlib -- no torch, no GPU, no model. Runs entirely locally on any
machine (including Apple Silicon) against an already-generated `raw_predictions.jsonl`.

Usage:
    uv run python scripts/probe_metrics_continuous.py \\
        --raw_predictions results/<run_id>/raw_predictions.jsonl \\
        --out_dir results/<run_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from continuous_metrics import DEFAULT_TOLERANCES, compute_metrics  # noqa: E402

NONDECODABLE_ITERS = (1, 4)  # z0, z3
DECODABLE_ITERS = (3, 5)  # z2, z4
PREDICTORS = ("ridge_pred", "mlp_pred")


def load_raw_predictions(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def group_by_iter_step(rows: list[dict]) -> dict[tuple[int, int], list[dict]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["iter"], r["step"])].append(r)
    return grouped


def compute_all_metrics(rows: list[dict], tolerances=DEFAULT_TOLERANCES) -> list[dict]:
    """One metrics dict per (iteration, step, predictor) cell."""
    grouped = group_by_iter_step(rows)
    out = []
    for (it, s), cell_rows in sorted(grouped.items()):
        y = [r["y_gold"] for r in cell_rows]
        for predictor in PREDICTORS:
            yhat = [r[predictor] for r in cell_rows]
            m = compute_metrics(y, yhat, tolerances=tolerances)
            out.append({"iter": it, "step": s, "predictor": predictor, **m})
    return out


def print_table(metrics: list[dict]) -> None:
    header = f"{'iter':>4} {'step':>4} {'predictor':>10} {'n':>4} {'r2':>8} {'pearson_r':>10} {'mae':>10} {'norm_mae':>9}"
    tol_keys = sorted(metrics[0]["tolerance_hit_rate"].keys(), key=float) if metrics else []
    header += "".join(f" {'hit@' + k:>9}" for k in tol_keys)
    print(header)
    for m in metrics:
        row = (f"{m['iter']:>4} {m['step']:>4} {m['predictor']:>10} {m['n']:>4} "
               f"{m['r2']:>8.3f} {m['pearson_r']:>10.3f} {m['mae']:>10.3f} {m['normalized_mae']:>9.3f}")
        row += "".join(f" {m['tolerance_hit_rate'][k]:>9.3f}" for k in tol_keys)
        print(row)


def summarize_group(metrics: list[dict], iters: tuple[int, ...], predictor: str) -> dict:
    cells = [m for m in metrics if m["iter"] in iters and m["predictor"] == predictor]
    if not cells:
        return {}

    def avg(key):
        vals = [c[key] for c in cells if c[key] == c[key]]  # drop NaN
        return sum(vals) / len(vals) if vals else None

    return {
        "iters": list(iters), "predictor": predictor, "n_cells": len(cells),
        "avg_r2": avg("r2"), "avg_pearson_r": avg("pearson_r"),
        "avg_mae": avg("mae"), "avg_normalized_mae": avg("normalized_mae"),
        "avg_hit_rate": {
            tau: sum(c["tolerance_hit_rate"][tau] for c in cells) / len(cells)
            for tau in cells[0]["tolerance_hit_rate"]
        },
    }


def plot_scatter(rows: list[dict], out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = group_by_iter_step(rows)
    focus_iters = sorted(set(NONDECODABLE_ITERS) | set(DECODABLE_ITERS))
    written = []
    for predictor, pred_label in zip(PREDICTORS, ("Ridge", "MLP")):
        steps = sorted({s for (_, s) in grouped if _ in focus_iters})
        if not steps:
            continue
        fig, axes = plt.subplots(len(focus_iters), len(steps),
                                  figsize=(4 * len(steps), 3.5 * len(focus_iters)), squeeze=False)
        for i, it in enumerate(focus_iters):
            group = "non-decodable (z{})".format(it - 1) if it in NONDECODABLE_ITERS else "decodable (z{})".format(it - 1)
            for j, s in enumerate(steps):
                ax = axes[i][j]
                cell_rows = grouped.get((it, s), [])
                if not cell_rows:
                    ax.set_visible(False)
                    continue
                y = [r["y_gold"] for r in cell_rows]
                yhat = [r[predictor] for r in cell_rows]
                ax.scatter(y, yhat, alpha=0.4, s=12)
                lo, hi = min(y + yhat), max(y + yhat)
                ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y=x")
                ax.set_title(f"iter={it} step={s}\n{group}", fontsize=9)
                ax.set_xlabel("gold")
                ax.set_ylabel("predicted")
        fig.suptitle(f"{pred_label} probe: predicted vs. gold, non-decodable {NONDECODABLE_ITERS} vs. decodable {DECODABLE_ITERS}")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = out_dir / f"scatter_{predictor}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        written.append(out_path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_predictions", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=None, help="defaults to raw_predictions' parent dir")
    ap.add_argument("--tolerances", type=float, nargs="+", default=list(DEFAULT_TOLERANCES))
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir or args.raw_predictions.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_raw_predictions(args.raw_predictions)
    print(f"loaded {len(rows)} raw (iter, step, example) rows from {args.raw_predictions}")

    metrics = compute_all_metrics(rows, tolerances=tuple(args.tolerances))
    print_table(metrics)

    summary = {
        "nondecodable": {p: summarize_group(metrics, NONDECODABLE_ITERS, p) for p in PREDICTORS},
        "decodable": {p: summarize_group(metrics, DECODABLE_ITERS, p) for p in PREDICTORS},
        "per_cell": metrics,
    }
    summary_path = out_dir / "continuous_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"wrote {summary_path}")

    for group_name, group_summary in (("NONDECODABLE", summary["nondecodable"]), ("DECODABLE", summary["decodable"])):
        for predictor, s in group_summary.items():
            if not s:
                continue
            print(f"{group_name} {predictor}: avg_r2={s['avg_r2']:.4f} avg_pearson_r={s['avg_pearson_r']:.4f} "
                  f"avg_mae={s['avg_mae']:.3f} avg_normalized_mae={s['avg_normalized_mae']:.3f} "
                  f"avg_hit_rate={s['avg_hit_rate']}")

    if not args.no_plots:
        written = plot_scatter(rows, out_dir)
        for p in written:
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
