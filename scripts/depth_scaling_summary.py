"""Summarize the depth2 depth-vs-layers scan (scripts/depth_scaling_fewshot.py runs): one row per model (and per
Ouro loop count) with layer count, params, per-level no-CoT accuracy, reach, 50% crossing and CoT reference; plus
rank correlations of reach with layers and with params, and a plot. No model deps.

    uv run --extra viz python scripts/depth_scaling_summary.py [--out results/depth2_layer_scan]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"


def nominal_params(name: str) -> float | None:
    """vLLM runs log n_params=None; fall back to the size in the model name (e.g. Qwen2.5-0.5B, pythia-160m)."""
    m = re.search(r"-(\d+(?:\.\d+)?)([bm])(?:-|$)", name.lower())
    return float(m.group(1)) * (1e9 if m.group(2) == "b" else 1e6) if m else None


def load_rows() -> list[dict]:
    rows = []
    for m in sorted(RESULTS.glob("*_baseline_fewshot_nocot_depth2-fewshot-*/manifest.json")):
        r = json.loads(m.read_text())
        ex = r["metrics"]["extra"]
        name = r["model"]["backbone"]
        cot = ex.get("cot_reference") or {}
        for tag, res in ex["nocot"].items():
            rows.append({
                "model": name if tag == "nocot" else f"{name} x{tag[1:]} loops",
                "family": name.split("/")[-1].split("-")[0],
                "layers": ex["n_layers"], "eff_layers": ex["effective_layers"][tag],
                "params": r["model"]["n_params"] or nominal_params(name),
                "acc": {int(k): v for k, v in res["per_level_accuracy"].items()},
                "reach": res["reach_above_floor"], "x50": res["crossing_50"], "x25": res["crossing_25"],
                "overall": res["overall"], "cot": cot.get("overall"),
                "cot_acc": {int(k): v for k, v in cot.get("per_level_accuracy", {}).items()},
                "run_id": r["run_id"],
            })
    return rows


def spearman(x, y) -> float:
    rx, ry = (np.argsort(np.argsort(v)).astype(float) for v in (x, y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None, help="dir for summary.md + depth_scan.png")
    a = p.parse_args()
    rows = sorted(load_rows(), key=lambda r: (r["family"], r["eff_layers"], r["params"] or 0))
    if not rows:
        raise SystemExit("no depth2 few-shot runs found")
    levels = sorted({L for r in rows for L in r["acc"]})
    fmt = lambda v: "-" if v is None else f"{v:.2f}"
    lines = ["| model | layers (eff.) | params (B) | " + " | ".join(f"L{L}" for L in levels)
             + " | reach | 50% at | CoT ref |", "|" + "---|" * (len(levels) + 6)]
    for r in rows:
        prm = f"{r['params'] / 1e9:.2f}" if r["params"] else "-"
        lines.append(f"| {r['model']} | {r['layers']} ({r['eff_layers']}) | {prm} | "
                     + " | ".join(fmt(r["acc"].get(L)) for L in levels)
                     + f" | {r['reach']} | {fmt(r['x50'])} | {fmt(r['cot'])} |")
    # Reach is near-constant within a family, so pooled correlations mostly measure family. Report within-family
    # rank correlations of the level-1 and level-2 accuracy with layers and with params instead.
    lines += ["", "| family | n | metric | Spearman vs layers | Spearman vs params |", "|---|---|---|---|---|"]
    for fam in sorted({r["family"] for r in rows}):
        fr = [r for r in rows if r["family"] == fam and "loops" not in r["model"] and r["params"]]
        if len(fr) < 3:
            continue
        for L in (1, 2):
            acc = [r["acc"][L] for r in fr]
            lines.append(f"| {fam} | {len(fr)} | acc L{L} | {spearman([r['layers'] for r in fr], acc):.2f} | "
                         f"{spearman([r['params'] for r in fr], acc):.2f} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.md").write_text(text + "\n")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fams = sorted({r["family"] for r in rows})
        fig, axes = plt.subplots(1, len(fams) + 1, figsize=(4.2 * (len(fams) + 1), 3.8))
        for ax, fam in zip(axes, fams):
            fr = [r for r in rows if r["family"] == fam]
            cmap = plt.get_cmap("viridis")
            for i, r in enumerate(fr):
                ax.plot(sorted(r["acc"]), [r["acc"][L] for L in sorted(r["acc"])], marker="o", ms=3, color=cmap(i / max(1, len(fr) - 1)),
                        label=f"{r['model'].split('/')[-1]} ({r['eff_layers']}L)")
            ax.set_title(fam)
            ax.set_xlabel("rules (depth)")
            ax.set_ylim(0, 1)
            ax.axhline(0.08, color="grey", lw=0.8, ls=":")
            ax.legend(fontsize=6)
        axes[0].set_ylabel("no-CoT accuracy")
        ax = axes[-1]
        for fam in fams:
            fr = [r for r in rows if r["family"] == fam]
            ax.plot([r["eff_layers"] for r in fr], [r["reach"] for r in fr], marker="o", label=fam)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("(effective) layers")
        ax.set_ylabel("reach (levels above floor)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "depth_scan.png", dpi=150)
        print(f"wrote {out}/summary.md, depth_scan.png")


if __name__ == "__main__":
    main()
