#!/usr/bin/env python3
"""GSM8K-Aug pilot (GSM8K_PILOT.md): paired CODI - CODI-0 gain on GSM8K test, from the logged predictions.

Same paired statistics as scripts/rulechain_gain_analysis.py (exact McNemar, Newcombe paired 95% CI, Holm, normalized
gain G / (1 - acc_ref)), adapted to unbalanced levels (level = number of rationale steps, 0-8): overall test, per step
count (Holm over levels), pooled bands, and a paired item bootstrap of the overall G. No d50: step count is not a
controlled difficulty axis here. Also reports the teacher-forced and teacher (CoT) accuracies the eval logged.

Usage: uv run python scripts/gsm8k_gain_analysis.py [--boot 2000] [--md out.md] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rulechain_gain_analysis import holm, norm_gain, paired_table, pfmt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUNS = {
    "codi": "20261104-012327_codi_gsm8k-aug-pilot-gpt2_gsm8k_aug_codi",
    "codi0": "20261103-225715_codi_gsm8k-aug-pilot-gpt2_gsm8k_aug_codi0",
}
BANDS = {"L0-2": (0, 2), "L3-4": (3, 4), "L5-8": (5, 8)}


def load(name: str) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, open(REPO / "results" / RUNS[name] / "predictions.jsonl"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=5980)
    ap.add_argument("--md")
    ap.add_argument("--json")
    args = ap.parse_args()
    lat, ref = load("codi"), load("codi0")
    assert lat.keys() == ref.keys(), "predictions are not on the same items"
    ids = sorted(lat)
    lvl = np.array([lat[i]["level"] for i in ids])
    get = lambda d, k: np.array([bool(d[i].get(k)) for i in ids])  # noqa: E731
    a, b = get(lat, "correct"), get(ref, "correct")

    out: dict = {"runs": RUNS, "n": len(ids), "boot": args.boot, "seed": args.seed}
    out["overall"] = {**paired_table(a, b), "norm_gain": norm_gain(a, b)}
    rng = np.random.default_rng(args.seed)
    bs = rng.integers(0, len(ids), size=(args.boot, len(ids)))
    out["overall"]["G_boot_ci95"] = [float(x) for x in np.percentile(a[bs].mean(1) - b[bs].mean(1), [2.5, 97.5])]
    levels = sorted(set(lvl.tolist()))
    per = {k: {**paired_table(a[lvl == k], b[lvl == k]), "norm_gain": norm_gain(a[lvl == k], b[lvl == k])} for k in levels}
    for k, p in zip(levels, holm([per[k]["mcnemar_p"] for k in levels])):
        per[k]["holm_p"] = p
    out["per_level"] = per
    out["bands"] = {n: paired_table(a[(lvl >= lo) & (lvl <= hi)], b[(lvl >= lo) & (lvl <= hi)]) for n, (lo, hi) in BANDS.items()}
    out["other"] = {m: {"tf_correct": float(get(d, "tf_correct").mean()), "teacher_correct": float(get(d, "teacher_correct").mean()),
                        "unparseable": float(np.mean([d[i]["predicted_answer"] in (None, "") for i in ids]))}
                    for m, d in (("codi", lat), ("codi0", ref))}

    f = lambda x: f"{x:+.3f}" if x else "0"  # noqa: E731
    ci = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]"  # noqa: E731
    o = out["overall"]
    L = ["# GSM8K-Aug pilot: paired CODI - CODI-0 gain (test)\n",
         f"GSM8K_PILOT.md. GSM8K test, n = {out['n']}. Level = rationale step count. G = CODI - CODI-0. "
         f"Bootstrap B = {args.boot} (seed {args.seed}).\n",
         "## Overall\n", "| CODI | CODI-0 | G | 95% CI (Newcombe) | 95% CI (bootstrap) | CODI-only / CODI-0-only | McNemar p | norm. gain |",
         "|---|---|---|---|---|---|---|---|",
         f"| {o['acc_latent']:.3f} | {o['acc_ref']:.3f} | {f(o['G'])} | {ci(o['G_ci95'])} | {ci(o['G_boot_ci95'])} | "
         f"{o['latent_only']} / {o['ref_only']} | {pfmt(o['mcnemar_p'])} | {f(o['norm_gain'])} |\n",
         "## By step count\n", "| Steps | n | CODI | CODI-0 | G | 95% CI | CODI-only / CODI-0-only | McNemar p | Holm p |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, p in per.items():
        L.append(f"| {k} | {p['n']} | {p['acc_latent']:.3f} | {p['acc_ref']:.3f} | {f(p['G'])} | {ci(p['G_ci95'])} | "
                 f"{p['latent_only']} / {p['ref_only']} | {pfmt(p['mcnemar_p'])} | {pfmt(p['holm_p'])} |")
    L += ["", "| Band | n | CODI | CODI-0 | G | 95% CI | CODI-only / CODI-0-only | McNemar p |", "|---|---|---|---|---|---|---|---|"]
    for n, p in out["bands"].items():
        L.append(f"| {n} | {p['n']} | {p['acc_latent']:.3f} | {p['acc_ref']:.3f} | {f(p['G'])} | {ci(p['G_ci95'])} | "
                 f"{p['latent_only']} / {p['ref_only']} | {pfmt(p['mcnemar_p'])} |")
    L += ["", "## Other accuracies (test)\n", "| Model | generated | teacher-forced at distillation position | teacher (writes CoT) | unparseable |",
          "|---|---|---|---|---|"]
    for m, d in out["other"].items():
        acc = (o["acc_latent"] if m == "codi" else o["acc_ref"])
        L.append(f"| {m} | {acc:.3f} | {d['tf_correct']:.3f} | {d['teacher_correct']:.3f} | {d['unparseable']:.3f} |")
    md = "\n".join(L) + "\n"
    print(md)
    if args.md:
        Path(args.md).write_text(md)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
