#!/usr/bin/env python3
"""RuleChain round 1: paired gain tests, bootstrap intervals, envelope and d50 (TRAINING_PROTOCOL.md section 3).

CPU only, reads the logged test predictions (`results/<run_id>/predictions.jsonl`) and the valid.json selection logs.

What it computes
  * Per level, for each latent model vs its matched reference (Coconut vs no_thoughts, CODI vs CODI-0) and vs the
    envelope: G(k) = acc_latent - acc_ref, Newcombe paired 95% CI, exact McNemar p, Holm-adjusted over the 12 levels,
    normalized gain G / (1 - acc_ref) with a paired item bootstrap CI. Pooled tests over level bands.
  * d50 per model = `depth_curve.halfway_level` of the 4PL fit (the same definition as the logged d50s), with a
    percentile bootstrap over items (resampled within level). Paired d50 differences use the same resampled items for
    both models (items are aligned by id).
  * Envelope = per level, the best no-latent model. Test envelope: max over plain no-CoT (plateau), no_thoughts, CODI-0,
    bootstrapped jointly over items. Valid envelope (the protocol's definition): from the per-level accuracies in the
    selection logs (no per-item valid predictions were saved), bootstrapped parametrically (independent binomials per
    model and level). Student-only diverged and is not a member.
  * Health check: each matched reference vs the envelope, per level.

Usage: uv run python scripts/rulechain_gain_analysis.py [--boot 2000] [--json out.json] [--md out.md]
       uv run python scripts/rulechain_gain_analysis.py --pilot [--boot 2000] [--json ...] [--md ...]

--pilot: the k <= 8 pilot (PILOT_K8.md), levels 1-8: each Coconut arm vs its matched no_thoughts (c2m3 vs no_thoughts_m3,
c2m6 and c1m6 vs no_thoughts_m6) and the arm-vs-arm contrasts c2m3 - c1m6 and c2m6 - c1m6. Same per-level paired tests,
Holm, normalized gain, pooled bands (L1-3, L4-6, L7-8) and paired d50 bootstrap; no envelope (all pilot references are
from one stack).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_curve import fit_sigmoid4, halfway_level, mcnemar_exact_p, newcombe_paired_ci  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LOGS = Path("/local-ssd/ktn48/lrm_work/logs")
LEVELS = list(range(1, 13))
FLAT_RANGE = 0.10  # as depth_curve.curve_metrics

RUNS = {
    "cot": "20261101-154307_explicit_cot_v2-depth2-coconutstack-cot-r2",
    "coconut": "20261102-022150_coconut_v2-depth2-gpt2_depth2_coconut_c1m6",
    "no_thoughts": "20261101-153922_baseline_no_cot_v2-depth2-gpt2_depth2_no_thoughts_m6",
    "no_cot": "20261101-154024_baseline_no_cot_v2-depth2-gpt2_depth2_no_cot_plateau",
    "codi": "20261102-000204_codi_v2-depth2-gpt2_depth2_codi_bs64-sel",
    "codi0": "20261102-060537_codi_v2-depth2-gpt2_depth2_codi0",
}
ENVELOPE_MEMBERS = ["no_cot", "no_thoughts", "codi0"]
PAIRS = [("coconut", "no_thoughts"), ("codi", "codi0")]
VALID_LOGS = {  # selection log, selected checkpoint is on the SELECTED line
    "no_cot": "select_gpt2_depth2_no_cot_plateau.txt",
    "no_thoughts": "select_gpt2_depth2_no_thoughts_m6.txt",
    "codi0": "select_gpt2_depth2_codi0.txt",
    "coconut": "select_gpt2_depth2_coconut_c1m6.txt",
    "codi": "select_gpt2_depth2_codi_bs64.txt",
}
VALID_N = {k: (167 if k <= 8 else 166) for k in LEVELS}  # depth2/valid.json level sizes


# ------------------------------------------------------------------------------------------------ loading
def load_test(name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """-> (correct[level_idx, item] bool, ids aligned, unparseable rate)."""
    recs = [json.loads(l) for l in open(REPO / "results" / RUNS[name] / "predictions.jsonl")]
    by_level: dict[int, list[tuple[str, bool]]] = {k: [] for k in LEVELS}
    unparse = 0
    for r in recs:
        by_level[int(r["level"])].append((r["id"], bool(r["correct"])))
        unparse += r.get("predicted_answer") in (None, "")
    n = {len(v) for v in by_level.values()}
    assert len(n) == 1, f"{name}: unequal level sizes {n}"
    ids = np.array([[i for i, _ in sorted(by_level[k])] for k in LEVELS])
    corr = np.array([[c for _, c in sorted(by_level[k])] for k in LEVELS])
    return corr, ids, unparse / len(recs)


def load_valid(name: str) -> dict[int, float]:
    text = (LOGS / VALID_LOGS[name]).read_text()
    sel = re.findall(r"SELECTED (\S+)", text)[-1]
    ck = Path(sel).name
    line = [l for l in text.splitlines() if l.startswith(ck + ":")][-1]
    accs = json.loads(line[line.index("{"):])
    return {int(k): float(v) for k, v in accs.items()}


# ------------------------------------------------------------------------------------------------ d50
def d50(accs: np.ndarray, n: np.ndarray) -> float:
    """Halfway level of the 4PL fit; nan if flat or out of range (as curve_metrics)."""
    if accs.max() - accs.min() < FLAT_RANGE:
        return float("nan")
    fit = fit_sigmoid4(LEVELS, accs, n)
    mid, _, _ = halfway_level(fit, float(accs.max()), LEVELS[0], LEVELS[-1])
    return float("nan") if mid is None else mid


def pct_ci(x: np.ndarray) -> list[float]:
    x = x[~np.isnan(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [math.nan, math.nan]


def holm(ps: list[float]) -> list[float]:
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def paired_table(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: bool arrays over the same items (a = latent, b = reference)."""
    both = int((a & b).sum()); a_only = int((a & ~b).sum()); b_only = int((~a & b).sum())
    neither = len(a) - both - a_only - b_only
    lo, hi = newcombe_paired_ci(both, a_only, b_only, neither)
    return {"n": len(a), "acc_latent": float(a.mean()), "acc_ref": float(b.mean()), "G": float(a.mean() - b.mean()),
            "G_ci95": [lo, hi], "latent_only": a_only, "ref_only": b_only,
            "mcnemar_p": mcnemar_exact_p(a_only, b_only)}


def norm_gain(a: np.ndarray, b: np.ndarray) -> float:
    ref = b.mean()
    return float("nan") if ref >= 1 else float((a.mean() - ref) / (1 - ref))


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=5980)
    ap.add_argument("--json")
    ap.add_argument("--md")
    ap.add_argument("--pilot", action="store_true", help="k <= 8 pilot runs (PILOT_K8.md)")
    args = ap.parse_args()
    if args.pilot:
        return pilot_main(args)
    rng = np.random.default_rng(args.seed)

    test, unparse = {}, {}
    ref_ids = None
    for name in RUNS:
        corr, ids, up = load_test(name)
        if ref_ids is None:
            ref_ids = ids
        assert (ids == ref_ids).all(), f"{name}: item ids differ"
        test[name], unparse[name] = corr, up
    n_item = test["cot"].shape[1]
    n_lvl = np.full(len(LEVELS), n_item)

    # bootstrap indices: resample items within level, shared by all models (paired)
    B = args.boot
    idx = rng.integers(0, n_item, size=(B, len(LEVELS), n_item))

    def boot_acc(corr: np.ndarray) -> np.ndarray:  # -> [B, levels]
        return np.take_along_axis(corr[None].repeat(B, 0), idx, axis=2).mean(2)

    acc = {k: v.mean(1) for k, v in test.items()}
    bacc = {k: boot_acc(v) for k, v in test.items()}
    env_test = np.max([acc[m] for m in ENVELOPE_MEMBERS], axis=0)
    benv_test = np.max([bacc[m] for m in ENVELOPE_MEMBERS], axis=0)
    env_test_who = [ENVELOPE_MEMBERS[i] for i in np.argmax([acc[m] for m in ENVELOPE_MEMBERS], axis=0)]

    # valid envelope (point + parametric bootstrap)
    valid = {m: np.array([load_valid(m)[k] for k in LEVELS]) for m in VALID_LOGS}
    vn = np.array([VALID_N[k] for k in LEVELS])
    env_valid = np.max([valid[m] for m in ENVELOPE_MEMBERS], axis=0)
    env_valid_who = [ENVELOPE_MEMBERS[i] for i in np.argmax([valid[m] for m in ENVELOPE_MEMBERS], axis=0)]
    bvalid = {m: rng.binomial(vn[None].repeat(B, 0), np.round(valid[m] * vn) / vn) / vn for m in ENVELOPE_MEMBERS}
    benv_valid = np.max([bvalid[m] for m in ENVELOPE_MEMBERS], axis=0)

    out: dict = {"boot": B, "seed": args.seed, "runs": RUNS, "unparseable_rate": unparse}

    # d50s
    d50s = {}
    curves = {**{k: (acc[k], bacc[k], n_lvl) for k in RUNS if k != "cot"},
              "envelope_test": (env_test, benv_test, n_lvl), "envelope_valid": (env_valid, benv_valid, vn)}
    bd50 = {}
    for name, (a, ba, nn) in curves.items():
        bd50[name] = np.array([d50(ba[i], nn) for i in range(B)])
        d50s[name] = {"d50": d50(a, nn), "ci95": pct_ci(bd50[name]),
                      "boot_undefined_frac": float(np.isnan(bd50[name]).mean())}
    out["d50"] = d50s
    out["envelope"] = {"test": {"acc": env_test.tolist(), "member": env_test_who},
                       "valid": {"acc": env_valid.tolist(), "member": env_valid_who,
                                 "note": "selection-log per-level accuracies; parametric bootstrap"}}

    # paired gains
    comps = {f"{a}-{b}": (a, b) for a, b in PAIRS}
    gains = {}
    for label, (lat, ref) in comps.items():
        rows = []
        for li, k in enumerate(LEVELS):
            t = paired_table(test[lat][li], test[ref][li])
            bg = (bacc[lat][:, li] - bacc[ref][:, li]) / np.where(bacc[ref][:, li] < 1, 1 - bacc[ref][:, li], np.nan)
            t.update(level=k, norm_gain=norm_gain(test[lat][li], test[ref][li]), norm_gain_ci95=pct_ci(bg))
            rows.append(t)
        for r, p in zip(rows, holm([r["mcnemar_p"] for r in rows])):
            r["mcnemar_p_holm"] = p
        bands = {"all": LEVELS, "L1-3": [1, 2, 3], "L4-6": [4, 5, 6], "L7-12": list(range(7, 13))}
        pooled = {}
        for bname, lv in bands.items():
            li = [k - 1 for k in lv]
            pooled[bname] = paired_table(test[lat][li].ravel(), test[ref][li].ravel())
        dd = bd50[lat] - bd50[ref]
        gains[label] = {"per_level": rows, "pooled": pooled,
                        "d50_diff": d50s[lat]["d50"] - d50s[ref]["d50"], "d50_diff_ci95": pct_ci(dd)}
    # latent vs test envelope (unpaired across models only through the max; bootstrap CI)
    for lat in ("coconut", "codi"):
        g = acc[lat] - env_test
        bg = bacc[lat] - benv_test
        gains[f"{lat}-envelope_test"] = {"per_level": [{"level": k, "G": float(g[i]), "G_ci95": pct_ci(bg[:, i])}
                                                       for i, k in enumerate(LEVELS)],
                                         "d50_diff": d50s[lat]["d50"] - d50s["envelope_test"]["d50"],
                                         "d50_diff_ci95": pct_ci(bd50[lat] - bd50["envelope_test"])}
    out["gains"] = gains

    # health check: matched reference vs envelope (valid, the protocol's definition; and test)
    out["health"] = {ref: {"valid_gap": (valid[ref] - env_valid).tolist(), "test_gap": (acc[ref] - env_test).tolist()}
                     for _, ref in PAIRS}

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    md = render(out, acc, valid)
    if args.md:
        Path(args.md).write_text(md)
    print(md)


def f3(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.3f}" if x < 0 or x > 0 else "0"


def ci(c):
    return "—" if any(math.isnan(v) for v in c) else f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def pfmt(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def render(out, acc, valid) -> str:
    L = []
    L.append(f"# RuleChain round 1: paired gain analysis\n\nBootstrap B = {out['boot']} (seed {out['seed']}), items "
             "resampled within level, shared across models. Test set, 500 items per level.\n")
    L.append("## d50 (halfway level of the 4PL fit) with 95% bootstrap CI\n")
    L.append("| Curve | d50 | 95% CI | boot undefined |\n|---|---|---|---|")
    for k, v in out["d50"].items():
        L.append(f"| {k} | {v['d50']:.2f} | [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] | {v['boot_undefined_frac']:.1%} |")
    L.append("\n## Envelope (best no-latent model per level)\n")
    L.append("| | " + " | ".join(f"L{k}" for k in LEVELS) + " |\n|---|" + "---|" * len(LEVELS))
    for split in ("valid", "test"):
        e = out["envelope"][split]
        L.append(f"| {split} acc | " + " | ".join(f"{a:.3f}" for a in e["acc"]) + " |")
        L.append(f"| {split} member | " + " | ".join(m.replace("no_thoughts", "no_th") for m in e["member"]) + " |")
    L.append("\n## Health check: matched reference − envelope (protocol: within a few points)\n")
    L.append("| Reference | split | " + " | ".join(f"L{k}" for k in LEVELS) + " |\n|---|---|" + "---|" * len(LEVELS))
    for ref, h in out["health"].items():
        for split in ("valid", "test"):
            L.append(f"| {ref} | {split} | " + " | ".join(f3(x) for x in h[f"{split}_gap"]) + " |")
    for label, g in out["gains"].items():
        L.append(f"\n## {label}\n")
        L.append(f"d50 difference {g['d50_diff']:+.2f}, 95% CI {ci(g['d50_diff_ci95'])}.\n")
        if "pooled" not in g:
            L.append("| Level | G | 95% CI |\n|---|---|---|")
            for r in g["per_level"]:
                L.append(f"| {r['level']} | {f3(r['G'])} | {ci(r['G_ci95'])} |")
            continue
        L.append("| Level | latent | ref | G | 95% CI (Newcombe) | lat-only / ref-only | McNemar p | Holm p | "
                 "norm. gain | 95% CI |\n|---|---|---|---|---|---|---|---|---|---|")
        for r in g["per_level"]:
            L.append(f"| {r['level']} | {r['acc_latent']:.3f} | {r['acc_ref']:.3f} | {f3(r['G'])} | {ci(r['G_ci95'])} | "
                     f"{r['latent_only']} / {r['ref_only']} | {pfmt(r['mcnemar_p'])} | {pfmt(r['mcnemar_p_holm'])} | "
                     f"{f3(r['norm_gain'])} | {ci(r['norm_gain_ci95'])} |")
        L.append("\n| Pooled | n | G | 95% CI | lat-only / ref-only | McNemar p |\n|---|---|---|---|---|---|")
        for b, r in g["pooled"].items():
            L.append(f"| {b} | {r['n']} | {f3(r['G'])} | {ci(r['G_ci95'])} | {r['latent_only']} / {r['ref_only']} | "
                     f"{pfmt(r['mcnemar_p'])} |")
    L.append("\n## Unparseable rate (test)\n")
    L.append(" · ".join(f"{k} {v:.3f}" for k, v in out["unparseable_rate"].items()))
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ pilot (PILOT_K8.md)
PILOT_RUNS = {
    "no_thoughts_m6": "20261102-215045_baseline_no_cot_v2-depth2-pilot_k8_no_thoughts_m6",
    "no_thoughts_m3": "20261102-222423_baseline_no_cot_v2-depth2-pilot_k8_no_thoughts_m3",
    "c2m3": "20261103-033009_coconut_v2-depth2-pilot_k8_coconut_c2m3",
    "c2m6": "20261103-110209_coconut_v2-depth2-pilot_k8_coconut_c2m6",
    "c1m6": "20261103-105353_coconut_v2-depth2-pilot_k8_coconut_c1m6",
}
PILOT_PAIRS = [("c2m3", "no_thoughts_m3"), ("c2m6", "no_thoughts_m6"), ("c1m6", "no_thoughts_m6"),
               ("c2m3", "c1m6"), ("c2m6", "c1m6")]
PILOT_BANDS = {"all": list(range(1, 9)), "L1-3": [1, 2, 3], "L4-6": [4, 5, 6], "L7-8": [7, 8]}


def pilot_main(args) -> None:
    global LEVELS, RUNS
    LEVELS, RUNS = list(range(1, 9)), PILOT_RUNS
    rng = np.random.default_rng(args.seed)
    test, unparse, ref_ids = {}, {}, None
    for name in RUNS:
        corr, ids, up = load_test(name)
        ref_ids = ids if ref_ids is None else ref_ids
        assert (ids == ref_ids).all(), f"{name}: item ids differ"
        test[name], unparse[name] = corr, up
    n_item = next(iter(test.values())).shape[1]
    n_lvl = np.full(len(LEVELS), n_item)
    B = args.boot
    idx = rng.integers(0, n_item, size=(B, len(LEVELS), n_item))
    bacc = {k: np.take_along_axis(v[None].repeat(B, 0), idx, axis=2).mean(2) for k, v in test.items()}
    bd50 = {k: np.array([d50(bacc[k][i], n_lvl) for i in range(B)]) for k in RUNS}
    d50s = {k: {"d50": d50(test[k].mean(1), n_lvl), "ci95": pct_ci(bd50[k]),
                "boot_undefined_frac": float(np.isnan(bd50[k]).mean())} for k in RUNS}
    gains = {}
    for lat, ref in PILOT_PAIRS:
        rows = []
        for li, k in enumerate(LEVELS):
            t = paired_table(test[lat][li], test[ref][li])
            bg = (bacc[lat][:, li] - bacc[ref][:, li]) / np.where(bacc[ref][:, li] < 1, 1 - bacc[ref][:, li], np.nan)
            t.update(level=k, norm_gain=norm_gain(test[lat][li], test[ref][li]), norm_gain_ci95=pct_ci(bg))
            rows.append(t)
        for r, pv in zip(rows, holm([r["mcnemar_p"] for r in rows])):
            r["mcnemar_p_holm"] = pv
        pooled = {b: paired_table(test[lat][[k - 1 for k in lv]].ravel(), test[ref][[k - 1 for k in lv]].ravel())
                  for b, lv in PILOT_BANDS.items()}
        gains[f"{lat}-{ref}"] = {"per_level": rows, "pooled": pooled, "d50_diff": d50s[lat]["d50"] - d50s[ref]["d50"],
                                 "d50_diff_ci95": pct_ci(bd50[lat] - bd50[ref])}
    out = {"boot": B, "seed": args.seed, "runs": RUNS, "unparseable_rate": unparse, "d50": d50s, "gains": gains}
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    L = [f"# RuleChain k <= 8 pilot: paired gain analysis\n\nPILOT_K8.md. Bootstrap B = {B} (seed {args.seed}), items "
         "resampled within level, shared across models. Test L1-8, 500 items per level. G = first − second.\n",
         "## d50 with 95% bootstrap CI\n", "| Run | d50 | 95% CI |\n|---|---|---|"]
    L += [f"| {k} | {v['d50']:.2f} | [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] |" for k, v in d50s.items()]
    L.append("\n## Per-level accuracy (test)\n")
    L.append("| Run | " + " | ".join(f"L{k}" for k in LEVELS) + " | all |\n|---|" + "---|" * (len(LEVELS) + 1))
    L += [f"| {k} | " + " | ".join(f"{a:.3f}" for a in v.mean(1)) + f" | {v.mean():.3f} |" for k, v in test.items()]
    for label, g in gains.items():
        L.append(f"\n## {label}\n\nd50 difference {g['d50_diff']:+.2f}, 95% CI {ci(g['d50_diff_ci95'])}.\n")
        L.append("| Level | first | second | G | 95% CI (Newcombe) | first-only / second-only | McNemar p | Holm p | "
                 "norm. gain | 95% CI |\n|---|---|---|---|---|---|---|---|---|---|")
        for r in g["per_level"]:
            L.append(f"| {r['level']} | {r['acc_latent']:.3f} | {r['acc_ref']:.3f} | {f3(r['G'])} | {ci(r['G_ci95'])} | "
                     f"{r['latent_only']} / {r['ref_only']} | {pfmt(r['mcnemar_p'])} | {pfmt(r['mcnemar_p_holm'])} | "
                     f"{f3(r['norm_gain'])} | {ci(r['norm_gain_ci95'])} |")
        L.append("\n| Pooled | n | G | 95% CI | first-only / second-only | McNemar p |\n|---|---|---|---|---|---|")
        for b, r in g["pooled"].items():
            L.append(f"| {b} | {r['n']} | {f3(r['G'])} | {ci(r['G_ci95'])} | {r['latent_only']} / {r['ref_only']} | "
                     f"{pfmt(r['mcnemar_p'])} |")
    L.append("\n## Unparseable rate (test)\n\n" + " · ".join(f"{k} {v:.3f}" for k, v in unparse.items()))
    md = "\n".join(L) + "\n"
    if args.md:
        Path(args.md).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
