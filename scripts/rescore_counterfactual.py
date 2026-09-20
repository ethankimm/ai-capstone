#!/usr/bin/env python3
"""Re-score the five logged causal-patching runs under Metric B (`matches_cf`) instead
of the `steered_to_donor` field they were logged with, which is invalid or
inconsistently defined across scripts -- see `steered_to_donor_audit.md` (E0 in its
experiment plan) for the full write-up. No GPU, no new model runs: every number here is
recomputed offline from the runs' own `predictions.jsonl` / `patch_pairs.jsonl` plus a
lookup of the recipient's gold `<<expr=val>>` chain (CODI: `load_gsm8k_aug`; Coconut:
the local Dilgren & Wiegreffe gold-reasoning-trace file).

  uv run python scripts/rescore_counterfactual.py [--coconut-data-dir DIR] [--reps 200]

Prints one table per run and writes the same tables as a "## Metric B addendum" section
appended to that run's `notes.md` (idempotent: re-running replaces a previously
appended addendum rather than stacking a second copy).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug
from latentreasoning.eval.counterfactual import (
    counterfactual_candidates, num_equal, parse_steps, score_patch, score_patch_unaligned,
)
from latentreasoning.runlog.manifest import RESULTS_DIR

ADDENDUM_MARKER = "## Metric B addendum"
DEFAULT_COCONUT_DATA_DIR = Path.home() / "Projects/are-lrms-easily-interpretable/data"


def wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = x / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


# ---- example lookups ------------------------------------------------------------------------
def load_codi_examples(split: str) -> dict[int, dict]:
    examples = load_gsm8k_aug(split=split, n=None, seed=None)
    out = {}
    for ex in examples:
        out[ex.idx] = {
            "question": ex.question, "answer": ex.answer,
            "chain": parse_steps(ex.rationale),
            "values": ex.intermediate_values,
        }
    return out


def load_coconut_examples(data_dir: Path) -> dict[int, dict]:
    rows = json.loads((data_dir / "gsm_valid-gold-reasoning-trace_test.json").read_text())
    out = {}
    for i, r in enumerate(rows):
        chain = parse_steps(" ".join(r["steps"]))
        out[i] = {"question": r["question"], "answer": r["answer"].strip(), "chain": chain,
                   "values": [s["val"] for s in chain]}
    return out


# ---- per-run scorers --------------------------------------------------------------------------
def score_step_aligned(records: list[dict], examples: dict[int, dict], get_step, get_donor_value,
                        get_base, get_patched) -> list[dict]:
    """`records` with a KNOWN patched chain step -- exact Metric B."""
    out = []
    for r in records:
        ex = examples.get(r["recipient_idx"])
        if ex is None:
            continue
        step = get_step(r)
        scored = score_patch(
            answer_base=get_base(r), answer_patched=get_patched(r),
            recipient_gold=ex["answer"], recipient_chain=ex["chain"],
            donor_value=get_donor_value(r), step=step, recipient_values=ex["values"],
        )
        out.append(scored)
    return out


def score_unaligned(records: list[dict], recipient_examples: dict[int, dict], donor_examples: dict[int, dict],
                     get_base, get_patched, get_donor_final=None) -> list[dict]:
    """`records` with NO known patch step (donor chosen by different final answer only)
    -- Metric B at any step, injecting any of the donor's own gold intermediate values."""
    out = []
    for r in records:
        rex = recipient_examples.get(r["recipient_idx"])
        dex = donor_examples.get(r["donor_idx"])
        if rex is None:
            continue
        donor_values = dex["values"] if dex is not None else []
        donor_final = get_donor_final(r) if get_donor_final else (dex["answer"] if dex else None)
        scored = score_patch_unaligned(
            answer_base=get_base(r), answer_patched=get_patched(r),
            recipient_gold=rex["answer"], donor_final=donor_final,
            recipient_chain=rex["chain"], donor_values=donor_values,
            recipient_values=rex["values"], donor_intermediate_values=donor_values,
        )
        out.append(scored)
    return out


def permutation_null_unaligned(records: list[dict], recipient_examples: dict[int, dict],
                                donor_examples: dict[int, dict], get_base, get_patched,
                                reps: int = 200, seed: int = 0) -> float:
    """Chance rate for `score_unaligned`'s matches_cf: shuffle which donor's gold chain
    is used to build each pair's candidate set (same patched answers, reassigned
    donors), `reps` times, average the hit rate."""
    rng = random.Random(seed)
    recipient_idxs = [r["recipient_idx"] for r in records]
    donor_idxs = [r["donor_idx"] for r in records]
    bases = [get_base(r) for r in records]
    patched = [get_patched(r) for r in records]
    rates = []
    for _ in range(reps):
        shuffled_donors = list(donor_idxs)
        rng.shuffle(shuffled_donors)
        hits = 0
        n_defined = 0
        for ridx, base, ans, didx in zip(recipient_idxs, bases, patched, shuffled_donors):
            rex = recipient_examples.get(ridx)
            dex = donor_examples.get(didx)
            if rex is None or dex is None:
                continue
            scored = score_patch_unaligned(
                answer_base=base, answer_patched=ans, recipient_chain=rex["chain"],
                donor_values=dex["values"],
            )
            if scored["matches_cf"] is not None:
                n_defined += 1
                hits += scored["matches_cf"]
        if n_defined:
            rates.append(hits / n_defined)
    return sum(rates) / len(rates) if rates else 0.0


# ---- summarizing / rendering -------------------------------------------------------------------
def summarize(scored: list[dict]) -> dict:
    n = len(scored)
    outcomes = Counter(s["outcome"] for s in scored)
    defined = [s for s in scored if s["matches_cf"] is not None]
    hits = sum(bool(s["matches_cf"]) for s in defined)
    lo, hi = wilson_ci(hits, len(defined))
    return {
        "n": n, "n_cf_defined": len(defined), "matches_cf_hits": hits,
        "matches_cf_rate": hits / len(defined) if defined else None,
        "matches_cf_ci": [lo, hi], "outcomes": dict(outcomes),
    }


def render_table(title: str, rows: list[tuple[str, dict]], null_rows: dict[str, float] | None = None) -> str:
    lines = [f"### {title}", "", "| group | n | n(cf defined) | matches_cf | 95% CI |" +
             (" perm-null |" if null_rows else " |")]
    lines.append("|---|---|---|---|---|" + ("---|" if null_rows else ""))
    for label, s in rows:
        rate = f"{s['matches_cf_rate']:.3f}" if s["matches_cf_rate"] is not None else "-"
        ci = f"[{s['matches_cf_ci'][0]:.3f}, {s['matches_cf_ci'][1]:.3f}]"
        row = f"| {label} | {s['n']} | {s['n_cf_defined']} | {rate} | {ci} |"
        if null_rows:
            nv = null_rows.get(label)
            row += f" {nv:.3f} |" if nv is not None else " - |"
        lines.append(row)
    lines.append("")
    for label, s in rows:
        oc = ", ".join(f"{k} {v}" for k, v in sorted(s["outcomes"].items(), key=lambda kv: -kv[1]))
        lines.append(f"- **{label}** taxonomy: {oc}")
    lines.append("")
    return "\n".join(lines)


def write_addendum(run_id: str, body: str) -> None:
    path = RESULTS_DIR / run_id / "notes.md"
    text = path.read_text()
    if ADDENDUM_MARKER in text:
        text = text.split(ADDENDUM_MARKER)[0].rstrip() + "\n"
    header = (f"\n{ADDENDUM_MARKER} (rescored 2026-09-19, `scripts/rescore_counterfactual.py`)\n\n"
              f"See `steered_to_donor_audit.md`. `steered_to_donor` as originally logged measures "
              f"Metric A (`answer_patched == donor.answer` -- already the case for this run except "
              f"where noted); the table below adds Metric B (`matches_cf`): does the answer equal the "
              f"counterfactual obtained by substituting the injected value into the RECIPIENT's own "
              f"remaining chain and re-evaluating.\n\n")
    path.write_text(text.rstrip() + "\n" + header + body)


# ---- run-specific drivers ----------------------------------------------------------------------
def rescore_codi_decode_patch_full_eval(examples: dict[int, dict], reps: int) -> None:
    run_id = "20260919-184323_codi_decode-patch-full-eval"
    records = [json.loads(l) for l in (RESULTS_DIR / run_id / "patch_pairs.jsonl").open()]
    records = [r for r in records if r["is_focus_iter"]]

    def scored_for(base_key, patched_key):
        return score_step_aligned(
            records, examples,
            get_step=lambda r: r["step"] - 1,
            get_donor_value=lambda r: r["donor_value_at_step"],
            get_base=lambda r: r[base_key], get_patched=lambda r: r[patched_key],
        )

    rows = [
        ("real donor", summarize(scored_for("answer_base", "answer_patched"))),
        ("control: random example, random iter", summarize(scored_for("answer_base", "answer_control_any"))),
        ("control: random example, live iter", summarize(scored_for("answer_base", "answer_control_live"))),
    ]
    body = render_table(f"Step-aligned single-slot patch (n={len(records)} focus-iter pairs)", rows)
    print(f"\n=== {run_id} ===\n{body}")
    write_addendum(run_id, body)


def rescore_coconut_decode_patch_pilot(examples: dict[int, dict], reps: int) -> None:
    run_id = "20260920-031246_coconut_decode-patch-pilot"
    records = [json.loads(l) for l in (RESULTS_DIR / run_id / "predictions.jsonl").open()]
    single = [r for r in records if r.get("condition") == "single"]

    scored_all = score_step_aligned(
        single, examples,
        get_step=lambda r: r["step"] - 1,
        get_donor_value=lambda r: (examples.get(r["donor_idx"]) or {}).get("values", [None] * 20)[r["step"] - 1]
        if r["step"] - 1 < len(examples.get(r["donor_idx"], {}).get("values", [])) else None,
        get_base=lambda r: r["answer_base"], get_patched=lambda r: r["answer_patched"],
    )
    by_pass: dict[int, list[dict]] = {}
    for r, s in zip(single, scored_all):
        by_pass.setdefault(r["pass"], []).append(s)

    rows_all = [("all passes", summarize(scored_all))]
    rows_by_pass = [(f"pass {p}", summarize(s)) for p, s in sorted(by_pass.items())]
    body = (render_table(f"Raw single-slot patch, all passes (n={len(single)})", rows_all) +
            render_table("By pass", rows_by_pass))
    print(f"\n=== {run_id} ===\n{body}")
    write_addendum(run_id, body)


def rescore_codi_interchange_placeholder(codi_examples: dict[int, dict], reps: int) -> None:
    run_id = "20260920-031925_codi_interchange-placeholder-pilot"
    records = [json.loads(l) for l in (RESULTS_DIR / run_id / "predictions.jsonl").open()]
    single = [r for r in records if r.get("condition") == "single"]
    by_iter: dict[int, list[dict]] = {}
    for r in single:
        by_iter.setdefault(r["iter"], []).append(r)

    rows = []
    null_rows = {}
    for it in sorted(by_iter):
        recs = by_iter[it]
        scored = score_unaligned(recs, codi_examples, codi_examples,
                                  get_base=lambda r: r["answer_base"], get_patched=lambda r: r["answer_patched"],
                                  get_donor_final=lambda r: r["donor_gold"])
        label = f"iter {it}"
        rows.append((label, summarize(scored)))
        null_rows[label] = permutation_null_unaligned(recs, codi_examples, codi_examples,
                                                        get_base=lambda r: r["answer_base"],
                                                        get_patched=lambda r: r["answer_patched"], reps=reps)

    scored_total = score_unaligned(single, codi_examples, codi_examples,
                                    get_base=lambda r: r["answer_base"], get_patched=lambda r: r["answer_patched"],
                                    get_donor_final=lambda r: r["donor_gold"])
    rows.append(("total", summarize(scored_total)))
    null_rows["total"] = permutation_null_unaligned(single, codi_examples, codi_examples,
                                                      get_base=lambda r: r["answer_base"],
                                                      get_patched=lambda r: r["answer_patched"], reps=reps)
    body = render_table(f"Raw single-slot patch, unaligned (n={len(single)})", rows, null_rows)
    print(f"\n=== {run_id} ===\n{body}")
    write_addendum(run_id, body)


def rescore_das(run_id: str, mechanism: str, recipient_examples: dict[int, dict],
                 donor_examples: dict[int, dict], reps: int) -> None:
    records = [json.loads(l) for l in (RESULTS_DIR / run_id / "predictions.jsonl").open()]
    by_site: dict[int, list[dict]] = {}
    for r in records:
        by_site.setdefault(r["site"], []).append(r)

    rows = []
    null_rows = {}
    for site in sorted(by_site):
        recs = by_site[site]
        scored = score_unaligned(recs, recipient_examples, donor_examples,
                                  get_base=lambda r: r["answer_base"], get_patched=lambda r: r["answer_patched"],
                                  get_donor_final=lambda r: r["donor_gold"])
        label = f"site {site}"
        rows.append((label, summarize(scored)))
        null_rows[label] = permutation_null_unaligned(recs, recipient_examples, donor_examples,
                                                        get_base=lambda r: r["answer_base"],
                                                        get_patched=lambda r: r["answer_patched"], reps=reps)

    scored_total = score_unaligned(records, recipient_examples, donor_examples,
                                    get_base=lambda r: r["answer_base"], get_patched=lambda r: r["answer_patched"],
                                    get_donor_final=lambda r: r["donor_gold"])
    rows.append(("total", summarize(scored_total)))
    null_rows["total"] = permutation_null_unaligned(records, recipient_examples, donor_examples,
                                                      get_base=lambda r: r["answer_base"],
                                                      get_patched=lambda r: r["answer_patched"], reps=reps)
    body = render_table(f"DAS learned-subspace patch, unaligned (n={len(records)})", rows, null_rows)
    print(f"\n=== {run_id} ===\n{body}")
    write_addendum(run_id, body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coconut-data-dir", default=str(DEFAULT_COCONUT_DATA_DIR))
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()

    print("loading CODI test split...")
    codi_test = load_codi_examples("test")
    print(f"  {len(codi_test)} examples")
    print("loading CODI validation split (downloads full train split once)...")
    codi_validation = load_codi_examples("validation")
    print(f"  {len(codi_validation)} examples")
    print("loading Coconut gold-reasoning-trace examples...")
    coconut_examples = load_coconut_examples(Path(args.coconut_data_dir))
    print(f"  {len(coconut_examples)} examples")

    rescore_codi_decode_patch_full_eval(codi_test, args.reps)
    rescore_coconut_decode_patch_pilot(coconut_examples, args.reps)
    rescore_codi_interchange_placeholder(codi_test, args.reps)
    rescore_das("20260920-050602_codi_das-pilot", "codi", codi_validation, codi_validation, args.reps)
    rescore_das("20260920-053935_coconut_das-pilot", "coconut", coconut_examples, coconut_examples, args.reps)

    print("\ndone -- review the addenda, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
