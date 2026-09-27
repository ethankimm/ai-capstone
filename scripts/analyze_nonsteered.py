#!/usr/bin/env python3
"""Where do non-steered patched answers go? Buckets every patched answer from E3
(same-problem minimal pairs, level 1) and E2 (cross-problem qualified donors, level 4)
by which chain it is consistent with -- turning "latents are context-entangled" from an
interpretation of a null into a measurement. No GPU: saved predictions only.

Buckets, first match wins (numeric equality, `counterfactual.num_equal`):

  unparseable            no number extracted
  unchanged              == recipient's own unpatched answer ("recipient original")
  donor_final            E3: == twin's final answer (steered). E2: == donor's final answer
  counterfactual         E2 only: == recipient chain re-run with the donor's step value at
                         the patched step (Metric B, `matches_cf`)
  partial_propagation    E3 only: the perturbed operand changed in only a proper, non-empty
                         subset of the steps that use it literally (the rest keep the
                         recipient's number), then re-executed
  wrong_step             the donor's value, but at the wrong place in the recipient's chain:
                         E3: a changed twin step value v_j injected at a step != j and
                         re-evaluated. E2: the donor step value injected at a step != the
                         patched one.
  donor_intermediate     == a non-final donor (E3: twin) step value that is not also a
                         recipient value
  recipient_intermediate == a non-final recipient step value
  off_by_delta           E3 only: == recipient answer +/- the operand delta
  other                  none of the above ("unrelated")

Chance reference: a permutation null. For each record, the patched answer of a random
other record in the same condition is bucketed against THIS record's chains; averaged
over `--reps` shuffles. A bucket only means something where it beats that.

E3 source runs are the committed minimal-pair runs; CODI's single-slot / prefix sites
there use the pre-fix site indexing (donor z_i fed where the recipient's z_{i-1} goes --
see `patch_minimal_pair_codi.py`), so CODI single-slot rows describe that intervention,
not the aligned one. Same for CODI E2.

  uv run python scripts/analyze_nonsteered.py [--coconut-data-dir DIR] [--reps 200] [--no-save]
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter
from pathlib import Path

from latentreasoning.data.gsm8k_aug import Example, load_gsm8k_aug
from latentreasoning.data.minimal_pairs import generate_minimal_pair
from latentreasoning.eval.counterfactual import (
    counterfactual_answer, fmt_num, num_equal, parse_steps, safe_eval, substitute,
)
from latentreasoning.runlog.manifest import RESULTS_DIR, DatasetInfo, ModelInfo, RunRecord, new_run_id

DEFAULT_COCONUT_DATA_DIR = Path.home() / "Projects/are-lrms-easily-interpretable/data"
SOURCES = {
    "codi": {"e3": "20260920-190420_codi_minimal-pair-patch", "e2": "20260920-085206_codi_qualified-patch"},
    "coconut": {"e3": "20260920-195725_coconut_minimal-pair-patch", "e2": "20260920-085317_coconut_qualified-patch"},
}
E3_BUCKETS = ("unparseable", "unchanged", "donor_final", "partial_propagation", "wrong_step", "donor_intermediate",
              "recipient_intermediate", "off_by_delta", "other")
E2_BUCKETS = ("unparseable", "unchanged", "counterfactual", "donor_final", "wrong_step",
              "donor_intermediate", "recipient_intermediate", "other")
# buckets that are "the donor's information, misapplied" vs chance-level catch-alls
NULL_BUCKETS = ("partial_propagation", "wrong_step", "donor_intermediate", "recipient_intermediate", "off_by_delta")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_examples(mechanism: str, coconut_data_dir: Path) -> dict[int, Example]:
    if mechanism == "codi":
        return {ex.idx: ex for ex in load_gsm8k_aug(split="test", n=None, seed=None)}
    rows = json.loads((coconut_data_dir / "gsm_valid-gold-reasoning-trace_test.json").read_text())
    return {i: Example(question=r["question"], rationale=" ".join(r["steps"]), answer=r["answer"].strip(), idx=i)
            for i, r in enumerate(rows)}


def any_eq(x: str | None, vals) -> bool:
    return x is not None and any(num_equal(x, v) for v in vals)


# ---- E3 ----------------------------------------------------------------------------------

class _Pick:
    """rng stand-in so `generate_minimal_pair` re-picks the operand E3 actually perturbed."""

    def __init__(self, tok: str):
        self.tok = tok

    def choice(self, cands):
        return next(c for c in cands if c[0] == self.tok)


def partial_propagation_answers(steps: list[dict], tok: str, new_tok: str, ans_idx: int) -> set[str]:
    """Final answers when the perturbed operand is replaced in only a proper, non-empty
    subset of the steps whose expression uses it literally (changed step RESULTS still
    propagate downstream as in `minimal_pairs._reexecute`)."""
    users = [j for j, st in enumerate(steps) if any_eq(tok, [m for m in _nums(st["expr"])])]
    out: set[str] = set()
    for r in range(1, len(users)):
        for subset in itertools.combinations(users, r):
            mapping: dict[str, str] = {}
            vals = []
            ok = True
            for j, st in enumerate(steps):
                m = dict(mapping)
                if j in subset:
                    m[tok] = new_tok
                v = safe_eval(substitute(st["expr"], m))
                if v is None:
                    ok = False
                    break
                v_str = fmt_num(v)
                vals.append(v_str)
                if not num_equal(v_str, st["val"]) and st["val"] not in mapping:
                    mapping[st["val"]] = v_str
            if ok:
                out.add(vals[ans_idx])
    return out


def _nums(expr: str) -> list[str]:
    from latentreasoning.eval.counterfactual import NUM_RE
    return NUM_RE.findall(expr)


def e3_context(rec: dict, ex: Example) -> dict:
    """Recipient chain, twin chain and the candidate answer sets for one E3 record."""
    mp = generate_minimal_pair(ex, rec["step"], rec["delta"], rng=_Pick(rec["perturbed_number"]))
    assert mp is not None and mp.twin.question == rec["twin_question"] and num_equal(mp.twin.answer, rec["twin_answer"]), \
        f"could not regenerate twin for recipient {rec['recipient_idx']}"
    r_steps = parse_steps(ex.rationale)
    t_steps = parse_steps(mp.twin.rationale)
    base = rec["answer_base"]
    ans_idx = next((j for j in range(len(r_steps) - 1, -1, -1) if num_equal(r_steps[j]["val"], ex.answer)),
                   len(r_steps) - 1)

    wrong: set[str] = set()
    changed = [j for j in range(len(r_steps)) if not num_equal(r_steps[j]["val"], t_steps[j]["val"])]
    for j in changed:
        for k in range(len(r_steps) - 1):
            if k == j:
                continue
            cf = counterfactual_answer(r_steps, k, t_steps[j]["val"], base)
            if cf is not None:
                wrong.add(cf)
    new_tok = fmt_num(float(rec["perturbed_number"].replace(",", "")) + rec["delta"])
    partial = partial_propagation_answers(r_steps, rec["perturbed_number"], new_tok, ans_idx)
    exclude = [base, rec["twin_answer"]]
    partial = {w for w in partial if not any_eq(w, exclude)}
    wrong = {w for w in wrong if not any_eq(w, exclude)}

    r_vals = [st["val"] for j, st in enumerate(r_steps) if j != ans_idx]
    t_vals = [st["val"] for j, st in enumerate(t_steps) if j != ans_idx and not any_eq(st["val"], [s["val"] for s in r_steps])]
    base_f = safe_eval(base.replace(",", "")) if base else None
    off = [fmt_num(base_f + rec["delta"]), fmt_num(base_f - rec["delta"])] if base_f is not None else []
    return {"base": base, "donor_final": rec["twin_answer"], "partial_propagation": partial, "wrong_step": wrong,
            "donor_intermediate": t_vals, "recipient_intermediate": r_vals, "off_by_delta": off}


def bucket_e3(pred: str | None, ctx: dict) -> str:
    if pred is None:
        return "unparseable"
    if num_equal(pred, ctx["base"]):
        return "unchanged"
    if num_equal(pred, ctx["donor_final"]):
        return "donor_final"
    for b in ("partial_propagation", "wrong_step", "donor_intermediate", "recipient_intermediate", "off_by_delta"):
        if any_eq(pred, ctx[b]):
            return b
    return "other"


# ---- E2 ----------------------------------------------------------------------------------

def e2_context(pair: dict, steps_by_idx: dict[int, list[dict]], gold_by_idx: dict[int, str], donor_key: str) -> dict:
    r_steps = steps_by_idx[pair["recipient_idx"]]
    d_idx = pair["donor_idx"] if donor_key == "real" else pair["random_donor_idx"]
    d_steps = steps_by_idx[d_idx]
    d_val = pair["donor_value_at_step"] if donor_key == "real" else pair["random_donor_value_at_step"]
    base, k = pair["answer_base"], pair["step"]
    cf = counterfactual_answer(r_steps, k, d_val, base) if d_val is not None else None
    wrong = set()
    if d_val is not None:
        for j in range(len(r_steps) - 1):
            if j != k:
                w = counterfactual_answer(r_steps, j, d_val, base)
                if w is not None:
                    wrong.add(w)
    d_final = gold_by_idx[d_idx]
    wrong = {w for w in wrong if not any_eq(w, [base, cf, d_final])}
    r_vals = [s["val"] for s in r_steps if not num_equal(s["val"], base)]
    d_vals = [s["val"] for s in d_steps if not num_equal(s["val"], d_final) and not any_eq(s["val"], [x["val"] for x in r_steps])]
    return {"base": base, "counterfactual": cf, "donor_final": d_final, "wrong_step": wrong,
            "donor_intermediate": d_vals, "recipient_intermediate": r_vals}


def bucket_e2(pred: str | None, ctx: dict) -> str:
    if pred is None:
        return "unparseable"
    if num_equal(pred, ctx["base"]):
        return "unchanged"
    if ctx["counterfactual"] is not None and num_equal(pred, ctx["counterfactual"]):
        return "counterfactual"
    if num_equal(pred, ctx["donor_final"]):
        return "donor_final"
    for b in ("wrong_step", "donor_intermediate", "recipient_intermediate"):
        if any_eq(pred, ctx[b]):
            return b
    return "other"


# ---- shared tabulation ---------------------------------------------------------------------

def tabulate(preds: list, ctxs: list[dict], bucket_fn, buckets: tuple, reps: int, rng: random.Random) -> dict:
    labels = [bucket_fn(p, c) for p, c in zip(preds, ctxs)]
    n = len(labels)
    counts = Counter(labels)
    # non-steered, changed answers: everything except unchanged / the target bucket(s)
    moved = [i for i, lab in enumerate(labels) if lab not in ("unchanged", "unparseable", "donor_final", "counterfactual")]
    moved_counts = Counter(labels[i] for i in moved)
    null = {b: 0.0 for b in buckets}
    if moved and reps:
        pool = [preds[i] for i in moved]
        for _ in range(reps):
            shuffled = pool[:]
            rng.shuffle(shuffled)
            for i, p in zip(moved, shuffled):
                null[bucket_fn(p, ctxs[i])] += 1
        null = {b: v / (reps * len(moved)) for b, v in null.items()}
    return {
        "n": n,
        "rates": {b: counts.get(b, 0) / n for b in buckets} if n else {},
        "counts": {b: counts.get(b, 0) for b in buckets},
        "n_moved_not_steered": len(moved),
        "moved_rates": {b: moved_counts.get(b, 0) / len(moved) for b in buckets} if moved else {},
        "moved_null_rates": null if moved else {},
    }, labels


def fmt_row(name: str, t: dict, keys=NULL_BUCKETS) -> str:
    r = t["rates"]
    head = f"{name:26s} n={t['n']:4d} " + " ".join(f"{b[:10]}={r.get(b, 0):.2f}" for b in r)
    if t["n_moved_not_steered"]:
        mr, nr = t["moved_rates"], t["moved_null_rates"]
        head += f"\n{'':26s} moved-not-steered n={t['n_moved_not_steered']}: " + " ".join(
            f"{b[:10]}={mr.get(b, 0):.2f}(null {nr.get(b, 0):.2f})" for b in keys if b in mr)
    return head


def analyze(mechanism: str, coconut_data_dir: Path, reps: int, seed: int, src: dict) -> tuple[dict, list[dict]]:
    rng = random.Random(seed)
    by_idx = load_examples(mechanism, coconut_data_dir)
    out: dict = {"e3": {}, "e2": {}}
    pred_rows: list[dict] = []

    # E3
    e3 = read_jsonl(RESULTS_DIR / src["e3"] / "predictions.jsonl")
    ctxs = [e3_context(r, by_idx[r["recipient_idx"]]) for r in e3]
    conds = [("all_slot", lambda r: r["all_slot"])]
    conds += [(f"single_slot_{s}", lambda r, s=s: r["single_slot"][s]) for s in e3[0]["single_slot"]]
    conds += [(f"prefix_{s}", lambda r, s=s: r["prefix"][s]) for s in e3[0]["prefix"]]
    # aligned E3 reruns (2026-09-26) also carry leave-one-out and pre-fix legacy conditions
    conds += [(f"leave_one_out_{s}", lambda r, s=s: r["leave_one_out"][s]) for s in e3[0].get("leave_one_out", {})]
    if "all_slot_legacy" in e3[0]:
        conds += [("all_slot_legacy", lambda r: r["all_slot_legacy"])]
    labels_by_cond = {}
    for name, get in conds:
        table, labels = tabulate([get(r)["pred"] for r in e3], ctxs, bucket_e3, E3_BUCKETS, reps, rng)
        out["e3"][name] = table
        labels_by_cond[name] = labels
        print(fmt_row(f"E3 {name}", table))
    for i, r in enumerate(e3):
        pred_rows.append({"experiment": "E3", "recipient_idx": r["recipient_idx"], "step": r["step"],
                          "twin_answer": r["twin_answer"], "answer_base": r["answer_base"],
                          "n_partial_candidates": len(ctxs[i]["partial_propagation"]),
                          "n_wrong_step_candidates": len(ctxs[i]["wrong_step"]),
                          "buckets": {c: labels_by_cond[c][i] for c in labels_by_cond}})

    # E2 (level 4): cross-problem donors, every site pooled and per site
    e2_preds = read_jsonl(RESULTS_DIR / src["e2"] / "predictions.jsonl")
    steps_by_idx = {r["idx"]: r["steps"] for r in e2_preds}
    gold_by_idx = {i: ex.answer for i, ex in by_idx.items()}
    pairs = read_jsonl(RESULTS_DIR / src["e2"] / "patch_pairs.jsonl")
    site_key = "iter" if "iter" in pairs[0] else "pass"  # CODI iteration / Coconut pass
    for cond, donor_key in (("real", "real"), ("random", "random"), ("mean", "real")):
        # mean ablation has no donor: bucket it against the REAL donor's chain as a null
        ctx2 = [e2_context(p, steps_by_idx, gold_by_idx, donor_key) for p in pairs]
        preds = [p[f"answer_{cond}"] for p in pairs]
        table, labels = tabulate(preds, ctx2, bucket_e2, E2_BUCKETS, reps, rng)
        sites = sorted({p[site_key] for p in pairs})
        table["by_site"] = {}
        for s in sites:
            ix = [i for i, p in enumerate(pairs) if p[site_key] == s]
            t_s, _ = tabulate([preds[i] for i in ix], [ctx2[i] for i in ix], bucket_e2, E2_BUCKETS, 0, rng)
            table["by_site"][str(s)] = t_s
        out["e2"][cond] = table
        print(fmt_row(f"E2 {cond} (all sites)", table, keys=("wrong_step", "donor_intermediate", "recipient_intermediate")))
        for i, p in enumerate(pairs):
            if cond == "real":
                pred_rows.append({"experiment": "E2", "recipient_idx": p["recipient_idx"], "donor_idx": p["donor_idx"],
                                  "site": p[site_key], "step": p["step"], "buckets": {}})
            pred_rows[len(e3) + i]["buckets"][cond] = labels[i]
    return out, pred_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coconut-data-dir", type=Path, default=DEFAULT_COCONUT_DATA_DIR)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--mechanism", choices=("codi", "coconut"), help="only this mechanism")
    ap.add_argument("--e3-run", help="E3 run_id to analyze instead of the committed default (needs --mechanism)")
    ap.add_argument("--slug", default="nonsteered-buckets")
    args = ap.parse_args()

    for mechanism in ((args.mechanism,) if args.mechanism else ("codi", "coconut")):
        src = dict(SOURCES[mechanism])
        if args.e3_run:
            src["e3"] = args.e3_run
        print(f"\n===================== {mechanism}")
        tables, rows = analyze(mechanism, args.coconut_data_dir, args.reps, args.seed, src)
        if args.no_save:
            continue
        e3_manifest = json.loads((RESULTS_DIR / src["e3"] / "manifest.json").read_text())
        record = RunRecord(
            run_id=new_run_id(mechanism, args.slug),
            mechanism=mechanism,
            stage="full_run",
            model=ModelInfo(**e3_manifest["model"]),
            dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=e3_manifest["dataset"]["n_examples"], seed=0),
            metrics={
                "compute_steps": e3_manifest["metrics"]["compute_steps"],
                "extra": {
                    "design": "offline: bucket every E3 (level-1 minimal pair) and E2 (level-4 cross-problem) "
                              "patched answer by the chain it is consistent with, vs a permutation null",
                    "e3_buckets": list(E3_BUCKETS), "e2_buckets": list(E2_BUCKETS),
                    "e3": tables["e3"], "e2": tables["e2"],
                    "null": f"permutation of moved-not-steered answers within condition, {args.reps} reps",
                    "source_runs": [src["e3"], src["e2"]],
                },
            },
            hyperparams={"reps": args.reps},
            seed=args.seed,
            hardware="local-cpu (offline, no model)",
            notes="Offline: where non-steered patched answers go (recipient original / donor final / donor "
                  "value at the wrong step / unrelated), E3 + E2, with a permutation null.",
        )
        path = record.save(predictions=rows)
        print(f"saved {path.parent}")


if __name__ == "__main__":
    main()
