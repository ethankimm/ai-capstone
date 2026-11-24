#!/usr/bin/env python3
"""Offline test of H3 ("the thought carries the program, the context carries the values")
on the logged CROSS-problem activation-patching runs -- see `program_transfer_audit.md`
for the write-up and `steered_to_donor_audit.md` for the predecessor. No GPU, no model:
every number is recomputed from the runs' saved `predictions.jsonl` / `patch_pairs.jsonl`
plus the gold `<<a op b=c>>` chains (CODI: `load_gsm8k_aug`; Coconut: the local Dilgren &
Wiegreffe gold-reasoning-trace file).

H3 predicts: patched answer == the DONOR's operation chain (program) executed on the
RECIPIENT's question numbers. Variants, scored only on changed answers (patched != base,
parseable, and != the recipient's gold, so "the patch fixed a wrong answer" never counts):

  pos_final / pos_any   positional operand map (i-th distinct number in D's question ->
                        i-th in R's); final value / any step value of the executed chain
  any_final / any_inter any injective assignment of R's question numbers to D's
                        question-number slots (constants kept, step dataflow kept)
  pos_trunc / any_trunc step-aligned runs only: D's program truncated at the patched step

Nulls: donor-shuffle permutation (patched answers fixed, donors permuted over distinct
(recipient, donor) pairs within run x condition x site -- the DAS runs reuse each pair at
three k values, so k is NOT its own cell), `--reps` times; the runs' own control conditions scored
the same way, and a chance baseline for `steered_to_donor_audit.md` section 3.4's
"arithmetic of two of the recipient's own numbers".

  uv run python scripts/rescore_program_transfer.py [--coconut-data-dir DIR] [--reps 200]
       [--json-out PATH]

Read-only with respect to `results/`: prints markdown tables, writes nothing unless
`--json-out` is given.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug
from latentreasoning.eval.counterfactual import NUM_RE, num_equal, parse_steps
from latentreasoning.runlog.manifest import RESULTS_DIR

DEFAULT_COCONUT_DATA_DIR = Path.home() / "Projects/are-lrms-easily-interpretable/data"
MAX_ASSIGNMENTS = 200_000  # injective assignments per pair before we subsample
VARIANTS = ("pos_final", "pos_any", "any_final", "any_inter", "pos_trunc", "any_trunc")

WORD_NUMS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
WORD_NUMS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
                  "eighty": 80, "ninety": 90, "hundred": 100, "dozen": 12})
WORD_RE = re.compile(r"\b(" + "|".join(WORD_NUMS) + r")\b", re.IGNORECASE)
# unlike counterfactual.NUM_RE, also accepts a leading-dot decimal (".25"), common in the chains
NUM_SPLIT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?|\.\d+)")


def wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = x / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def to_float(s) -> float | None:
    if s is None:
        return None
    try:
        return float(str(s).replace(",", "").strip())
    except ValueError:
        return None


# ---- programs ---------------------------------------------------------------------------------
def question_numbers(question: str) -> list[float]:
    """Distinct numbers in the question, in order of first appearance: digit literals
    (NUM_RE, commas stripped) and cardinal number words ("three", "dozen")."""
    found = [(m.start(), float(m.group().replace(",", ""))) for m in NUM_RE.finditer(question)
             if m.group().replace(",", "").replace(".", "").isdigit()]
    found += [(m.start(), float(WORD_NUMS[m.group().lower()])) for m in WORD_RE.finditer(question)]
    out: list[float] = []
    for _, v in sorted(found):
        if not any(abs(v - o) < 1e-9 for o in out):
            out.append(v)
    return out


class Program:
    """A gold `<<expr=val>>` chain with every numeric operand resolved to a slot:
    ('s', t) = result of earlier step t (preferred), ('q', j) = j-th distinct question
    number, ('c', v) = constant. Each step compiles to a Python expression over variables
    so it can be run vectorized over many operand assignments at once (numpy arrays)."""

    def __init__(self, question: str, rationale_steps: list[dict]):
        self.qnums = question_numbers(question)
        self.steps_src = rationale_steps
        self.codes, self.slots = [], []  # per step: compiled code, {varname: slot}
        self.ok = bool(rationale_steps)
        for i, st in enumerate(rationale_steps):
            pieces = NUM_SPLIT_RE.split(st["expr"].replace("x", "*").replace("×", "*"))
            expr, slots = [], {}
            for p in pieces:
                if p and NUM_SPLIT_RE.fullmatch(p):
                    v = float(p.replace(",", ""))
                    slot = None
                    for t in range(i - 1, -1, -1):
                        if num_equal(rationale_steps[t]["val"], p):
                            slot = ("s", t)
                            break
                    if slot is None:
                        for j, q in enumerate(self.qnums):
                            if abs(q - v) < 1e-9:
                                slot = ("q", j)
                                break
                    if slot is None:
                        slot = ("c", v)
                    name = f"v{len(slots)}"
                    slots[name] = slot
                    expr.append(name)
                else:
                    expr.append(p)
            src = "".join(expr)
            if not re.fullmatch(r"[v\d\s.+\-*/()]+", src):
                self.ok = False
                break
            try:
                self.codes.append(compile(src, "<step>", "eval"))
            except SyntaxError:
                self.ok = False
                break
            self.slots.append(slots)
        self.q_used = sorted({s[1] for sl in self.slots for s in sl.values() if s[0] == "q"})

    def run(self, qvals: dict[int, np.ndarray], n: int) -> list[np.ndarray]:
        """Execute over n assignments; qvals maps question slot -> array (n,). Returns one
        (n,) float array per step (nan where undefined)."""
        results: list[np.ndarray] = []
        with np.errstate(all="ignore"):
            for code, slots in zip(self.codes, self.slots):
                env = {}
                for name, (kind, x) in slots.items():
                    if kind == "s":
                        env[name] = results[x]
                    elif kind == "q":
                        env[name] = qvals[x]
                    else:
                        env[name] = np.full(n, x)
                try:
                    r = eval(code, {"__builtins__": {}}, env)  # noqa: S307 -- charset checked
                except ZeroDivisionError:
                    r = np.full(n, np.nan)
                r = np.broadcast_to(np.asarray(r, dtype=float), (n,)).copy()
                r[~np.isfinite(r)] = np.nan
                results.append(r)
        return results


_PERM_CACHE: dict[tuple[int, int], np.ndarray] = {}


def assignments(n: int, m: int) -> np.ndarray:
    if (n, m) not in _PERM_CACHE:
        total = 1
        for i in range(m):
            total *= n - i
        if total <= MAX_ASSIGNMENTS:
            arr = np.array(list(itertools.permutations(range(n), m)), dtype=np.int64).reshape(-1, m)
        else:  # deterministic subsample of injective assignments
            rng = np.random.default_rng(0)
            arr = np.array([rng.permutation(n)[:m] for _ in range(MAX_ASSIGNMENTS)], dtype=np.int64)
        _PERM_CACHE[(n, m)] = arr
    return _PERM_CACHE[(n, m)]


def isin(values: np.ndarray, target: float) -> bool:
    return bool(np.any(np.isclose(values, target, rtol=1e-6, atol=1e-6)))


class H3Scorer:
    """hit(donor, recipient, target, trunc_step) -> {variant: bool}, cached."""

    def __init__(self, examples: dict[int, dict]):
        self.ex = examples
        self._prog: dict[int, Program] = {}
        self._cache: dict[tuple, dict] = {}

    def prog(self, idx: int) -> Program:
        if idx not in self._prog:
            e = self.ex[idx]
            self._prog[idx] = Program(e["question"], e["chain"])
        return self._prog[idx]

    def executions(self, didx: int, ridx: int):
        """(positional per-step values or None, any-assignment per-step arrays or None)."""
        key = ("exec", didx, ridx)
        if key in self._cache:
            return self._cache[key]
        dp, rq = self.prog(didx), self.prog(ridx).qnums
        pos = anyr = None
        if dp.ok:
            if not dp.q_used or max(dp.q_used) < len(rq):
                qv = {j: np.array([rq[j]]) for j in dp.q_used}
                pos = [r[0] for r in dp.run(qv, 1)]
            m = len(dp.q_used)
            if m == 0:  # program uses constants / step results only
                anyr = dp.run({}, 1)
            elif m <= len(rq):
                A = assignments(len(rq), m)
                rq_arr = np.array(rq)
                qv = {j: rq_arr[A[:, c]] for c, j in enumerate(dp.q_used)}
                anyr = dp.run(qv, len(A))
        self._cache[key] = (pos, anyr)
        return pos, anyr

    def hit(self, didx: int, ridx: int, target: float, trunc: int | None) -> dict:
        key = (didx, ridx, round(target, 6), trunc)
        if key in self._cache:
            return self._cache[key]
        pos, anyr = self.executions(didx, ridx)
        out = dict.fromkeys(VARIANTS, False)
        if pos is not None:
            out["pos_final"] = isin(np.array([pos[-1]]), target)
            out["pos_any"] = isin(np.array(pos), target)
            if trunc is not None and trunc < len(pos):
                out["pos_trunc"] = isin(np.array([pos[trunc]]), target)
        if anyr is not None:
            out["any_final"] = isin(anyr[-1], target)
            out["any_inter"] = any(isin(a, target) for a in anyr)
            if trunc is not None and trunc < len(anyr):
                out["any_trunc"] = isin(anyr[trunc], target)
        self._cache[key] = out
        return out


# ---- section 3.4 "arith of recipient's own numbers" ------------------------------------------
def arith_values(ex: dict) -> np.ndarray:
    if "_arith" not in ex:
        nums = list(question_numbers(ex["question"])) + [v for v in (to_float(s["val"]) for s in ex["chain"]) if v is not None]
        a = np.array(nums, dtype=float)
        if len(a) < 2:
            ex["_arith"] = np.array([])
        else:
            i, j = np.where(~np.eye(len(a), dtype=bool))
            x, y = a[i], a[j]
            with np.errstate(all="ignore"):
                v = np.concatenate([x + y, x - y, x * y, np.where(y != 0, x / y, np.nan)])
            ex["_arith"] = v[np.isfinite(v)]
    return ex["_arith"]


def known_values(rex: dict, dex: dict | None) -> list[float]:
    """Values that put an answer outside `other_number` (recipient gold / intermediates,
    donor final / intermediates)."""
    vals = [to_float(rex["answer"])] + [to_float(s["val"]) for s in rex["chain"]]
    if dex is not None:
        vals += [to_float(dex["answer"])] + [to_float(s["val"]) for s in dex["chain"]]
    return [v for v in vals if v is not None]


def is_other_number(v: float, base: float | None, rex: dict, dex: dict | None) -> bool:
    if base is not None and abs(v - base) < 1e-9:
        return False
    return not any(abs(v - k) < 1e-6 * max(1, abs(k)) for k in known_values(rex, dex))


# ---- data ------------------------------------------------------------------------------------
def load_codi_examples(split: str) -> dict[int, dict]:
    return {ex.idx: {"question": ex.question, "answer": ex.answer, "chain": parse_steps(ex.rationale)}
            for ex in load_gsm8k_aug(split=split, n=None, seed=None)}


def load_coconut_examples(data_dir: Path) -> dict[int, dict]:
    rows = json.loads((data_dir / "gsm_valid-gold-reasoning-trace_test.json").read_text())
    return {i: {"question": r["question"], "answer": r["answer"].strip(),
                "chain": parse_steps(" ".join(r["steps"]))} for i, r in enumerate(rows)}


def read_jsonl(run_id: str, name: str) -> list[dict] | None:
    p = RESULTS_DIR / run_id / name
    if not p.exists():
        return None
    return [json.loads(line) for line in p.open() if line.strip()]


def unit(run, mech, ds, condition, site, cell_extra, r, patched_key, base_key="answer_base",
         donor_key="donor_idx", step=None, scored_against="real donor"):
    return {"run": run, "mech": mech, "ds": ds, "condition": condition, "site": site,
            "cell": (run, condition, site, cell_extra), "recipient_idx": r["recipient_idx"],
            "donor_idx": r[donor_key], "step": step, "answer_base": r.get(base_key),
            "answer_patched": r.get(patched_key), "scored_against": scored_against}


def collect_units(missing: list[str]) -> list[dict]:
    U: list[dict] = []

    run = "20260919-184323_codi_decode-patch-full-eval"
    recs = read_jsonl(run, "patch_pairs.jsonl")
    if recs is None:
        missing.append(f"{run}/patch_pairs.jsonl")
    else:
        for r in recs:
            site = f"iter {r['iter']}"
            s = r["step"] - 1  # 1-indexed in this run
            U.append(unit(run, "codi", "test", "real", site, None, r, "answer_patched", step=s))
            U.append(unit(run, "codi", "test", "ctrl: random ex, random iter", site, None, r,
                          "answer_control_any", step=s, scored_against="real donor (not patched in)"))
            U.append(unit(run, "codi", "test", "ctrl: random ex, live iter", site, None, r,
                          "answer_control_live", step=s, scored_against="real donor (not patched in)"))

    run = "20260920-031925_codi_interchange-placeholder-pilot"
    recs = read_jsonl(run, "predictions.jsonl")
    if recs is None:
        missing.append(f"{run}/predictions.jsonl")
    else:
        for r in recs:
            if r["condition"] == "single":
                U.append(unit(run, "codi", "test", "real single-slot", f"iter {r['iter']}", None, r, "answer_patched"))
            else:
                U.append(unit(run, "codi", "test", "real grouped", "{z0,z3} nondecodable", None, r,
                              "answer_patched_nondecodable_group"))
                U.append(unit(run, "codi", "test", "real grouped", "{z2,z4} decodable", None, r,
                              "answer_patched_decodable_group"))

    for run, mech, ds in (("20260920-050602_codi_das-pilot", "codi", "validation"),
                          ("20260920-053935_coconut_das-pilot", "coconut", "coconut")):
        recs = read_jsonl(run, "predictions.jsonl")
        if recs is None:
            missing.append(f"{run}/predictions.jsonl")
            continue
        for r in recs:
            # k is NOT part of the shuffle cell: the same pairs recur at every k (see shuffle_nulls)
            U.append(unit(run, mech, ds, "real (DAS)", f"site {r['site']}", None, r, "answer_patched"))
            U[-1]["k"] = r["k"]

    for run, mech, sitekey in (("20260920-085206_codi_qualified-patch", "codi", "iter"),
                               ("20260920-085317_coconut_qualified-patch", "coconut", "pass")):
        ds = "test" if mech == "codi" else "coconut"
        recs = read_jsonl(run, "patch_pairs.jsonl")
        if recs is None:
            missing.append(f"{run}/patch_pairs.jsonl")
            continue
        for r in recs:
            site, s = f"{sitekey} {r[sitekey]}", r["step"]  # 0-indexed in these runs
            U.append(unit(run, mech, ds, "real", site, None, r, "answer_real", step=s))
            U.append(unit(run, mech, ds, "ctrl: random donor", site, None, r, "answer_random", step=s,
                          donor_key="random_donor_idx", scored_against="random donor (its own)"))
            U.append(unit(run, mech, ds, "ctrl: random donor vs real-donor program", site, None, r,
                          "answer_random", step=s, scored_against="real donor (not patched in)"))
            U.append(unit(run, mech, ds, "ctrl: mean ablation", site, None, r, "answer_mean", step=s,
                          scored_against="real donor (not patched in)"))

    run = "20260920-031246_coconut_decode-patch-pilot"
    recs = read_jsonl(run, "predictions.jsonl")
    if recs is None:
        missing.append(f"{run}/predictions.jsonl")
    else:
        for r in recs:
            if r.get("condition") == "single":
                U.append(unit(run, "coconut", "coconut", "real single-slot", f"pass {r['pass']}", None, r,
                              "answer_patched", step=r["step"] - 1))
            elif r.get("condition") == "grouped":
                U.append(unit(run, "coconut", "coconut", "real grouped", "most-decodable passes", None, r,
                              "answer_patched_most_decodable"))
                U.append(unit(run, "coconut", "coconut", "real grouped", "least-decodable passes", None, r,
                              "answer_patched_least_decodable"))
    return U


def verify_indices(U: list[dict], data: dict[str, dict]) -> list[str]:
    """Index-convention checks: recorded golds vs loader golds, and base accuracy."""
    lines = []
    for run in dict.fromkeys(u["run"] for u in U):
        us = [u for u in U if u["run"] == run and u["condition"].startswith("real")]
        ds = data[us[0]["ds"]]
        recs = {}
        for name in ("patch_pairs.jsonl", "predictions.jsonl"):
            for r in read_jsonl(run, name) or []:
                if "recipient_idx" in r:
                    recs[(r["recipient_idx"], r.get("donor_idx"))] = r
        n_gold = n_gold_ok = n_dgold = n_dgold_ok = 0
        for r in recs.values():
            if "recipient_gold" in r:
                n_gold += 1
                n_gold_ok += num_equal(r["recipient_gold"], ds[r["recipient_idx"]]["answer"])
            if "donor_gold" in r:
                n_dgold += 1
                n_dgold_ok += num_equal(r["donor_gold"], ds[r["donor_idx"]]["answer"])
        base_ok = sum(num_equal(u["answer_base"], ds[u["recipient_idx"]]["answer"]) for u in us)
        extra = ""
        preds = read_jsonl(run, "predictions.jsonl") or []
        steps_rows = [p for p in preds if "steps" in p and "idx" in p]
        if steps_rows:
            ok = sum([s["val"] for s in p["steps"]] == [s["val"] for s in ds[p["idx"]]["chain"]] for p in steps_rows)
            extra += f"; saved gold chains match loader {ok}/{len(steps_rows)}"
        gold_rows = [p for p in preds if "gold_answer" in p and "idx" in p]
        if gold_rows:
            ok = sum(num_equal(p["gold_answer"], ds[p["idx"]]["answer"]) for p in gold_rows)
            extra += f"; decode-pass gold_answer match {ok}/{len(gold_rows)}"
        lines.append(f"- `{run}` ({us[0]['ds']}): recipient_gold match {n_gold_ok}/{n_gold}, "
                     f"donor_gold match {n_dgold_ok}/{n_dgold}, answer_base==loader gold "
                     f"{base_ok}/{len(us)}{extra}")
    return lines


# ---- scoring ---------------------------------------------------------------------------------
def changed_mask(U: list[dict], data: dict[str, dict]) -> np.ndarray:
    m = np.zeros(len(U), dtype=bool)
    for i, u in enumerate(U):
        p, b = to_float(u["answer_patched"]), to_float(u["answer_base"])
        g = to_float(data[u["ds"]][u["recipient_idx"]]["answer"])
        m[i] = p is not None and (b is None or abs(p - b) > 1e-9) and (g is None or abs(p - g) > 1e-9)
    return m


def hits_for(U, changed, donors, scorers) -> np.ndarray:
    H = np.zeros((len(U), len(VARIANTS)), dtype=bool)
    for i in np.flatnonzero(changed):
        u = U[i]
        d = donors[i]
        if d is None:
            continue
        h = scorers[u["ds"]].hit(d, u["recipient_idx"], to_float(u["answer_patched"]), u["step"])
        H[i] = [h[v] for v in VARIANTS]
    return H


def shuffle_nulls(U, changed, scorers, reps: int, seed: int = 0) -> np.ndarray:
    """(reps, n_units, n_variants) hits with donors permuted within each cell
    (run x condition x site). The permutation is over DISTINCT (recipient, donor) pairs:
    the DAS runs patch the same 60 pairs at k=8/32/64, so a pair that hits at one k tends
    to hit at all three; permuting per record (or per k) would treat those as
    independent and make the p-values anti-conservative."""
    rng = random.Random(seed)
    cells = defaultdict(list)
    for i, u in enumerate(U):
        cells[u["cell"]].append(i)
    cell_pairs = {c: list(dict.fromkeys((U[i]["recipient_idx"], U[i]["donor_idx"]) for i in idxs))
                  for c, idxs in cells.items()}
    out = np.zeros((reps, len(U), len(VARIANTS)), dtype=bool)
    for rep in range(reps):
        donors = [None] * len(U)
        for c, idxs in cells.items():
            pairs = cell_pairs[c]
            ds = [d for _, d in pairs]
            rng.shuffle(ds)
            new = dict(zip(pairs, ds))
            for i in idxs:
                donors[i] = new[(U[i]["recipient_idx"], U[i]["donor_idx"])]
        out[rep] = hits_for(U, changed, donors, scorers)
    return out


def fmt_rate(x, n):
    if n == 0:
        return "-"
    lo, hi = wilson_ci(x, n)
    return f"{x}/{n} = {x / n:.3f} [{lo:.3f}, {hi:.3f}]"


def table(U, sel_groups, changed, H, N, variants, title) -> str:
    lines = [f"#### {title}", "",
             "| group | n | n changed | " + " | ".join(
                 f"{v}: real [95% CI] | {v}: shuffle mean (95% range) | p" for v in variants) + " |",
             "|---|---|---|" + "---|---|---|" * len(variants)]
    for label, idx in sel_groups:
        idx = np.array(idx, dtype=int)
        nc = int(changed[idx].sum())
        row = f"| {label} | {len(idx)} | {nc} |"
        for v in variants:
            k = VARIANTS.index(v)
            x = int(H[idx, k].sum())
            null = N[:, idx, k].sum(axis=1) / max(nc, 1)
            p = float(np.mean(N[:, idx, k].sum(axis=1) >= x)) if nc else float("nan")
            lo, hi = np.percentile(null, [2.5, 97.5]) if nc else (0, 0)
            row += f" {fmt_rate(x, nc)} | {null.mean():.3f} ({lo:.3f}-{hi:.3f}) | {p:.3f} |"
        lines.append(row)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coconut-data-dir", default=str(DEFAULT_COCONUT_DATA_DIR))
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--json-out", default=None, help="optional: dump summary numbers as JSON")
    args = ap.parse_args()

    print("loading data (CODI test, CODI validation, Coconut gold traces)...")
    data = {"test": load_codi_examples("test"), "validation": load_codi_examples("validation"),
            "coconut": load_coconut_examples(Path(args.coconut_data_dir))}
    scorers = {k: H3Scorer(v) for k, v in data.items()}

    missing: list[str] = []
    U = collect_units(missing)
    print(f"{len(U)} scored units; missing files: {missing or 'none'}")
    print("\n### Index-convention checks\n")
    print("\n".join(verify_indices(U, data)))

    changed = changed_mask(U, data)
    H = hits_for(U, changed, [u["donor_idx"] for u in U], scorers)
    print(f"\nrunning donor-shuffle null, {args.reps} reps...")
    N = shuffle_nulls(U, changed, scorers, args.reps)

    summary = {}
    # program coverage
    print("\n### Program-parse coverage (donor side)\n")
    for ds, sc in scorers.items():
        didx = {u["donor_idx"] for u in U if u["ds"] == ds}
        if not didx:
            continue
        progs = [sc.prog(d) for d in didx]
        ok = sum(p.ok for p in progs)
        const_ops = sum(any(s[0] == "c" for sl in p.slots for s in sl.values()) for p in progs if p.ok)
        m = [len(p.q_used) for p in progs if p.ok]
        print(f"- {ds}: {ok}/{len(progs)} donor chains compile; {const_ops} use >=1 constant operand; "
              f"question-number slots used: mean {np.mean(m):.2f}, max {max(m)}")

    for mech in ("codi", "coconut"):
        print(f"\n## {mech.upper()}\n")
        mech_idx = [i for i, u in enumerate(U) if u["mech"] == mech]
        conds = list(dict.fromkeys(U[i]["condition"] for i in mech_idx))
        # headline: pooled over runs, real cross-problem patches only
        real_idx = [i for i in mech_idx if U[i]["condition"].startswith("real")]
        groups = [("ALL real patches, pooled", real_idx)] + [
            (f"{c} (all runs)", [i for i in mech_idx if U[i]["condition"] == c]) for c in conds]
        print(table(U, groups, changed, H, N, ["pos_final", "pos_any", "any_final", "any_inter"],
                    f"{mech}: pooled by condition"))
        summary[mech] = {}
        for label, idx in groups:
            idx = np.array(idx)
            nc = int(changed[idx].sum())
            summary[mech][label] = {v: {"hits": int(H[idx, k].sum()), "n_changed": nc,
                                        "null_mean": float(N[:, idx, k].sum(axis=1).mean() / max(nc, 1)),
                                        "p": float(np.mean(N[:, idx, k].sum(axis=1) >= H[idx, k].sum()))}
                                    for k, v in enumerate(VARIANTS)}
        for run in dict.fromkeys(U[i]["run"] for i in mech_idx):
            run_idx = [i for i in mech_idx if U[i]["run"] == run]
            groups = []
            for c in dict.fromkeys(U[i]["condition"] for i in run_idx):
                c_idx = [i for i in run_idx if U[i]["condition"] == c]
                groups.append((f"**{c}** (vs {U[c_idx[0]]['scored_against']}) total", c_idx))
                for s in dict.fromkeys(U[i]["site"] for i in c_idx):
                    groups.append((f"{c} / {s}", [i for i in c_idx if U[i]["site"] == s]))
            aligned = any(U[i]["step"] is not None for i in run_idx)
            print(table(U, groups, changed, H, N, ["pos_final", "pos_any", "any_final", "any_inter"],
                        f"`{run}`"))
            if aligned:
                print(table(U, [g for g in groups if "total" in g[0]], changed, H, N,
                            ["pos_trunc", "any_trunc"], f"`{run}` -- donor program truncated at the patched step"))

    # ---- pair-level headline, degenerate hits, multiple comparisons
    print("\n## Pair-level headline (a (recipient, donor) pair counts once however many k / records it has)\n")
    print("| mech | variant | changed pairs | pairs hit, real [95% CI] | shuffle mean (95% range) | p |")
    print("|---|---|---|---|---|---|")
    for mech in ("codi", "coconut"):
        idx = [i for i, u in enumerate(U) if u["mech"] == mech and u["condition"].startswith("real") and changed[i]]
        groups = defaultdict(list)
        for i in idx:
            groups[(U[i]["run"], U[i]["condition"], U[i]["site"], U[i]["recipient_idx"], U[i]["donor_idx"])].append(i)
        gl = list(groups.values())
        for v in ("pos_final", "pos_any", "any_final", "any_inter"):
            k = VARIANTS.index(v)
            x = sum(bool(H[g, k].any()) for g in gl)
            null = np.array([sum(bool(N[r, g, k].any()) for g in gl) for r in range(N.shape[0])])
            lo, hi = np.percentile(null / len(gl), [2.5, 97.5])
            print(f"| {mech} | {v} | {len(gl)} | {fmt_rate(x, len(gl))} | {null.mean() / len(gl):.3f} "
                  f"({lo:.3f}-{hi:.3f}) | {np.mean(null >= x):.3f} |")
        summary[mech]["pair_level_n"] = len(gl)

    print("\n## What the H3 hits are (real condition, changed answers)\n")
    for mech in ("codi", "coconut"):
        for v in ("any_final", "any_inter"):
            k = VARIANTS.index(v)
            hits = [i for i, u in enumerate(U) if u["mech"] == mech and u["condition"].startswith("real") and H[i, k]]
            if not hits:
                continue
            in_rq = sum(any(abs(to_float(U[i]["answer_patched"]) - q) < 1e-9
                            for q in scorers[U[i]["ds"]].prog(U[i]["recipient_idx"]).qnums) for i in hits)
            small = sum(abs(to_float(U[i]["answer_patched"])) <= 10 for i in hits)
            print(f"- {mech} {v}: {len(hits)} hits; patched answer is one of R's own question numbers in "
                  f"{in_rq}/{len(hits)}; |answer| <= 10 in {small}/{len(hits)}")

    print("\n## Multiple comparisons: site-level cells with >=10 changed answers and p < 0.05\n")
    for kind in ("real", "ctrl"):
        cells = defaultdict(list)
        for i, u in enumerate(U):
            if u["condition"].startswith(kind):
                cells[(u["run"], u["condition"], u["site"])].append(i)
        cells = {c: np.array(v) for c, v in cells.items() if changed[v].sum() >= 10}
        for v in ("pos_final", "pos_any", "any_final", "any_inter"):
            k = VARIANTS.index(v)
            sig = [c for c, idx in cells.items() if np.mean(N[:, idx, k].sum(axis=1) >= H[idx, k].sum()) < 0.05]
            print(f"- {kind} {v}: {len(sig)}/{len(cells)} cells -> {[f'{c[0][:15]} {c[2]}' for c in sig]}")

    # ---- example hits
    print("\n## Example hits (real condition; one per distinct pair; any_final or positional first)\n")
    rng = random.Random(1)
    seen, cand = set(), []
    for i, u in enumerate(U):
        key = (u["run"], u["recipient_idx"], u["donor_idx"])
        if u["condition"].startswith("real") and changed[i] and key not in seen and (
                H[i, VARIANTS.index("pos_any")] or H[i, VARIANTS.index("any_final")]):
            seen.add(key)
            cand.append(i)
    rng.shuffle(cand)
    cand.sort(key=lambda i: -int(H[i, VARIANTS.index("pos_any")]))  # stable: positional hits first
    for i in sorted(cand[:10], key=lambda i: (U[i]["mech"], U[i]["run"])):
        u = U[i]
        ex, sc = data[u["ds"]], scorers[u["ds"]]
        rex, dex = ex[u["recipient_idx"]], ex[u["donor_idx"]]
        pos, _ = sc.executions(u["donor_idx"], u["recipient_idx"])
        hv = [v for k, v in enumerate(VARIANTS) if H[i, k]]
        print(f"- {u['mech']} `{u['run']}` {u['condition']} / {u['site']}"
              f"{' k=' + str(u['k']) if 'k' in u else ''}: R={u['recipient_idx']} D={u['donor_idx']} "
              f"base={u['answer_base']} patched={u['answer_patched']} R_gold={rex['answer']} D_gold={dex['answer']}\n"
              f"  R chain: {' '.join('<<'+s['expr']+'='+s['val']+'>>' for s in rex['chain'])}; "
              f"R qnums: {[fmt(x) for x in sc.prog(u['recipient_idx']).qnums]}\n"
              f"  D chain: {' '.join('<<'+s['expr']+'='+s['val']+'>>' for s in dex['chain'])}; "
              f"D qnums: {[fmt(x) for x in sc.prog(u['donor_idx']).qnums]}\n"
              f"  D program on R (positional): {[fmt(x) for x in pos] if pos else None}; hits: {hv}")

    # ---- section 3.4 base rate
    print("\n## Section 3.4 base rate: 'arith (+-*/) of two of the recipient's own numbers'\n")
    print("| mech | run | condition | n other_number (observed) | arith hit, observed | "
          "random other record's patched answer (other_number filter) | random other record's base answer (filter) "
          "| own base answer (no filter) |")
    print("|---|---|---|---|---|---|---|---|")
    arith_summary = {}
    for mech in ("codi", "coconut"):
        for run in dict.fromkeys(u["run"] for u in U if u["mech"] == mech):
            for cond in ("real", "control"):
                # control answers counted once (the "random donor vs real-donor program" rows
                # duplicate the random-donor answers)
                idx = [i for i, u in enumerate(U) if u["run"] == run and "vs real-donor" not in u["condition"]
                       and u["condition"].startswith("real" if cond == "real" else "ctrl")]
                if not idx:
                    continue
                r = arith_rows(U, idx, changed, data, args.reps)
                arith_summary[f"{run}|{cond}"] = r
                print(f"| {mech} | `{run[:15]}..{run.split('_', 2)[-1]}` | {cond} | {r['n_other']} | "
                      f"{fmt_rate(r['obs_hits'], r['n_other'])} | {r['null_patched']:.3f} (n={r['n_null_patched']:.0f}) | "
                      f"{r['null_base']:.3f} (n={r['n_null_base']:.0f}) | {fmt_rate(r['own_base_hits'], r['n_changed'])} |")
        pooled = [i for i, u in enumerate(U) if u["mech"] == mech and u["condition"].startswith("real")]
        r = arith_rows(U, pooled, changed, data, args.reps)
        arith_summary[f"{mech}|pooled real"] = r
        print(f"| {mech} | **pooled, real** | real | {r['n_other']} | {fmt_rate(r['obs_hits'], r['n_other'])} | "
              f"{r['null_patched']:.3f} (n={r['n_null_patched']:.0f}) | {r['null_base']:.3f} (n={r['n_null_base']:.0f}) | "
              f"{fmt_rate(r['own_base_hits'], r['n_changed'])} |")

    print(f"\nmissing / unusable files: {missing or 'none'}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"h3": summary, "arith": arith_summary, "missing": missing}, indent=1))


def fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return int(x) if abs(x - round(x)) < 1e-9 else round(float(x), 4)


def arith_rows(U, idx, changed, data, reps, seed=0) -> dict:
    """Observed arith-of-recipient-numbers rate on the other_number bucket vs. the same
    classifier applied to (a) another record's changed patched answer, (b) another
    record's base answer -- both through the same other_number filter relative to THIS
    recipient/donor -- and (c) this record's own base answer (no filter)."""
    rng = random.Random(seed)
    ch = [i for i in idx if changed[i]]
    other, own_base_hits = [], 0
    for i in ch:
        u = U[i]
        ex = data[u["ds"]]
        rex, dex = ex[u["recipient_idx"]], ex.get(u["donor_idx"])
        p, b = to_float(u["answer_patched"]), to_float(u["answer_base"])
        if is_other_number(p, b, rex, dex):
            other.append(i)
        if b is not None and isin(arith_values(rex), b):
            own_base_hits += 1
    obs = sum(isin(arith_values(data[U[i]["ds"]][U[i]["recipient_idx"]]), to_float(U[i]["answer_patched"]))
              for i in other)
    pool_p = [to_float(U[i]["answer_patched"]) for i in ch]
    pool_b = [to_float(U[i]["answer_base"]) for i in idx if to_float(U[i]["answer_base"]) is not None]
    rates_p, rates_b, n_p, n_b = [], [], [], []
    for _ in range(reps):
        for pool, rates, ns in ((pool_p, rates_p, n_p), (pool_b, rates_b, n_b)):
            hits = n = 0
            for i in ch:
                u = U[i]
                ex = data[u["ds"]]
                rex, dex = ex[u["recipient_idx"]], ex.get(u["donor_idx"])
                v = rng.choice(pool)
                if not is_other_number(v, to_float(u["answer_base"]), rex, dex):
                    continue
                n += 1
                hits += isin(arith_values(rex), v)
            rates.append(hits / n if n else 0.0)
            ns.append(n)
    return {"n_changed": len(ch), "n_other": len(other), "obs_hits": int(obs),
            "null_patched": float(np.mean(rates_p)), "n_null_patched": float(np.mean(n_p)),
            "null_base": float(np.mean(rates_b)), "n_null_base": float(np.mean(n_b)),
            "own_base_hits": own_base_hits}


if __name__ == "__main__":
    main()
