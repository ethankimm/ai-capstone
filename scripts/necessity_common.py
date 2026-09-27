"""Shared scoring / sharding for the necessity runs (`necessity_codi.py`,
`necessity_coconut.py`): early termination + ablate-all / ablate-single with zero, mean
and norm-matched-noise replacement, plus sampled pass@k on the ablate-all conditions.

Per example, a mechanism script produces
  records[cond] = {"pred": str|None, "correct": bool}      (greedy)
  samples[cond] = [bool] * k                                (temperature samples of the answer
                                                             only -- the latents are deterministic)
and this module turns them into the per-condition summary:

  accuracy (+ Wilson CI), unparseable rate, answer_changed vs the full-budget baseline,
  PWC  = preserved-when-correct = P(condition correct | baseline correct),
  flips: correct->wrong (count, rate over baseline-correct) and wrong->correct (count,
         rate over baseline-wrong), paired exact McNemar on those two counts,
  pass@1 / pass@k (unbiased estimator, Chen et al. 2021) for sampled conditions.

Sharding: `--num_shards N --shard_index i` runs examples[i::N] and writes
`<shard_dir>/shard_<i>.json`; `--merge_dir <shard_dir>` combines them and saves the run.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPLACEMENTS = ("zero", "mean", "noise")


def wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = x / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def condition_names(n_sites: int) -> list[str]:
    conds = [f"trunc_{k}" for k in range(n_sites + 1)]
    conds += [f"ablate_all_{r}" for r in REPLACEMENTS]
    conds += [f"ablate_single_{r}_{s}" for r in REPLACEMENTS for s in range(n_sites)]
    return conds


def sampled_condition_names(n_sites: int) -> list[str]:
    return [f"trunc_{n_sites}", "trunc_0"] + [f"ablate_all_{r}" for r in REPLACEMENTS]


def summarize(records: dict[str, dict], samples: dict[str, dict], baseline: str, k: int) -> dict:
    """records[cond][idx] = {"pred","correct"}; samples[cond][idx] = [bool]."""
    base = records[baseline]
    out = {}
    for cond, rows in records.items():
        idxs = [i for i in rows if i in base]
        n = len(idxs)
        correct = sum(rows[i]["correct"] for i in idxs)
        base_ok = [i for i in idxs if base[i]["correct"]]
        base_bad = [i for i in idxs if not base[i]["correct"]]
        c2w = sum(1 for i in base_ok if not rows[i]["correct"])
        w2c = sum(1 for i in base_bad if rows[i]["correct"])
        s = {
            "n": n, "accuracy": correct / n, "accuracy_wilson_ci": list(wilson_ci(correct, n)),
            "unparseable_rate": sum(rows[i]["pred"] is None for i in idxs) / n,
            "answer_changed_vs_baseline": sum(rows[i]["pred"] != base[i]["pred"] for i in idxs) / n,
            "pwc": (len(base_ok) - c2w) / len(base_ok) if base_ok else None,
            "pwc_wilson_ci": list(wilson_ci(len(base_ok) - c2w, len(base_ok))),
            "flips_correct_to_wrong": c2w, "flips_wrong_to_correct": w2c,
            "flip_rate_correct_to_wrong": c2w / len(base_ok) if base_ok else None,
            "flip_rate_wrong_to_correct": w2c / len(base_bad) if base_bad else None,
            "mcnemar_p_exact": mcnemar_exact_p(w2c, c2w),
        }
        if cond in samples and samples[cond]:
            srows = samples[cond]
            p1 = [pass_at_k(len(v), sum(v), 1) for v in srows.values()]
            pk = [pass_at_k(len(v), sum(v), k) for v in srows.values()]
            pk_base_ok = [pass_at_k(len(srows[i]), sum(srows[i]), k) for i in srows if base[i]["correct"]]
            pk_base_bad = [pass_at_k(len(srows[i]), sum(srows[i]), k) for i in srows if not base[i]["correct"]]
            s.update({
                "sampled_n": len(srows), "pass_at_1": sum(p1) / len(p1), f"pass_at_{k}": sum(pk) / len(pk),
                f"pass_at_{k}_given_baseline_correct": sum(pk_base_ok) / len(pk_base_ok) if pk_base_ok else None,
                f"pass_at_{k}_given_baseline_wrong": sum(pk_base_bad) / len(pk_base_bad) if pk_base_bad else None,
            })
        out[cond] = s
    return out


def print_table(summary: dict, k: int) -> None:
    print(f"\n{'condition':24s}    n    acc    PWC   c->w  w->c   McNemar p   changed | pass@1 pass@{k}")
    for cond, s in summary.items():
        extra = f"| {s['pass_at_1']:.3f}  {s[f'pass_at_{k}']:.3f}" if "pass_at_1" in s else ""
        pwc = f"{s['pwc']:.3f}" if s["pwc"] is not None else "  -  "
        print(f"{cond:24s} {s['n']:5d}  {s['accuracy']:.3f}  {pwc}  {s['flips_correct_to_wrong']:4d}  "
              f"{s['flips_wrong_to_correct']:4d}  {s['mcnemar_p_exact']:10.3g}   {s['answer_changed_vs_baseline']:.3f}  {extra}")


def write_shard(shard_dir: str, shard_index: int, payload: dict) -> Path:
    path = Path(shard_dir) / f"shard_{shard_index}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=str))
    return path


def load_shards(merge_dir: str) -> dict:
    """Combine shard payloads; checks every shard used the same replacement vectors."""
    shards = [json.loads(p.read_text()) for p in sorted(Path(merge_dir).glob("shard_*.json"))]
    assert shards, f"no shard_*.json in {merge_dir}"
    n_shards = shards[0]["meta"]["num_shards"]
    assert len(shards) == n_shards, f"found {len(shards)} shards, expected {n_shards}"
    norms = shards[0]["meta"]["mean_norms"]
    for s in shards[1:]:
        assert all(abs(a - b) < 1e-3 for a, b in zip(norms, s["meta"]["mean_norms"])), "shards used different means"
    merged = {"meta": shards[0]["meta"], "records": {}, "samples": {}, "elapsed_sec": []}
    for s in shards:
        for key in ("records", "samples"):
            for cond, rows in s[key].items():
                merged[key].setdefault(cond, {}).update({int(i): v for i, v in rows.items()})
        merged["elapsed_sec"].append(s["meta"]["elapsed_sec"])
    return merged


def predictions_rows(records: dict, samples: dict, gold: dict) -> list[dict]:
    """One row per example: greedy pred/correct for every condition, sample correctness counts."""
    idxs = sorted(next(iter(records.values())).keys())
    rows = []
    for i in idxs:
        row = {"idx": i, "gold_answer": gold[i],
               "greedy": {c: records[c][i] for c in records if i in records[c]},
               "sampled_n_correct": {c: sum(samples[c][i]) for c in samples if i in samples[c]}}
        rows.append(row)
    return rows
