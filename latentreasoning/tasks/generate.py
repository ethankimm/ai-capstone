"""Build train/valid/valid_monitor/test splits for the synthetic depth and width tasks.

    uv run python -m latentreasoning.tasks.generate --task all \
        --out ~/Documents/Penn/CIS5980/synthetic_data

Writes ``<out>/<task>/{train,valid,valid_monitor,test,test_full,stats}.json``.

Split rules (per task, per level):

* Levels whose content space is < ``ENUM_THRESHOLD`` are enumerated and shuffled
  (seeded): test = min(test_per_level, 40% of space), validation = min(valid quota,
  10% of space), train = the rest, truncated to the train quota, or -- if smaller --
  oversampled (every unique item repeated floor/ceil times, seeded) to reach the quota.
  Depth enumerated levels are therefore uniform over the space, not drawn with the
  per-step sampler. Width L2 (enumerated) uses a weighted shuffle by the answer-balanced
  sampler's probability, and its train oversampling = each remaining item once plus
  weighted draws with replacement.
* Larger levels are sampled with the per-step sampler, rejecting anything whose
  canonical key was already drawn (test first, then validation, then train), so all
  splits are disjoint and duplicate-free.
* Train/valid quotas are split evenly across levels (remainder to the lowest levels).
  ``valid_monitor`` = the first ``monitor_per_level`` validation items of each level
  (fewer if the level's validation set is smaller).

Canonical keys: depth = (x0, ops, operands, commute flags); width = the ordered list of
operand pairs (equivalent to question-text equality).

``test.json`` is plain (id/task/level/question/steps/answer); ``test_full.json`` adds
``twins`` (one per ladder target; see ``synthetic.build_ladder``) and ``decoys``; each
twin/decoy carries ``in_train`` (its canonical key occurs in the train split).
Output is byte-reproducible for a given seed and generator version.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

from latentreasoning.tasks import synthetic as S

GENERATOR_VERSION = "1.2.0"
DEFAULT_SEED = 5980
ENUM_THRESHOLD = 50_000
TASK_LEVELS = {"depth": list(S.DEPTH_LEVELS), "width": list(S.WIDTH_LEVELS)}

PLAIN_KEYS = ("id", "task", "level", "question", "steps", "answer")


def _rng(seed: int, *parts) -> random.Random:
    return random.Random("|".join(str(x) for x in (seed, *parts)))


def _split_quota(total: int, levels: list[int]) -> dict[int, int]:
    base, rem = divmod(total, len(levels))
    return {lv: base + (1 if i < rem else 0) for i, lv in enumerate(levels)}


def _space_size(task: str, level: int) -> int:
    return S.depth_space_size(level) if task == "depth" else S.width_space_size(level)


def _enumerate(task: str, level: int):
    return S.enumerate_depth(level) if task == "depth" else S.enumerate_width(level)


def _sample(task: str, level: int, rng: random.Random):
    return S.sample_depth(rng, level) if task == "depth" else S.sample_width(rng, level)


def _record(task: str, level: int, split: str, i: int, problem) -> dict:
    rec = S.render(problem)
    rec.update(id=f"{task}-{split}-L{level:02d}-{i:05d}", task=task, level=level)
    return rec


def build_task(
    task: str,
    seed: int = DEFAULT_SEED,
    train_total: int = 100_000,
    valid_total: int = 2_000,
    test_per_level: int = 500,
    monitor_per_level: int = 30,
    levels: list[int] | None = None,
) -> tuple[dict[str, list[dict]], dict]:
    """Return ({split: records}, stats). Pure function of its arguments."""
    levels = levels or TASK_LEVELS[task]
    train_q = _split_quota(train_total, levels)
    valid_q = _split_quota(valid_total, levels)

    splits = {"train": [], "valid": [], "valid_monitor": [], "test": []}
    test_problems: list[tuple[int, object]] = []
    level_stats: dict[str, dict] = {}
    train_keys: set = set()

    for level in levels:
        space = _space_size(task, level)
        enumerated = space < ENUM_THRESHOLD
        if enumerated:
            pool = _enumerate(task, level)
            assert len(pool) == space, (task, level, len(pool), space)
            if task == "width":
                # Weighted shuffle (Efraimidis-Spirakis) so test/valid follow the
                # answer-balanced sampler's distribution rather than uniform-over-space.
                r = _rng(seed, task, level, "enum-shuffle")
                keyed = [(math.log(1.0 - r.random()) / S.width_problem_prob(p), i) for i, p in enumerate(pool)]
                pool = [pool[i] for _, i in sorted(keyed, reverse=True)]
            else:
                _rng(seed, task, level, "enum-shuffle").shuffle(pool)
            n_test = min(test_per_level, int(0.4 * space))
            n_valid = min(valid_q[level], int(0.1 * space))
            test = pool[:n_test]
            valid = pool[n_test:n_test + n_valid]
            rest = pool[n_test + n_valid:]
            if len(rest) >= train_q[level]:
                train = rest[:train_q[level]]
            elif task == "width":
                # every remaining item once, then weighted draws (with replacement) by
                # sampler probability up to the quota
                r = _rng(seed, task, level, "oversample")
                w = [S.width_problem_prob(p) for p in rest]
                train = rest + r.choices(rest, weights=w, k=train_q[level] - len(rest))
            else:
                reps, extra = divmod(train_q[level], len(rest))
                train = rest * reps + _rng(seed, task, level, "oversample").sample(rest, extra)
        else:
            rng = _rng(seed, task, level, "sample")
            seen: set = set()

            def draw(n):
                got = []
                while len(got) < n:
                    p = _sample(task, level, rng)
                    k = p.key()
                    if k not in seen:
                        seen.add(k)
                        got.append(p)
                return got

            test = draw(test_per_level)
            valid = draw(valid_q[level])
            train = draw(train_q[level])

        n_unique_train = len({p.key() for p in train})
        train_keys.update(p.key() for p in train)
        splits["train"] += [_record(task, level, "train", i, p) for i, p in enumerate(train)]
        valid_recs = [_record(task, level, "valid", i, p) for i, p in enumerate(valid)]
        splits["valid"] += valid_recs
        splits["valid_monitor"] += valid_recs[:monitor_per_level]
        splits["test"] += [_record(task, level, "test", i, p) for i, p in enumerate(test)]
        test_problems += [(level, p) for p in test]
        level_stats[str(level)] = {
            "space_size": space,
            "enumerated": enumerated,
            "test": len(test),
            "valid": len(valid),
            "valid_monitor": min(monitor_per_level, len(valid)),
            "train_rows": len(train),
            "train_unique": n_unique_train,
            "oversampling_factor": round(len(train) / n_unique_train, 4),
        }

    _rng(seed, task, "train-order").shuffle(splits["train"])

    # Twins and decoys for test items.
    test_full = []
    for rec, (level, p) in zip(splits["test"], test_problems):
        rng = _rng(seed, task, rec["id"], "twins")
        ladder = S.build_ladder(task, p, rng)
        decoys = S.depth_decoys(p, rng) if task == "depth" else S.width_decoys(p, rng)
        full = dict(rec)
        full["twins"] = [dict(r, in_train=k in train_keys) for k, r in ladder]
        full["decoys"] = [dict(r, in_train=k in train_keys) for k, r in decoys]
        test_full.append(full)
    splits["test_full"] = test_full

    stats = {
        "task": task,
        "seed": seed,
        "generator_version": GENERATOR_VERSION,
        "enum_threshold": ENUM_THRESHOLD,
        "ladder_targets": list(S.LADDER_TARGETS),
        "ladder_tolerance_factor": S.LADDER_TOLERANCE,
        "counts": {k: len(v) for k, v in splits.items()},
        "levels": level_stats,
        "content": _content_stats(task, splits),
        "twins": _twin_stats(test_full),
        "decoys": _decoy_stats(task, test_full),
    }
    return splits, stats


def _content_stats(task: str, splits) -> dict:
    out = {}
    by_level: dict[int, dict[str, dict]] = {}
    for split in ("train", "test"):
        for r in splits[split]:
            by_level.setdefault(r["level"], {})[r["question"]] = r  # unique content
    for level, recs in sorted(by_level.items()):
        answers = Counter(int(r["answer"]) for r in recs.values())
        n = sum(answers.values())
        d = {
            "n_unique_train_test": n,
            "n_distinct_answers": len(answers),
            "answer_mean": round(statistics.mean(answers.elements()), 3),
            "top5_answers": [[a, round(c / n, 4)] for a, c in answers.most_common(5)],
        }
        if task == "depth":
            ops = Counter()
            zero_steps = 0
            n_steps = 0
            for r in recs.values():
                for st in r["steps"]:
                    _, op, _, v = S.parse_step(st)
                    ops[op] += 1
                    zero_steps += v == 0
                    n_steps += 1
            d["op_mix"] = {op: round(ops[op] / n_steps, 4) for op in S.OPS}
            d["frac_steps_value_zero"] = round(zero_steps / n_steps, 4)
        out[str(level)] = d
    return out


def _twin_stats(test_full) -> dict:
    per_target: dict[float, list] = {}
    per_level: dict[int, dict] = {}
    for ex in test_full:
        lv = per_level.setdefault(ex["level"], {"n_base": 0, "n_no_twins": 0, "step_changed": Counter(), "in_train": 0, "n": 0, "within": Counter()})
        lv["n_base"] += 1
        if not ex["twins"]:
            lv["n_no_twins"] += 1
        for t in ex["twins"]:
            per_target.setdefault(t["target_rel_change"], []).append(t)
            lv["step_changed"][t["step_changed"]] += 1
            lv["in_train"] += t["in_train"]
            lv["n"] += 1
            lv["within"][t["target_rel_change"]] += t["within_tolerance"]
    targets = {}
    for t, tw in sorted(per_target.items()):
        targets[str(t)] = {
            "n": len(tw),
            "frac_within_tolerance": round(sum(x["within_tolerance"] for x in tw) / len(tw), 4),
            "median_rel_change_step": statistics.median(x["rel_change_step"] for x in tw),
            "median_rel_change_answer": statistics.median(
                x["rel_change_answer"] for x in tw if x["rel_change_answer"] is not None
            ),
            "frac_duplicate_of_earlier_target": round(
                sum(x["duplicate_of_target"] is not None for x in tw) / len(tw), 4
            ),
        }
    levels = {}
    for level, lv in sorted(per_level.items()):
        n_with = lv["n_base"] - lv["n_no_twins"]
        levels[str(level)] = {
            "n_base_without_twins": lv["n_no_twins"],
            "frac_twins_in_train": round(lv["in_train"] / lv["n"], 4) if lv["n"] else None,
            "step_changed_hist": {str(k): v for k, v in sorted(lv["step_changed"].items())},
            "frac_within_tolerance_by_target": {
                str(t): round(lv["within"][t] / n_with, 4) if n_with else None
                for t in S.LADDER_TARGETS
            },
        }
    return {"per_target": targets, "per_level": levels}


def _decoy_stats(task: str, test_full) -> dict:
    out = {}
    for ex in test_full:
        lv = out.setdefault(str(ex["level"]), {"n_base": 0, "types": Counter(), "in_train": 0, "n": 0, "n_decoys_hist": Counter()})
        lv["n_base"] += 1
        lv["n_decoys_hist"][len(ex["decoys"])] += 1
        for t in {d["decoy_type"] for d in ex["decoys"]}:
            lv["types"][t] += 1
        for d in ex["decoys"]:
            lv["in_train"] += d["in_train"]
            lv["n"] += 1
    res = {}
    for level, lv in out.items():
        res[level] = {
            "frac_base_with_type": {t: round(c / lv["n_base"], 4) for t, c in sorted(lv["types"].items())},
            "mean_decoys_per_base": round(lv["n"] / lv["n_base"], 4),
            "n_decoys_hist": {str(k): v for k, v in sorted(lv["n_decoys_hist"].items())},
            "frac_decoys_in_train": round(lv["in_train"] / lv["n"], 4) if lv["n"] else None,
        }
    return res


def dumps_records(records: list[dict], plain: bool = False) -> str:
    """Deterministic JSON list: one record per line, sorted keys."""
    if plain:
        records = [{k: r[k] for k in PLAIN_KEYS} for r in records]
    lines = [json.dumps(r, sort_keys=True, ensure_ascii=True) for r in records]
    return "[\n" + ",\n".join(lines) + "\n]\n"


def write_task(out_dir: Path, task: str, **kwargs) -> dict:
    splits, stats = build_task(task, **kwargs)
    tdir = Path(out_dir).expanduser() / task
    tdir.mkdir(parents=True, exist_ok=True)
    files = {
        "train.json": dumps_records(splits["train"], plain=True),
        "valid.json": dumps_records(splits["valid"], plain=True),
        "valid_monitor.json": dumps_records(splits["valid_monitor"], plain=True),
        "test.json": dumps_records(splits["test"], plain=True),
        "test_full.json": dumps_records(splits["test_full"]),
    }
    stats["sha256"] = {}
    for name, text in files.items():
        data = text.encode()
        (tdir / name).write_bytes(data)
        stats["sha256"][name] = hashlib.sha256(data).hexdigest()
    (tdir / "stats.json").write_text(json.dumps(stats, sort_keys=True, indent=2) + "\n")
    return stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", choices=["depth", "width", "all"], default="all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--train-total", type=int, default=100_000)
    ap.add_argument("--valid-total", type=int, default=2_000)
    ap.add_argument("--test-per-level", type=int, default=500)
    ap.add_argument("--monitor-per-level", type=int, default=30)
    args = ap.parse_args(argv)
    tasks = ["depth", "width"] if args.task == "all" else [args.task]
    for task in tasks:
        stats = write_task(
            Path(args.out), task, seed=args.seed, train_total=args.train_total,
            valid_total=args.valid_total, test_per_level=args.test_per_level,
            monitor_per_level=args.monitor_per_level,
        )
        print(f"{task}: {stats['counts']} -> {Path(args.out).expanduser() / task}")


if __name__ == "__main__":
    main()
