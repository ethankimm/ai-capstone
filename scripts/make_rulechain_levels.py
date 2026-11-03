#!/usr/bin/env python3
"""RuleChain (depth2) restricted to a level range, for the CODI "do latents work when they are needed?" runs.

Uses `synthetic_v2.build_task` with the canonical seed (5980) and test_per_level 500, so the per-level RNG draws the
same test items first: the test split here is identical to the canonical depth2 test items of these levels (checked
below), so results compare directly with round 1. Valid and train are drawn after test from the same per-level
stream, so every split is disjoint from the canonical test; the first ~166 valid items per level equal the canonical
valid items, train is new (much larger than round 1's ~8.3k per level, so each item is seen only a few times).

  uv run python scripts/make_rulechain_levels.py --levels 4-7 --train-total 200000 \\
      --out ../synthetic_data_v2_l4to7 --canonical ../synthetic_data_v2
Writes <out>/depth2/{train,valid,valid_monitor,test}.json + stats.json, and the hash file for check_data_hashes.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from latentreasoning.tasks.generate import dumps_records
from latentreasoning.tasks.synthetic_v2 import DEFAULT_SEED, build_task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="4-7")
    ap.add_argument("--train-total", type=int, default=200_000)
    ap.add_argument("--train-per-level", default=None,
                    help='per-level train counts overriding --train-total, e.g. "1:10000,2:10000,4:40000"')
    ap.add_argument("--valid-total", type=int, default=2_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--canonical", default=None, help="canonical synthetic_data_v2 root, to verify the test split")
    ap.add_argument("--hash-file", default=None, help="default: configs/synthetic/v2_l<lo>to<hi>_data_sha256.json")
    a = ap.parse_args()
    lo, hi = map(int, a.levels.split("-"))
    levels = list(range(lo, hi + 1))
    quota = None
    if a.train_per_level:
        quota = {int(k): int(v) for k, v in (kv.split(":") for kv in a.train_per_level.split(","))}
        assert sorted(quota) == levels, f"--train-per-level must cover exactly levels {levels}"
        a.train_total = sum(quota.values())
    splits, stats = build_task("depth2", seed=DEFAULT_SEED, train_total=a.train_total, valid_total=a.valid_total,
                               test_per_level=500, levels=levels, train_quota=quota)
    if quota:
        stats["train_per_level"] = quota
    d = Path(a.out).expanduser() / "depth2"
    d.mkdir(parents=True, exist_ok=True)
    sha = {}
    for name, recs in splits.items():
        text = dumps_records(recs)
        (d / f"{name}.json").write_text(text)
        sha[f"depth2/{name}.json"] = hashlib.sha256(text.encode()).hexdigest()
    stats["levels_restricted"] = levels
    (d / "stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True))
    print(stats["counts"])

    if a.canonical:
        canon = json.loads((Path(a.canonical).expanduser() / "depth2/test.json").read_text())
        key = lambda r: (r["level"], r["question"], r["answer"])  # noqa: E731
        want = sorted(key(r) for r in canon if lo <= r["level"] <= hi)
        got = sorted(key(r) for r in splits["test"])
        assert got == want, "test split differs from the canonical test items of these levels"
        canon_q = {r["question"] for r in canon}
        assert not canon_q & {r["question"] for r in splits["train"] + splits["valid"]}, "train/valid overlap test"
        print(f"test split identical to canonical L{lo}-L{hi} test ({len(got)} items); train/valid disjoint from it")

    hf = Path(a.hash_file) if a.hash_file else \
        Path(__file__).resolve().parents[1] / f"configs/synthetic/v2_l{lo}to{hi}_data_sha256.json"
    hf.write_text(json.dumps({"_note": f"sha256 of RuleChain (depth2) restricted to levels {lo}-{hi}, generator "
                              f"{stats['generator_version']}, seed {DEFAULT_SEED}, train_total {a.train_total}"
                              f"{', train_per_level ' + a.train_per_level if a.train_per_level else ''}, made by "
                              "scripts/make_rulechain_levels.py. Use with HASH_FILE=<this> "
                              "scripts/check_data_hashes.py.", "files": sha}, indent=1) + "\n")
    print(f"hashes -> {hf}")


if __name__ == "__main__":
    main()
