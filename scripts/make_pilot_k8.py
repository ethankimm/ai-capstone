#!/usr/bin/env python3
"""Pilot data for PILOT_K8.md: RuleChain (depth2) restricted to k <= 8, with a 25k-item train subset.

Reads <data>/depth2/{train,valid,test,valid_monitor}.json and writes <data>/depth2_k8/:
  train.json          25,000 items, 3,125 per level for k = 1..8, sampled without replacement (seed 5980)
  valid.json          every valid item with k <= 8 (1,336)
  test.json           every test item with k <= 8 (4,000; the same items as round 1's L1-8)
  valid_monitor.json  every monitor item with k <= 8 (240)
  SHA256SUMS
Item order within each file follows the source file. Deterministic: rerunning gives identical bytes.

Usage: python3 scripts/make_pilot_k8.py /local-ssd/ktn48/lrm_work/synthetic_data_v2
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

MAX_K = 8
TRAIN_PER_LEVEL = 3125
SEED = 5980


def main() -> None:
    root = Path(sys.argv[1])
    src, out = root / "depth2", root / "depth2_k8"
    out.mkdir(exist_ok=True)
    rng = random.Random(SEED)

    train = json.loads((src / "train.json").read_text())
    keep = set()
    for k in range(1, MAX_K + 1):
        idx = [i for i, x in enumerate(train) if x["level"] == k]
        keep.update(rng.sample(idx, TRAIN_PER_LEVEL))
    files = {"train.json": [x for i, x in enumerate(train) if i in keep]}
    for name in ("valid.json", "test.json", "valid_monitor.json"):
        files[name] = [x for x in json.loads((src / name).read_text()) if x["level"] <= MAX_K]

    sums = []
    for name, rows in files.items():
        data = json.dumps(rows).encode()
        (out / name).write_bytes(data)
        sums.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
        print(name, len(rows), dict(sorted(Counter(x["level"] for x in rows).items())))
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print("\n".join(sums))


if __name__ == "__main__":
    main()
