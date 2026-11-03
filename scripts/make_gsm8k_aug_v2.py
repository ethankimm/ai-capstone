#!/usr/bin/env python3
"""GSM8K-Aug in the RuleChain (synthetic v2) JSON schema, for running TRAINING_PROTOCOL.md unchanged except for the data
(GSM8K_PILOT.md). Writes <out>/gsm8k_aug/:
  train.json          100,000 items sampled from `load_gsm8k_aug("train")` (repo validation carve-out excluded), seed 5980
  valid.json          the repo's fixed 1,000-item validation split (`load_gsm8k_aug("validation")`)
  test.json           GSM8K test, all 1,319 items
  SHA256SUMS
Each row: {"question", "steps": ["<<a+b=c>>", ...], "answer", "level", "task": "gsm8k_aug", "id"} where steps = the
rationale split on whitespace (as Coconut's preprocessing/gsm_icot.py does) and level = number of steps, so the synthetic
eval scripts report accuracy per step count. Item order: train in sampled order, valid/test in HF order.
Deterministic: rerunning gives identical bytes (hashes registered in configs/synthetic/v2_data_sha256.json).

Usage: uv run python scripts/make_gsm8k_aug_v2.py /local-ssd/ktn48/lrm_work/synthetic_data_v2
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug

TRAIN_N = 100_000
SEED = 5980
TASK = "gsm8k_aug"


def rows(examples, split: str) -> list[dict]:
    out = []
    for ex in examples:
        steps = ex.rationale.strip().split()
        out.append({"question": ex.question, "steps": steps, "answer": ex.answer.strip(), "level": len(steps),
                    "task": TASK, "id": f"{TASK}-{split}-{ex.idx:06d}"})
    return out


def main() -> None:
    out = Path(sys.argv[1]) / TASK
    out.mkdir(parents=True, exist_ok=True)
    train = load_gsm8k_aug("train", seed=None)
    train = random.Random(SEED).sample(train, TRAIN_N)
    files = {
        "train.json": rows(train, "train"),
        "valid.json": rows(load_gsm8k_aug("validation", seed=None), "valid"),
        "test.json": rows(load_gsm8k_aug("test", seed=None), "test"),
    }
    sums = []
    for name, rs in files.items():
        data = json.dumps(rs).encode()
        (out / name).write_bytes(data)
        sums.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
        print(name, len(rs), dict(sorted(Counter(r["level"] for r in rs).items())))
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print("\n".join(sums))


if __name__ == "__main__":
    main()
