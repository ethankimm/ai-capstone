#!/usr/bin/env python3
"""Zero-model sanity-floor baseline (last number in the question) over the bundled
offline GSM8K-Aug sample. No torch/GPU. Proves the data -> eval -> run-record pipeline
works and gives every real mechanism a floor to clear.

Usage: uv run python scripts/run_baseline.py [--n N]
"""
from __future__ import annotations

import argparse

from latentreasoning.baselines.heuristic import name as mechanism_name, predict
from latentreasoning.data.gsm8k_aug import load_local_sample
from latentreasoning.eval.harness import run_eval
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="subset size (default: full bundled sample)")
    args = parser.parse_args()

    examples = load_local_sample(n=args.n)
    result = run_eval(predict, examples)
    print(f"n={result.n}  final_answer_accuracy={result.final_answer_accuracy:.3f}  "
          f"unparseable_rate={result.unparseable_rate:.3f}  sec/ex={result.sec_per_example:.4f}")

    record = RunRecord(
        run_id=new_run_id(mechanism_name, "gsm8k-aug-sample"),
        mechanism=mechanism_name,
        stage="pilot",
        model=ModelInfo(backbone="none", checkpoint="none"),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=result.n, seed=0),
        metrics={
            "final_answer_accuracy": result.final_answer_accuracy,
            "unparseable_rate": result.unparseable_rate,
            "sec_per_example": result.sec_per_example,
            "compute_steps": 0,
        },
        notes="Sanity-floor baseline over the bundled offline GSM8K-Aug sample.",
    )
    path = record.save(predictions=result.records)
    print(f"wrote {path.parent}/  -> fill in notes.md, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
