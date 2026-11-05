#!/usr/bin/env python3
"""Re-render RuleChain questions with the start value after the rules (TRAINING_PROTOCOL section 11). Only the question text
changes: chains, ids, levels, steps, answers and splits stay the same, so results pair item by item with the original format.
  Start with 57. Rule 1: ... Rule 2: ... What is the final number?
  Rule 1: ... Rule 2: ... Start with 57. What is the final number?
Usage: python scripts/make_rulechain_startlast.py <src root> <dst root> <file> [<file> ...]   (files relative to the root, e.g. depth2/test.json)"""
import json
import re
import sys
from pathlib import Path

Q = re.compile(r"^(Start with \d+\.) (Rule 1: .*) (What is the final number\?)$")


def start_last(q: str) -> str:
    m = Q.match(q.strip())
    if not m:
        raise ValueError(f"not a RuleChain question: {q[:80]}")
    return f"{m.group(2)} {m.group(1)} {m.group(3)}"


def main() -> None:
    src, dst, files = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    for name in files:
        rows = json.loads((src / name).read_text())
        out = [dict(r, question=start_last(r["question"])) for r in rows]
        (dst / name).parent.mkdir(parents=True, exist_ok=True)
        (dst / name).write_text(json.dumps(out, ensure_ascii=False, indent=0) + "\n")
        print(name, len(out))


if __name__ == "__main__":
    main()
