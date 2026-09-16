#!/usr/bin/env python3
"""Regenerate results/README.md and EXPERIMENTS.md from results/<run_id>/. Run after
logging a run or resolving a merge; commit the output."""
from latentreasoning.runlog.index import rebuild

if __name__ == "__main__":
    rebuild()
    print("rewrote results/README.md and EXPERIMENTS.md")
