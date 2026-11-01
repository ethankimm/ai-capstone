#!/usr/bin/env python3
"""Early stopping for Coconut `run.py`, which has none: watch its captured log and kill
the torchrun process group once per-epoch monitor accuracy has not improved by at least
`--min-delta` for `--patience` epochs, or after `--max-epochs` epochs. With
`save_only_improve: True` the best-on-monitor checkpoint is already on disk.

  python scripts/pod/coconut_watch.py --log /workspace/logs/x.log --pid <torchrun pid>
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthetic_common import parse_coconut_log  # noqa: E402


def should_stop(accs: list[float], patience: int, min_delta: float, max_epochs: int,
                min_epochs: int = 0) -> str | None:
    if len(accs) >= max_epochs:
        return f"max_epochs={max_epochs} reached"
    if len(accs) > patience and len(accs) >= min_epochs:
        best_before = max(accs[:-patience])
        if max(accs[-patience:]) < best_before + min_delta:
            return f"no gain >= {min_delta} in last {patience} epochs (best before {best_before:.4f})"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min-delta", type=float, default=0.005)
    ap.add_argument("--max-epochs", type=int, default=10)
    ap.add_argument("--interval", type=int, default=60)
    # never stop on patience before this many epochs (the no_cot plateau run must reach checkpoint_12, the
    # data-matched point; a flat stretch near zero accuracy early on would otherwise end it)
    ap.add_argument("--min-epochs", type=int, default=0)
    args = ap.parse_args()
    while True:
        try:
            os.kill(args.pid, 0)
        except ProcessLookupError:
            print("watch: process exited on its own", flush=True)
            return
        text = Path(args.log).read_text(errors="replace") if Path(args.log).exists() else ""
        accs = [v["accuracy"] for v in parse_coconut_log(text)["val_accuracy"]]
        reason = should_stop(accs, args.patience, args.min_delta, args.max_epochs, args.min_epochs)
        if reason:
            print(f"watch: stopping after {len(accs)} epochs: {reason}; accs={accs}", flush=True)
            # give run.py a moment to finish writing a checkpoint saved right after the eval
            time.sleep(90)
            try:
                os.killpg(os.getpgid(args.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
