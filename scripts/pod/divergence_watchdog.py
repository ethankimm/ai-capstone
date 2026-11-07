#!/usr/bin/env python3
"""Kill a CODI training run that diverges, so a blow-up costs minutes instead of the rest of the run.

Parses only the HF Trainer's logging-step lines (10-step averages), which the patched CODI prints as
    {'loss': 0.8123, 'epoch': 1.23}
    {'grad_norm': 0.091..., 'epoch': 1.23}
(the per-micro-batch `loss=...` lines are ignored: single micro-batches spike to ~12x the median in healthy runs).
Kills train.py (pkill -f; one run per pod) and appends DIVERGED to LOG.watchdog when, after WARMUP logging steps:
  * loss or grad_norm is non-finite, or
  * grad_norm > GRAD_MAX (healthy l1to7w CODI max after warmup: 7.5; the diverged CODI-0 went 1185 -> 6e17 -> inf; raised from 50 for the
    step decoder, whose gradients count in the norm and are clipped anyway; replay still kills that CODI-0 at step 4590 and nothing else), or
  * loss > HARD x the median of the previous WINDOW losses, or
  * loss > SOFT x that median for SUSTAIN consecutive logs.
Checked against the 2026-11-03 l1to7w logs (--simulate): the surviving CODI run is never killed; the diverged CODI-0
is killed at its second blow-up (step ~4590).

  nohup python3 scripts/pod/divergence_watchdog.py /workspace/logs/<name>_codi.log > /workspace/logs/watchdog_codi.log 2>&1 &
  python3 scripts/pod/divergence_watchdog.py <log> --simulate     # replay a finished log, print where it would kill
"""
import math
import re
import statistics
import subprocess
import sys
import time

WINDOW, WARMUP, HARD, SOFT, SUSTAIN, GRAD_MAX = 200, 100, 20.0, 3.0, 50, 500.0
PAT = re.compile(r"\{'loss': ([^,]+), 'epoch': ([0-9.]+)\}\n\{'grad_norm': ([^,]+), 'epoch'")


def parse(text: str) -> list[tuple[float, float, float]]:
    return [(float(a), float(e), float(g)) for a, e, g in PAT.findall(text.replace("\r", "\n"))]


def verdict(rows: list[tuple[float, float, float]]) -> str | None:
    if len(rows) <= WARMUP:
        return None
    loss, epoch, grad = rows[-1]
    if not (math.isfinite(loss) and math.isfinite(grad)):
        return f"non-finite loss {loss} / grad_norm {grad} at epoch {epoch}"
    if grad > GRAD_MAX:
        return f"grad_norm {grad:.3g} > {GRAD_MAX} at epoch {epoch}"
    if len(rows) > WINDOW + SUSTAIN:
        base = statistics.median(r[0] for r in rows[-WINDOW - SUSTAIN:-SUSTAIN])
        if loss > HARD * base:
            return f"loss {loss:.3f} > {HARD} x median {base:.3f} at epoch {epoch}"
        if all(r[0] > SOFT * base for r in rows[-SUSTAIN:]):
            return f"loss > {SOFT} x median {base:.3f} for {SUSTAIN} logs (last {loss:.3f}) at epoch {epoch}"
    return None


def main() -> None:
    log = sys.argv[1]
    if "--simulate" in sys.argv:
        rows = parse(open(log, errors="replace").read())
        for i in range(1, len(rows) + 1):
            v = verdict(rows[:i])
            if v:
                print(f"would kill at logging step {i} (train step ~{i * 10}): {v}")
                return
        print(f"never killed ({len(rows)} logging steps)")
        return
    while True:
        try:
            rows = parse(open(log, errors="replace").read())
        except FileNotFoundError:
            rows = []
        v = verdict(rows)
        if v:
            subprocess.run(["pkill", "-f", "[t]rain.py"])
            with open(log + ".watchdog", "a") as f:
                f.write(f"DIVERGED {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {v}\n")
            print("DIVERGED", v, flush=True)
            return
        if rows and subprocess.run(["pgrep", "-f", "[t]rain.py"], capture_output=True).returncode != 0:
            print("train.py gone, watchdog exiting", flush=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
