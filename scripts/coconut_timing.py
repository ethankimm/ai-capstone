#!/usr/bin/env python3
"""Project full-run time and cost of a Coconut curriculum run from a `debug: True` timing run (run.py with 5k train
examples per epoch).

  python scripts/coconut_timing.py --log /workspace/logs/<scratch>.log [--mem <scratch>.mem] --train-n 100000 \
      --debug-n 5000 --rate 0.27 --max-latent-stage 6 --final-epochs 3

Every stage is timed, none extrapolated: stage cost grows with the number of latents (each latent is one more sequential
forward pass), so a projection from early stages under-counts (the v1 cut was based on stage-1 s/step and was wrong; the
first c1m6 version of this script counted stage 4 as final and under-quoted by ~2x). With `resume 1` and epochs_per_stage 1,
printed epoch e (1-based, as run.py prints it) trains stage e-1; stages 1..M are the curriculum, stage > M is the final
all-latent stage. coconut_v2.sh `timing` runs num_epochs 8 for M = 6, i.e. stages 1-6 plus one final-stage epoch.
Each measured debug epoch is scaled by train_n / debug_n; the final stage is counted --final-epochs times; the CoT stage 0
is not part of the run (resume 1). Per-epoch validation time (run.py decodes the monitor set every epoch) is NOT in these
numbers: add the logged validation time, or budget ~10% on top. Prints hours and dollars at --rate $/h.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_common import parse_coconut_log  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True)
ap.add_argument("--mem", default=None)
ap.add_argument("--train-n", type=int, default=100000)
ap.add_argument("--debug-n", type=int, default=5000)
ap.add_argument("--rate", type=float, default=0.27, help="$/h")
ap.add_argument("--max-latent-stage", type=int, default=6, help="M: stages 1..M are the curriculum, > M is final")
ap.add_argument("--final-epochs", type=int, default=3)
a = ap.parse_args()

text = Path(a.log).read_text(errors="replace")
secs = parse_coconut_log(text)["seconds_train_epoch"]  # {epoch (1-based, as printed): seconds}
if not secs:
    sys.exit("no completed training epoch found in the log (did the timing run crash? tail the log)")
scale = a.train_n / a.debug_n
M = a.max_latent_stage
print("printed epoch -> stage -> debug seconds -> projected full-epoch hours")
per_stage = {}
for e in sorted(secs):
    stage = e - 1
    h = secs[e] * scale / 3600
    label = "final" if stage > M else str(stage)
    print(f"  {e}: stage {label:>5}: {secs[e]:>5}s -> {h:5.2f} h")
    per_stage["final" if stage > M else stage] = h

missing = [s for s in range(1, M + 1) if s not in per_stage] + ([] if "final" in per_stage else ["final"])
total = sum(per_stage[s] for s in range(1, M + 1) if s in per_stage) + a.final_epochs * per_stage.get("final", 0.0)
print(f"PROJECTED ({M} curriculum stages + {a.final_epochs} final-stage epochs, training only): "
      f"{total:.1f} h = ${total * a.rate:.2f} at ${a.rate}/h")
if missing:
    print(f"WARNING: stages {missing} were not measured -- the projection is INCOMPLETE and too low; "
          f"rerun the timing job with num_epochs = M + 2 = {M + 2}")
if a.mem and Path(a.mem).exists():
    vals = [int(x) for x in Path(a.mem).read_text().split() if x.strip().isdigit()]
    if vals:
        print(f"peak GPU memory during the timing run: {max(vals)} MiB")
m = re.search(r"_IncompatibleKeys\([^\n]*\)", text)
print("load_state_dict:", m.group(0)[:300] if m else "(no _IncompatibleKeys line found in the log)")
