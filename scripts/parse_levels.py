#!/usr/bin/env python3
"""stdin: output of eval_synthetic_coconut.py; prints 'mean_all <v> mean_L1-5 <v> min_level <v> <per-level json>' (used by coconut_v2.sh;
`select` ranks checkpoints by field 2 = mean over all levels, TRAINING_PROTOCOL.md)."""
import json
import re
import sys

t = sys.stdin.read()
m = re.search(r"per-level: (\{.*\})", t)
if not m:
    print("NO per-level line:\n" + t[-800:])
    sys.exit(2)
pl = {int(k): v for k, v in json.loads(m.group(1)).items()}
w = [pl[k] for k in range(1, 6) if k in pl]
print("mean_all %.4f" % (sum(pl.values()) / len(pl)), "mean_L1-5 %.4f" % (sum(w) / len(w)), "min_level %.4f" % min(pl.values()), json.dumps(pl))
