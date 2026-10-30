#!/usr/bin/env python3
"""Verify the registered synthetic v2 data files (configs/synthetic/v2_data_sha256.json) under a data root.
  python scripts/check_data_hashes.py /workspace/synthetic_data_v2 [depth2 modal]
Exit 0 only if every registered file for the requested tasks exists and matches. Run at the start of every pod job."""
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
tasks = set(sys.argv[2:])
reg = json.loads((Path(__file__).resolve().parents[1] / "configs/synthetic/v2_data_sha256.json").read_text())["files"]
bad = 0
for key, want in sorted(reg.items()):
    if tasks and key.split("/")[0] not in tasks:
        continue
    f = root / key
    got = hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else "MISSING"
    ok = got == want
    bad += not ok
    print(("ok  " if ok else "BAD ") + key + " " + got[:12])
sys.exit(1 if bad else 0)
