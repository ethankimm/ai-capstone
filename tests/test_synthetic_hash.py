"""sha256 assert for the registered synthetic v2 data files + the timing parser (no GPU/model deps)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import synthetic_common as sc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_registry_covers_depth2_and_modal():
    reg = json.loads((ROOT / "configs/synthetic/v2_data_sha256.json").read_text())["files"]
    for task in ("depth2", "modal"):
        for f in ("train", "valid", "valid_monitor", "test"):
            key = f"{task}/{f}.json"
            assert key in reg and len(reg[key]) == 64


def test_unregistered_file_not_checked(tmp_path):
    d = tmp_path / "depth"  # v1 dir name, not in the registry
    d.mkdir()
    (d / "test.json").write_text("[]")
    assert sc.assert_known_hash(d / "test.json") is None


def test_mismatch_raises(tmp_path):
    d = tmp_path / "depth2"
    d.mkdir()
    (d / "test.json").write_text('[{"question": "tampered"}]')
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        sc.assert_known_hash(d / "test.json")


def test_real_files_match_if_present():
    data = Path.home() / "Documents/Penn/CIS5980/synthetic_data_v2"
    if not (data / "depth2/test.json").exists():
        pytest.skip("v2 data not on this machine")
    for task in ("depth2", "modal"):
        for f in ("test", "valid", "valid_monitor"):
            assert sc.assert_known_hash(data / task / f"{f}.json")


def test_coconut_timing_projection(tmp_path):
    lines = []
    for e, sec in zip((2, 3, 4, 5), ("00:05:00", "00:09:00", "00:12:00", "00:12:00")):
        lines.append(f"Training Epoch: {e}/5, batch 40/40 completed (loss: 1.2:  100%|#########| 40/40 [{sec}<00:00, 7.9s/it]")
    log = tmp_path / "t.log"
    log.write_text("\n".join(lines))
    def run(m):
        return subprocess.run([sys.executable, str(ROOT / "scripts/coconut_timing.py"), "--log", str(log), "--rate", "0.27",
                               "--max-latent-stage", str(m)], capture_output=True, text=True, check=True).stdout
    # M = 3: stages 1,2,3 once + final stage (printed epoch 5) three times: (300 + 540 + 720 + 3*720)/3600 * 20 = 20.67 h
    out = run(3)
    assert "PROJECTED (3 curriculum stages" in out and "20.7 h" in out and "WARNING" not in out
    # M = 6 (c1m6): the same log stops at stage 4, so stages 5, 6 and the final stage are unmeasured -> loud warning
    out = run(6)
    assert "WARNING: stages [5, 6, 'final'] were not measured" in out
