# Necessity check harness (2026-11-08)

General route split: per latent site j, `both` (donor z_j fed in), `input` (donor z_j fed in, pass j's keys/values reset to the
recipient's own: only the recurrent input carries it), `kv` (own z_j, donor's keys/values at pass j), `kv_noans` (as kv, answer masked
from pass j), plus `both_all` / `kv_all`. See HANDOFF_2026-11-08_NECESSITY.md for the framing.

- `core.py`: model-agnostic loop (`run`, `route_split`, `summarise`) over an adapter.
- `adapters.py`: RuleChainCODI (our checkpoints, t15's loop), GSM8KCODI (zen-E, t12's loop), GridCODI / GridCoconut (Dilgren & Wiegreffe
  grid, ProsQA / ProntoQA / GSM8K; checkpoints in `~/Documents/Penn/CIS5980/grid_checkpoints`, grid repo at `~/Projects/are-lrms-easily-interpretable`).
- `pairs.py`: minimal pairs for ProsQA / ProntoQA (rewire one relation on the gold path so the answer flips, equal token length; the
  built-in reasoners reproduce gold on all 500 / 800 test items).
- `regress.py`: reproduces t15 (gated L3, all 26 conditions, 286 items) and t12 (first 40 pairs) item for item. Passed 2026-11-08.

Run with the CODI venv through `codi_audit_artifacts/run_guarded.sh` (one job at a time, `caffeinate -dims`):

    caffeinate -dims sh run_guarded.sh <log> $REPO/scripts/necessity/regress.py rulechain mps gated 3
