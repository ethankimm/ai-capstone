# CODI null audit (2026-11-04)

Scripts behind run `*_codi_v2-l1to7w-codi-null-audit` and `CODI_NULL_AUDIT.md`. `a*` read logged predictions and train logs,
no model. `b*` load checkpoints: run them with the CODI venv, passing this folder (it holds `harness.py`, `rc.py` and takes
the JSON output) and, where asked, a device:

    ../codi/.venv/bin/python scripts/codi_null_audit/b3_course.py scripts/codi_null_audit mps

`a*` scripts expect baseline predictions from `origin/kevin/lrm-training-1`, `-2` and `kevin/sim-cot` under `<dir>/preds/`.

Memory: on a 24 GB Mac run one job at a time. With MPS set both `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6` and
`PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5`, and use batches of 32 or fewer. Calling the LM-head wrapper on question tokens builds
[B, T, 50k] logits, which took swap to 49 GB here, so `harness.py` uses the transformer body.

`a11_sim.py` is a noisy-tracker simulator that FAILED its held-out checks; it is kept because the report cites that failure.
