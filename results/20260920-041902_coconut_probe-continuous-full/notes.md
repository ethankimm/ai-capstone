## 2026-09-20 — Coconut: continuous probe metrics at full scale (entire local gold-trace file) (coconut, run_id: 20260920-041902_coconut_probe-continuous-full)

**Goal:** Full-scale confirmation of `20260920-040411_coconut_probe-continuous-pilot`
(continuous-metrics pilot at train_n=500/eval_n=300). Uses the **entire**
`gsm_valid-gold-reasoning-trace_test.json` file (1194 examples, index-parity split ->
597/597) instead of a subsample of it — this is all the local gold-trace data available
for Coconut, so this run is the ceiling on n for this data source, not a further-scalable
"full" the way CODI's shared `gsm8k_aug` validation split is.

**Mechanism / model:** `coconut`, gpt2 / `hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33`,
compute_steps=6.

**Data:** `gsm_valid-gold-reasoning-trace_test.json`, index-parity split (`idx % 2`,
`split_seed=0`) into train_n=597 / eval_n=597 — both full halves of the file (the pilot
used 500/300, a subsample of the same halves). Same split-provenance caveat as the pilot
applies: this is not the same split source as CODI's `gsm8k_aug` train/validation split.

**Hyperparams:** Unchanged from the pilot: ridge_lambda=1.0, mlp_hidden=64, mlp_epochs=200,
mlp_lr=1e-3, max_step=3, signed-log1p target transform.

**Command:**
```
cd /workspace/ai-capstone && .venv_coconut/bin/python scripts/probe_coconut.py \
    --checkpoint_path <hf cache path for connordilgren/gpt2-gsm8k-coconut checkpoint_33> \
    --data_dir /workspace/coconut_data \
    --slug probe-continuous-full --stage full_run --hardware "RunPod RTX A4500 (secure)" \
    --num_latents 6 --train_n 597 --eval_n 597
```
then locally: `uv run python scripts/probe_metrics_continuous.py --raw_predictions results/20260920-041902_coconut_probe-continuous-full/raw_predictions.jsonl`

(Same fresh pod as the CODI full run (`a2qfxcs5t3yajj`), run in parallel in the separate
pinned `.venv_coconut` (torch==2.5.1/transformers==4.46.2). Extraction: ~29s (train,
597 examples) + ~28s (eval, 597 examples) — scaled from the pilot's 24s/14s at 500/300,
consistent with Coconut's cheaper single-shared-forward-pass extraction vs. CODI's
per-iteration loop. Combined pod uptime for both mechanisms' full runs ~12 min ≈ $0.05;
terminated immediately after rsyncing results back.)

**Headline results** (`results/20260920-041902_coconut_probe-continuous-full/continuous_metrics.json`):

| | ridge avg R² | ridge avg Pearson r | ridge avg norm-MAE | mlp avg R² | mlp avg Pearson r | mlp avg norm-MAE |
|---|---|---|---|---|---|---|
| **non-decodable {1,4}** (z0,z3) | -63.8 | 0.19 | 0.50 | -0.20 | 0.21 | 0.11 |
| **decodable {3,5}** (z2,z4) | -246,833 | 0.05 | 11.99 | +0.04 | 0.20 | 0.09 |

hit@0.20: 6–12% across group/predictor combinations. Full per-cell table:
`continuous_metrics.json`. Scatter plots: `scatter_ridge_pred.png` / `scatter_mlp_pred.png`.

**Interpretation:**
- **MLP tells the same "floor" story as the pilot and as CODI**: R² near zero (-0.20 to
  +0.04, the only positive-average-R² cell across every run this session), Pearson r
  0.20–0.21, hit@0.20 ~9–12% — weak-to-no signal, consistent across both group labels
  and both scales (pilot and full).
- **Ridge's averaged R² is essentially uninformative at this n and must not be quoted
  as a headline number.** The "decodable" group average of -246,833 is driven by a
  single catastrophic cell: iter=5 step=1 ridge has R²=-1,479,884, MAE=769,344 (normalized
  MAE=69.2 — the prediction error is on average **69x the target's own standard
  deviation**), while its own Pearson r is 0.000 and n=597 is not small. This is a
  closed-form ridge regression on 768 standardized features with `ridge_lambda=1.0`
  fitting on 597 training rows — under-regularized relative to the feature dimension for
  this particular (iteration, step) cell's train/eval distribution shift, producing a
  handful of eval-set predictions with an enormous magnitude that dominate the squared-
  error sum. This is a genuine, reproducible instability in this specific ridge fit
  (re-running `probe_metrics_continuous.py` against the same cached `raw_predictions.jsonl`
  reproduces it exactly, since scoring is deterministic on cached predictions), not a
  metrics-engine bug — but it means **R² is not a trustworthy summary statistic for
  ridge on this data at this regularization strength**, full stop. Pearson r and
  normalized MAE (both bounded, more robust to a handful of outlier predictions) are the
  metrics to read for ridge; R² should be treated as diagnostic-only per cell, never
  averaged into a headline number, for this predictor.
- **This sharpens (rather than just repeats) the pilot's R²-fragility caveat**: the pilot's
  worst ridge cell was R²=-342 (also iter=2 step=2, also flagged as an outlier-driven
  blowup); at full scale a *different* cell (iter=5 step=1) blows up far more severely.
  The instability isn't tied to one specific cell or fixed by more data — it's a property
  of closed-form ridge with `lambda=1.0` on 768-d standardized features at this n, and
  should inform any future probe work: either increase `ridge_lambda` materially, or
  report Pearson r / hit-rate as primary and drop averaged R² for ridge specifically.
- **Bottom line unchanged from the pilot and from CODI**: no recoverable intermediate-value
  signal in Coconut's raw latent-pass vectors at any of the 4 focus positions, by the
  metric that's actually reliable here (Pearson r, consistently 0.05–0.21, i.e. "no
  meaningful linear relationship").

**Gotchas hit:**
- The ridge R² blowup above (iter=5 step=1) — worth a "gotcha" entry specifically because
  it changes how this run's headline table should be read (see Interpretation). No
  extraction-side gotchas; same pinned Coconut venv/checkpoint as the pilot.

**Caveats:**
- Ridge-averaged R² for the "decodable" group is not meaningful as reported in the table
  above (driven by one catastrophic cell) — read Pearson r / normalized MAE instead, or
  the per-cell table in `continuous_metrics.json` with that one cell excluded.
- Split-provenance mismatch with CODI (same as pilot) — this run's train/eval split is
  index-parity on a separate, smaller local file, not `gsm8k_aug`.
- This is the ceiling on available local gold-trace data (1194 examples total); a larger
  n would require regenerating the data via
  `are-lrms-easily-interpretable/preprocessing/prepare_gsm8k.py`, out of scope here.

**Next:**
- If ridge probes are revisited for either mechanism, increase `ridge_lambda` (currently
  1.0, clearly insufficient at this feature dimension/n) and treat averaged R² across
  cells as unreliable regardless of that fix — report per-cell R² only as a diagnostic
  alongside Pearson r, never averaged as a headline number for ridge.
- Companion run `20260920-042323_codi_probe-continuous-full` (CODI full) reaches the
  same MLP-is-the-more-trustworthy-predictor, Pearson-r-is-the-steadier-metric
  conclusion, without as severe a ridge blowup (CODI's worst full-run ridge cell is
  R²=-4.4, not in the same universe as Coconut's -1.5M) — worth a follow-up question
  (not investigated here) on whether Coconut's live hidden vectors have higher
  effective dimensionality / less train-eval distribution overlap than CODI's, making
  ridge specifically less stable there.
