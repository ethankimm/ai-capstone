## 2026-09-20 — CODI: continuous probe metrics (R², Pearson r, MAE, tolerance sweep) vs the binary-threshold floor (codi, run_id: 20260920-040622_codi_probe-continuous-pilot)

**Goal:** Follow-up to `20260920-032053_codi_probe-pilot` (ridge/MLP probes on z0..z5 both
at floor under a hard `tol=0.01` relative-error hit rate, alongside logit-lens accuracy on
the same examples). That pilot's own "Next" note flagged the risk: a binary 1% threshold
collapses "no recoverable signal" and "signal exists but not to 1% precision" into the
same zero. This run reruns the identical probe fit (unchanged hyperparams) with a raw
per-example prediction cache added, then scores it with a continuous-metrics engine
(R², Pearson r, MAE, normalized MAE, and a relaxed tolerance sweep at 1/5/10/20%) instead
of just the single hard threshold, to see whether a softer readout finds signal the binary
metric was hiding.

**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3` (released checkpoint,
same as every other CODI run this week), compute_steps=6, paper inference protocol
(LoRA r=128/α=32, projection 768+LN, greedy).

**Data:** Probes fit on `gsm8k-aug` `train` (n=3000, `train_seed=0`), reported on the
fixed `validation` split (n=300, seed=0) — identical split to `20260920-032053_codi_probe-pilot`.

**Hyperparams:** Unchanged from the prior pilot: ridge_lambda=1.0, mlp_hidden=64 (2-layer,
ReLU), mlp_epochs=200, mlp_lr=1e-3, max_step=3, signed-log1p target transform (inverted to
real scale before scoring — both the `tol` hit-rate and the new continuous metrics operate
on real-scale values). `tol=0.01` is still recorded in the manifest for compatibility but
is not the metric this run is about.

**Command:**
```
cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/probe_codi.py \
    --ckpt_dir <hf cache path for zen-E/CODI-gpt2@fd641b3> \
    --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \
    --slug probe-continuous-pilot --stage pilot --hardware "RunPod RTX A4500 (secure)" \
    --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \
    --lora_r 128 --lora_alpha 32 --lora_init --greedy True \
    --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
    --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \
    --output_dir /tmp/unused --train_n 3000 --eval_n 300
```
then, entirely locally (no GPU, no model):
```
uv run python scripts/probe_metrics_continuous.py \
    --raw_predictions results/20260920-040622_codi_probe-continuous-pilot/raw_predictions.jsonl
```
(pod `p3h64x3usf4jmv`, EU-RO-1, RTX A4500 secure $0.25/hr — same GPU tier the prior CODI
pilot used, chosen again since A5000/A6000 showed zero secure stock at provisioning time.
Same pod also set up a second venv and ran the Coconut counterpart of this run,
`20260920-040411_coconut_probe-continuous-pilot` — see that run's notes for the split.
Setup: `codi_setup.sh` clone+patch+pinned venv build, checkpoint `snapshot_download`, a
throwaway smoke test at `--train_n 40 --eval_n 20` to catch errors before the real run
(ran clean, deleted, not logged — n<10, a code-path check). Real run: feature extraction
298s (train, 3000 examples) + 27s (eval, 300 examples), then 18 probe fits, negligible.
Pod total uptime ~18 min covering both mechanisms' env setup + smoke tests + real runs
≈ $0.08 combined; terminated immediately after rsyncing both runs' results back.)

**Headline results** (`results/20260920-040622_codi_probe-continuous-pilot/continuous_metrics.json`,
averaged over the (iteration, step) cells in each group):

| | ridge avg R² | ridge avg Pearson r | ridge avg norm-MAE | mlp avg R² | mlp avg Pearson r | mlp avg norm-MAE |
|---|---|---|---|---|---|---|
| **non-decodable {1,4}** (z0,z3) | -1.09 | 0.13 | 0.22 | -0.23 | 0.27 | 0.16 |
| **decodable {3,5}** (z2,z4) | -1.32 | 0.26 | 0.18 | -2.34 | 0.38 | 0.21 |

Tolerance-sweep hit rates stay low across the board even at the loosest tolerance:
hit@0.20 ranges 12–16% across the four group/predictor combinations (vs. hit@0.01 of
0.5–0.7%) — full per-cell table and hit-rates at all four tolerances in
`continuous_metrics.json`. Per-(iteration,step,predictor) detail: `predictions.jsonl` /
`raw_predictions.jsonl` / `continuous_metrics.json`. Scatter plots (predicted vs. gold,
all 6 focus (iter, step) panels) at `scatter_ridge_pred.png` / `scatter_mlp_pred.png`.

**Interpretation:**
- **The continuous metrics do not surface signal the binary threshold was hiding.**
  Average R² is negative in 3 of the 4 group/predictor cells (ridge at both groups, MLP
  at "decodable") — meaning the probe's predictions are, on average, *worse* than always
  predicting the training-set mean value. MLP at the "non-decodable" group is the only
  cell with R² clearly toward zero rather than sharply negative (-0.23), still far from
  a positive, informative fit. This is the sharper version of the prior pilot's floor
  finding: it isn't "signal below 1% precision," it's closer to "no linearly/shallowly
  recoverable signal at all" at this train_n/hyperparameter setting.
- **No clean non-decodable/decodable separation appears on any of these metrics.** Both
  groups sit at comparably poor R²/Pearson r/hit-rates; the "decodable" group (z2,z4) is
  not obviously better-probed than the "non-decodable" group (z0,z3), consistent with the
  earlier pilot's own caution that these groups are the *ablation*-derived split, not the
  logit-lens-derived one (iterations 2/4/6 are where the logit lens itself actually
  decodes; z3=iter4 sits in the "non-decodable" bucket here but is one of the logit-lens
  "live" positions — see the prior pilot's notes for the full labeling caveat).
- **R² is not a robust metric on GSM8K-Aug's heavy-tailed target values** — several cells
  (e.g. iter=1 step=3 ridge: R²=-2.24; iter=2 step=3 ridge: R²=-22.8; iter=5 step=1/2
  ridge: R²≈-4.3/-4.6) show wildly negative R² driven by a handful of large-magnitude
  gold values the probe badly mispredicts, since R² weights squared error. Pearson r on
  those same cells is a steadier read (0.1–0.3, i.e. "weak or no linear relationship,"
  rather than R²'s "catastrophically worse than the mean"). Read Pearson r as the primary
  continuous signal here and R² as a secondary, outlier-sensitive check — both still say
  "no useful recoverable signal," but R²'s magnitude alone overstates how bad specific
  cells are relative to the rest.
- **Normalized MAE (MAE / std(gold)) is the most interpretable scale-free number**: ~0.16
  to ~0.22 across groups, i.e. typical absolute error is comparable to (16–22% smaller
  than) the entire spread of gold values at that (iteration, step) — not a small residual
  relative to the problem's own scale.

**Gotchas hit:**
- None new on the extraction side (same pinned CODI venv/checkpoint as prior runs).
- Building the raw per-example cache required adding a `raw` key to `probe_one`'s return
  dict and writing it to a *separate* `raw_predictions.jsonl` (not `predictions.jsonl`,
  which stays aggregate-only per the existing schema) — the continuous-metrics script
  reads that file, not the manifest's `predictions.jsonl`.

**Caveats:**
- Single seed, no probe-hyperparameter tuning (identical caveat to the prior pilot —
  this run only changes the *scoring*, not the fit).
- R²'s outlier sensitivity (above) means the averaged headline table understates how
  uniformly poor most cells are and overstates a few outlier cells' badness; always check
  Pearson r alongside R² for this dataset's value distribution.
- `max_step=3` still caps per-example step count; step-3 numbers are the noisiest
  (n=147 vs. n=299 at step 1).

**Next:**
- The basis-drift hypothesis is now doubly unsupported (binary floor + continuous
  floor); the natural next step per `next_experiments.md` #1 is donor-interchange
  patching with content-bearing donors at z0/z3, which tests causal faithfulness
  directly rather than through a readout probe.
- If probe accuracy is still worth pursuing: a probe combining multiple iterations/steps
  jointly (rather than one (iteration, step) cell at a time) might find signal a
  per-cell-isolated probe can't — out of scope here.
- Companion run `20260920-040411_coconut_probe-continuous-pilot` runs the identical
  continuous-metrics engine on Coconut and finds the same "no recoverable signal" result.
