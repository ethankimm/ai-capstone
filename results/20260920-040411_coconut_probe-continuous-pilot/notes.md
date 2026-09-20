## 2026-09-20 — Coconut: linear ridge + 2-layer MLP probes on z0..z5, scored with continuous metrics (coconut, run_id: 20260920-040411_coconut_probe-continuous-pilot)

**Goal:** First probe-fitting run for Coconut (previously only decoded/patched, never
regression-probed) — the Coconut counterpart to `probe_codi.py` /
`20260920-032053_codi_probe-pilot` and its continuous-metrics follow-up
`20260920-040622_codi_probe-continuous-pilot`. Tests whether Coconut's raw 768-d live
latent-pass hidden vectors (before the `lm_head` unembedding the paper's own
vocabulary-projection logit lens reads through) linearly or shallowly-nonlinearly encode
the intermediate calculator values, scored with the same continuous-metrics engine
(R², Pearson r, MAE, normalized MAE, tolerance sweep 1/5/10/20%) as the CODI companion
run, so the two mechanisms are compared on identical scoring logic even though they use
different underlying architectures for producing the latent vectors.

**New code this run:** `scripts/probe_coconut.py` (adapted from `probe_codi.py` to
Coconut's `run_passes` API via `scripts/coconut_common.py`) and `scripts/probe_common.py`
(ridge/MLP/signed-log1p primitives factored out so both mechanisms' probe scripts share
identical fitting code, reducing risk of the two implementations drifting apart).
`scripts/probe_metrics_continuous.py` is unchanged from the CODI run — it is
mechanism-agnostic by construction (reads a `raw_predictions.jsonl` of
`(iter, step, idx, y_gold, ridge_pred, mlp_pred)` rows; doesn't care which mechanism
produced them).

**Mechanism / model:** `coconut`, gpt2 / `hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33`
(released checkpoint, Dilgren & Wiegreffe COLM 2026 collection — same one
`decode_patch_coconut.py` uses), compute_steps=6.

**Data:** `gsm_valid-gold-reasoning-trace_test.json` (1194 examples, local-only — the same
file `decode_patch_coconut.py` uses; there is no separate held-out train/validation split
for Coconut the way `gsm8k_aug` gives CODI one). Split by index parity (`idx % 2`,
`decode_patch_coconut.py`'s own `mapping_split` convention) into probe-fit (train_n=500)
and probe-report (eval_n=300) halves, `split_seed=0`. **This is NOT the same split
provenance as CODI's `gsm8k_aug` train/validation split** — probe accuracy between the
two mechanisms is not literally apples-to-apples on data-split source, only on the
metrics/tolerance-sweep logic, which is identical.

**Hyperparams:** ridge_lambda=1.0, mlp_hidden=64 (2-layer, ReLU), mlp_epochs=200,
mlp_lr=1e-3, max_step=3, signed-log1p target transform. Same values as the CODI runs.

**Command:**
```
cd /workspace/ai-capstone && .venv_coconut/bin/python scripts/probe_coconut.py \
    --checkpoint_path <hf cache path for connordilgren/gpt2-gsm8k-coconut checkpoint_33> \
    --data_dir /workspace/coconut_data \
    --slug probe-continuous-pilot --stage pilot --hardware "RunPod RTX A4500 (secure)" \
    --num_latents 6 --train_n 500 --eval_n 300
```
then, entirely locally (no GPU, no model):
```
uv run python scripts/probe_metrics_continuous.py \
    --raw_predictions results/20260920-040411_coconut_probe-continuous-pilot/raw_predictions.jsonl
```
(same pod as `20260920-040622_codi_probe-continuous-pilot` — `p3h64x3usf4jmv`, EU-RO-1,
RTX A4500 secure $0.25/hr — but a **separate venv** (`.venv_coconut`, torch==2.5.1 /
transformers==4.46.2, pinned per `coconut_common.py`'s docstring: newer transformers
returns a `Cache` object rather than legacy `past_key_values` tuples that
`coconut_common.py`'s KV-cache slicing needs). Setup: `uv venv` + pinned pip install
(~1 min), checkpoint `hf_hub_download` (~seconds), a throwaway smoke test at
`--train_n 30 --eval_n 20` (ran clean, deleted, not logged — n<10, a code-path check).
Real run: feature extraction 24s (train, 500 examples) + 14s (eval, 300 examples) — much
faster than CODI's per-example cost since Coconut's `run_passes` does one shared forward
pass per example rather than CODI's separate per-iteration latent loop — then 18 probe
fits, negligible. Combined pod uptime for both mechanisms (env setup + smoke tests + both
real runs) ~18 min ≈ $0.08; terminated immediately after rsyncing results back.)

**Headline results** (`results/20260920-040411_coconut_probe-continuous-pilot/continuous_metrics.json`):

| | ridge avg R² | ridge avg Pearson r | ridge avg norm-MAE | mlp avg R² | mlp avg Pearson r | mlp avg norm-MAE |
|---|---|---|---|---|---|---|
| **non-decodable {1,4}** (z0,z3) | -2.71 | 0.19 | 0.20 | -0.03 | 0.25 | 0.11 |
| **decodable {3,5}** (z2,z4) | -0.12 | 0.15 | 0.14 | 0.00 | 0.14 | 0.11 |

Tolerance-sweep hit rates: hit@0.20 ranges 7.6–12.9% across the four group/predictor
combinations (vs. hit@0.01 of 0.06–0.74%). Per-(iteration,step,predictor) detail:
`predictions.jsonl` / `raw_predictions.jsonl` / `continuous_metrics.json`. Scatter plots:
`scatter_ridge_pred.png` / `scatter_mlp_pred.png`.

**Interpretation:**
- **Same "no recoverable signal" conclusion as CODI**, despite the architecturally
  different way Coconut produces its latent vectors (splicing a live hidden state back
  into one growing sequence vs. CODI's separate per-iteration loop). MLP R² sits near
  zero (-0.03 / +0.005) rather than sharply negative — the least-bad result across both
  mechanisms' runs — but still far from a positive, informative fit, and Pearson r stays
  weak (0.14–0.25) at every group/predictor combination.
- **No non-decodable/decodable separation appears here either.** Unlike CODI, this split
  uses the SAME fixed z0/z3-vs-z2/z4 convention for consistency with the shared analysis
  script, even though Coconut's own decodability profile (per `decode_patch_coconut.py`'s
  data-driven most-/least-decodable pass ranking) is NOT assumed to follow that fixed
  pattern — D&W's own finding is that Coconut+GPT2 encodes gold traces in most passes
  when correct, unlike CODI's odd/even split. Reading this run's group labels as
  "decodable" vs. "non-decodable" in Coconut's own empirical sense would be a
  mislabeling; they are only the plan's fixed z-index convention, kept for cross-
  mechanism comparability on the same downstream script.
- **R² outlier sensitivity is even more visible here than in the CODI run**: iter=2 step=2
  ridge cell has R²=-342 (a handful of large-magnitude gold values the ridge probe badly
  overshoots on, given GSM8K-Aug's heavy tail) while its own Pearson r is 0.036 — i.e.
  "no real linear relationship," not "catastrophically anti-correlated." R² magnitude
  alone should not be read as a severity ranking across cells; Pearson r is steadier.
- **Split-provenance caveat matters for future comparison**: this run's train/eval split
  is index-parity on a different, smaller local file (1194 examples total) than CODI's
  shared `gsm8k_aug` train (huge)/validation (1000, fixed) split. Don't quote CODI vs.
  Coconut probe accuracy against each other as if they used the same held-out data — only
  the metrics computation itself is directly comparable.

**Gotchas hit:**
- None blocking. `coconut_common.py`'s `torch.load(weights_only=False)` FutureWarning
  fires (expected, same as `decode_patch_coconut.py`'s prior runs) — not an error, no
  action needed at this pinned torch version.
- `probe_coconut.py`'s `extract_features` reuses `run_passes`' already-computed
  `live_hidden`/`logits` per pass (no extra forward pass needed for the logit-lens
  comparison), mirroring `probe_codi.py`'s reuse of `decode_patch_codi.py`'s per-iteration
  outputs — kept the extraction cost low (24s/14s vs. CODI's 298s/27s) since Coconut's
  loop shares one KV cache across passes rather than CODI's separate per-iteration calls.

**Caveats:**
- Single seed (`split_seed=0`), no probe-hyperparameter tuning — identical caveat to the
  CODI runs, this is a pilot not a tuned probe.
- Split-provenance mismatch with CODI (above) — flag before comparing probe numbers
  cross-mechanism.
- `max_step=3` caps per-example step count; n shrinks from 300 (step 1) to 217 (step 3).
- This is the first-ever Coconut probe run — no prior binary-tol=0.01 baseline exists to
  compare this continuous-metrics result against the way the CODI companion run does;
  there is nothing to say this "sharpens" for Coconut, only that it's a first measurement.

**Next:**
- If pursuing further: a proper data-driven decodable/non-decodable split for Coconut
  (per `decode_patch_coconut.py`'s own most_decodable_passes/least_decodable_passes
  ranking) rather than reusing CODI's fixed z0/z3-vs-z2/z4 convention, to test probe
  recoverability against Coconut's *own* empirical decodability groups rather than an
  imported one.
- A larger `train_n`/`eval_n` pilot on the full 1194-example gold-trace set (currently
  only ~500/300 of it used) would tighten per-cell Wilson-style uncertainty, though given
  how uniformly poor the signal is here that's unlikely to change the qualitative
  conclusion.
- Companion run `20260920-040622_codi_probe-continuous-pilot` (CODI) reaches the same
  "no recoverable signal, R² unreliable on outliers, Pearson r a steadier read" pattern.
