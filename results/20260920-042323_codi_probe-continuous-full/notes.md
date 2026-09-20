## 2026-09-20 — CODI: continuous probe metrics at full scale (eval_n=1000, the whole validation split) (codi, run_id: 20260920-042323_codi_probe-continuous-full)

**Goal:** Full-scale confirmation of `20260920-040622_codi_probe-continuous-pilot`
(continuous-metrics pilot at eval_n=300). Since probe fitting is cheap (feature
extraction is the only real cost, and it scales linearly with eval_n) there was no
reason to stop at a pilot-sized n once the pilot's floor result looked this clean —
this run uses the *entire* fixed `gsm8k-aug` `validation` split (n=1000) instead of a
300-example subsample, so this is the number to cite, not the pilot's.

**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3`, compute_steps=6,
same paper inference protocol as every other CODI run this week.

**Data:** Probes fit on `gsm8k-aug` `train` (n=3000, unchanged from the pilot — probe
capacity, not train-set size, was never the bottleneck here), reported on the **full**
`validation` split (n=1000, seed=0 — everyone's shared held-out set, in its entirety).

**Hyperparams:** Unchanged from the pilot: ridge_lambda=1.0, mlp_hidden=64, mlp_epochs=200,
mlp_lr=1e-3, max_step=3, signed-log1p target transform.

**Command:**
```
cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/probe_codi.py \
    --ckpt_dir <hf cache path for zen-E/CODI-gpt2@fd641b3> \
    --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \
    --slug probe-continuous-full --stage full_run --hardware "RunPod RTX A4500 (secure)" \
    --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \
    --lora_r 128 --lora_alpha 32 --lora_init --greedy True \
    --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
    --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \
    --output_dir /tmp/unused --train_n 3000 --eval_n 1000
```
then locally: `uv run python scripts/probe_metrics_continuous.py --raw_predictions results/20260920-042323_codi_probe-continuous-full/raw_predictions.jsonl`

(Fresh pod `a2qfxcs5t3yajj`, EU-RO-1, RTX A4500 secure $0.25/hr — same tier as the pilot
run, new pod since the pilot's pod had already been terminated. Env setup (CODI +
Coconut venvs, both checkpoints) took ~3 min; this run's own extraction was 288s (train,
unchanged from pilot) + 90s (eval, up from 27s at eval_n=300 — scales close to linearly
as expected), then 18 probe fits, negligible. Combined pod uptime for both mechanisms'
full runs ~12 min ≈ $0.05; terminated immediately after rsyncing results back.)

**Headline results** (`results/20260920-042323_codi_probe-continuous-full/continuous_metrics.json`):

| | ridge avg R² | ridge avg Pearson r | ridge avg norm-MAE | mlp avg R² | mlp avg Pearson r | mlp avg norm-MAE |
|---|---|---|---|---|---|---|
| **non-decodable {1,4}** (z0,z3) | -0.47 | 0.11 | 0.13 | -0.08 | 0.14 | 0.11 |
| **decodable {3,5}** (z2,z4) | -1.43 | 0.12 | 0.12 | -0.11 | 0.18 | 0.11 |

hit@0.20 (loosest tolerance): 14–16% across group/predictor combinations, vs. hit@0.01
of 0.4–0.7%. Full per-cell table: `continuous_metrics.json`. Scatter plots:
`scatter_ridge_pred.png` / `scatter_mlp_pred.png`.

**Interpretation:**
- **Matches the pilot's conclusion, tighter.** Same qualitative floor: negative average R²
  in 3 of 4 group/predictor cells, Pearson r uniformly weak (0.11–0.18), no
  non-decodable/decodable separation. The full run's per-cell numbers are noticeably
  *less extreme* than the pilot's (worst ridge cell here is R²=-4.4 at iter=5 step=1,
  n=998, vs. the pilot's R²=-22.8 at n=147) — larger eval n damps the outlier-driven R²
  blowups the pilot's smaller per-step samples were prone to (see pilot notes' caveat on
  R² and heavy-tailed GSM8K values). This is the expected effect of more data, not a
  different underlying finding: Pearson r, the steadier metric, barely moves between
  pilot and full (pilot 0.13–0.38 vs. full 0.11–0.18 — both "weak, no real linear
  relationship").
- Normalized MAE also tightens slightly (pilot ~0.16–0.22 vs. full ~0.11–0.13) for the
  same reason — fewer per-cell outliers pulling the average up.
- **This is now the number to cite for CODI**, not the pilot — full validation split,
  same conclusion, better-behaved tail.

**Gotchas hit:** None new; same pinned CODI venv/checkpoint/extraction path as the pilot.

**Caveats:** Same as the pilot (single seed, no probe-hyperparameter tuning,
`max_step=3` step-3 n still smallest at 454). R² remains an outlier-sensitive metric on
this dataset even at full scale (iter=5 cells still show R²≈-4); Pearson r is the more
reliable read throughout.

**Next:** This closes out the continuous-metrics probe question for CODI at the current
hyperparameters. Per `next_experiments.md` #1, donor-interchange patching with
content-bearing donors at z0/z3 is the next step toward the *causal*-faithfulness
question this probe/decoding work can't answer on its own.
