# RuleChain round 1: results summary

Runs from `TRAINING_PROTOCOL.md` (RuleChain = `depth2`, GPT-2 small, generator 2.0.0, seed 5980), trained
2026-10-31 to 2026-11-01 (PT) on cci-p124 (H100 NVL), GPUs 7 and 4. Every number below is from the single logged test
eval (500 items per level, 6,000 total) of the checkpoint picked on `valid.json`; the run records in `results/` hold the
details. One seed per model, so these support existence claims only (protocol section 1).

## Headline

**Neither latent method beats its matched no-latent reference on RuleChain.**

- **Coconut** (6 latents, 1 per rule) is at or below `no_thoughts` at every level, clearly below at k = 2.
- **CODI** (6 latents) matches **CODI-0** (same training, 0 latents) at every level. CODI's large lead over plain no-CoT
  comes from its training signal (teacher CE + distillation into the answer position), not from the latent passes.
- With that signal, a **single forward pass with no latents reaches about 5 rules** (CODI-0 d50 4.90), against 2.35 for
  plain no-CoT. The envelope's single-pass reach is therefore about 4.9, not 2-3 (valid envelope d50 4.95 [4.75, 5.13]).
- Paired tests refine this: Coconut is significantly *below* `no_thoughts` at L2-3, and CODI has a small significant gain
  at L3, below d50. Both are single-seed; seed variance at L2 is of the same size (see below).

## Results

d50 = number of rules where accuracy crosses 50% (logistic midpoint).

| # | Run | Test acc | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8-12 | d50 | Record |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CoT (Coconut stack) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ≥ 0.998 | — | `20261101-154307_…cot-r2` |
| 2 | Coconut c1m6, 6 latents | 0.142 | 0.982 | 0.278 | 0.116 | 0.062 | 0.050 | 0.028 | 0.046 | ≤ 0.034 | 1.77 | `20261102-022150_…coconut_c1m6` |
| 3 | `no_thoughts` (ref. for 2) | 0.164 | 0.994 | 0.414 | 0.164 | 0.082 | 0.066 | 0.044 | 0.042 | ≤ 0.038 | 1.88 | `20261101-153922_…no_thoughts_m6` |
| 4 | no-CoT to plateau (= data-matched `checkpoint_12`) | 0.172 | 0.760 | 0.482 | 0.240 | 0.160 | 0.100 | 0.076 | 0.032 | ≤ 0.048 | 2.35 | `20261101-154024_…no_cot_plateau` |
| 5 | CODI, 6 latents (64×2 rerun) | 0.401 | 1.000 | 1.000 | 0.992 | 0.758 | 0.434 | 0.228 | 0.112 | ≤ 0.090 | 4.79 | `20261102-000204_…codi_bs64-sel` |
| 6 | CODI-0, 0 latents (ref. for 5) | 0.405 | 1.000 | 0.988 | 0.960 | 0.764 | 0.494 | 0.240 | 0.120 | ≤ 0.082 | 4.90 | `20261102-060537_…codi0` |
| 7 | Student-only (CODI health check) | — | diverged at epoch 7.3 twice; dropped | | | | | | | | — | `20261102-054552_…codi-student-diverged` |

### Gains vs the matched reference (test)

| Level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Coconut − `no_thoughts` | −0.012 | **−0.136** | −0.048 | −0.020 | −0.016 | −0.016 | +0.004 | −0.002 | −0.008 | +0.002 | +0.006 | −0.008 |
| CODI − CODI-0 | 0 | +0.012 | +0.032 | −0.006 | −0.060 | −0.012 | −0.008 | +0.008 | +0.002 | −0.010 | +0.002 | −0.010 |

**Paired tests** (`scripts/rulechain_gain_analysis.py`, full tables in `RULECHAIN_GAIN_ANALYSIS.md`; exact McNemar,
Newcombe CIs, Holm over 12 levels, 2,000 item bootstraps):

- **Coconut − `no_thoughts`:** L2 −0.136 [−0.182, −0.089], Holm p 3e-7; pooled L1-3 −0.065 (p 7e-10); L7-12 −0.001.
  d50 difference −0.11 [−0.17, −0.06]. The latents cost accuracy at L2-3 rather than giving nothing.
- **CODI − CODI-0:** L2 +0.012 (6 vs 0 discordant), **L3 +0.032 (17 vs 1), Holm p 0.002**, normalized gain 0.80 at L3;
  L5 −0.060 does not survive Holm (0.37). Overall −0.004 [−0.014, +0.005]; d50 difference −0.12 [−0.24, +0.01]. The only
  gain sits below single-pass reach, the reverse of the protocol's onset prediction.
- **These intervals cover item sampling only.** `no_thoughts` seed 1 (round 2, `20261102-190130_…no_thoughts_m6_s1`)
  differs from seed 0 at L2 by −0.084 [−0.131, −0.036] (p 0.0008) with an identical recipe, so at L2 seed variance is of
  the same order as the effects. Neither effect is a finding until the round-2 seeds replicate it.

### Envelope preview (best no-latent model per level, test numbers; the protocol's envelope uses valid)

L1 1.000 (CODI-0) · L2 0.988 (CODI-0) · L3 0.960 (CODI-0) · L4 0.764 (CODI-0) · L5 0.494 (CODI-0) · L6 0.240 (CODI-0) ·
L7 0.120 (CODI-0). CODI-0 sets the envelope at every level from L2 up; its d50 (4.90) is the envelope's. The formal
envelope and d50 bootstrap (protocol section 3) are not computed yet.

## What the protocol's checks say

- **G1 (CoT ≥ 0.97 at every level on valid):** passed, 1.000 everywhere.
- **Coconut's expected fall (own d50 in about [8, 10]):** missed, d50 1.77. Per protocol section 3, the latents are not
  working one per rule, so per-latent probing for step s_j is not justified on this checkpoint.
- **Under-trained flags:** Coconut and `no_thoughts` (valid still rising over the 3 final-stage epochs). CODI's valid mean
  also rose slightly at the end (0.392 → 0.395); the rule is written for Coconut, so it is noted, not flagged.
- **Reference health check:** student-only (run 7) diverged, so CODI-0 was not checked against it. CODI-0 *is* the
  envelope at every level. `no_thoughts` fails the envelope check badly (−0.55 to −0.80 at L2-4), so Coconut is also
  reported against the envelope (L2 −0.71, L3 −0.84, d50 −3.1); that gap mixes training stacks.

### Coconut monitor curve by stage (360 items, each stage in its own format)

| | St. 1 | St. 2 | St. 3 | St. 4 | St. 5 | St. 6 | Final 1 | Final 2 | Final 3 |
|---|---|---|---|---|---|---|---|---|---|
| Coconut | 0.978 | 0.294 | 0.164 | 0.161 | 0.133 | 0.131 | 0.122 | 0.158 | 0.164 |
| `no_thoughts` | 1.000 | 0.297 | 0.194 | 0.164 | 0.147 | 0.142 | 0.167 | 0.158 | 0.142 |

Coconut tracked its no-latent reference stage by stage: removing written steps, not adding latents, set the result.

## Deviations from the protocol

| Change | Why | Effect |
|---|---|---|
| Ran on cci-p124 (H100 NVL), not RunPod | Hardware available | Paths only (`coconut_v2.sh` env overrides, `render_cfg` path rewrite, free-port fallback) |
| Coconut stack micro-batch 32 × 4 | Allowed on large cards (`MICRO_BS=32`) | Same effective batch 128 |
| CODI rerun named `gpt2_depth2_codi_bs64`, every epoch kept (`--save_total_limit 40`) | First run diverged with no pre-divergence checkpoint left | Selection rule unchanged (best of last 3 on valid) |
| `coconut_watch.py --min-epochs 12` | Plateau rule could stop run 4 before `checkpoint_12` existed | Run 4 stopped at exactly epoch 12 |
| `scripts/codi_zero_loss_log.patch` | Upstream CODI crashes at the first checkpoint when a loss weight is 0 | Logging only; loss unchanged |
| No-CoT (run 4) resumed from `checkpoint_1` | Crashed after a SIGSTOP pause (NCCL watchdog timeout) | Lost ~22% of epoch 2; optimizer state reset |
| CODI (run 5) test set evaluated twice | Path-sort bug first picked the epoch-12 checkpoint (`20261101-235727`, kept) | Selection used `valid.json` only; no leakage |
| Student-only (run 7) dropped | Diverged at epoch 7.3; resume from epoch 7 diverged again | User decision; protocol lists it as first to drop |
| GPU sharing in the first ~16 h | Throughput | Speed only, no effect on results |

## Training stability (CODI recipe, lr 3e-3, LoRA r128)

| Run | Batch | Outcome |
|---|---|---|
| CODI (first attempt) | 32 × 4 | Diverged at epoch 6.1 (NaN from 6.69); `20261101-154337_…codi-32x4-diverged` |
| CODI | 64 × 2 | Completed; transient teacher-loss spikes at epochs 1.5 and 9.2 recovered |
| CODI-0 | 64 × 2 | Completed cleanly |
| Student-only | 64 × 2 | Diverged at epoch 7.3; resume diverged again at 7.31 |

2 of 4 runs diverged. Student-only diverged at the protocol's exact batch, so the micro-batch is not the cause; the learning
rate is on the edge for this task. The only batch-dependent CODI term (`ref_selected.std()`, `src/model.py:387`) is nearly
the same at 32 and 64.

## Open decisions for the team

1. ~~**Paired gain analysis**~~ done 2026-11-02 (above); needs a seed-level version once round 2 is in.
2. **Coconut on a longer schedule** (2-3 epochs per stage, more final epochs), so the null is not just a short-schedule result.
3. **CODI stack at a lower lr** (e.g. 1e-3) for CODI, CODI-0 and student-only together, to check the null gain holds when
   training is stable.
4. **Resample-swap necessity test on CODI's latents**: whether it is still informative when G ≈ 0. It could show whether
   the latents carry content even though they add no accuracy.
5. **Second seed** of the four latent/reference runs: started 2026-11-02 09:15 PT, **stopped 12:55 PT by user decision**
   for the k ≤ 8 Coconut pilot (`PILOT_K8.md`). Only `no_thoughts` s1 finished (logged); Coconut s1 stopped after stage 3,
   CODI s12 at step 7,029 / 31,280, CODI-0 s12 not started. Still needed for any round-1 claim.

## Where things are

- Run records: `results/<run_id>/` (manifest, notes, predictions, train logs); index in `EXPERIMENTS.md`.
- Checkpoints and logs (not in git): `/local-ssd/ktn48/lrm_work/` (`coconut_runs/`, `codi_runs/`, `logs/`), env in
  `lrm_work/env.sh`.
- Local commits `ee99abd` … on `main` (not pushed).
