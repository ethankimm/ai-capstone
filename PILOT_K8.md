# Pilot: can Coconut learn to use its latents on RuleChain? (k ≤ 8, longer schedule, GSM8K-style latents)

Status: **done** (2026-11-03 03:02 PT). **Verdict: null — no Coconut arm beats its matched no-latent reference at L4-6.**
Ran 2026-11-02 12:57 PT to 2026-11-03 03:02 PT on cci-p124, GPUs 4 and 7. Branch `kevin/lrm-training-2`.

## Why

Round 1 (`RULECHAIN_RESULTS.md`) found no latent gain: Coconut c1m6 is *below* its matched `no_thoughts` at L2-3, and
CODI matches CODI-0. Round 1 used Coconut's **ProsQA** recipe (1 latent per step, here with 1 epoch per stage), the
setting where Dilgren & Wiegreffe (2026) and Rizvi-Martel et al. (2026) find latents unnecessary. RuleChain is arithmetic,
closer to GSM8K-Aug, the one setting where Dilgren & Wiegreffe find Coconut's latents in use. Coconut's GSM8K recipe uses
**2 latents per step × 3 replaced steps = 6 latents, 3 epochs per stage**. Every published Coconut configuration uses 6
latents in total. This pilot asks whether that recipe (or 2 latents per rule over 6 rules) gets Coconut to beat its matched
reference at all; it is a positive-control search, not a replacement for round 1's protocol.

Seed 2 of round 1 (Coconut s1, CODI / CODI-0 s12) was **stopped by user decision** at 12:55 PT 2026-11-02 to free the GPUs:
Coconut c1m6_s1 had finished stage 3 (monitor 0.981 / 0.317 / 0.189), CODI s12 was at step 7,029 / 31,280, CODI-0 s12 never
started. None reached an eval, so there are no run records; `no_thoughts_m6_s1` finished and is logged
(`20261102-190130_…`).

## Design

| | Round 1 | **Pilot** |
|---|---|---|
| Train data | 100k items, k = 1-12 | **25k items, k = 1-8**, 3,125 per level (`scripts/make_pilot_k8.py`, seed 5980) |
| Valid / test | `depth2/valid.json` / `test.json` | k ≤ 8 subsets of the same files (1,336 / 4,000 items; test L1-8 = the same items as round 1) |
| Init | CoT `gpt2_depth2_cot/checkpoint_3` | same |
| Latents per rule (`c_thought`) × `max_latent_stage` | 1 × 6 | **2 × 3 = 6** (published GSM8K recipe); 2 × 6 = 12 (wider budget); 1 × 6 = 6 (schedule-only) |
| Epochs per stage | 1 | **3** (`resume 3` skips stage 0) |
| Final all-latent stage | 3 epochs | **5 epochs** (M 6: epochs 22-26; M 3: epochs 13-17) |
| Selection / test | best of last 3 on valid, one test | same: best of the last 3 final-stage checkpoints (`24/25/26` for M 6, `15/16/17` for M 3) on k ≤ 8 valid, one test |
| lr / wd / effective batch | 1e-4 / 0.01 / 128 | same (c2 arm at micro-batch 16 × 8 for memory; others 32 × 4) |

With M = 6, rules 7-8 never get a latent, as in round 1; with M = 3 (the GSM8K recipe), rules 4-8 never do. The
reference's curriculum depends on M (stage s drops the first s written steps), so each M has its own `no_thoughts`.

**Correction (2026-11-02 13:10 PT):** the first version of this file called 2 × 6 "the GSM8K recipe". It is not: the
recipe is 2 × 3 = 6 latents; 2 × 6 doubles the latent budget to 12, which breaks the protocol's 6-latent budget and mixes
latent width with total budget. The 2 × 3 arm and its reference were added (user decision) and run before the 1 × 6 arm.

### Runs

| Run | GPU | Config | Latents | Reference | Role |
|---|---|---|---|---|---|
| `pilot_k8_coconut_c2m3` | 4 (2nd) | `pilot_k8_coconut_c2m3.yaml` | 2 × 3 = 6 | `no_thoughts_m3` | **Published GSM8K recipe** |
| `pilot_k8_coconut_c2m6` | 7 | `pilot_k8_coconut_c2m6.yaml` | 2 × 6 = 12 | `no_thoughts_m6` | 2 latents for every rule up to 6; double budget |
| `pilot_k8_coconut_c1m6` | 4 (3rd) | `pilot_k8_coconut_c1m6.yaml` | 1 × 6 = 6 | `no_thoughts_m6` | Schedule-only: round 1's layout on the longer schedule |
| `pilot_k8_no_thoughts_m6` | 4 (1st) | `pilot_k8_no_thoughts_m6.yaml` | 0 | — | Reference for the M 6 arms |
| `pilot_k8_no_thoughts_m3` | 4 (2nd) | `pilot_k8_no_thoughts_m3.yaml` | 0 | — | Reference for the M 3 arm |

Configs in `configs/synthetic/`. Queues: `lrm_work/queue_pilot_k8_gpu7.sh`, `lrm_work/queue_pilot_k8_gpu4_v2.sh` (replaced
`queue_pilot_k8_gpu4.sh` at 13:07 PT, taking over the running `no_thoughts_m6`); helpers `lrm_work/pilot_k8_lib.sh`; log
`lrm_work/logs/queue_pilot_k8.log`. Expected finish (PT): `no_thoughts_m6` ~14:05, `no_thoughts_m3` ~14:50, `c2m3`
~19:30, `c2m6` ~01:00 Tue, `c1m6` ~02:15 Tue.

## Results

Paired tests on test L1-8 (500 items per level), exact McNemar, Newcombe CIs, Holm over 8 levels, 2,000 item bootstraps
(`scripts/rulechain_gain_analysis.py --pilot`; full tables in `PILOT_K8_GAIN_ANALYSIS.md`). G = first − second.

| Comparison | All L1-8 | L1-3 | **L4-6 (decision band)** | L7-8 | L2 (Holm p) | d50 diff |
|---|---|---|---|---|---|---|
| c2m3 − no_thoughts_m3 | −0.004 (p 0.39) | −0.018 (0.07) | **−0.002 [−0.015, +0.011] (0.84)** | +0.013 (0.06) | −0.036 (0.89) | −0.10 [−0.12, +0.07] |
| c2m6 − no_thoughts_m6 | −0.021 (8e-6) | −0.052 (4e-8) | **−0.003 [−0.017, +0.010] (0.70)** | −0.002 (0.89) | **−0.126 (1.5e-7)** | −0.10 [−0.15, −0.05] |
| c1m6 − no_thoughts_m6 | −0.026 (1e-7) | −0.063 (4e-11) | **−0.007 [−0.020, +0.007] (0.38)** | +0.002 (0.90) | **−0.170 (1.8e-13)** | −0.12 [−0.17, −0.07] |
| c2m3 − c1m6 (both 6 latents) | −0.011 (0.02) | −0.025 (0.008) | −0.001 (0.92) | −0.003 (0.77) | −0.038 (0.63) | −0.08 |
| c2m6 − c1m6 (same M) | +0.005 (0.35) | +0.011 (0.23) | +0.003 (0.69) | −0.004 (0.67) | +0.044 (0.36) | +0.02 |

| Run | Test L1-8 | d50 [95% CI] | Run record |
|---|---|---|---|
| `pilot_k8_no_thoughts_m6` | 0.216 | 1.90 [1.86, 1.96] | `20261102-215045_…` |
| `pilot_k8_no_thoughts_m3` | 0.184 | 1.80 [1.71, 1.81] | `20261102-222423_…` |
| `pilot_k8_coconut_c2m3` | 0.180 | 1.71 [1.67, 1.80] | `20261103-033009_…` |
| `pilot_k8_coconut_c2m6` | 0.195 | 1.81 [1.78, 1.84] | `20261103-110209_…` |
| `pilot_k8_coconut_c1m6` | 0.191 | 1.79 [1.75, 1.82] | `20261103-105353_…` |

Unparseable rate 0.000 for all. Every Coconut arm is flagged under-trained by the protocol rule (valid still rising between
the last two checkpoints), though every monitor curve is flat over the final stage.

### Verdict against the decision rules

- **Positive: no.** No arm is above its matched reference at L4-6; all three L4-6 gains are within ±0.007 with CIs
  inside ±0.02, against references at 0.03-0.06 (~95% headroom). Neither the published GSM8K recipe (2 × 3), nor 2
  latents for every rule (2 × 6, double budget), nor round 1's layout on a 3× longer schedule (1 × 6) makes the latents
  add accuracy above single-pass reach.
- **Budget allocation (c2m3 vs c1m6): no difference at L4-6**; c2m3 is slightly lower at L1-3. **Width (c2m6 vs c1m6): no
  difference.** **Schedule (pilot c1m6 vs round-1 c1m6, each vs its reference): no change** — round 1 L2 −0.136 / L1-3
  −0.065, pilot L2 −0.170 / L1-3 −0.063.
- **Null → next candidates** per this file: step-supervised latents (SIM-CoT-style) or the from-scratch regime.

### The L2 deficit replicates

With M = 6, Coconut is clearly below `no_thoughts` at L2 in all three independently trained models: round 1 c1m6 −0.136,
pilot c1m6 −0.170, pilot c2m6 −0.126 (each Holm p < 1e-6). Each exceeds round-1 `no_thoughts`' seed-to-seed L2 swing
(0.084). The M = 3 arm shows the same sign (−0.036), not significant, against a reference that is itself weak at L2.
Training with latents costs direct-answer accuracy at k = 2 without the latents doing that computation. This is the
round-1 L2 result that seed 2 was meant to check; it now holds across three models, two layouts and two schedules
(still one seed each, and all share the CoT init).

## What counts as a result

- **Positive:** a Coconut arm above its matched `no_thoughts` (`_m3` or `_m6`) by a paired test at L4-6 (above single-pass reach for this
  stack, within the latent budget). That would be the first RuleChain model whose latents add accuracy, and the candidate
  for the resample-swap necessity test and for decoding (proposal steps 2-3).
- **c2m3 vs c1m6** (both 6 latents): how the budget is spent (2 per rule for 3 rules vs 1 per rule for 6). **c2m6 vs
  c1m6** (same M): latent width, confounded with total budget (12 vs 6). **c1m6 vs round-1 c1m6**, each against its own
  reference: the longer schedule.
- **Null:** every arm ≈ its reference again → the GSM8K recipe does not rescue Coconut here either; the next candidates are
  step-supervised latents (SIM-CoT-style) or the from-scratch regime.

Caveats fixed in advance: one seed; 25k items is a quarter of round 1's data, so absolute accuracies are not comparable to
round 1, only gains against the matched pilot reference are; the CoT init saw all 100k items and all 12 levels.
