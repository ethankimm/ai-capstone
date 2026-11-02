# Pilot: can Coconut learn to use its latents on RuleChain? (k ≤ 8, GSM8K recipe)

Status: **running**, launched 2026-11-02 12:57 PT on cci-p124, GPUs 4 and 7. Branch `kevin/lrm-training-2`.

## Why

Round 1 (`RULECHAIN_RESULTS.md`) found no latent gain: Coconut c1m6 is *below* its matched `no_thoughts` at L2-3, and
CODI matches CODI-0. Round 1 used Coconut's **ProsQA** recipe (1 latent per step, here with 1 epoch per stage), the
setting where Dilgren & Wiegreffe (2026) and Rizvi-Martel et al. (2026) find latents unnecessary. RuleChain is arithmetic,
closer to GSM8K-Aug, the one setting where Dilgren & Wiegreffe find Coconut's latents in use. Coconut's GSM8K recipe uses
**2 latents per step and 3 epochs per stage**. This pilot asks whether that recipe gets Coconut to beat its matched
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
| Latents per rule (`c_thought`) × `max_latent_stage` | 1 × 6 | **2 × 6 = 12 latents** (main arm); 1 × 6 (schedule-only arm) |
| Epochs per stage | 1 | **3** (`resume 3` skips stage 0) |
| Final all-latent stage | 3 epochs | **5 epochs** (epochs 22-26) |
| Selection / test | best of last 3 on valid, one test | same: best of `checkpoint_24/25/26` on k ≤ 8 valid, one test |
| lr / wd / effective batch | 1e-4 / 0.01 / 128 | same (c2 arm at micro-batch 16 × 8 for memory; others 32 × 4) |

Rules 7-8 never get a latent (M = 6), as in round 1; L7-8 test what the final pass does beyond the latents.

### Runs

| Run | GPU | Config | Role |
|---|---|---|---|
| `pilot_k8_coconut_c2m6` | 7 | `configs/synthetic/pilot_k8_coconut_c2m6.yaml` | **Main arm**: GSM8K recipe |
| `pilot_k8_no_thoughts_m6` | 4 | `configs/synthetic/pilot_k8_no_thoughts_m6.yaml` | Matched reference for both arms (same data, schedule, init; no latents) |
| `pilot_k8_coconut_c1m6` | 4 (after the reference) | `configs/synthetic/pilot_k8_coconut_c1m6.yaml` | Schedule-only arm: separates 2 latents per rule from the longer schedule |

Queues: `lrm_work/queue_pilot_k8_gpu{4,7}.sh` (helpers `lrm_work/pilot_k8_lib.sh`), log `lrm_work/logs/queue_pilot_k8.log`.

## What counts as a result

- **Positive:** a Coconut arm above `pilot_k8_no_thoughts_m6` by a paired test at L4-6 (above single-pass reach for this
  stack, within the latent budget). That would be the first RuleChain model whose latents add accuracy, and the candidate
  for the resample-swap necessity test and for decoding (proposal steps 2-3).
- **c2 > c1 at matched schedule** says latent width matters; **c1 > round-1 c1** at matched reference says the schedule
  matters.
- **Null:** both arms ≈ reference again → the GSM8K recipe does not rescue Coconut here either; the next candidates are
  step-supervised latents (SIM-CoT-style) or the from-scratch regime.

Caveats fixed in advance: one seed; 25k items is a quarter of round 1's data, so absolute accuracies are not comparable to
round 1, only gains against the matched pilot reference are; the CoT init saw all 100k items and all 12 levels.
