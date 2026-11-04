# Pilot: does the RuleChain protocol find a CODI latent gain on GSM8K-Aug?

Status: **done** (2026-11-03 17:23 PT). **Verdict: sanity passed: CODI beats CODI-0 by +0.074 on GSM8K test (McNemar p 2e-11).**
Ran 2026-11-03 12:07-17:23 PT on cci-p124, GPUs 4 and 7. Branch `kevin/lrm-training-2`.

## Why

On RuleChain, neither Coconut nor CODI beats its matched no-latent reference (`RULECHAIN_RESULTS.md`, `PILOT_K8.md`).
CODI's lead over plain no-CoT came entirely from its training signal: CODI-0 (same losses, 0 latents) matched it at every
level. GSM8K-Aug is the setting where earlier work finds latents in use. This pilot is a sanity check of the protocol:
run `TRAINING_PROTOCOL.md`'s CODI stack with **nothing changed except the dataset**. A CODI − CODI-0 gain here means
the protocol can detect a latent gain where one is expected, so RuleChain's null is about the task. If GSM8K is null
too, the protocol and its matched reference become the suspects. Coconut is left out for now (user decision): its
curriculum would add a choice between our c1m6 schedule and the published c2m3 one.

## Design

| | RuleChain round 1 | **This pilot** |
|---|---|---|
| Train data | 100k RuleChain items | **100k items sampled from GSM8K-Aug train** (seed 5980), excluding the repo's 1,000-item validation carve-out |
| Valid / test | `depth2/valid.json` / `test.json` | repo's fixed 1,000-item validation split / GSM8K test (all 1,319) |
| Levels (reporting only) | rule count k | number of `<<…>>` steps in the rationale (test: 0-8; 18 test items have no steps) |
| Selection | best of last 3 on valid, unweighted mean over levels | best of last 3 on valid, **overall accuracy** (step counts are unbalanced; the mean over levels would weight the 1-3 items at 7-9 steps like 400 items at 2) |
| Everything else | protocol | same: LoRA r128/α32, lr 3e-3 cosine, 3% warmup, wd 0.1, 40 epochs, 64 × 2, bf16, seed 11, 6 latents (CODI-0: 0), `include_last_cot False`, `fix_attn_mask` + `pad_invariant`, projection 768 + LayerNorm, every epoch kept, NaN guard |

Data: `scripts/make_gsm8k_aug_v2.py` writes `gsm8k_aug/{train,valid,test}.json` in the RuleChain schema (steps = the
rationale split on whitespace, as in Coconut's `gsm_icot.py`). Hashes are in `configs/synthetic/v2_data_sha256.json`.
CODI reads the data through the same synthetic-task code path as RuleChain (`gsm8k_aug` added to `SYNTHETIC_TASKS` in
`scripts/codi_synthetic.patch`).

**Differences from CODI's published GSM8K-Aug run**, all from keeping the RuleChain code path:
- 100k of 385k training items.
- Teacher CoT steps joined by `\n` rather than a space; question `.strip()` + `\n`.
- Items with non-digit-leading answers are kept: 440 negative answers, plus 40 items such as `~7.14`, `2/5`, `$180`.
  CODI's `icot` loader drops these.
- Our padding fix (`fix_attn_mask` + `pad_invariant`).
- Selection on valid.

Published reference points (full data): CODI GPT-2 43.7% (Shao et al.); no-CoT GPT-2 16.5% (Hao et al.).

### Runs

| Run | GPU | Latents | Role | Training (PT) |
|---|---|---|---|---|
| `gpt2_gsm8k_aug_codi` | 4 | 6 | CODI | done 17:22, tested 17:23 |
| `gpt2_gsm8k_aug_codi0` | 7 | 0 | Matched reference (CODI-0) | done 14:56, tested 14:57 |

Queue `lrm_work/queue_gsm8k_codi.sh GPU NAME NUM_LATENT`; select/test `lrm_work/codi_select_test_gsm8k.sh`; log
`lrm_work/logs/queue_gsm8k.log`. Smoke runs (2 steps, both latent counts) and an eval-path check passed before launch.

## What counts as a result

- **Sanity passed:** CODI above CODI-0 on test by a paired test (exact McNemar on the 1,319 items), with gains reported
  per step count. GSM8K is then a working positive control, and RuleChain's null is about the task.
- **Null:** CODI ≈ CODI-0. Either GSM8K's latent use in earlier work does not survive the matched reference (CODI's gain
  is its training signal here too), or something in the protocol removes it. Next: student-only, to separate "CODI-0 is
  strong" from "no-CoT is strong", and possibly the published `icot` formatting.
- CODI's own accuracy vs the published 43.7% checks that the recipe trains properly on 100k items. Being somewhat lower is
  expected.

Caveats fixed in advance: one seed; a quarter of the published training data.

## Results

Paired tests on GSM8K test (1,319 items): exact McNemar, Newcombe CIs, Holm over step counts, 2,000 item bootstraps
(`scripts/gsm8k_gain_analysis.py`; full tables in `GSM8K_GAIN_ANALYSIS.md`). G = CODI − CODI-0.

| Run | Test acc | Teacher (CoT) | Selected (valid acc) | Trained (PT, 2026-11-03) | Run record |
|---|---|---|---|---|---|
| `gpt2_gsm8k_aug_codi` | **0.261** | 0.429 | checkpoint-31280, epoch 40 (0.573) | 12:07-17:22 | `20261104-012327_…` |
| `gpt2_gsm8k_aug_codi0` | **0.187** | 0.370 | checkpoint-30498, epoch 39 (0.510) | 12:07-14:56 | `20261103-225715_…` |

| | All | 0-2 steps | 3-4 steps | 5-8 steps | 2 steps (Holm p) | 3 steps (Holm p) |
|---|---|---|---|---|---|---|
| n | 1,319 | 440 | 654 | 225 | 357 | 364 |
| G | **+0.074 [+0.052, +0.095]** | +0.111 | +0.075 | −0.004 | +0.104 (2e-4) | +0.118 (6e-6) |
| CODI-only / CODI-0-only | 155 / 58 (p 2e-11) | 70 / 21 | 79 / 30 | 6 / 7 | 57 / 20 | 59 / 16 |

Normalized gain overall 0.09. Unparseable 0.000 for both. Training was clean (no NaN; loss 11 → 0.3).

### Verdict

- **Sanity passed.** On GSM8K the protocol finds a clear CODI latent gain over its matched reference. The same setup on
  RuleChain gave G ≈ 0 at every level. The protocol and the CODI-0 reference can detect a latent gain, so RuleChain's
  null is about the task.
- **Where the gain sits:** at 1-3 steps (+0.10 to +0.12). It vanishes from 4 steps up, where both models are near the
  floor. The latents help where CODI-0 is partly right, not at the hardest items.
- **Recipe health:** the teacher's CoT accuracy (0.429) is about the published GPT-2 CoT figure (42.9%, Hao et al.), and
  CODI-0 (0.187) is about published no-CoT (16.5%). CODI itself (0.261) is well below the published 43.7%. That gap is
  most likely from using 100k of 385k items and our code path; not tested.
- **Valid ≠ test.** Valid accuracy (0.51-0.57) is about twice test accuracy. GSM8K-Aug has about 50 augmentations of each
  GSM8K train problem, so the held-out valid items are near-duplicates of training items. 29% of valid items share an
  exact step sequence with a training item, against 10% of test items. Valid still ranks checkpoints, but do not read it
  as an accuracy estimate.

Caveats: one seed; CODI only (Coconut not run on GSM8K).

**Next candidates:**
- A resample-swap necessity test on these CODI latents, since there is now a gain to explain.
- Decoding the latents (proposal steps 2-3) on this checkpoint.
- Optionally a second seed, or Coconut under the protocol on GSM8K.

## Latent diagnostics (2026-11-03 PT)

Task brief: repeat Henning's RuleChain latent ablation (`20261104-022824`, origin/main) on the selected CODI
checkpoint-31280, then probe what the latents encode.

- **Ablation** (`20261104-034313_…latent-ablation`, teacher-forced first answer token, 1,195 of 1,319 items scored, clean 0.283):
  - **No fixed point.** The latents alternate in a 2-cycle: z_1/z_3/z_5 vs z_2/z_4, with odd-even centered cosine about 0.2.
  - Every odd site carries item content (resample z_1 / z_3 / z_5: −0.085 / −0.062 / −0.063). Even sites cost ≤ 0.03; z_0 is unused.
  - The mean-ablated tail declines in steps (s = 5…1: −0.028, −0.037, −0.073, −0.078, −0.165), with significant increments
    from z_1, z_3 and z_5. On RuleChain, tail ≥ 3 cost 0.01.
  - Mean-ablating every latent (0.115) is below CODI-0 (0.204 on the same items).
- **Probes + logit lens** (`20261104-034608_…latent-probes`):
  - v_1 peaks at z_1 (probe 0.45, lens 0.43; majority 0.03); v_2 peaks at z_3 (0.21).
  - Even sites decode CoT punctuation (">>", "."), not values.
  - v_3, v_4 and the final answer are near the majority baseline at every site.
- **Reading:** GSM8K's latent gain comes from a short serial computation over the odd passes, covering about the first
  one or two step results. It is not one extra pass of parallel compute, and it is not a full chain.
