# Pilot: does the RuleChain protocol find a CODI latent gain on GSM8K-Aug?

Status: **training** (launched 2026-11-03 12:07 PT on cci-p124, GPUs 4 and 7). Branch `kevin/lrm-training-2`.

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

| Run | GPU | Latents | Role | Training ETA (PT) |
|---|---|---|---|---|
| `gpt2_gsm8k_aug_codi` | 4 | 6 | CODI | ~17:40, then select + test |
| `gpt2_gsm8k_aug_codi0` | 7 | 0 | Matched reference (CODI-0) | ~15:05, then select + test |

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

Pending.
