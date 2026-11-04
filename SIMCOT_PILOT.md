# Pilot: does step-level supervision (SIM-CoT) make Coconut's latents chain on RuleChain?

Status: **done** (2026-11-04 06:42 PT). **Verdict: positive. SIM-CoT c1m6 is above `no_thoughts_m6` by +0.731 at L4-6
(1,106 vs 9 items), and it collapses exactly at k = 7, the first rule without its own latent.** Necessity tests pending.
Ran 2026-11-03 22:26 PT to 2026-11-04 06:42 PT on cci-p124, GPU 7. Branch `kevin/sim-cot`.

## Why

Every RuleChain latent model so far has been null against its matched no-latent reference:
- Coconut with the ProsQA and GSM8K recipes (`RULECHAIN_RESULTS.md`, `PILOT_K8.md`).
- CODI, whose latents sit at a fixed point (Henning, `20261104-022824`).

On GSM8K, CODI's latents hold only v_1 and v_2 (`GSM8K_LATENT_DIAGNOSTICS.md`). None of these runs had a training signal
that tells latent j to hold step j. SIM-CoT (Wei et al., ICLR 2026) adds exactly that, and attributes implicit-CoT failure
to latent homogeneity. The pilot asks: with step supervision, does Coconut beat its matched reference above single-pass
reach? `PILOT_K8.md` and Henning's notes both name this as the next candidate.

## Method

SIM-CoT adds an auxiliary decoder during training only:
- The decoder is a separate GPT-2 with fresh pretrained weights. It shares the input embedding with the main model and is
  discarded at inference.
- It decodes latent k into the text of step k (`<<a op b = c>>`, then EOS).
- Loss = Coconut's LM loss + 1.0 × the mean over latents of the step loss. The step loss is summed over tokens and
  averaged over sequences, as in their code.

Latent ↔ step mapping (their code, unchanged):

| Stage | Latent j decodes |
|---|---|
| Curriculum stage s (latents 1..s) | step j |
| Final stage (6 latents) | latents 1-5: steps 1-5. Latent 6: steps 6..k joined |
| Item with k < 6 rules (final stage) | spare latents decode the answer `### N` |

## Code

- [InternLM/SIM-CoT](https://github.com/InternLM/SIM-CoT) @ `cd3291d`, Coconut fork. Its model forward is identical to
  upstream Coconut's apart from the added step loss (checked by diff).
- Patch `scripts/simcot_explain_loss.patch` fixes a bug:
  - **The released `run.py` never computes the step loss.** It builds training data with `get_cot_latent_dataset` +
    `MyCollator`, so `explainable_ids_list` never reaches the forward, and `gsm_simcot.yaml` as released trains plain
    Coconut with an idle decoder.
  - The patch uses their own `get_cot_with_explainable_latent_dataset` + `MyExplainableCollator`, which give identical
    input_ids/labels plus the step tokens.
  - It also logs the step loss.
  - The bug is not reported in their issues. Issue #6 (SIM-CoT trained from the repo does poorly) may be related.
- Checks:
  - RuleChain steps parse into exactly one `<<…>>` group per step (5,000 / 5,000 train items).
  - 2-epoch smoke run (stage 6 + final stage, 2k items): the step loss falls 78 → 14.5.
  - `scripts/eval_synthetic_coconut.py` loads the checkpoint with missing = unexpected = 0. The decoder weights are
    ignored, and inference is plain Coconut.

## Design

Identical to `pilot_k8_coconut_c1m6`; only the SIM-CoT keys are added (`configs/synthetic/pilot_k8_simcot_c1m6.yaml`):

| | |
|---|---|
| Data | k ≤ 8 pilot: 25k train (3,125 per level), valid 1,336, test L1-8 (4,000, = round 1's L1-8) |
| Latents | c_thought 1 × max_latent_stage 6, `pad_latent_to_max` |
| Schedule | CoT init (`gpt2_depth2_cot/checkpoint_3`), resume 3, 3 epochs per stage, 5 final epochs (22-26), lr 1e-4, wd 0.01, effective batch 128 (32 × 4), fp32, seed 0 |
| Selection / test | best of checkpoint_24/25/26 on depth2_k8/valid.json, one logged test |

**Deviation from SIM-CoT's GPT-2 recipe.** They continue from a trained Coconut checkpoint (c_thought 2 × 5 stages,
about 30 more epochs; their issue #4). We start from the CoT model, as our pilot c1m6 does, so that the only difference
from an existing run is the step loss.

**References (already logged, no new runs):**
- `pilot_k8_no_thoughts_m6` (`20261102-215045`): the matched no-latent reference.
- `pilot_k8_coconut_c1m6` (`20261103-105353`): same recipe without step loss.

## Decision rule (fixed before results)

- **Positive:** SIM-CoT above `no_thoughts_m6` by a paired test (exact McNemar) pooled over **L4-6**, the same band and
  test as `PILOT_K8.md`. The latents then add accuracy above single-pass reach, for the first time on RuleChain.
  - **Follow-up:** a resample swap of latent j within level (does the answer follow the donor's step?) and the
    `ablate_latents_rulechain_codi.py`-style tail ablation, to show the latents carry it.
  - Probing is circular here (the latents are trained to decode steps), so it is not evidence.
- **Effect of step supervision:** SIM-CoT vs `pilot_k8_coconut_c1m6`, same paired tests. This also checks whether the L2
  deficit (Coconut below `no_thoughts` at L2 in all three M = 6 models) persists.
- **Expected fall:** if each latent does one rule, accuracy should stay high up to about k = M + reach (6 + ~2). Our
  Coconut runs so far fall at d50 ≈ 1.8.
- **Null:** SIM-CoT ≈ `no_thoughts_m6` at L4-6. Then step supervision does not make GPT-2 small chain rules through its
  latents here either. That strengthens the negative result, with the caveat that we start from CoT, not from a trained
  Coconut model.

Caveats fixed in advance: one seed; 25k items; the step loss's scale (summed over tokens) is SIM-CoT's choice, not tuned.

## Results

Paired tests on test L1-8 (500 items per level): exact McNemar, Newcombe CIs, Holm over 8 levels, 2,000 item bootstraps
(`scripts/rulechain_gain_analysis.py --simcot`; full tables in `SIMCOT_GAIN_ANALYSIS.md`).

| Run | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | all | d50 [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| **SIM-CoT c1m6** | 0.992 | **0.980** | **0.936** | **0.866** | **0.788** | **0.678** | 0.008 | 0.034 | **0.660** | **6.12** [6.05, 6.18] |
| `no_thoughts_m6` (matched) | 0.986 | 0.430 | 0.114 | 0.064 | 0.032 | 0.042 | 0.032 | 0.032 | 0.216 | 1.90 [1.86, 1.96] |
| plain c1m6 (no step loss) | 0.990 | 0.260 | 0.090 | 0.046 | 0.032 | 0.040 | 0.034 | 0.034 | 0.191 | 1.79 [1.75, 1.82] |
| round-1 CODI-0 (100k items, k 1-12) | 1.000 | 0.988 | 0.960 | 0.764 | 0.494 | 0.240 | 0.120 | 0.082 | 0.581 | 4.86 [4.74, 4.99] |

| Comparison | L1-3 | **L4-6 (decision band)** | L7-8 | d50 diff |
|---|---|---|---|---|
| SIM-CoT − no_thoughts_m6 | +0.459 (694 / 5) | **+0.731 [+0.707, +0.753] (1,106 / 9, p 3e-314)** | −0.011 (p 0.15) | +4.21 |
| SIM-CoT − plain c1m6 | +0.523 | +0.738 (1,116 / 9) | −0.013 | +4.33 |
| SIM-CoT − round-1 CODI-0 | −0.013 | L4 +0.102, L5 +0.294, L6 +0.438 (Holm p ≤ 4e-5) | L7 −0.112, L8 −0.048 | +1.26 |

Run record `20261104-144235_coconut_v2-depth2-pilot_k8_simcot_c1m6`.
- Selected checkpoint_26 (valid 0.675). Flagged under-trained: valid still rising, 0.668 → 0.675.
- Unparseable 0.000.
- Monitor rose through every stage to 0.68. Plain c1m6's monitor collapsed to ~0.17 from stage 2 on.

### Verdict against the decision rules

- **Positive.** Large gain at L4-6 over the matched reference.
  - This is the first RuleChain model whose latents (with step supervision) beat single-pass reach.
  - It is also above the strongest no-latent model measured (CODI-0) from L4 to L6.
- **Effect of step supervision:** the same recipe without the step loss (plain c1m6) is null. The step loss alone turns
  a null into +0.74 at L4-6. **The L2 deficit is gone** (0.980 vs 0.260).
- **Expected fall:**
  - Accuracy stays high through k = 6 (= M), then drops to 0.008 at k = 7. That is below `no_thoughts` (0.032, Holm
    p 0.035) and far below CODI-0 (0.120).
  - Rule 7 is the first rule without its own latent (SIM-CoT trains latent 6 to decode steps 6..k joined).
  - The model now runs the whole chain through its latents and does not even add one rule in the answer pass. This is
    the one-rule-per-latent signature, not "M + single-pass reach" (which would have put the fall near k ≈ 8).
- **Still required by the decision rule:** necessity. The step loss is extra supervision `no_thoughts` does not get.
  The k = 7 collapse argues that the latents carry the chain, but a resample swap / tail ablation has to show it.

**Next:**
1. Necessity: resample swap per latent within level, follows-donor vs keeps-own, plus mean / tail ablation.
2. Does latent j carry rule j: swap only latent j and test whether the answer follows rule j's donor value.
3. Second seed; vary M (e.g. M = 8) to check that the cliff moves with the number of latents.
