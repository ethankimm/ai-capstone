# Pilot: does step-level supervision (SIM-CoT) make Coconut's latents chain on RuleChain?

Status: **training** (launched 2026-11-03 22:26 PT on cci-p124, GPU 7; expected done around 09:00 PT 2026-11-04, then
select + test). Branch `kevin/sim-cot`.

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

Pending.
