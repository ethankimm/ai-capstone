## 2026-11-08 — GSM8K grid need test: Coconut needs its latent passes (0 latents .081 -> 6 latents .331, every pass adds), and beats no-CoT (.169); the gain peaks at 2-3 gold steps (+.44 / +.32 over 0 latents) and vanishes at 5+ steps (coconut, run_id: 20261108-083818_coconut_gsm8k-grid-need-by-path-length)

**Goal:** The one grid task where latents should matter (October random swap: load-bearing). Need = Coconut 6 latents minus Coconut 0 latents, by gold step count; framing: a need beyond single-pass reach predicts the model must pass the input-only check (t17 GSM8K Coconut). Pre-registered (t16, commit 3574431): need >= .10 and growing with step count.
**Mechanism / model:** `connordilgren/gpt2-gsm8k-coconut` (checkpoint_33, HF cache), k = 0..6; `connordilgren/gpt2-gsm8k-no-cot` (checkpoint_25).
**Data:** GSM8K test (grid `gsm_original_test.json`), all 1319, binned by number of gold `<<...>>` steps.
**Hyperparams / command:** as the ProsQA record; answer after "###", numeric match; `grid_accuracy.py gsm8k <method> mps <out>`; raw JSON `codi_audit_artifacts/t16_gsm8k_*.json`.
**Results:**

| gold steps (n) | 1 (83) | 2 (357) | 3 (364) | 4 (290) | 5 (138) | 6 (57) | all |
|---|---|---|---|---|---|---|---|
| Coconut, 6 latents | .36 | .56 | .40 | .17 | .09 | .02 | .331 |
| Coconut, 0 latents | .13 | .12 | .08 | .04 | .06 | .04 | .081 |
| no-CoT | .14 | .29 | .15 | .13 | .09 | .04 | .169 |

Coconut by latent count (all): .081 .083 .152 .190 .234 .272 .331 for k = 0..6.
**Interpretation:** First half of the prediction holds (need .25 overall, each extra pass adds .04-.07). Second half fails: the gain does not grow with step count; it is largest at 2-3 steps and gone at 5+, where every model is near the floor. Over no-CoT, Coconut gains most at 2 and 3 steps (+.27 / +.25), where no-CoT is low but not at the floor: that is the "beyond single-pass reach" zone, so this checkpoint is the test case for the necessity prediction (t17).
**Caveats:** 0 latents is off-distribution (.081 is below no-CoT's .169, so "need" over 0 latents overstates; over no-CoT the gain is .16). Grid Coconut paper number 34.1 reproduces (.331). One checkpoint.
**Next:** t17 GSM8K Coconut route split on the 20260927-003331 twin pairs.
