## 2026-11-08 — Released CODI GPT-2 on GSM8K, matched reference for t22: test .425, zero passes .224, settling pass 1.8 / 3.0 / 3.7 / 4.3; SIM's step decoder applied to CODI's latents reads nothing (.15 / .02 / .03); the logit lens reads v_1 at the outputs of passes 1, 3 and 5 (.76-.78) on 2-step items (codi, run_id: 20261108-232436_codi_gsm8k-released-codi-pass-sweep-decoder)

**Goal:** t22a/d and t23's readability side (pre-registered, t_predictions.md t22-t23, commit 6a1e88b): released CODI through the same harness as released SIM-CODI (`20261108-232435`).
**Mechanism / model:** `zen-E/CODI-gpt2` @ fd641b3, fp32 MPS, batch 1; SIM-CoT's decoder weights applied to its latents as a control.
**Data:** GSM8K test, 1,319.
**Command:** `sim_gsm8k.py <repo> sweep codi mps <out>`; analysis `sim_gsm8k_analyze.py`; record `record22.py sweep_codi`.
**Headline results:**
- Test .425 (.419 in 20260927-004340, same checkpoint, eval-mode path). Accuracy after k = 0..6 passes: .224 .249 .234 .372 .368 .417 .425. Mean settling pass n = 2/3/4/5+: 1.76 / 2.98 / 3.73 / 4.32, as logged in 20260927-005051.
- SIM decoder on CODI latents: .15 / .02 / .03 at steps 1-3 (right items); wrong answer equals a decoded result .05 vs .04 chance.
- Lens (top-1 token, best of layers 4-12) reads v_1 on 2-step right items: question tail .71; outputs of passes 0-5: .05 .76 .05 .78 .04 .78.
**Predictions:** sweep reproduced: held. Decoder control <= .2: held. t23 lens at pass 5 >= .4: held (.78).
**Interpretation:** The decoder is specific to SIM's latent code, so its reads on SIM are not the question text leaking through. The lens pattern follows CODI's alternation: outputs of odd passes (which become z_2, z_4, z_6) read v_1; even ones read operators. The pass-5 output becomes z_6, which no computation consumes, so its readability is unused by construction (t23).
**Caveats:** One checkpoint; lens is top-1 at single positions.
**Next:** route split `20261108-232438`.
