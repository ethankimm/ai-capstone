## 2026-11-08 — Content reliance in SIM-CODI matches plain CODI's split: latents ignored at L1 (all-six donors followed .000, mean ablation 1.000), relied on from L2 (follows .998-1.000, mean ablation to .01-.02), so reliance does not track need in either method (codi, run_id: 20261108-013131_codi_v2-l1to7w-startlast-sim-content-reliance)

**Goal:** HANDOFF_2026-11-07 step 2 (paper F2, "reliance does not track need", across methods). Does SIM-CODI, like plain start-last CODI (`20261107-220408_codi_v2-l1to7w-startlast-content-reliance`), ignore its latents at L1 and rely on their content from L2, below reach? Start-last CODI-0 solves L2 at .98 with no latents.
**Mechanism / model:** SIM-CODI start-last, checkpoint-16335 (`20261107-225603`), loaded without its step decoder (answer path unchanged; clean test L1-L6 1.000, L7 .998 as the pod eval).
**Data:** canonical **test** L1-L7, 500 per level (499 recipients at L7), to match the plain record row by row. Means: 2,000 train items of the same level (seed 21).
**Command:** On the laptop (MPS, float32, `run_guarded.sh`, `caffeinate`): `~/Documents/Penn/CIS5980/codi_audit_artifacts/t2_sim_resample.py`, a copy of `b29_ungated_resample.py` with only the checkpoint and output names changed (diff checked); record `record10.py`. Smoke (L2, 40 items): identity (patching an item's own six latents back) keeps every answer.
**Design (b29's):** all six latent inputs replaced at once. Different-value donor: same level, differs at v_{L-1}, predicted answer = the recipient's last rule on the donor's v_{L-1}, in 0..99 and != gold. Same-value donor: same v_{L-1}, a different start value where possible (format control). Mean: each z_s set to its same-level train mean. Recipients: answered right.

**Headline results** (SIM-CODI vs plain CODI from b29, same test items and donor seeds):

| L | SIM clean | SIM diff donor: follows / keeps / donor's answer | SIM same-value donor keeps (different start share) | SIM mean acc | plain clean | plain follows / keeps | plain same keeps | plain mean acc |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | .000 / 1.000 / .012 | 1.000 (0.00; at L1 v_{L-1} is the start value) | 1.000 | 1.000 | .000 / 1.000 | 1.000 | 1.000 |
| 2 | 1.000 | **.998** / .000 / .142 | .998 (1.00) | .022 | .998 | .994 / .002 | 1.000 | .044 |
| 3 | 1.000 | 1.000 / .000 / .118 | 1.000 (1.00) | .020 | .476 | .374 / .017 | .616 | .032 |
| 4 | 1.000 | 1.000 / .000 / .120 | 1.000 (1.00) | .016 | .306 | .275 / .033 | .423 | .038 |
| 5 | 1.000 | 1.000 / .000 / .108 | 1.000 (1.00) | .024 | .160 | .150 / .025 | .325 | .040 |
| 6 | 1.000 | 1.000 / .000 / .130 | 1.000 (1.00) | .022 | .116 | .034 / .017 | .193 | .020 |
| 7 | .998 | 1.000 / .000 / .120 | 1.000 (1.00) | .012 | .076 | .105 / .079 | .105 | .034 |

- **L1: latents ignored, in both methods.** A donor's six latents (different start value) never move the answer (.000 follows, 1.000 keeps own), and all-six mean ablation leaves 1.000. The answer tokens apply the single rule to the start value in the question.
- **L2 and up: content reliance, in both methods.** SIM follows the donor .998-1.000 at every level L2-L7. A same-value donor with a different start value keeps the answer (.998-1.000), so what the latents pass on is the value v_{L-1}, not format or the start value. Mean ablation drops to chance (.012-.024). The donor's own answer comes out only .11-.14: the answer tokens apply the recipient's last rule to the donor's value.
- At L3+ plain CODI's clean accuracy is low, and its "follows" is bounded by it. SIM shows the same reliance at full accuracy.

**Predictions (`t_predictions.md`, written before any output):** "L2+ follows near 1.0, same-value donor keeps ~1.0, mean ablation to chance from L2": right. "L1 the open cell, guess follows partly (.3-.9)": **wrong**, L1 ignores the latents completely.
**Interpretation:** F2 now holds across two methods. Both models depend on latent content at L2, where the same backbone without latents (CODI-0) already scores .98, and both ignore the latents at L1. So reliance follows the training setup, not the depth the task needs. The boundary sits at the same place despite very different training signals (no step labels vs a step decoder on every latent). For SIM, reliance from L2 is partly imposed: the step loss trains z_0 to carry v_1 at L2. But L1 has no trained step, and there SIM, like plain, lets the answer tokens do the one rule themselves. This connects to step 1 (`20261108-012524`): at L3+ SIM's first untrained latent z_{L-1} is fully used on swap, but at L1 the same position (z_0 = z_{L-1}) is not. The answer reads a carried value only when a pass before it had a rule to apply.
**Gotchas hit:** none. At L1 the same-value control is trivial (same v_0 = same start value, share 0.00 different), as in b29.
**Caveats:** One seed; test split (as the plain row), so this record is not on valid like steps 1 and 3. All-six swaps cannot tell which latent carries the value (step 1 does that per latent on valid). Plain's L3+ rows rest on 38-238 recipients.
**Next:** Handoff step 3 (dynamics) is logged with this; step 4 (formation over epochs) and step 5's second check (L7 with 7+ passes) remain.
