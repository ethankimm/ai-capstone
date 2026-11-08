## 2026-11-08 — GSM8K Coconut route split: values sit at localised latents (step-1 edits move the answer through z1, .72; step-2 edits through z4, .68), carried entirely by those latents' keys/values; input-only .00-.03 everywhere. Fails the registered input-only check despite needing its latents (coconut, run_id: 20261108-091416_coconut_gsm8k-grid-route-split)

**Goal:** Necessity check on the one grid model that needs its latents (t16: 0 -> 6 latents .081 -> .331, no-CoT .169). Pre-registered (t17, commit 3574431): the framing predicts a pass at >= 1 site if the need is beyond single-pass reach; guess: input only >= half the full swap at an early site.
**Mechanism / model:** `connordilgren/gpt2-gsm8k-coconut` checkpoint_33, 6 latents, adapter `GridCoconut` (latent input = previous pass's last hidden state).
**Data:** the 231 twin pairs of 20260927-003331 (one number in the question perturbed, chain re-executed); recipients recovered from the GSM8K test set by question text (that record's `recipient_idx` does not index the HF test list); 225 with equal token length, all 225 qualified (recipient right, twin answered as implied).
**Hyperparams:** sites j = 0..5; per pair 27 conditions in one batch (equal to single runs on the first 10 pairs: agreement 1.000; unedited row = clean on 225/225). fp32, greedy, MPS.
**Command:** `scripts/necessity/grid_split.py gsm8k coconut mps <out> 1000` via `run_guarded.sh`; raw `codi_audit_artifacts/t17_gsm8k_coconut.json`.
**Results (share following the twin):**

| site | full swap | input only | entries only | entries, hidden from answer |
|---|---|---|---|---|
| z0 | .02 | .03 | .00 | .00 |
| z1 | .56 | .00 | .56 | .46 |
| z2 | .00 | .00 | .00 | .00 |
| z3 | .03 | .03 | .01 | .00 |
| z4 | .34 | .01 | .34 | .14 |
| z5 | .00 | .00 | .04 | .02 |
| all six | .83 | — | .89 | — |

By edited step: step 1 (n = 143) z1 .72, z4 .20; step 2 (n = 65) z1 .26, z4 .68; step 3 (n = 14) z1 .36, z4 .36.
**Interpretation:** Same profile as released GSM8K CODI (20261108-061509): localised value slots, carried by each slot's keys/values, input only ~0. z1's value also reaches the answer through later latents' attention (hidden from the answer .46 of .56); z4's is mostly read directly (.14 of .34). Under the pre-registered check (input only) this model FAILS at every site although it needs its latents, so it is the case the pre-registration said would contradict the framing if its gain lies beyond single-pass reach. **Definitional caveat raised with Henning (not resolved here):** in the full swap only z_j differs from the recipient's run, so the twin's value enters through the recurrent input z_j (the previous pass's final layer) and is stored in pass j's own entries; that path does cross "final layer -> next input" once. The input-only condition resets pass j's own entries, so it tests the NEXT hop (z_j -> z_{j+1}), i.e. a multi-hop chain, not whether z_j is used. Under a one-hop reading (full single-site swap of the input), GSM8K Coconut and GSM8K CODI pass at their value slots. `input_all` (computed by the script) is 0 by construction and is dropped from the record.
**Caveats:** twins perturb early steps mostly (hop 0/1: 208 of 225). One checkpoint. Whether the gain is beyond single-pass reach is not settled (t16: gain over no-CoT largest at 2-3 steps, where no-CoT is low but above the floor).
**Next:** decide the check's definition; decode z1 / z4 (inputs and entries) and run the decoded-value donor test.
