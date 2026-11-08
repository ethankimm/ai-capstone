## 2026-11-08 — Released CODI GPT-2 route split at all six sites (matched reference for t22b): values at z0 / z2 / z4 travel through attention entries (input only .01), odd sites carry nothing (both and entries .01), all-sites .85; reproduces t12 on 312 pairs (codi, run_id: 20261108-232438_codi_gsm8k-released-codi-route-split-all-sites)

**Goal:** t22b reference and t23 use side (pre-registered, t_predictions.md t22-t23, commit 6a1e88b). Extends t12 (`20261108-061509`, even sites only) to all six sites and to the regenerated candidate pool.
**Mechanism / model:** `zen-E/CODI-gpt2` @ fd641b3; SIM's decoder applied after swaps as a control.
**Data:** 1,471 regenerated twin candidates; 312 qualified for CODI (n = 2: 115, n = 3: 122, n >= 4: 75).
**Command:** `sim_gsm8k.py <repo> route codi mps <out>`; record `record22.py route_codi`.
**Headline results:**
- All sites both .846, entries .846.
- Per site, both / input / kv / kv_noans: z0 .20 / .01 / .20 / .17; z1 .01 / .01 / .01 / .01; z2 .22 / .01 / .21 / .07; z3 .01 / .01 / .01 / .01; z4 .36 / .01 / .35 / .01; z5 .01 / .01 / .01 / .01 (t12: z0 .199, z2 .215, z4 .358, input .007).
- 2-step recipients: z4 .22, z2 .13, z5 .00 (both and entries).
**Predictions (t23):** z5 swap <= .10 with the lens reading v_1 at pass 5 >= .4 on the same items: held (z5 .00, lens .78 in `20261108-232436`). But the state the lens reads is the pass-5 output, which becomes z_6 and is consumed by nothing, so this is readable-and-unused by construction, a weaker case than the RuleChain spare latents. z4 <= .15 at n = 2: missed (.22).
**Interpretation:** CODI's slot regime holds at every site: even latents store values read by attention, odd latents are neither readable as values nor used. Together with SIM (`20261108-232437`), both released GPT-2 latent models on GSM8K fail the input-only check while their latents are used as a set (.83-.85).
**Caveats:** One checkpoint; decoder-after-swap rows are a control (SIM decoder on CODI latents).
**Next:** none.
