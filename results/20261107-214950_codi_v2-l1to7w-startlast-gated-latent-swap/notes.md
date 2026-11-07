## 2026-11-07 — Per-latent swap on rule-gated CODI: every latent z_1..z_{L-1} decides the answer alone (follows the donor .98-1.00, mean ablation to chance); latents after z_{L-1} are idle, though a probe reads the value in them (codi, run_id: 20261107-214950_codi_v2-l1to7w-startlast-gated-latent-swap)

**Goal:** The decoder control (20261107-211610) found every step readable from its own latent. Is each latent also used, tested one latent at a time rather than argued from the gate? And which latents are idle?
**Mechanism / model:** `codi`, gpt2, 6 latents. Rule-gated start-last CODI checkpoint-17820, gate on. fp32, MPS.
**Data:** Valid L3-L7, 286 / 286 / 286 / 285 / 285 items; scored pairs use recipients answered right, n = 279-286 per cell. Mean latents from 2,000 train items per level (seed 21).
**Command:** `codi_audit_artifacts/b28_gated_swap.py` under `run_guarded.sh` and `caffeinate -dims`; record from `record7.py`. Swap: replace the input latent z_j (j = 1-5) with a same-level donor's z_j. k = min(j, L-1) is the step z_j carries. Donor: first in a seeded shuffle with a different v_k whose predicted answer (the recipient's rules k+1 to L applied to the donor's v_k) is in 0-99 and differs from the recipient's gold. Mean ablation: z_j set to its train mean, which is the draft's necessity measure.
**Headline results:**
- **Checks.** Patching an item's own z_j back in leaves every answer unchanged (smoke run). Clean accuracy 1.000 at L3-L6, .979 at L7.
- **z_1 to z_{L-1}, the latents that carry the chain:** follows donor 1.000 at every position at L3-L6, and .982-.993 at L7. Keeps own .000 everywhere. Mean ablation leaves .024-.063 (chance).
- **z_L to z_5, after the chain (L3: z_3-z_5; L4: z_4, z_5; L5: z_5):** follows donor .000, keeps own 1.000, mean-ablated accuracy 1.000.
- **"Donor answer"** (.00-.15) only counts items where the predicted answer happens to equal the donor's own answer, so it is not a separate route.
- **Route.** The answer reads v_{L-1} from the pass whose input is z_{L-1}, the first pass after the last rule-reading pass. At L6 and L7 that is the last pass, so all of z_1-z_5 are used.

**Interpretation:**
- In the gated model, readable and used coincide one latent at a time for z_1 to z_{L-1}: each latent alone decides the answer from its step onward, and no other route to its value exists.
- Latents after z_{L-1} are idle. Swapping or ablating them changes nothing, yet the linear probe reads v_{L-1} from them at 1.00 (20261107-211610). The logit lens reads values only at the sites in use. So in this model a probe-based monitor would report values the model does not use, while the logit lens tracks use.
- Against ungated start-last CODI, single latents follow the donor .00-.01 there (20261106-105634), because its latents are near copies. Single-latent tests therefore cannot show reliance in that model, but here they can.

**Gotchas hit:** The Mac went to sleep on battery mid-run (pmset: sleep at 13:36 and 13:38), so the run took about 30 minutes instead of about 5. It saves after each level, and results are unaffected.
**Caveats:** One seed, valid only. The swap moves the whole latent, not a value direction, so this does not show which features in z_j carry the value. Pairs are restricted to items the model got right with an in-range predicted answer.
**Next:** On ungated start-last CODI below reach, swap all latents from a different-value donor against mean ablation, to separate content reliance from format damage at L1-L2. On this model, block the route in use and test whether the idle latents can stand in as a backup.
