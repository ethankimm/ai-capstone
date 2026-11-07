## 2026-11-07 — Rule-gated CODI, idle latents are not a backup: hiding the pass the answer reads leaves .08-.21, hiding everything but the idle passes leaves chance, though a probe reads v_{L-1} from them at 1.00 (codi, run_id: 20261107-220408_codi_v2-l1to7w-startlast-gated-idle-backup)

**Goal:** Idle latents (after z_{L-1}) are ignored under swap and mean ablation (20261107-214950), yet a linear probe reads v_{L-1} from them at 1.00 (20261107-211610). Are they a working backup the model can fall back on, or inert?
**Mechanism / model:** `codi`, gpt2. Rule-gated start-last CODI checkpoint-17820, gate on. fp32, MPS.
**Data:** Valid L3-L5 (levels with idle passes), 286 per level.
**Command:** `codi_audit_artifacts/b30_gated_backup.py` under `run_guarded.sh` and `caffeinate -dims`; record from `record8.py`. Only the answer tokens' attention to latent passes changes.
**Headline results (exact, L3 / L4 / L5):**
- Clean: 1.000 / 1.000 / 1.000.
- Hide the pass the answer reads (input z_{L-1}): .206 / .108 / .084.
- Hide passes 0 to L-1, leaving only the idle passes: .038 / .042 / .028.
- Hide only the idle passes (control): 1.000 / 1.000 / 1.000.
- Hide every latent pass (floor): .028 / .021 / .010.

**Interpretation:**
- The idle passes are inert. With every other route hidden, the answer cannot use them, so a probe's 1.00 read there is a value the model neither uses nor can fall back on. For a monitor this is a pure false alarm. The logit lens does not read these sites.
- The residual .08-.21 after hiding the pass in use most likely comes from the pass before it, which computed v_{L-1} in its upper layers (inferred, not tested).

**Gotchas hit:** None.
**Caveats:** One seed, valid only. Blocking attention removes the route entirely; this does not test whether brief retraining would make the idle latents usable.
**Next:** None for this model.
