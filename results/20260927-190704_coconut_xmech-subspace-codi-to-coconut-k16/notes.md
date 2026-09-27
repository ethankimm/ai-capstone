## 2026-09-27 — CODI → Coconut through the DAS value subspace: mapped coordinates carry the donor's values at 55–90% of Coconut's own subspace-only ceiling (cf_joint 0.05–0.07 vs own 0.09–0.11; shuffled/random 0), but subspace-only patches are ~3× weaker than whole vectors (coconut, run_id: 20260927-190704_coconut_xmech-subspace-codi-to-coconut-k16)

**Goal:** Round-3 P4 caveat: the unrestricted CODI↔Coconut map reads all six source latents and writes whole
target vectors, so it could route question-derived information rather than the computed values. Restrict
both ends to the DAS value subspaces. This record: codi-to-coconut, k=16; sibling at the other k: `20260927-190716_coconut_xmech-subspace-codi-to-coconut-k32`.
**Design** (`xmech_common.run_subspace_transplant`; `--mode subspace` of `xmech_codi.py` / `xmech_coconut.py`):
same eval-split ladder recipients/donors as the unrestricted P4 runs (259 recipients, base-correct in both
models; L2 same ops, L3 same length, L4 any), same dumps (re-made on this pod: accuracy CODI 0.419, Coconut
0.331 on eval, identical to round 3). Rotations from `20260927-184022` (CODI z0+z2+z4) and `20260927-182915`
(Coconut passes 1+4). Every `*_sub` condition writes k coordinates into the TARGET's DAS subspace at its
carrier sites and keeps the recipient's own live complement (a cross-problem DAS interchange):
**own_sub** = the target's own donor coordinates (ceiling for a subspace-only patch); **mapped_sub** = ridge
from the source donor's DAS coordinates at its carriers → target coordinates (fit on the 8000-question fit
split); **shuffled_sub** = same ridge on permuted pairs; **complement_sub** = ridge from the source donor's
coordinates *outside* its DAS subspace; **random_sub** = own donor coordinates in a random k-dim subspace;
**mapped_full** = the unrestricted round-3 condition (map of all 6 source sites → whole carrier vectors),
re-run on the same recipients. Scoring as P1 (cf_joint = recipient's program on the donor's step values;
permutation null in brackets).
**Mechanism / model:** Coconut `hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33` (target) and CODI `hf:zen-E/CODI-gpt2@fd641b3` (source).
**Command:** `xmech_coconut.py --mode subspace --checkpoint_path .../checkpoint_33 --questions xmech_questions.jsonl --codi_latents codi_latents.pt --out_latents coconut_latents.pt --codi_rotation rot_0+2+4_k16.pt --coconut_rotation rot_1+4_k16.pt --slug xmech-subspace-codi-to-coconut-k16 --stage full_run` (full line in `eval_command.txt`). Pod `ft7u4einf6ci9s` (RTX A5000 secure, $0.27/hr);
the four subspace transplants ran in parallel, ~50 min, most of it ridge fitting on CPU.
**Headline results** (259 recipients; both k in one table, this record's k=16):

| cf_joint (null ≤0.006) | L2 | L3 | L4 |
|---|---|---|---|
| own_sub k16 / k32 | 0.085 / 0.104 | 0.108 / 0.120 | 0.085 / 0.120 |
| **mapped_sub** k16 / k32 | **0.066 / 0.085** | **0.062 / 0.108** | **0.046 / 0.069** |
| complement_sub k16 / k32 | 0.073 / 0.089 | 0.073 / 0.116 | 0.050 / 0.077 |
| shuffled_sub k16 / k32 | 0.000 / 0.008 | 0.000 / 0.004 | 0.000 / 0.000 |
| random_sub k16 / k32 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| mapped_full (unrestricted) | 0.309 | 0.301 | 0.278 |

random_sub leaves 98–100% of answers unchanged; shuffled_sub leaves ~63%. self_mapped_sub (CODI recipient's own
coordinates mapped back into it) unchanged 91% (k16) / 92% (k32). Coordinate-map eval R²: pass 1 0.34 / 0.40,
pass 4 0.18 / 0.22 (complement input: 0.47 / 0.24; shuffled ≤0).

**Interpretation:**
- **Subspace-only patches transfer much less across problems than whole vectors, even within Coconut.**
  own_sub (Coconut's own donor, subspace only) reaches 0.09–0.12 cf_joint vs ~0.30 for whole-vector own
  carriers (round 3). DAS was trained on minimal pairs where the complement barely differs; across problems,
  the recipient's own complement pulls back toward its own values.
- **Within that ceiling, CODI's values are linearly translatable into Coconut's value subspace**: mapped_sub
  reaches 55–90% of own_sub (e.g. L3 k32 0.108 vs 0.120), against 0–0.8% for the shuffled map and 0% for a
  random subspace. So the pure value channel crosses mechanisms, not just question-derived context.
- The input restriction does not matter: mapping from CODI's non-DAS coordinates works as well
  (complement_sub ≈ mapped_sub). Consistent with `20260927-185312`: CODI stores the value redundantly
  across the vector, so its DAS subspace is not the only place a map can read it from.
- The unrestricted map's 0.28–0.31 (replicating round 3's 0.24–0.32) comes mostly from writing whole
  vectors. This run cannot separate "other question-derived information" from "a stronger intervention":
  own_sub, the equally restricted ceiling, is also 3× below own whole-vector.
**Caveats:** n=259, one donor draw per level (±~3 pp at 0.1). The subspace patch keeps the recipient's
*live* complement at later carriers (it reflects earlier patches), as in DAS training. The DAS subspaces were
trained on within-problem minimal pairs, which may not be the right basis for cross-problem transfer.
**Next:** Train the rotation on cross-problem (ladder) pairs instead of minimal pairs and see whether
own_sub approaches whole-vector transfer; if it does, rerun this restriction with that basis.
