## 2026-09-27 — Coconut → CODI through the k=32 DAS value subspace: mapped cf_joint 0.04–0.06 vs own subspace 0.10–0.21 (shuffled ≤0.015, random 0) (codi, run_id: 20260927-191455_codi_xmech-subspace-coconut-to-codi-k32)

**Goal:** Round-3 P4 caveat: the unrestricted CODI↔Coconut map reads all six source latents and writes whole
target vectors, so it could route question-derived information rather than the computed values. Restrict
both ends to the DAS value subspaces. This record: coconut-to-codi, k=32; sibling at the other k: `20260927-191525_codi_xmech-subspace-coconut-to-codi-k16`.
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
**Mechanism / model:** CODI `hf:zen-E/CODI-gpt2@fd641b3` (target) and Coconut `hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33` (source).
**Command:** `xmech_codi.py $CODI_FLAGS --mode subspace --questions ... --codi_latents ... --coconut_latents ... --codi_rotation rot_0+2+4_k32.pt --coconut_rotation rot_1+4_k32.pt --slug xmech-subspace-coconut-to-codi-k32 --stage full_run` (full line in `eval_command.txt`). Pod `ft7u4einf6ci9s` (RTX A5000 secure, $0.27/hr);
the four subspace transplants ran in parallel, ~50 min, most of it ridge fitting on CPU.
**Headline results** (259 recipients; both k in one table, this record's k=32):

| cf_joint (null ≤0.006) | L2 | L3 | L4 |
|---|---|---|---|
| own_sub k16 / k32 | 0.120 / 0.178 | 0.135 / 0.208 | 0.073 / 0.100 |
| **mapped_sub** k16 / k32 | **0.015 / 0.039** | **0.031 / 0.050** | **0.031 / 0.062** |
| complement_sub k16 / k32 | 0.031 / 0.073 | 0.046 / 0.073 | 0.046 / 0.081 |
| shuffled_sub k16 / k32 | 0.000 / 0.004 | 0.008 / 0.012 | 0.000 / 0.015 |
| random_sub k16 / k32 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| mapped_full (unrestricted) | 0.263 | 0.282 | 0.266 |

random_sub leaves 98–100% unchanged; shuffled_sub ~47–63%. self_mapped_sub unchanged 86% (k16) / 79% (k32).
Coordinate-map eval R²: z0 0.38 / 0.41, z2 0.21 / 0.22, z4 0.16 / 0.16 (complement input: 0.54 / 0.30 / 0.23).

**Interpretation:**
- **Coconut → CODI through the value subspace is weak.** mapped_sub gives 0.015–0.06 cf_joint, 12–62% of
  CODI's own subspace-only ceiling (own_sub 0.07–0.21), above the shuffled map (≤0.015) and random subspace (0)
  but small. With whole vectors the same direction gives 0.26–0.28.
- CODI's own subspace-only interchange carries a cross-problem donor's values at 0.07–0.21 (k=32 > k=16),
  i.e. again a fraction of whole-vector own carriers (round 2/3: 0.20–0.39).
- Coconut's non-DAS coordinates predict CODI's DAS coordinates *better* than Coconut's DAS coordinates do
  (complement_sub 0.03–0.08 > mapped_sub; R² z0 0.54 vs 0.38). Coconut's 16/32 DAS dims hold the value
  (`20260927-185326`) but not everything CODI's value subspace encodes.
- The direction asymmetry (into Coconut works better) matches the probe results: Coconut's value subspace is
  compact and localized, so it is an easy target to write to; CODI's is not.
**Caveats:** n=259, one donor draw per level (±~3 pp at 0.1). The subspace patch keeps the recipient's
*live* complement at later carriers (it reflects earlier patches), as in DAS training. The DAS subspaces were
trained on within-problem minimal pairs, which may not be the right basis for cross-problem transfer.
**Next:** Train the rotation on cross-problem (ladder) pairs instead of minimal pairs and see whether
own_sub approaches whole-vector transfer; if it does, rerun this restriction with that basis.
