## 2026-09-28 — CODI → Coconut through the LADDER-trained DAS subspace, k=32: mapped coordinates now reach 68-104% of Coconut's own subspace-only ceiling (cf_joint 0.166-0.197 vs own_sub 0.189-0.243), roughly double round 4's minimal-pair-basis numbers (coconut, run_id: 20260928-012717_coconut_xmech-subspace-ladder-codi-to-coconut-k32)

**Goal:** Follow-up to Experiment B (`20260927-234206_coconut_das-ladder`,
`20260928-005033_codi_das-ladder`): both mechanisms' DAS rotations trained on CROSS-PROBLEM ladder
donors reached ~58-70% of the whole-vector cross-problem ceiling (own-mechanism), vs round 4's
minimal-pair-trained rotations at 28-53% — meeting the pre-registered "approaches ceiling" branch
for rerunning the CROSS-MECHANISM subspace transplant (`run_subspace_transplant`,
round 4's `20260927-190704/-190716`) with the new rotations instead.
**Design** (`xmech_common.run_subspace_transplant`, unchanged from round 4; `xmech_coconut.py
--mode subspace`): identical to round 4's codi-to-coconut subspace runs, except `--codi_rotation`
and `--coconut_rotation` point at the LADDER-trained rotations
(`rotations_ladder/{codi,coconut}/rot_*_k32.pt`, from this round's Experiment B) instead of the
minimal-pair ones. Same 259 recipients, same conditions: **own_sub** (Coconut's own donor
coordinates, cross-problem DAS interchange — ceiling for a subspace-only patch with THIS basis);
**mapped_sub** (ridge from CODI donor's ladder-DAS coordinates at z0,z2,z4 → Coconut's ladder-DAS
coordinates at passes 1,4); **shuffled_sub** (same ridge, permuted pairs); **complement_sub** (ridge
from CODI's coordinates OUTSIDE its ladder-DAS subspace); **random_sub** (own donor coordinates in
a random k-dim subspace); **mapped_full** (the unrestricted P4 condition, re-run here for a
same-recipient comparison).
**Mechanism / model:** Coconut `hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33` (target), CODI
`hf:zen-E/CODI-gpt2@fd641b3` (source). Rotations: CODI `rot_0+2+4_k32.pt` (from
`20260928-005033_codi_das-ladder`), Coconut `rot_1+4_k32.pt` (from
`20260927-234206_coconut_das-ladder`).
**Command:** `xmech_coconut.py --checkpoint_path .../checkpoint_33 --mode subspace --codi_rotation
rotations_ladder/codi/rot_0+2+4_k32.pt --coconut_rotation rotations_ladder/coconut/rot_1+4_k32.pt
--slug xmech-subspace-ladder-codi-to-coconut-k32 --stage full_run` (full line in
`eval_command.txt`). Pod `jbzj5cd4diacxb` (RTX A40 secure, $0.49/hr, EU-SE-1); ran alongside 3
sibling subspace runs (this experiment's k16, and both directions' k16/k32) once the 8
Experiment-A/B jobs had finished, so no CPU contention this time (~15 min total).
**Headline results** (259 recipients; sibling at k=16: `20260928-012726`; round-4 minimal-pair
comparator: `20260927-190716_coconut_xmech-subspace-codi-to-coconut-k32`):

| cf_joint (null ≤0.006) | L2 | L3 | L4 |
|---|---|---|---|
| own_sub (ladder basis, k32) | 0.189 | 0.220 | 0.243 |
| **mapped_sub (ladder basis, k32)** | **0.197** | **0.189** | **0.166** |
| complement_sub (ladder basis, k32) | 0.224 | 0.208 | 0.185 |
| shuffled_sub | 0.008 | 0.000 | 0.000 |
| random_sub | 0.000 | 0.000 | 0.000 |
| mapped_full (unrestricted, this run) | 0.309 | 0.301 | 0.278 |
| mapped_sub, round-4 minimal-pair basis, k32 (`20260927-190716`) | 0.085 | 0.108 | 0.069 |

mapped_sub roughly DOUBLES round 4's minimal-pair-basis number at every level (0.197 vs 0.085,
0.189 vs 0.108, 0.166 vs 0.069). mapped_sub reaches 68-104% of THIS run's own_sub (vs round 4's
55-90% of ITS own_sub — a similar or slightly better relative efficiency, on a much higher absolute
base). self_mapped_sub (Coconut recipient's own coordinates mapped back into itself) unchanged
89.2%. Coordinate-map eval R²: pass 1 0.451, pass 4 0.255 (complement input: 0.499/0.263; shuffled
≤-0.01) — close to round 4's minimal-pair-basis R² (0.34-0.47 / 0.18-0.24), i.e. the ladder basis's
gain is NOT primarily from a better-fitting ridge map; it is from the DAS subspace itself
generalizing better to cross-problem donors (see the own_sub jump, which needs no cross-mechanism
map at all).
**Interpretation:**
- **Training the DAS rotation on cross-problem donors, not just improves the same-mechanism
  subspace-only ceiling (Experiment B) — it roughly doubles the CROSS-MECHANISM subspace-restricted
  transfer too**, in absolute cf_joint terms (0.166-0.197 vs 0.069-0.085). The value channel that
  crosses mechanisms was always there (round 3/4 found the unrestricted, whole-vector version); this
  confirms it is genuinely the SAME kind of information the within-problem DAS subspace was
  supposed to isolate, just requiring a cross-problem-trained subspace to access reliably at k=32.
- mapped_sub now essentially MATCHES or exceeds own_sub at L2 (104%) and is close at L3 (86%),
  vs round 4's 82%/90% for the SAME levels with the old basis — i.e. relative efficiency
  (mapped/own) is similar-to-better, and the whole scale moved up together.
- The input restriction still doesn't matter: mapping from CODI's non-DAS (complement) coordinates
  works AS WELL AS or better than mapping from its DAS coordinates (complement_sub 0.185-0.224 >
  mapped_sub 0.166-0.197 at every level) — replicating round 4's finding
  (`20260927-185312`: CODI stores the value redundantly across the vector) with the new basis.
**Caveats:** n=259, one donor draw per level. random_sub/shuffled_sub near zero throughout confirm
the improvement is a real basis effect, not a general loosening of the null.
**Next:** See sibling k=16 (`20260928-012726`) and the reverse direction
(`20260928-012726`... `20260928-013447/-013453`) for the full 2-direction x 2-k picture. This
closes out the round-4 "wrong basis" open problem: a cross-problem-trained DAS subspace
substantially narrows (though does not fully close) the gap to whole-vector transfer, in both the
same-mechanism and cross-mechanism settings.
