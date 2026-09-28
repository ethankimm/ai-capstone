## 2026-09-28 — Coconut → CODI through the LADDER-trained DAS subspace, k=32: mapped coordinates reach 26-72% of CODI's own subspace-only ceiling (cf_joint 0.069-0.089 vs own_sub 0.124-0.270), 40-77% above round 4's minimal-pair basis (codi, run_id: 20260928-013447_codi_xmech-subspace-ladder-coconut-to-codi-k32)

**Goal:** Reverse direction of `20260928-012717` — read that record for the full motivation. This
record: coconut-to-codi, k=32.
**Design** (`xmech_common.run_subspace_transplant`; `xmech_codi.py --mode subspace`): identical to
round 4's coconut-to-codi subspace runs, except the rotations are the LADDER-trained ones
(`rotations_ladder/{codi,coconut}/rot_*_k32.pt`) instead of the minimal-pair ones. Same 259
recipients, same conditions (own_sub / mapped_sub / shuffled_sub / complement_sub / random_sub /
mapped_full).
**Mechanism / model:** CODI `hf:zen-E/CODI-gpt2@fd641b3` (target, aligned sites), Coconut
`hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33` (source). Rotations: CODI `rot_0+2+4_k32.pt`,
Coconut `rot_1+4_k32.pt` (both from this round's Experiment B).
**Command:** `xmech_codi.py $CODI_FLAGS --mode subspace --codi_rotation rot_0+2+4_k32.pt
--coconut_rotation rot_1+4_k32.pt --slug xmech-subspace-ladder-coconut-to-codi-k32 --stage
full_run` (full line in `eval_command.txt`). Pod `jbzj5cd4diacxb` (RTX A40 secure, $0.49/hr,
EU-SE-1); ran alongside 3 sibling subspace runs, no CPU contention.
**Headline results** (259 recipients; sibling at k=16: `20260928-013453`; round-4 minimal-pair
comparator: `20260927-191455_codi_xmech-subspace-coconut-to-codi-k32`):

| cf_joint (null ≤0.009) | L2 | L3 | L4 |
|---|---|---|---|
| own_sub (ladder basis, k32) | 0.270 | 0.263 | 0.124 |
| **mapped_sub (ladder basis, k32)** | **0.069** | **0.081** | **0.089** |
| complement_sub (ladder basis, k32) | 0.112 | 0.112 | 0.108 |
| shuffled_sub | 0.019 | 0.015 | 0.004 |
| random_sub | 0.000 | 0.000 | 0.000 |
| mapped_full (unrestricted, this run) | 0.259 | 0.278 | 0.270 |
| mapped_sub, round-4 minimal-pair basis, k32 (`20260927-191455`) | 0.039 | 0.050 | 0.062 |

mapped_sub improves over round 4's minimal-pair-basis number at every level (0.069 vs 0.039, +77%;
0.081 vs 0.050, +62%; 0.089 vs 0.062, +44%) — a real but smaller relative gain than the codi-to-coconut
direction (`20260928-012717`, roughly a doubling), consistent with Experiment B's own asymmetry
(CODI's ladder-basis own_sub gain over minimal-pair was smaller than Coconut's). mapped_sub reaches
26-72% of this run's own_sub. Coordinate-map eval R²: z0 0.471, z2 0.304, z4 0.254 (complement
input: 0.557/0.362/0.303; shuffled ≤-0.02) — again close to round 4's minimal-pair-basis R², so the
gain is from the subspace generalizing better, not from the ridge map fitting better.
**Interpretation:**
- **The direction asymmetry from round 4 persists with the better basis**: into Coconut, mapped_sub
  is close to (or above) own_sub; into CODI, mapped_sub tops out around 26-72% of own_sub even with
  the improved basis. This matches the probe-based explanation carried over from round 4: Coconut's
  value subspace is compact and easy to write into; CODI's is not (it's redundant across the whole
  vector), so a value written only into CODI's k-dim subspace competes with everything else in the
  live vector at that site.
- Coconut's non-DAS (complement) coordinates again predict CODI's DAS coordinates BETTER than
  Coconut's own DAS coordinates do (complement_sub 0.108-0.112 > mapped_sub 0.069-0.089 at every
  level; R² 0.557 vs 0.471 at z0) — same pattern as round 4, replicated with the new basis.
**Caveats:** n=259, one donor draw per level. own_sub's L4 (0.124) is notably lower than L2/L3
(0.270/0.263) here, same pattern seen in Experiment B's own CODI run (`20260928-005033`) — L4's
random, most-different donor is harder for CODI's subspace-only patch specifically.
**Next:** See sibling k=16 (`20260928-013453`) and the forward direction
(`20260928-012717`/`-012726`). Together these four runs close out the round-4 "wrong basis" open
problem for both directions and both tested k.
