## 2026-09-28 — CODI → Coconut through the LADDER-trained DAS subspace, k=16: mapped coordinates reach 59-93% of Coconut's own subspace-only ceiling (cf_joint 0.120-0.158 vs own_sub 0.170-0.220) (coconut, run_id: 20260928-012726_coconut_xmech-subspace-ladder-codi-to-coconut-k16)

**Goal:** k=16 sibling of `20260928-012717` (k=32) — read that record for the full motivation and
design. This record: codi-to-coconut, k=16.
**Design:** identical to `20260928-012717` except k=16 (`rotations_ladder/{codi,coconut}/rot_*_k16.pt`).
**Mechanism / model:** Coconut (target), CODI (source). Rotations: CODI `rot_0+2+4_k16.pt`, Coconut
`rot_1+4_k16.pt` (both from this round's Experiment B, `20260928-005033`/`20260927-234206`).
**Command:** `xmech_coconut.py --mode subspace --codi_rotation rot_0+2+4_k16.pt --coconut_rotation
rot_1+4_k16.pt --slug xmech-subspace-ladder-codi-to-coconut-k16 --stage full_run` (full line in
`eval_command.txt`). Pod `jbzj5cd4diacxb` (RTX A40 secure, $0.49/hr, EU-SE-1); ran alongside 3
sibling subspace runs, no CPU contention from the earlier 8-job batch (already finished).
**Headline results** (259 recipients; round-4 minimal-pair comparator:
`20260927-190704_coconut_xmech-subspace-codi-to-coconut-k16`):

| cf_joint (null ≤0.006) | L2 | L3 | L4 |
|---|---|---|---|
| own_sub (ladder basis, k16) | 0.170 | 0.220 | 0.205 |
| **mapped_sub (ladder basis, k16)** | **0.158** | **0.124** | **0.120** |
| complement_sub (ladder basis, k16) | 0.189 | 0.166 | 0.151 |
| shuffled_sub | 0.012 | 0.008 | 0.008 |
| random_sub | 0.000 | 0.000 | 0.000 |
| mapped_full (unrestricted, this run) | 0.309 | 0.301 | 0.278 |
| mapped_sub, round-4 minimal-pair basis, k16 (`20260927-190704`) | 0.066 | 0.062 | 0.046 |

mapped_sub is more than DOUBLE round 4's minimal-pair-basis number at every level (0.158 vs 0.066,
0.124 vs 0.062, 0.120 vs 0.046) — an even larger relative jump than at k=32. mapped_sub reaches
59-93% of this run's own_sub. Coordinate-map eval R²: pass 1 0.451, pass 4 0.246 (complement input:
0.547/0.286; shuffled ≤-0.01).

**Interpretation:**
- Same conclusion as the k=32 sibling, at a smaller subspace: k=16 already captures most of the
  ladder basis's improvement over the minimal-pair basis (roughly 2x at both k), consistent with
  Experiment B's own finding that k=16 already captures most of the same-mechanism gain.
- complement_sub (0.151-0.189) again matches or exceeds mapped_sub (0.120-0.158) at every level,
  same pattern as k=32 and as round 4 — CODI's value is not confined to its own DAS subspace at
  either k.
**Caveats:** n=259, one donor draw per level. random_sub/shuffled_sub near zero confirm the
improvement is a real basis effect.
**Next:** See `20260928-012717` (k=32, same direction) and the reverse direction
(`20260928-013453`/`-013447`) for the complete picture.
