## 2026-09-28 — Coconut → CODI through the LADDER-trained DAS subspace, k=16: mapped coordinates reach 20-50% of CODI's own subspace-only ceiling (cf_joint 0.058-0.077 vs own_sub 0.116-0.293), 22-97% above round 4's minimal-pair basis (codi, run_id: 20260928-013453_codi_xmech-subspace-ladder-coconut-to-codi-k16)

**Goal:** k=16 sibling of `20260928-013447` (k=32) — read that record for the full motivation and
design. This record: coconut-to-codi, k=16.
**Design:** identical to `20260928-013447` except k=16 (`rotations_ladder/{codi,coconut}/rot_*_k16.pt`).
**Mechanism / model:** CODI (target, aligned sites), Coconut (source). Rotations: CODI
`rot_0+2+4_k16.pt`, Coconut `rot_1+4_k16.pt` (both from this round's Experiment B).
**Command:** `xmech_codi.py $CODI_FLAGS --mode subspace --codi_rotation rot_0+2+4_k16.pt
--coconut_rotation rot_1+4_k16.pt --slug xmech-subspace-ladder-coconut-to-codi-k16 --stage
full_run` (full line in `eval_command.txt`). Pod `jbzj5cd4diacxb` (RTX A40 secure, $0.49/hr,
EU-SE-1); ran alongside 3 sibling subspace runs, no CPU contention.
**Headline results** (259 recipients; round-4 minimal-pair comparator:
`20260927-191525_codi_xmech-subspace-coconut-to-codi-k16`):

| cf_joint (null ≤0.012) | L2 | L3 | L4 |
|---|---|---|---|
| own_sub (ladder basis, k16) | 0.224 | 0.293 | 0.116 |
| **mapped_sub (ladder basis, k16)** | **0.077** | **0.062** | **0.058** |
| complement_sub (ladder basis, k16) | 0.120 | 0.100 | 0.077 |
| shuffled_sub | 0.008 | 0.012 | 0.004 |
| random_sub | 0.000 | 0.000 | 0.000 |
| mapped_full (unrestricted, this run) | 0.259 | 0.278 | 0.270 |
| mapped_sub, round-4 minimal-pair basis, k16 (`20260927-191525`) | 0.015 | 0.031 | 0.031 |

mapped_sub improves substantially over round 4's minimal-pair-basis number at every level (0.077 vs
0.015, +413%; 0.062 vs 0.031, +100%; 0.058 vs 0.031, +87% — the k=16 gain is even larger in relative
terms than k=32's, though both k remain well below own_sub). mapped_sub reaches 20-50% of this
run's own_sub. Coordinate-map eval R²: z0 0.491, z2 0.327, z4 0.279 (complement input:
0.596/0.407/0.343; shuffled ≤-0.02).
**Interpretation:**
- Same direction asymmetry as k=32 (`20260928-013447`): Coconut's value subspace is easier to write
  a cross-mechanism value into than CODI's, at either k tested.
- The relative improvement over round 4's minimal-pair basis is LARGER at k=16 than k=32 in this
  direction (up to +413% vs +77%) because round 4's minimal-pair-basis k16 number
  (`20260927-191525`, 0.015) was unusually low to begin with — closer to its own shuffled-pair floor
  (0.000-0.015) than to a real signal. The ladder basis turns k=16 from "barely above noise" into a
  small but clearly real effect (0.058-0.077, vs shuffled 0.004-0.012).
**Caveats:** n=259, one donor draw per level.
**Next:** This completes the round-4 "wrong basis" follow-up: 4 subspace-ladder runs (2 directions
x k∈{16,32}), all logged. Combined with Experiment B's own-mechanism result and Experiment A's
question-only control, this is a full 12-run round; see `MEMORY.md`/results index for the summary.
