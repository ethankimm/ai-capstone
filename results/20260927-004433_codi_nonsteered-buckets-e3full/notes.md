## 2026-09-27 — Where non-steered answers go, aligned CODI E3 at n=317: single-use operands steer 92%, multi-use operands half-propagate (codi, run_id: 20260927-004433_codi_nonsteered-buckets-e3full)

**Goal:** Repeat the bucket analysis (`20260926-235222_codi_nonsteered-buckets`) on the aligned, full-n E3
rerun `20260927-004340_codi_minimal-pair-patch-aligned`. The E2 (level-4) tables in this record are the
same as in the earlier record (same E2 source, shifted-indexing caveat).
**Command:** `uv run python scripts/analyze_nonsteered.py --mechanism codi --e3-run 20260927-004340_codi_minimal-pair-patch-aligned --slug nonsteered-buckets-e3full --reps 200`
**Headline results:** all-slot (aligned) moved-not-steered answers: n=36, **partial_propagation 64%**
(permutation null 3%).

| condition | operand used in ≥2 steps (n=60) | used once (n=257) |
|---|---|---|
| all-slot: steered / partial / unchanged | 0.50 / **0.38** / 0.03 | **0.92** / — / 0.05 |
| z0 alone: steered / partial | 0.05 / **0.50** | 0.23 / — |
| z2 alone: steered / partial | 0.07 / 0.23 | 0.23 / — |
| z4 alone: steered / partial | 0.28 / 0.13 | 0.38 / — |
| z1, z3, z5 alone | ≥0.98 unchanged | ≥0.99 unchanged |

**Interpretation:** Confirms the n=137 finding on aligned sites and 2.3× the pairs: when the perturbed
number enters one step, transplanting the twin's latents steers 92% of the time; when it enters two or
more steps, only 50%, and the dominant failure (38%) is the recipient chain with the new number used in
some steps and the old number in others. z0 alone produces exactly that half-propagated answer on half
the multi-use pairs. The latents carry the value of the step they compute; other uses of the operand
are read from the question tokens. That is the measured form of "context-entangled".
**Caveats:** first-match buckets; "other" remains the largest bucket for single-site non-steered answers.
