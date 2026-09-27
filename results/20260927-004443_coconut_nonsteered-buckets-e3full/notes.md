## 2026-09-27 — Where non-steered answers go, Coconut E3 at n=231: same multi-use operand failure, concentrated on pass 1 (coconut, run_id: 20260927-004443_coconut_nonsteered-buckets-e3full)

**Goal:** Repeat the bucket analysis (`20260926-235222_coconut_nonsteered-buckets`) on the full-n E3 rerun
`20260927-003331_coconut_minimal-pair-patch-full`. E2 tables unchanged from the earlier record.
**Command:** `uv run python scripts/analyze_nonsteered.py --mechanism coconut --e3-run 20260927-003331_coconut_minimal-pair-patch-full --slug nonsteered-buckets-e3full --reps 200`
**Headline results:** all-slot moved-not-steered n=16, partial_propagation 38% (null 3%).

| condition | operand used in ≥2 steps (n=46) | used once (n=185) |
|---|---|---|
| all-slot: steered / partial / unchanged | 0.59 / 0.13 / 0.17 | **0.89** / — / 0.09 |
| pass 1 alone: steered / partial | **0.07 / 0.48** | 0.68 / — |
| pass 4 alone: steered / partial | 0.37 / 0.17 | 0.33 / — |
| passes 0, 2, 3 alone | ≥0.98 unchanged | ≥0.95 unchanged |

**Interpretation:** Same pattern as CODI (`20260927-004433`): 89% steering for single-use operands vs 59%
for multi-use ones. Pass 1 alone steers 68% of single-use pairs but only 7% of multi-use pairs, where it
yields the half-propagated answer 48% of the time. Pass 1 carries one use of the number (the step it
computes), not the number as a variable.
**Caveats:** multi-use subset n=46; first-match buckets.
