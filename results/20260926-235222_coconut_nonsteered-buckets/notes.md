## 2026-09-26 — Where non-steered patched answers go, Coconut: pass 1 carries one use of the perturbed number, not the number (coconut, run_id: 20260926-235222_coconut_nonsteered-buckets)

**Goal:** As for CODI (`20260926-235222_codi_nonsteered-buckets`): bucket every patched answer of
Coconut E3 (`20260920-195725`) and E2 (`20260920-085317`) by which chain it is consistent with, against
a permutation null. RESEARCH_PLAN round 1, item 3.
**Mechanism / model:** `coconut`, released checkpoint_33. No model run.
**Command:** `uv run python scripts/analyze_nonsteered.py --reps 200`.
**Headline results:**

E3 all-slot (n=105): 77.1% steered, 13.3% unchanged; only 10 moved-not-steered answers
(partial_propagation 2, wrong_step 1, donor_intermediate 1, off_by_delta 1, other 5).
E3 **single-slot pass 1** (the pass that carries 48.6% alone): of 23 moved-not-steered answers,
**partial_propagation 52% (null 2%)**.

Split by whether the perturbed number is used in ≥2 steps:

| condition | subset | n | steered | partial_prop. | unchanged | other |
|---|---|---|---|---|---|---|
| all-slot | ≥2 uses | 29 | 0.52 | 0.07 | 0.21 | 0.14 |
| all-slot | 1 use | 76 | **0.87** | — | 0.11 | 0.01 |
| pass 1 only | ≥2 uses | 29 | **0.07** | **0.41** | 0.21 | 0.31 |
| pass 1 only | 1 use | 76 | 0.64 | — | 0.33 | 0.01 |
| pass 4 only | ≥2 uses | 29 | 0.38 | 0.14 | 0.41 | 0.07 |
| pass 4 only | 1 use | 76 | 0.25 | — | 0.72 | 0.01 |

E2 (level 4, 282 pairs): real donor 34% unchanged, 1.4% counterfactual; moved answers 93% other,
donor_intermediate 2% (null 1.6%), recipient_intermediate 5% (null 2%). The random donor and mean
ablation look the same (recipient_intermediate 9% / 12%). No donor-specific signal.

**Interpretation:** Pass 1's "48.6% alone" is really two numbers: 64% when the perturbed operand is
used once, 7% when it is used twice, where pass 1 instead yields the half-propagated answer 41% of the
time. So pass 1 carries the value of the step it computes, and the other use of the operand still
comes from the question text. That localizes and qualifies the Coconut "single addressable slot"
reading: addressable per step, not per variable. Even all-slot drops from 87% to 52% on multi-use
operands. Level 4 moves are disruption, not transmission, as for CODI.
**Caveats:** small multi-use subset (29); only 10 moved-not-steered all-slot answers. Buckets are
first-match; `other` is still the largest non-steered bucket for single-slot pass 1 on multi-use pairs (31%).
**Next:** repeat on the full-n E3 rerun (`patch_minimal_pair_coconut.py`, 2026-09-26).
