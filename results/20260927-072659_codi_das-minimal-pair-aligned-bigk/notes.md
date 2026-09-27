## 2026-09-27 — E4 large-k extension, CODI: DAS training makes steering WORSE than an untrained rotation at k=512 (0.22 vs 0.71) — the DAS nulls are an optimization failure, not evidence (codi, run_id: 20260927-072659_codi_das-minimal-pair-aligned-bigk)

**Goal:** Extend `20260927-070156_codi_das-minimal-pair-aligned` (k ≤ 64, ≤6%) to k ∈ {128, 256, 512} on
the z0+z2+z4 group, matching the range of Coconut's saturation sweep (`20260921-195626`, k up to 256).
**Mechanism / model / data:** as the parent run (same train/eval pools, seeds, lr 1e-3, 5 epochs, 200 train
/ 100 eval pairs). Command in `eval_command.txt`. Same pod (RTX A6000 secure), ~22 min alone.
**Headline results** (n=100, matches_twin / answer_changed):

| k | untrained random R | trained R |
|---|---|---|
| 128 | 0.00 / 0.02 | 0.08 / 0.55 |
| 256 | 0.11 / 0.18 | 0.15 / 0.47 |
| 512 | **0.71** / 0.85 | **0.22** / 0.59 |
| 768 (raw swap) | 0.84 / 0.95 | — |

**Interpretation:**
- At k=512 training cuts steering from 71% (random rotation) to 22%. An objective that rewards the
  twin's answer should never do worse than its random initialisation on the same pairs, so the training
  recipe is failing (lr/epochs/objective — the teacher-forced CE is still ~10 nats), not finding "no
  subspace". A random 512-dim subspace already carries most of the value (71 of 84 points).
- Consequence: **every DAS null so far (CODI pilot, CODI E4 old + aligned, Coconut E4 + saturation sweep)
  used this same training recipe and should not be cited as evidence against a linear subspace** until the
  recipe is fixed and shown to beat its untrained reference.
**Gotchas hit:** Only visible because this run added untrained references; the earlier DAS runs had none.
**Caveats:** n=100, one seed.
**Next:** Debug DAS training before any more DAS claims: lower lr (1e-4), more epochs, check that the
trained R beats the untrained one at k=512 on the TRAIN pairs first, and consider an interchange loss on
the steered token only. Re-run CODI and Coconut E4 with the fixed recipe and the untrained references.
