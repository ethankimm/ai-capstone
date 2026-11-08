## 2026-11-08 — Pause-token control for start-last CODI, canonical L1-L12 (codi, run_id: 20261108-182830_codi_v2-l1to7w-startlast-pause-depth2-codi-canonical)

**Goal:** Secondary out-of-range check for `20261108-182654_codi_v2-l1to7w-startlast-pause-depth2-codi` (section 15.3): the same selected checkpoint-17820 on canonical L1-L12, 500/level, so L8-L12 are beyond what was trained.
**Mechanism / model:** as the main record (`--pause_latents True`, 6 latents, start-last data).
**Command:** run by the arm pipeline after the test eval; `eval_command.txt`.
**Headline results:** `final_answer_accuracy` 0.254 over 6,000 items; see `metrics.extra.per_level_accuracy` for L1-L12. L1-L7 reproduce the main record; L8-L12 (untrained): 0.054, 0.042, 0.030, 0.034, 0.030.
**Interpretation:** Not read yet beyond the main record. L8-L12 are out of range (canonical test only) and not part of the primary.
**Gotchas hit:** none. **Caveats:** one seed, as the main record.
**Next:** Compare L8-L12 with the start-last CODI canonical record `20261106-083950` when the mechanism tests are done.
