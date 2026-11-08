## 2026-11-08 — ProsQA grid need test: Coconut scores .980 with 0 latent passes and with 6, in every path-length bin; CODI .820 vs .816. Latent passes add nothing; Coconut's lead over no-CoT (.760) comes from training, not from the passes (coconut, run_id: 20261108-083818_coconut_prosqa-grid-need-by-path-length)

**Goal:** Necessity framing, step 3/4a (HANDOFF_2026-11-08_NECESSITY.md): do the grid's latent models need their latent passes on ProsQA, and does any need grow with path length beyond what a single pass reaches? Pre-registered in `scripts/codi_null_audit/prereg/t_predictions.md` (t16, committed 3574431 before the full run).
**Mechanism / model:** Dilgren & Wiegreffe GPT-2 grid: `connordilgren/gpt2-prosqa-{coconut,codi,no-cot,cot}`, all from the same training recipe family. Coconut run with k = 0..6 latent passes; CODI with 6 and 0 (question + <bot> + <eot>).
**Data:** ProsQA test, all 500, binned by gold path length (len(steps): 3 / 4 / 5 / 6 = 202 / 217 / 69 / 12).
**Hyperparams:** batch 1, fp32, greedy, MPS; answers exact match after lower-casing (CODI after "The answer is:", others after "###").
**Command:** `scripts/necessity/grid_accuracy.py prosqa <method> mps <out>` through `codi_audit_artifacts/run_guarded.sh` (chain `n1_chain.sh`); raw JSON `codi_audit_artifacts/t16_prosqa_*.json`.
**Results:**

| model | all | 3 | 4 | 5 | 6 (n=12) |
|---|---|---|---|---|---|
| Coconut, 6 latents | .980 | .97 | .99 | .97 | 1.00 |
| Coconut, 0 latents | .980 | .97 | .99 | .97 | 1.00 |
| CODI, 6 latents | .816 | .77 | .85 | .84 | .75 |
| CODI, 0 latents | .820 | .77 | .86 | .83 | .83 |
| no-CoT | .760 | .75 | .77 | .71 | .92 |
| CoT | .742 | .81 | .73 | .58 | .75 |

Coconut is identical at every k = 0..6 (.980). Need (6 minus 0 latents) = .000 for Coconut, -.004 for CODI.
**Interpretation:** Matches the prediction. The published Coconut-over-CoT gap (here .98 vs .74 / .76) is real in this grid, but the Coconut model gets it without running any latent pass, so it is a training effect (the curriculum teaches the question-reading pass to answer), not latent serial depth. Accuracy does not fall with path length for any model, so ProsQA at these lengths sits within single-pass reach. Prediction for the check: both latent models fail (t17).
**Gotchas hit:** The grid checkpoints used in October lived in a vanished scratchpad; re-downloaded to `~/Documents/Penn/CIS5980/grid_checkpoints`. CODI numbers reproduce 20261103-021023 exactly (.816 / .820).
**Caveats:** 0 latents is an off-distribution truncation for both latent models (it still scores the same here). One checkpoint per method. Bin 6 has 12 items. CoT's low score is mostly early commitment on distractor branches (not analysed here).
**Next:** t17 route split on minimal pairs (expected all ~0); t18 probes for path nodes and BFS frontiers.
