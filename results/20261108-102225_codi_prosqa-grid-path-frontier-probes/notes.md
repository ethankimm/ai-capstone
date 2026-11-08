## 2026-11-08 — ProsQA CODI probes: latents read path nodes no better than the question tail / <bot> (path_1 .18-.21 vs .16 / .19; front_1 AUROC .77 vs .74) and the LM-head readout is at base everywhere; the tail reads the first hop only weakly (codi, run_id: 20261108-102225_codi_prosqa-grid-path-frontier-probes)

**Goal:** Step 3: probes for the path's intermediate node at the question tail vs the latents. Pre-registered (t18, 0e3d7cc): latent probes within .05 / .03 of <bot>; tail / bot read path_1 >= .3 and front_1 AUROC >= .75.
**Mechanism / model:** `connordilgren/gpt2-prosqa-codi`, 6 latents (one near-constant vector, 20261103-021023).
**Data / command:** as the Coconut record; `grid_probe.py codi mps <out> 3000`; raw `codi_audit_artifacts/t18_codi.json`.
**Results (layer 12; shuffled controls .03-.08, .49-.57):** path_1 tail .16, <bot> .19, latents .18-.21; path_2 .09 / .11 / .15-.17; path_3 .16 / .15 / .13-.15; front_1 AUROC .65 / .74 / .77; front_4 .64 / .66 / .60. LM-head frontier mass at every latent = base (.13 / .21 / .15 / .08 vs .13 / .19 / .15 / .06).
**Interpretation:** Latents add nothing over <bot> beyond small gains (path_2 +.04-.06, front_1 +.03), consistent with near-constant latents; the item information they hold is the small residual (1.2% of energy). Prediction partly missed: the question tail reads the first hop more weakly than predicted (path_1 .16-.19 < .3; front_1 .65-.74 at the tail), matching t17's finding that this model barely follows the path (edits followed .025).
**Caveats:** as the Coconut record.
**Next:** none.
