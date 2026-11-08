## 2026-11-08 — lr-3e-3 L1-7w CODI with same-level means: all six latents ablated keeps L1 (1.000) and drops L2-L5 to .704 / .138 / .070 / .032; donor latents followed .77 / .98 at L2 / L3, while stabilised CODI-0 on the same data scores .994 / .860 (codi, run_id: 20261108-233039_codi_v2-lr3-samelevel-mean-ablation)

**Goal:** t24 (pre-registered, t_predictions.md, commit 6a1e88b), second model: same-level mean ablation of the relay-regime CODI.
**Mechanism / model:** lr-3e-3 L1-7w CODI, `codi_runs_l1to7w_codi/checkpoint-17820` (20261103-193425; its own CODI-0 diverged, 20261103-174232).
**Data:** l1to7w test, L1-L5, 500 per level; means from 2,000 train items per level (seed 21).
**Command:** `MODEL=lr3 run_guarded.sh t24_lr3.log t24_samelevel_mean.py ...`; record `record22.py t24_lr3`.
**Headline results** (clean / mean-ablated / follows donor):
- L1 1.000 / 1.000 / .000
- L2 .998 / .704 / .768
- L3 .992 / .138 / .984
- L4 .858 / .070 / .818
- L5 .660 / .032 / .585
- Reference without latents on the same data: stabilised CODI-0 (20261105-024111) 1.000 / .994 / .860 / .722 / .560 (a different recipe, lr 2e-3; the matched lr-3e-3 CODI-0 diverged).
**Predictions:** L1 >= .95: held. L2 <= .7: missed by .004 (.704). L3 <= .3, L4-L5 <= .15: held.
**Interpretation:** Same pattern as round 1: reliance from L2 where a no-latent model is near ceiling.
**Caveats:** No matched CODI-0 for this recipe; one seed.
**Next:** none.
