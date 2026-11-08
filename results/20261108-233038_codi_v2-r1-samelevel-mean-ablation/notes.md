## 2026-11-08 — Round-1 CODI with same-level means: all six latents ablated keeps L1 (1.000) and drops L2-L5 to .682 / .190 / .084 / .060, while matched CODI-0 scores .988 / .960 at L2 / L3; donor latents are followed .90 / .98 at L2 / L3. The pooled-mean result (.63 / .17 at L2 / L3) holds from L2 (codi, run_id: 20261108-233038_codi_v2-r1-samelevel-mean-ablation)

**Goal:** t24 (pre-registered, t_predictions.md, commit 6a1e88b): replace the level-pooled means behind the paper's "reliance without need" figure (round-1 CODI .99 -> .17 at L3) with same-level means, after pooling was found to flip start-last L1 (20261106-105537 correction).
**Mechanism / model:** round-1 CODI, `rulechain-checkpoints/gpt2_depth2_codi_bs64/checkpoint-31280` (20261102-000204), 6 latents, start value first, trained L1-L12.
**Data:** depth2 test, L1-L5, 500 per level; means from 2,000 train items per level (seed 21).
**Command:** `MODEL=r1 run_guarded.sh t24_r1.log codi_audit_artifacts/t24_samelevel_mean.py scripts/codi_null_audit mps <out>` (b29 with a model switch; copy in `prereg/`); smoke: writing back a run's own latents keeps every answer. Record `prereg/record22.py t24_r1`.
**Headline results** (clean / all six same-level mean-ablated / follows a different-value donor's six latents):
- L1 1.000 / 1.000 / .000
- L2 1.000 / .682 / .898
- L3 .992 / .190 / .984
- L4 .758 / .084 / .707
- L5 .442 / .060 / .430
- Same-value donors keep the answer .80-1.00. Matched round-1 CODI-0 (20261102-060537): L2 .988, L3 .960, L4 .764, L5 .494.
**Predictions:** L1 >= .95: held (1.000). L2 <= .7: held (.682). L3 <= .3: held (.190). L4-L5 <= .15: held (.084, .060).
**Interpretation:** The figure survives the correction. At L2-L3 the answer depends on the latents' content (donors followed .90-.98, ablation costs .32-.80) although the same training without latents solves those levels at .96-.99. Reliance is a routing choice of this recipe, not need.
**Caveats:** One seed; mean ablation is a distribution shift; the donor test is the cleaner use measure.
**Next:** quote these numbers instead of the pooled ones.
