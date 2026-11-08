## 2026-11-08 — Start-first CODI under the SIM-CODI battery (matched scripts): latents ignored (swap follows donor .00-.02, all six mean-ablated within .04 of clean), not readable (probe .12 vs shuffled .02), flat pass sweep (1 pass = 12 passes) (codi, run_id: 20261108-135708_codi_v2-l1to7w-stab-codi-latent-use-matched)

**Goal:** Matched comparison for SIM-CODI start-first's latent battery (20261108-134158), as section 14.3 requires: the same s1/s2/s3 scripts, items and probe cells on the start-first CODI reference, so "SIM-CODI uses its latents, CODI does not" is one measurement rather than two from different scripts (20261105-052233 and 20261105-053042 used other scripts).
**Mechanism / model:** Start-first CODI (stabilised recipe, section 10), checkpoint-17820, the model of 20261105-044444. `codi_runs_l1to7w_stab/codi/run/checkpoint-17820`. No step decoder; the "carries v_k" / "trained target" labels in the record are SIM-CODI's indexing, kept so cells line up, and mean nothing about CODI's training.
**Data:** `synthetic_data_v2_l1to7w` valid, L2-L7 for swaps and passes, L3-L6 for probes, as the SIM record. Swap pairs need a recipient answered right, so n falls with CODI's accuracy (L2 284 to L7 53).
**Command:** `codi_audit_artifacts/s1_sim_swap.py`, `s2_sim_probe.py`, `s3_sim_passes.py`, `s4_record.py` with `SIM_DATA=synthetic_data_v2_l1to7w SIM_TAG=_sfcodi SIM_CK=codi_runs_l1to7w_stab/codi/run/checkpoint-17820 SIM_MIN_CLEAN=0`, laptop MPS float32 under `run_guarded.sh`. Smoke: identity patch keeps every answer.
**Headline results:**
- **Clean valid:** L2 .993, L3 .906, L4 .773, L5 .601, L6 .414, L7 .186 (test record .994, .854, .728, .598, .406, .234).
- **Swap:** any single latent, L2-L7: follows donor .000-.019, keeps own .93-1.00.
- **Mean ablation:** single latents within .02 of clean; all six L2 .993 (clean .993), L3 .888 (.906), L4 .741 (.773), L5 .563 (.601), L6 .407 (.414), L7 .193 (.186). Same as 20261105-052233 (.689 to .671 pooled).
- **Probes:** v_k at z_{k-1}, pooled k = 2..L-1 on L3-L6, .119 vs shuffled .024; v_k at z_k .118. SIM-CODI: .520 and .823.
- **Pass sweep (k = 1..12):** flat at every level, e.g. L5 .57-.60, L6 .39-.43, L7 .19-.21.

**Interpretation:** By section 14.3's definitions start-first CODI is not readable, ignores its latents and shows no steps, so it is the clean contrast for SIM-CODI: the same data, recipe and latent count, and the only difference is the step decoder. Prefill does all of CODI's work here, and its reach ends where the single pass does (d50 5.25). This replicates 20261105-052233/053042 with the SIM battery.
**Gotchas hit:** None.
**Caveats:** One seed, valid only. Shrinking swap n at deep levels (53 pairs at L7). Frozen linear probes; a non-linear reader could find more, but the swap says the answer does not use it.
**Next:** None for this record; see 20261108-134158.
