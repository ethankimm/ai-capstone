## 2026-11-08 — Prefill probe: start-first CODI runs the chain in the question pass (v_2 .52-.58, v_3 .27-.30 at later rule ends), start-first SIM-CODI does not (past v_1 at the floor, like both start-last models): the step loss moved the chain out of the question pass, no dormant copy (codi, run_id: 20261108-190213_codi_v2-l1to7w-prefill-chain-probe)

**Goal:** t21c (pre-registered, t_predictions.md t21, commit 96a04d4): with the start value first, does SIM-CODI still compute the chain while reading the question (a dormant second copy a latent monitor would miss), or did the step loss move it into the latents? Plain start-first CODI is the reference that uses the question pass; start-last models are the floor (start value not yet seen at the rule ends).
**Mechanism / model:** simsf = SIM-CODI start-first ckpt-17820 (20261108-130905); stab = start-first CODI ckpt-17820 (20261105-044444); sim = SIM-CODI start-last ckpt-16335 (20261107-225603); startlast = start-last CODI ckpt-17820 (20261106-083843). Question pass only, no latents.
**Data:** probes fit on 2000 train questions per level (seed 30), scored on all valid items, L3-L6, each model on its own data (start-first or start-last).
**Command:** `run_guarded.sh t21c.log scripts/necessity/prefill_probe.py t21c_prefill.json mps simsf stab sim startlast` (log in `scripts/codi_null_audit/prereg/`); record via `prereg/record21.py t21c`.
**Headline results** (exact linear read, best of layers 4/8/12; shuffled-label controls .01-.04; most common value .03-.05):

| v_k read at the end of rule k (at any rule end >= k) | v_1 | v_2 | v_3 | v_4 |
|---|---|---|---|---|
| start-first CODI | .41-.46 (.65-.69) | .30-.35 (.52-.58) | .23-.25 (.27-.30) | .12-.13 (.17) |
| start-first SIM-CODI | .36-.46 (same) | .02-.05 | .02-.03 | .02-.03 |
| start-last SIM-CODI | .01-.02 | .02-.03 | .02-.04 | .02-.03 |
| start-last CODI | .01-.02 | .02-.03 | .02-.03 | .02-.03 |

At <bot>: v_1 .44-.52 for both SIM models; v_{L-1} at <bot> .15-.36 for start-first CODI, .00-.11 for the others.

**Interpretation:** Pre-registered either/or: **moved**. Plain start-first CODI computes the chain while reading the rules, about one rule behind (each value reads best a rule after it is defined), falling with depth, which matches its single-pass reach. SIM-CODI start-first, trained on the same data, computes only the first rule in the question pass (needed for z_0) and nothing after it: past v_1 its question states look like the start-last models', which cannot compute there at all. So the step loss did not add a latent trace next to the prefill route; it replaced the prefill route. For the paper: there is no dormant parallel computation here that a latent-only monitor would miss, and "trace used" is not a relay of prefill values (t21a, 20261108-134158). Threshold misses, all on the low side: start-first CODI v_1 at rule 1 >= .7 (got .41-.46; .65-.69 one rule later) and v_2 >= .5 (.30-.35 diagonal; .52-.58 later); SIM v_1 >= .5 (.36-.46). Start-last floor <= .10 held.
**Gotchas hit:** Slow tokenizer has no offset mapping; rule-end tokens found by tokenising each prefix (GPT-2 splits '.' into its own token; asserted).
**Caveats:** Linear exact reads; a non-linear reader could find more in SIM's question states, but its answer drops to chance without latents (20261108-134158), so nothing there is used. One seed per model.
**Next:** None for this record.
