## 2026-10-11 — Twin magnitude, CODI GPT-2: steering is flat over 5% to >100% answer changes; a small dip at <=5% (codi, run_id: 20261011-092740_codi_twin-magnitude-gsm8k)

**Goal:** Phase A step 1 (MULTI_DATASET.md §4). Every earlier twin changed one operand by 1-3; an offline reanalysis of the logged twin runs found steering drops when the answer changes by <=5% of its value. Does it, with changes up to +-20 and x0.5..x3, and is the drop about relative change, absolute change, or answer size?
**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3`, 6 latents, eval mode, bf16, greedy.
**Data:** gsm8k-aug test, all 1319; 11,514 twin candidates (4,540 pass the hidden-operand filter); 2,917 qualified pairs
(original and twin correct), 300 non-responsive pairs.
**Design:** same script, sites and qualification as E3, plus `--change_set 1,-1,2,-2,3,-3,5,-5,10,-10,20,-20,x0.5,x1.25,x1.5,x2,x3 --conditions all_slot`: one twin per (example, propagation-qualified step, change), deduplicated by twin text; multiplicative changes round half up. Only the aligned all-slot transplant is run. Per pair: absolute and relative answer change and fraction moved = (patched - original)/(twin - original). The hidden-operand filter (`minimal_pairs.passes_hidden_operand_filter`: no chain operand that is neither a question number nor an earlier result, and the perturbed number not equal to any step result) is recorded per pair, not applied at patch time; the analysis reports filtered (primary) and all pairs. **Definition is mine**: the Llama twin notes name the filter but don't define it. Analysis: `uv run python scripts/analyze_twin_magnitude.py results/<this run>`, which also excludes pairs whose twin answer equals the original (the changed step doesn't reach the answer; steering is undefined and `matches_twin` is False by construction). The `magnitude_bins` in the manifest INCLUDE those pairs; quote the analysis-script numbers below.
**Command:** `eval_command.txt`; wrapper `scripts/pod/phase_a_step1.sh run`. Pod `ymuz0w5buqfe26` (RTX A40 secure, CA-MTL-1,
$0.59/hr), run in parallel with the Coconut run; decode 177 s, twins ~25 min, patch 409 s.
**Headline results:** base accuracy 0.415 (E3 on A5000: 0.419, same invocation; 5/1319 examples, bf16 on a different GPU).
CFR 0.636. All-slot steer 0.812 over all qualified pairs (manifest), non-responsive pairs follow the twin's wrong answer 0.493.

Steer rate by relative answer change (27 of 2,917 pairs excluded: twin answer = original):

| rel. answer change | filtered n | filtered steer [95% CI] | all n | all steer [95% CI] | median frac. moved |
|---|---|---|---|---|---|
| <=5% | 119 | 0.840 [0.764, 0.895] | 223 | 0.753 [0.693, 0.805] | 1.00 |
| 5-15% | 273 | 0.919 [0.881, 0.946] | 472 | 0.847 [0.812, 0.877] | 1.00 |
| 15-40% | 439 | 0.891 [0.858, 0.917] | 781 | 0.837 [0.810, 0.862] | 1.00 |
| 40-100% | 461 | 0.852 [0.817, 0.882] | 928 | 0.809 [0.783, 0.833] | 1.00 |
| >100% | 249 | 0.823 [0.771, 0.866] | 486 | 0.813 [0.776, 0.845] | 1.00 |

Logistic steer ~ log rel change + log |answer|: filtered -0.141 (p=0.018) and -0.144 (p=0.001); all +0.002 (p=0.96) and
-0.070 (p=0.010).
**Interpretation:** Large changes transplant as well as small ones: steering stays at 0.81-0.92 from 5% to >100% answer
change, so the 83-84% E3 number was not an artefact of +-1..3 twins. The <=5% dip is real in the unfiltered pool (0.75 vs
0.85) but mostly disappears under the hidden-operand filter (0.84 vs 0.92, CIs touch), so part of the earlier "small changes
steer less" signal came from twins whose re-executed gold is unreliable. Larger original answers steer a little less in
both pools. The log-linear logistic is the wrong shape for a threshold; read the bins. Median fraction moved is 1.0 in
every bin: failures are mostly all-or-nothing, not partial moves.
**Gotchas hit:** pairs with twin answer = original answer had to be excluded from the analysis (see Design).
**Caveats:** single seed, greedy. Bins share recipients (several twins per problem), so the CIs are somewhat optimistic.
The filtered <=5% bin has n=119, below the 150 target.
**Next:** the same ladder on the synthetic depth task (step 3), where the change size is controlled by construction.
