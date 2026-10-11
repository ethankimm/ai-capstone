## 2026-10-11 — Twin magnitude, Coconut GPT-2: steering drops only when the answer changes by <=5%; flat above (coconut, run_id: 20261011-090628_coconut_twin-magnitude-gsm8k)

**Goal:** Phase A step 1 (MULTI_DATASET.md §4). Every earlier twin changed one operand by 1-3; an offline reanalysis of the logged twin runs found steering drops when the answer changes by <=5% of its value. Does it, with changes up to +-20 and x0.5..x3, and is the drop about relative change, absolute change, or answer size?
**Mechanism / model:** `coconut`, gpt2 / `hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33`, 6 latent passes, fp32, greedy.
**Data:** gold-trace test file, all 1194 (seed-0 order); 10,951 twin candidates (4,374 pass the hidden-operand filter);
2,072 qualified pairs, 300 non-responsive pairs.
**Design:** same script, sites and qualification as E3, plus `--change_set 1,-1,2,-2,3,-3,5,-5,10,-10,20,-20,x0.5,x1.25,x1.5,x2,x3 --conditions all_slot`: one twin per (example, propagation-qualified step, change), deduplicated by twin text; multiplicative changes round half up. Only the aligned all-slot transplant is run. Per pair: absolute and relative answer change and fraction moved = (patched - original)/(twin - original). The hidden-operand filter (`minimal_pairs.passes_hidden_operand_filter`: no chain operand that is neither a question number nor an earlier result, and the perturbed number not equal to any step result) is recorded per pair, not applied at patch time; the analysis reports filtered (primary) and all pairs. **Definition is mine**: the Llama twin notes name the filter but don't define it. Analysis: `uv run python scripts/analyze_twin_magnitude.py results/<this run>`, which also excludes pairs whose twin answer equals the original (the changed step doesn't reach the answer; steering is undefined and `matches_twin` is False by construction). The `magnitude_bins` in the manifest INCLUDE those pairs; quote the analysis-script numbers below.
**Command:** `eval_command.txt`; wrapper `scripts/pod/phase_a_step1.sh run`. Pod `ymuz0w5buqfe26` (RTX A40 secure, CA-MTL-1,
$0.59/hr), in parallel with the CODI run; decode 70 s, whole run ~20 min.
**Headline results:** base accuracy 0.336 (identical to E3). CFR 0.587. All-slot steer 0.845 over all qualified pairs
(manifest); non-responsive pairs follow the twin's wrong answer 0.473.

Steer rate by relative answer change (19 of 2,072 pairs excluded: twin answer = original):

| rel. answer change | filtered n | filtered steer [95% CI] | all n | all steer [95% CI] | median frac. moved |
|---|---|---|---|---|---|
| <=5% | 72 | 0.708 [0.595, 0.801] | 116 | 0.595 [0.504, 0.680] | 1.00 |
| 5-15% | 200 | 0.880 [0.828, 0.918] | 345 | 0.870 [0.830, 0.901] | 1.00 |
| 15-40% | 318 | 0.909 [0.872, 0.936] | 588 | 0.881 [0.852, 0.905] | 1.00 |
| 40-100% | 333 | 0.898 [0.861, 0.926] | 660 | 0.871 [0.843, 0.895] | 1.00 |
| >100% | 173 | 0.838 [0.776, 0.886] | 344 | 0.840 [0.798, 0.875] | 1.00 |

Logistic steer ~ log rel change + log |answer|: filtered +0.084 (p=0.27) and -0.225 (p=6e-5); all +0.154 (p=0.003) and
-0.185 (p=4e-7).
**Interpretation:** Confirms the offline reanalysis for Coconut, and sharpens it: steering is flat at ~0.84-0.91 from 5% to
>100% answer change and drops only in the <=5% bin (0.60 all, 0.71 filtered; CIs clear of the 5-15% bin in both pools).
So the drop is a threshold at small relative changes, not a trend, and large changes don't hurt. Coconut's dip is deeper
than CODI's (`20261011-092740_codi_twin-magnitude-gsm8k`: 0.84 filtered). Larger answers steer less independently of the
relative change. Median fraction moved = 1.0 everywhere; mean fraction moved in the <=5% bin is >1 because the failures
there overshoot by many multiples of the tiny twin-original gap.
**Gotchas hit:** as in the CODI run (excluded twin = original pairs).
**Caveats:** single seed, greedy, gold-trace subset. Bins share recipients. Filtered <=5% n=72, below the 150 target: the
dip is significant in the unfiltered pool, suggestive in the filtered one.
**Next:** step 3 ladder on the synthetic depth task.
