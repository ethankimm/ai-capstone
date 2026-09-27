## 2026-09-26 — Where non-steered patched answers go, CODI: level 1 misses are partial propagation of the perturbed number; level 4 moves are unrelated (codi, run_id: 20260926-235222_codi_nonsteered-buckets)

**Goal:** `henning_thoughts.md`: "donor states change answer but do not steer as per donor value ->
what are the answers going towards instead?" Turn "context-entangled" into a measurement by bucketing
every patched answer from E3 (level 1, same-problem twin, `20260920-190420`) and E2 (level 4,
cross-problem donor, `20260920-085206`). RESEARCH_PLAN round 1, item 3.
**Mechanism / model:** `codi`, released checkpoint. No model run, saved predictions only.
**Command:** `uv run python scripts/analyze_nonsteered.py --reps 200` (writes this + the Coconut record).
**Buckets** (first match wins; definitions in the script docstring): unchanged (recipient original) ·
donor_final (steered) · counterfactual (E2 Metric B) · **partial_propagation** (E3: the perturbed
operand changed in only some of the steps that use it) · **wrong_step** (donor value injected at the
wrong chain step) · donor_intermediate · recipient_intermediate · off_by_delta (E3: recipient answer
± delta) · other (unrelated). **Null:** each moved-not-steered answer is re-bucketed against a random
other record's chains (200 permutations); a bucket only means something above that.
**Headline results:**

E3 all-slot (n=137): 70.1% steered, 7.3% unchanged, 8.0% partial_propagation, 12.4% other.
Among the 31 moved-but-not-steered answers: **partial_propagation 35% (null 1.6%)**, recipient
intermediate 6% (null 2%), off_by_delta 3% (1%), other 55%.

Split by whether partial propagation is possible at all (perturbed number used in ≥2 steps):

| all-slot | n | steered | partial_propagation | unchanged | other |
|---|---|---|---|---|---|
| number used in ≥2 steps | 26 | 0.50 | **0.42** | 0.04 | 0.04 |
| number used once | 111 | 0.75 | — | 0.08 | 0.14 |

E2 (level 4, 370 pairs pooled over sites): real donor 68% unchanged, 1.1% counterfactual, 0% donor
final; of the 114 moved answers, 91% other, wrong_step 6% (null 0.3%) — but the **random** donor
gives the same 6% wrong_step and 4% donor_intermediate, and mean ablation 0% / 7% recipient
intermediate. Nothing distinguishes the real donor from a random one.

**Interpretation:** At level 1 the misses are not noise. When the perturbed number enters the chain
twice, the all-slot latent patch carries the new value into one use while the other use keeps the
recipient's number — about as often (42%) as it fully steers (50%). When the number enters once,
steering is 75%. Direct evidence that part of the computation reads operands from the question
tokens' KV cache rather than from the latents: the latents hold the value for the step they compute,
not a context-free copy of the operand. At level 4 the moved answers are unrelated numbers, and the
real donor's step value shows up no more often than a random donor's: cross-problem patches disrupt,
they don't transmit.
**Caveats:** CODI's committed E3/E2 runs use the pre-fix site indexing (donor z_i fed where the
recipient's z_{i-1} goes; see `patch_minimal_pair_codi.py`); all-slot is a full shifted transplant, and
single-slot / prefix rows describe that intervention, not the aligned one. Re-run this analysis on the
aligned E3 rerun. The multi-use subset is small (26); 55% of moved answers are still "other".
**Next:** rerun on the aligned CODI E3 (and larger n); Coconut counterpart
`20260926-235222_coconut_nonsteered-buckets`.
