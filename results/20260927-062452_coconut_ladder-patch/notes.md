## 2026-09-27 — P1 transfer ladder, Coconut: pass 1 carries step 0's value and pass 4 step 1's, portable across problems at every level; all six passes carry the donor's answer itself (coconut, run_id: 20260927-062452_coconut_ladder-patch)

**Goal:** RESEARCH_PLAN §4 P1, Coconut counterpart of `20260927-063622_codi_ladder-patch` (same donor
levels, scoring and null — see that record and `scripts/ladder_common.py`).
**Mechanism / model:** `coconut`, gpt2 / `hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33`, 6 latents.
Conditions: all_slot (passes 0..5), carriers (passes 1,4 — the value-carrying passes of E3
`20260927-003331`), pass 1 alone, pass 4 alone.
**Data:** gold-trace test file, all 1194; base accuracy 0.336; 296 recipients (chain lengths 2/3/4+:
173/101/22).
**Command:** `scripts/patch_ladder_coconut.py --eval_n 0 --stage full_run` (see `eval_command.txt`).
Pod `5wb7p2e1kr568o`, RTX A6000 secure; decode 75 s, patching ~4 min.
**Headline results** (n=296, rate / permutation null):

| level | condition | unchanged | donor_final | cf_joint | cf_single | other |
|---|---|---|---|---|---|---|
| L2 | all_slot | 0.04 | **0.37** / 0.011 | 0.15 / 0.006 | 0.01 | 0.42 |
| L2 | carriers 1,4 | 0.08 | 0.15 / 0.010 | **0.27** / 0.005 | 0.01 | 0.46 |
| L2 | pass 1 | 0.09 | 0.04 | **0.28** / 0.005 | **0.12** / 0.003 | 0.46 |
| L2 | pass 4 | 0.55 | 0.02 | 0.05 | 0.07 | 0.27 |
| L3 | all_slot | 0.04 | **0.26** / 0.010 | 0.04 | 0.01 | 0.61 |
| L3 | carriers | 0.06 | 0.03 | **0.34** / 0.005 | 0.01 | 0.51 |
| L3 | pass 1 | 0.10 | 0.01 | 0.25 | 0.12 | 0.49 |
| L4 | all_slot | 0.02 | **0.23** / 0.012 | 0.04 | 0.01 | 0.65 |
| L4 | carriers | 0.03 | 0.03 | **0.29** / 0.006 | 0.01 | 0.56 |
| L4 | pass 1 | 0.06 | 0.01 | 0.25 | 0.11 | 0.54 |

By chain length: **pass 1** → 2-step cf_joint 0.43–0.47 at every level; 3-step cf_single (= the step-0
substitution) 0.30–0.34 at every level. **Carriers** on 3-step chains: cf_joint 0.19 / 0.28 / 0.28
(L2/L3/L4). **all_slot** donor_final 2-step 0.38 / 0.28 / 0.26.

**Interpretation:**
- **Value per pass, portable across problems.** Pass 1 holds step 0's result: transplanted from ANY
  problem, the recipient carries on with the donor's step-0 value (2-step: cf_joint ~45%; 3-step: exactly
  the step-0 substitution, ~32%) — flat across L2/L3/L4, so the operator sequence does not matter. Adding
  pass 4 turns 3-step answers into full cf_joint (step 0 AND step 1 values replaced, 19–28%), so pass 4
  holds step 1's result. Null ≤0.5%.
- **All six passes also carry the donor's answer.** all_slot gives the donor's final answer 23–37% even
  for unrelated donors (L4), where CODI gives 4%. Coconut's full latent sequence contains the answer
  (consistent with Li et al.: early routing to the answer); CODI's does not — it recomputes the final
  operation from context.
- Compared with CODI: both mechanisms carry portable intermediate values at a few sites (Coconut 1/4,
  CODI 0/2/4). They differ in where the final operation lives: in Coconut's latents vs CODI's context.
**Caveats:** One donor draw per level; greedy; 4+ step chains n=22. Base accuracy on this file 0.336.
**Next:** P4 — cross-mechanism transplant at matched sites (CODI z0 → Coconut pass 1 for the step-0 value).
