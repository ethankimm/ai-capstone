## 2026-09-27 — P1 transfer ladder, CODI: intermediate values transfer across problems (cf_joint 44→37→21% down the ladder on 2-step chains), the program does not (codi, run_id: 20260927-063622_codi_ladder-patch)

**Goal:** RESEARCH_PLAN §4 P1: fill the ladder rungs between level 1 (same-problem twin, E3 aligned 83.9%)
and level 4 (unrelated problem). Same recipients at every level; donors: **L2** different problem, same
number of steps and same operator sequence; **L3** same number of steps, different operators; **L4** any
problem (random-donor control). `scripts/patch_ladder_codi.py` + `scripts/ladder_common.py`.
**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3`, 6 latents, eval mode, greedy.
Aligned sites (z_s → iteration s+1). Conditions: all_slot (z0..z5), carriers (z0,z2,z4 — the value-carrying
sites of aligned E3), each carrier alone.
**Data:** gsm8k-aug test, all 1319; base accuracy 0.419; 369 recipients (base-correct, verified chain,
with a base-correct L2, L3 and L4 donor of a different answer). Chain lengths 2/3/4+: 204/130/35.
**Scoring** (first match wins; see `ladder_common.py`): unchanged · donor_final · **cf_joint** (the
recipient's program re-run with every intermediate value replaced by the donor's value at the same step
index — values from the latents, final operation and its question operands from the recipient's context)
· cf_single (one donor step value substituted) · recipient/donor intermediate · other. Null: each answer
re-bucketed against another recipient's donor at the same level (200 permutations).
**Command:** see `eval_command.txt` (`--eval_n 0 --n_recipients 1000`). Pod `5wb7p2e1kr568o`, RTX A6000
secure, ~15 min in parallel with three other jobs.
**Headline results** (n=369, rate / permutation null):

| level | condition | unchanged | donor_final | cf_joint | cf_single | other |
|---|---|---|---|---|---|---|
| L2 | all_slot | 0.06 | 0.15 / 0.009 | **0.33** / 0.004 | 0.01 | 0.45 |
| L2 | carriers z0,z2,z4 | 0.06 | 0.13 / 0.009 | **0.34** / 0.004 | 0.01 | 0.44 |
| L3 | all_slot | 0.06 | 0.02 / 0.008 | 0.24 / 0.004 | 0.02 | 0.63 |
| L3 | carriers | 0.06 | 0.02 / 0.007 | **0.32** / 0.004 | 0.02 | 0.56 |
| L4 | all_slot | 0.05 | 0.04 / 0.007 | 0.14 / 0.004 | 0.02 | 0.71 |
| L4 | carriers | 0.05 | 0.02 / 0.008 | 0.18 / 0.005 | 0.01 | 0.68 |
| L2/L3/L4 | z0 alone | 0.53–0.55 | ≤0.02 | 0.02–0.03 | **0.08–0.12** / 0.001 | 0.29–0.33 |
| L2/L3/L4 | z2 alone | 0.63–0.68 | ≤0.01 | 0.02–0.03 | 0.01–0.03 | 0.26–0.28 |
| L2/L3/L4 | z4 alone | 0.53–0.59 | ≤0.01 | 0.04–0.07 | 0.00–0.02 | 0.28–0.35 |

By chain length (carriers; donor_final / cf_joint): 2 steps L2 0.06/0.44, L3 0.01/0.37, L4 0.01/0.21;
3 steps L2 0.22/0.25, L3 0.05/0.31, L4 0.04/0.18; 4+ steps (n=35) L2 0.17/0.14, L3 0/0.09, L4 0/0.03.

**Interpretation:**
- **Values are portable, programs are not.** Transplanting a different problem's z0/z2/z4 makes CODI
  finish its OWN computation on the DONOR's intermediate values (cf_joint 18–34%, null ≤0.5%). The
  donor's final answer only comes through when the donor has the same operator sequence and ≥3 steps
  (L2 3-step: 22%) — where the donor's program and the recipient's coincide. At L3/L4 donor_final is at
  its null.
- The ladder breaks gradually, not at a cliff: cf_joint 44% (L2) → 37% (L3) → 21% (L4) on 2-step
  chains. Same operator sequence helps, but most of the transfer survives a different operator.
  Level 1 (same problem, E3 aligned) is 84%.
- Carriers ≥ all_slot at L3/L4: adding the "inert" odd sites z1/z3/z5 from an unrelated problem hurts.
- Single carriers alone rarely finish the job (z0 alone → the step-0 substitution, cf_single 8–12%),
  consistent with aligned E3 where each even site carries part of the value.
- "other" stays 44–71%: most transplants from another problem produce an answer we can't attribute.
**Gotchas hit:** For 2-step chains cf_joint and "step-0 cf_single" are the same number; cf_joint takes
precedence, so the by-length split is the one to read. Donor_final precedes cf_joint when equal.
**Caveats:** One recipient–donor draw per level (pair_seed 0); greedy; 4+ step chains are thin (n=35).
Chains are the dataset's `<<>>` annotations filtered to self-consistent ones (`chain_ok`).
**Next:** Coconut counterpart `20260927-062452_coconut_ladder-patch`; P4 (CODI↔Coconut transplant) can now
start from value transfer at matched sites (CODI z0 ↔ Coconut pass 1 for step 0).
