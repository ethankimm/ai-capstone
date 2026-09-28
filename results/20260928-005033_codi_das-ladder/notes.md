## 2026-09-28 — Experiment B, CODI: a DAS rotation trained on cross-problem ladder donors reaches 58-77% of the whole-vector cross-problem ceiling, vs 12-53% for the minimal-pair-trained rotation (codi, run_id: 20260928-005033_codi_das-ladder)

**Goal:** CODI counterpart of `20260927-234206_coconut_das-ladder` — read that record for the full
motivation (round-4 open problem 2: does training the DAS rotation on cross-problem donors instead
of within-problem minimal pairs close the gap between subspace-only and whole-vector cross-problem
transfer found in round 4's `20260927-191455/-191525`?).
**Design** (`das_ladder_codi.py`, modelled on `das_minimal_pair_codi.py` — reuses its
`OrthogonalRotation`/`intervene`/`run_intervened`/`teacher_forced_ce`/`answer_target` unchanged):
- **Train:** decoded 3000 GSM8K-Aug train examples with this model alone (base accuracy 79.4% on
  the decoded pool), built ladder pairs for up to 500 recipients, 1000 (recipient, donor, level)
  training tuples with a defined cf_joint target. One shared rotation R per k over sites {0,2,4}
  jointly, trained 5 epochs (5000 steps/k), teacher-forced against `"The answer is: {cf_joint}"`.
- **Eval:** same 259 P4-identical recipients as the Coconut sibling run (from
  `xmech_common._ladder_pairs` on the shared dumps). Conditions: `full` (raw whole-vector swap),
  `untrained_k{16,32,64}` (random rotation, floor), `trained_k{16,32,64}` (ladder-trained). Scored
  with `ladder_common.summarize` (cf_joint bucket + permutation null).
**Mechanism / model:** CODI `hf:zen-E/CODI-gpt2@fd641b3`, aligned sites (z_s feeds iteration s+1).
**Command:** `xmech_codi.py`-style CODI flags + `das_ladder_codi.py --site_group 0,2,4 --k_values
16,32,64 --train_n_recipients 500 --epochs 5 ... --save_rotations rotations_ladder/codi --slug
das-ladder --stage full_run` (full line in `eval_command.txt`). Pod `jbzj5cd4diacxb` (RTX A40
secure, $0.49/hr, EU-SE-1); ran alongside 7 other jobs initially (see sibling run's caveat on
CPU-bound contention slowing the first ~90 min), then alone.
**Headline results** (259 recipients; train pool 500 ladder recipients / 1000 tuples, train
accuracy 79.4%):

| cf_joint (null ≤0.010) | L2 | L3 | L4 |
|---|---|---|---|
| full (whole-vector own, ceiling) | 0.363 | 0.386 | 0.193 |
| untrained_k16/32/64 (floor) | 0.000 | 0.000 | 0.000 |
| **trained_k16** | **0.228** | **0.293** | **0.112** |
| **trained_k32** | **0.270** | **0.263** | **0.124** |
| **trained_k64** | **0.278** | **0.266** | **0.135** |
| % of whole-vector ceiling, k32 | 74% | 68% | 64% |
| round-4 minimal-pair rotation own_sub, k32 (`20260927-191455`) | 0.178 (46%) | 0.208 (53%) | 0.100 (52%*) |

(*round-4's ceiling reference used round-3's "0.39/0.39/0.20"; this run's own ceiling is
0.363/0.386/0.193 — percentages above use each run's own ceiling for an apples-to-apples read.)

untrained leaves 98.8-99.6% unchanged (a random 16-64 dim subspace is essentially inert); the
trained rotation leaves 8.5-15% unchanged, close to `full`'s 6.6-7.7%.

**Interpretation:**
- **Same qualitative result as the Coconut sibling, smaller relative gain but still substantial**:
  the ladder-trained rotation reaches 58-77% of the whole-vector ceiling (using k32: 74%/68%/64% for
  L2/L3/L4) vs round 4's minimal-pair rotation at 46-53% (k32) — round 4's CODI-side own_sub was
  already the STRONGER of the two mechanisms' minimal-pair-basis results (see
  `20260927-191455/-191525`'s own interpretation: "CODI's own subspace-only interchange carries a
  cross-problem donor's values at 0.07–0.21"), so there was less headroom to close here than on the
  Coconut side (28-40% → 58-70%, a near-doubling).
- k=16 already captures most of the L2/L3 gain (k16 actually edges out k32/k64 at L3: 0.293 vs
  0.263/0.266) but L4 keeps improving through k64 (0.112→0.124→0.135) — consistent with L4's random
  donor needing a slightly larger subspace to carry enough of the more different problem's value.
- L4's ceiling itself is much lower here than L2/L3 (0.193 vs 0.363/0.386) — matches the established
  pattern (round 2/3) that CODI's whole-vector transfer is weaker for the least-similar donor, so
  the subspace-only L4 numbers (11-14%) are a smaller absolute effect even at a similar RELATIVE
  fraction of ceiling (64-70%).
**Caveats:** n=259 eval, n=1000 train tuples; one donor draw per level. Wall-clock inflated by
initial GPU/CPU contention with 7 other concurrent jobs (loss curves/results unaffected, see sibling
run's caveat). `full` here (0.363/0.386/0.193) is close to but not identical to the "0.39/0.39/0.20"
quoted in round-3 notes for the nominal same condition (this run's own eval decode, not a reused
cached prediction) — small run-to-run variance, doesn't change the qualitative picture.
**Next:** Both mechanisms show own_sub with the ladder basis approaching (Coconut) or substantially
closing (CODI) the whole-vector ceiling — the pre-registered condition for rerunning `--mode
subspace` with these rotations to test cross-mechanism `mapped_sub`. See the
`xmech-subspace-ladder-*` runs for that follow-up (in progress / logged separately).
