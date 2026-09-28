## 2026-09-28 — Experiment B, Coconut: a DAS rotation trained on cross-problem ladder donors reaches 58-70% of the whole-vector cross-problem ceiling, vs 28-40% for the minimal-pair-trained rotation (coconut, run_id: 20260927-234206_coconut_das-ladder)

**Goal:** Round-4 open problem 2: the DAS value-subspace rotations used throughout (E4,
`20260927-182915`) were trained on WITHIN-PROBLEM minimal pairs, where the complement (everything
outside the k-dim subspace) is nearly identical between recipient and donor. When that rotation
was used for a CROSS-PROBLEM subspace-only patch (round 4, `20260927-190704/-190716`), the result
was far weaker than a whole-vector cross-problem patch (own_sub 0.085-0.120 cf_joint vs whole-vector
own ~0.27-0.35) — hypothesized to be because the training basis doesn't suit cross-problem transfer
(the recipient's live complement, no longer near-identical to the donor's, pulls back toward the
recipient's own values). This asks: does training the SAME kind of DAS rotation directly on
CROSS-PROBLEM (ladder L2+L3) donor pairs close that gap?
**Design** (`das_ladder_coconut.py`, modelled on `das_minimal_pair_coconut.py` — reuses its
`OrthogonalRotation`/`intervene`/`run_intervened`/`run_pass_range`/`teacher_forced_ce` unchanged):
- **Train:** decoded 3000 `gsm_original_train.json` examples with this model alone (base accuracy
  92.6% on the decoded pool), built ladder pairs (`ladder_common.build_ladder_pairs`, L2 same ops +
  L3 same length/different ops) for up to 500 recipients, flattened to 1000 (recipient, donor,
  level) training tuples where the recipient's program re-run on the donor's step values
  (`ladder_common.targets(...)["cf_joint"]`) is defined. One shared rotation R per k, trained for 5
  epochs (5000 steps/k) via the same DAS mechanism as `das_minimal_pair_coconut.py`
  (`intervene(R, z_a=recipient's own live pass, z_b=donor's own live pass, k)` at passes {1,4}
  jointly), teacher-forced against `"### {cf_joint}"` (no leading space).
- **Eval:** the EXACT SAME 259 recipients as the unrestricted P4 run (`xmech_common._ladder_pairs`
  reproduced from `codi_latents.pt`+`coconut_latents.pt`'s correctness intersection, pair_seed 0) —
  for a same-recipient, apples-to-apples comparison to round 4's minimal-pair-rotation own_sub.
  Conditions: `full` (raw whole-vector swap of the donor's own passes 1,4 — any orthogonal R at
  k=768 gives this exactly), `untrained_k{16,32,64}` (random rotation, same k, floor), `trained_k{16,32,64}`
  (this run's ladder-trained rotation). Scored with `ladder_common.summarize` (cf_joint bucket +
  permutation null), matching P4's own metric — NOT `das_minimal_pair`'s `matches_twin`, since
  ladder donors are cross-problem (their own final answer isn't a valid target).
**Mechanism / model:** Coconut `hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33`.
**Command:** `scripts/das_ladder_coconut.py --checkpoint_path .../checkpoint_33 --data_dir
coconut_data --site_group 1,4 --k_values 16,32,64 --train_n_recipients 500 --epochs 5 ...
--save_rotations rotations_ladder/coconut --slug das-ladder --stage full_run` (full line in
`eval_command.txt`). Pod `jbzj5cd4diacxb` (RTX A40 secure, $0.49/hr, EU-SE-1); ran alongside 7 other
jobs (all 6 Experiment A conditions + `das_ladder_codi`) initially, then alone once those finished
(training sped up ~10x once GPU/CPU contention cleared — see caveats).
**Headline results** (259 recipients; train pool 500 ladder recipients / 1000 (recipient, donor,
level) tuples, train accuracy 92.6%):

| cf_joint (null ≤0.006) | L2 | L3 | L4 |
|---|---|---|---|
| full (whole-vector own, ceiling) | 0.270 | 0.313 | 0.351 |
| untrained_k16/32/64 (floor) | 0.000 | 0.000 | 0.000 |
| **trained_k16** | **0.170** | **0.220** | **0.205** |
| **trained_k32** | **0.189** | **0.220** | **0.243** |
| **trained_k64** | **0.181** | **0.208** | **0.243** |
| % of whole-vector ceiling, k32 | 70% | 70% | 69% |
| round-4 minimal-pair rotation own_sub, k32 (`20260927-190716`) | 0.104 (35%) | 0.120 (40%) | 0.120 (34%*) |

(*round-4's own ceiling for that comparison used the round-3 "own whole-vector" figure ~0.30 for
all levels; this run's own ceiling varies by level (0.270/0.313/0.351) — percentages above use each
run's own ceiling.)

untrained (random k-dim rotation) leaves 97-100% of answers unchanged — a random subspace carries
nothing, as expected; the trained rotation leaves only 10-16% unchanged (vs full's 6-9%), i.e. most
of the ceiling's "the answer actually changes" behavior is preserved too.

**Interpretation:**
- **Training the DAS rotation on cross-problem donors instead of within-problem minimal pairs
  roughly DOUBLES the subspace-only cross-problem transfer**: 58-70% of the whole-vector ceiling
  (this run) vs 28-40% (round 4's minimal-pair rotation, same eval recipients, same k). This
  confirms round 4's hypothesis: the minimal-pair training basis was mismatched to cross-problem
  transfer specifically because training never exposed the rotation to a genuinely different live
  complement.
- k=16 already captures most of the gain (63-70% of ceiling); k=32/64 add only a few more points.
  The value that generalizes across problems needs a moderately larger subspace than
  minimal-pair-DAS suggested (there, k=16 vs k=32 differed by a similar small margin, but both sat
  much lower in absolute terms) — i.e. the earlier "16 dims carry the value" finding
  (`20260927-182915`) was correct about the DIMENSIONALITY, but the earlier ROTATION (the specific
  16-dim subspace it found) generalizes far less well than one trained for this purpose.
- This still falls short of the whole-vector ceiling (a 30-40% gap remains at every k), so a
  subspace-only patch — even on the right basis — does not fully substitute for the unrestricted
  intervention; some transferable content still lives outside any single k≤64 linear subspace of
  this rotation, or the complement (still the recipient's own live value) genuinely interferes.
**Caveats:** n=259 eval, n=1000 train tuples (500 recipients x up to 2 levels); one donor draw per
level at eval. Training ran ~10x slower during the first ~90 min while sharing the GPU/CPU with 7
other jobs (this repo's convention allows several jobs per GPU, but ridge-fit-heavy jobs are
CPU-bound and 8 concurrent jobs saturated this pod's 9 vCPUs) — final wall-clock is not
representative of a dedicated-GPU run; loss curves and results are unaffected by this, only
runtime. `full`'s absolute values here (0.270/0.313/0.351) differ a few points from the "0.30" all
three levels quoted in round-3/4 summary notes for the nominal same condition — see the Experiment A
runs' caveats for the same observation; doesn't affect this run's own ratios.
**Next:** Given own_sub with the ladder basis reaches ~70% of ceiling (the pre-registered "approaches
whole-vector ceiling" branch), the natural follow-up is `xmech_coconut.py`/`xmech_codi.py --mode
subspace` re-run with these ladder rotations (instead of the minimal-pair ones) to see whether
CROSS-MECHANISM `mapped_sub` also improves — see `20260927-234206`'s sibling
(`20260928-005033_codi_das-ladder`) for the CODI side, and the `xmech-subspace-ladder-*` runs for
that follow-up.
