## 2026-09-27 — E4 with the teacher-forcing target fixed, CODI: a learned 16-dim subspace of z0/z2/z4 steers 70% (raw swap 83%) — the DAS "null" was a bug (codi, run_id: 20260927-094215_codi_das-minimal-pair-fixed)

**Goal:** Rerun aligned E4 after finding why DAS training made things worse than an untrained rotation
(`20260927-072659_codi_das-minimal-pair-aligned-bigk`: trained 0.22 vs untrained 0.71 at k=512).
**The bug:** CODI answers "The answer is: 16" after eot. `teacher_forced_ce` forced the bare number
("16" = token 1433) as the first token after eot, where the model emits "The" (464), and the number after
"is:" is the space-prefixed " 16" (1467) anyway. Every CODI DAS run trained toward a token sequence the
model never produces (losses 10–40 nats), fighting the answer format. Fixed: target = "The answer is: {n}"
(`answer_target` in `das_minimal_pair_codi.py`; also fixed in `das_codi.py`).
**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3`, aligned sites (z_s → iteration s+1),
transformer + LoRA frozen; lr 1e-3, 5 epochs, cosine.
**Data:** same pools as the previous E4 runs: 200 train pairs (gsm8k-aug train n=2500, acc 0.791), 100 eval
pairs (validation n=800, acc 0.791).
**Command:** `scripts/das_minimal_pair_codi.py … --slug das-minimal-pair-fixed --site_groups "4;0,2,4"
--k_values 8,16,32,64,128,256,512 --train_n_pairs 200 --eval_n_pairs 100 --epochs 5`. Pod `b7i2qndzf0w1dh`
(RTX L4 secure, $0.49/hr — A5000/A6000/A40 out of stock), ~1 h 45 min sharing the GPU with 3 jobs.
**Headline results** (n=100 eval pairs, matches_twin):

| group | untrained k≤256 | k=8 | 16 | 32 | 64 | 128 | 256 | 512 | raw swap k=768 |
|---|---|---|---|---|---|---|---|---|---|
| z4 | ≤0.01 | 0.13 | 0.18 | 0.23 | 0.29 | 0.29 | 0.25 | 0.29 | 0.34 |
| z0+z2+z4 | ≤0.04 | 0.38 | **0.70** | 0.72 | 0.71 | 0.73 | 0.69 | **0.74** | 0.83 |

(untrained k=512: 0.11 for z4, 0.67 for z0+z2+z4 — a random half of the space already carries most of it.)
**Interpretation:**
- **The perturbed value lives in a low-dimensional linear subspace.** With one rotation shared across
  z0/z2/z4, 16 of 768 dims (2%) reproduce 84% of the raw full-vector effect (0.70 / 0.83), and it saturates
  from k=16 on. z4 alone saturates near its own raw ceiling by k=64 (0.29 / 0.34).
- This reverses every earlier CODI DAS conclusion (`20260920-050602`, `20260920-235312`,
  `20260927-070156`, `20260927-072659`): those nulls came from the target bug (and, for the older
  ones, the site shift), not from the representation.
**Caveats:** n=100 (±~9 pp at 0.7); one seed; the untrained references use one random rotation per k.
**Next:** Read out the learned 16-dim subspace (does its projection linearly decode the step value? does it
align with the P4 cross-mechanism map?).
