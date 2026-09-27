## 2026-09-27 — E4 rerun with aligned sites, CODI: the raw z0/z2/z4 swap steers 84%, but no learned subspace up to k=64 recovers more than 6% (codi, run_id: 20260927-070156_codi_das-minimal-pair-aligned)

**Goal:** Rerun E4 (`20260920-235312_codi_das-minimal-pair`, flat null ≤4%) with the site-indexing fix.
The old run spliced the twin's z_i into a background of the recipient's z_{i-1}. Here site s = z_s feeds
iteration s+1, and both background and donor are z_s. Groups follow aligned E3 (`20260927-004340`):
**z4 alone** (strongest single site) and **z0+z2+z4 jointly** (the value-carrying sites; mirrors Coconut's
"1" and "1+4"). New untrained references on the same eval pairs: `full` (k=768 — any orthogonal R gives
exactly the raw full-vector swap) and `untrained` (random R at each k).
**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3`, 6 latents; transformer + LoRA frozen.
**Data:** train pairs from gsm8k-aug train n=2500 (acc 0.790, 200 qualified pairs / 296 checked); eval
pairs from validation n=800 (acc 0.790, 100 / 165). Same pools and seeds as the old run.
**Command:** `scripts/das_minimal_pair_codi.py … --slug das-minimal-pair-aligned --site_groups "4;0,2,4"
--k_values 8,16,32,64 --train_n_pairs 200 --eval_n_pairs 100 --epochs 5` (see `eval_command.txt`). Pod
`5wb7p2e1kr568o` RTX A6000 secure, ~40 min in parallel with the E2 and ladder runs.
**Headline results** (n=100 eval pairs, matches_twin / answer_changed):

| group | full k=768 (raw swap) | untrained k≤64 | k=8 | k=16 | k=32 | k=64 |
|---|---|---|---|---|---|---|
| z4 | **0.41** / 0.58 | 0.00 / ≤0.01 | 0.02 / 0.03 | 0.02 / 0.08 | 0.00 / 0.10 | 0.02 / 0.09 |
| z0+z2+z4 | **0.84** / 0.95 | 0.00 / ≤0.01 | 0.03 / 0.15 | 0.02 / 0.16 | 0.05 / 0.16 | **0.06** / 0.27 |
| old run (shifted), best cell | — | — | | | 0.04 (1+2+3+4) | |

Training loss (teacher-forced CE on the twin's answer) now reaches ~9–11 nats on z0+z2+z4 (old run:
plateau at 30–40).
**Interpretation:**
- The references validate the fix: the untrained raw swap on these validation pairs reproduces aligned
  E3 (84% vs 83.9% on the test set; z4 alone 41% vs 36%).
- DAS is still a null relative to that ceiling. A learned k≤64 subspace (≤8% of the dims) moves the
  answer more as k grows (answer_changed 15→27%) but lands on the twin's answer at most 6%. Either the
  value is spread over many more than 64 dimensions, or 200 training pairs × 5 epochs cannot find the
  rotation — the loss is still ~10 nats, so training is the more likely limit.
- The old "DAS null" for CODI stands qualitatively, but it is no longer evidence against the value being
  in the latents: the raw vectors at the same sites carry it 84% of the time.
**Caveats:** n=100 eval pairs (±~5 pp); one seed; no hyperparameter search (lr 1e-3, 5 epochs).
**Next:** large-k extension `*_codi_das-minimal-pair-aligned-bigk` (k = 128/256/512, z0+z2+z4), same pools.

**Update (large-k extension `20260927-072659_codi_das-minimal-pair-aligned-bigk`):** at k=512 the untrained rotation steers 71% and the trained one 22% — the DAS training recipe is failing, so the ≤6% here is not evidence against a subspace.

**Resolved (2026-09-27):** the DAS teacher-forcing target was wrong (CODI: bare number instead of "The answer is: N"; Coconut: " ###" instead of "###"), and the Coconut minimal-pair script trained and evaluated on overlapping test examples. With both fixed, DAS finds a ~16-dim value subspace in both mechanisms: `20260927-094215_codi_das-minimal-pair-fixed` (z0+z2+z4, k=16: 0.70 vs raw 0.83), `20260927-084449_coconut_das-minimal-pair-fixed` (passes 1+4, k=32: 0.67 vs raw 0.75). This run's null is an artifact.
