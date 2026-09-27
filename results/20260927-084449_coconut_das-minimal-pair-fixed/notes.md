## 2026-09-27 — E4 with the teacher-forcing target and data split fixed, Coconut: an 8-dim subspace of passes 1+4 steers 50%, 32 dims 67% (raw swap 75%) (coconut, run_id: 20260927-084449_coconut_das-minimal-pair-fixed)

**Goal:** Rerun Coconut's minimal-pair DAS (`20260920-233519`, `20260921-195626`: flat nulls) with two bugs
fixed, plus the untrained references added on 2026-09-27 to the CODI script.
**Bugs fixed:** (1) target was " ### {n}" (" ###" = token 44386); Coconut emits "### {n}" ("###" = 21017),
so training pushed toward a first token the model never produces (losses 12–34 nats). (2) train and eval
pools both came from the 1194-example gold-trace TEST file with train_pool_n ≥ 1194, so every eval
recipient was also a training recipient. Now train = `gsm_original_train.json` (n=2500, acc 0.921 — the
checkpoint's own training data), eval = `gsm_original_valid.json` (n=500, acc 0.366). Also fixed in
`das_coconut.py`.
**Mechanism / model:** `coconut`, gpt2 / `hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33`; lr 1e-3,
5 epochs; 200 train / 100 eval pairs.
**Command:** `scripts/das_minimal_pair_coconut.py … --slug das-minimal-pair-fixed --site_groups "1;1,4"
--k_values 8,16,32,64,128,256,512 --train_n_pairs 200 --eval_n_pairs 100 --epochs 5`. Pod `b7i2qndzf0w1dh`
(L4 secure), ~55 min sharing the GPU.
**Headline results** (n=100, matches_twin):

| group | untrained k≤256 | k=8 | 16 | 32 | 64 | 128 | 256 | 512 | raw swap k=768 |
|---|---|---|---|---|---|---|---|---|---|
| pass 1 | ≤0.01 | 0.22 | 0.39 | 0.41 | 0.43 | 0.44 | 0.49 | 0.49 | 0.57 |
| passes 1+4 | ≤0.08 | **0.50** | 0.61 | **0.67** | 0.62 | 0.65 | 0.66 | 0.67 | 0.75 |

**Interpretation:**
- Same picture as CODI (`20260927-094215_codi_das-minimal-pair-fixed`): the value sits in a low-dim linear
  subspace — 8 dims of one rotation shared by passes 1 and 4 give two-thirds of the raw-swap effect, 32
  dims 89%. Coconut saturates at slightly smaller k than CODI (8–32 vs 16).
- The earlier Coconut DAS nulls (pilot, E4, saturation sweep up to k=256) were the target/split bugs.
**Caveats:** n=100; one seed. Eval pairs come from a different file (valid) than aligned E3 (test), so the
raw-swap ceiling (0.75) differs from E3's 0.827.
**Next:** as for CODI — read out the subspace and compare it across mechanisms via the P4 map.
