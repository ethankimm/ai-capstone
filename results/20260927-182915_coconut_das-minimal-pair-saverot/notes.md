## 2026-09-27 — E4 rerun that saves the rotations, Coconut passes 1+4: k=16 steers 63%, k=32 65% (raw swap 75%, untrained 0%) — replicates the fixed E4 (coconut, run_id: 20260927-182915_coconut_das-minimal-pair-saverot)

**Goal:** As `20260927-184022_codi_das-minimal-pair-saverot`: `20260927-084449_coconut_das-minimal-pair-fixed`
did not save its rotations. Retrain the passes 1+4 group at k=16/32 with `--save_rotations`.
**Mechanism / model:** `coconut`, gpt2 / `hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33`, model frozen.
**Data / hyperparams:** identical to `20260927-084449` (200 train pairs from `gsm_original_train.json` n=2500,
acc 0.921; 100 eval pairs from `gsm_original_valid.json` n=500, acc 0.366; lr 1e-3, 5 epochs).
**Command:** `das_minimal_pair_coconut.py --checkpoint_path .../checkpoint_33 --data_dir /workspace/coconut_data
--slug das-minimal-pair-saverot --stage full_run --site_groups "1,4" --k_values 16,32 --train_n_pairs 200
--eval_n_pairs 100 --epochs 5 --save_rotations /workspace/rotations/coconut`. Pod `ft7u4einf6ci9s`
(RTX A5000 secure), 9 min sharing the GPU.
**Headline results** (n=100, matches_twin): k=16 **0.63**, k=32 **0.65**; raw full-vector swap 0.75;
untrained 0.00 at both k.
**Interpretation:** Replicates E4 (previous: k=16 0.61, k=32 0.67, raw 0.75).
**Artifacts:** `rot_1+4_k16.pt`, `rot_1+4_k32.pt` at `~/Documents/Penn/CIS5980/xmech_artifacts/rotations/coconut/`.
**Caveats:** n=100; the eval base accuracy (0.366) is on the valid file, as before.
**Next:** `20260927-185326_coconut_das-subspace-probe` and the xmech-subspace transplants.
