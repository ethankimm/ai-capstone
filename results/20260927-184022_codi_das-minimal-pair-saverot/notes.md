## 2026-09-27 — E4 rerun that saves the rotations, CODI z0+z2+z4: k=16 steers 63%, k=32 69% (raw swap 82%, untrained 0%) — replicates the fixed E4 (codi, run_id: 20260927-184022_codi_das-minimal-pair-saverot)

**Goal:** `20260927-094215_codi_das-minimal-pair-fixed` did not save its learned rotations, and the two
follow-ups need them (DAS-subspace decoding, P4 restricted to the value subspace). Retrain the z0+z2+z4
group at k=16 and k=32 with `--save_rotations`; the eval doubles as a second training run of E4.
**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3`, aligned sites, transformer + LoRA frozen.
**Data / hyperparams:** identical to `20260927-094215` (200 train pairs from gsm8k-aug train n=2500, 100 eval
pairs from validation n=800; lr 1e-3, 5 epochs, cosine; same seeds). Train acc 0.790, eval acc 0.789.
**Command:** `das_minimal_pair_codi.py $CODI_FLAGS --slug das-minimal-pair-saverot --stage full_run
--site_groups "0,2,4" --k_values 16,32 --train_n_pairs 200 --eval_n_pairs 100 --epochs 5
--save_rotations /workspace/rotations/codi` (full line in `eval_command.txt`). Pod `ft7u4einf6ci9s`
(RTX A5000 secure, $0.27/hr), 14 min sharing the GPU with 3 other jobs.
**Headline results** (n=100, matches_twin): k=16 **0.63**, k=32 **0.69**; raw full-vector swap 0.82;
untrained random rotation 0.00 at k=16 and k=32.
**Interpretation:** Replicates E4 within noise (previous run: k=16 0.70, k=32 0.72, raw 0.83; ±9 pp at n=100).
The trained rotation, not the patch size, carries the effect (untrained 0.00).
**Artifacts:** `rot_0+2+4_k16.pt`, `rot_0+2+4_k32.pt` ({"W": 768×768, "k", "group"}; the subspace is the first
k rows of W), kept outside the repo at `~/Documents/Penn/CIS5980/xmech_artifacts/rotations/codi/`.
**Caveats:** Training is not bit-deterministic across runs (0.70 → 0.63 at k=16 with the same seeds).
**Next:** `20260927-185312_codi_das-subspace-probe` (decoding) and the xmech-subspace transplants.
