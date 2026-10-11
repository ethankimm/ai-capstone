## 2026-10-11 — Synthetic depth task, no-CoT GPT-2 (same stack as filler): 0.360, midpoint depth 4.5; filler-6 is 2.2 pt worse (baseline_no_cot, run_id: 20261011-133416_baseline_no_cot_synthetic-depth-nocot-samestack)

**Goal:** Phase A step 2: matched no-CoT control for the depth filler run (H2 core control), and a second no-CoT depth
curve next to the Coconut-repo no-CoT run.
**Mechanism / model:** `baseline_no_cot`, gpt2, full fine-tune.
**Data:** synthetic depth v1.2.0: train 100k, monitor 349, test 5,576 (500/level, L1 = 76).
**Training:** `scripts/train_synthetic.py --method filler_tokens --compute-steps 0` (no filler, not faithful: question ->
answer, i.e. no-CoT) -- the same training script, datasets/collate, lr 1e-4, effective batch 128, linear decay over max 10
epochs and early-stopping rule (2 epochs without >= 0.5 pt monitor gain) as the faithful filler runs, so filler vs this
run isolates the filler positions (and the faithful mode's 50% CoT training half) from the training stack. Seed 42.
GPU shared with other jobs. Command: `eval_command.txt`; wrapper `scripts/pod/phase_a_step2.sh`.
**Logging note:** the script logged it as `filler_tokens` (compute_steps 0); relabelled to `baseline_no_cot` before commit
as the Phase A prompt requires (`metrics.extra.relabelled_from_run_id` holds the original id).
**Checkpoint:** `~/Documents/Penn/CIS5980/synthetic_checkpoints/synthetic-depth-nocot-samestack/best` (epoch 3).
**Headline results:** test accuracy **0.360**, AUC 0.385, midpoint at depth
**4.49**. Monitor by epoch: 0.183, 0.281, 0.367, 0.315, 0.364; best epoch
3, early-stopped after 5. Mean seconds/epoch 1101 (shared GPU).

| level | n | accuracy [95% CI] |
|---|---|---|
| 1 | 76 | 0.789 [0.685, 0.866] |
| 2 | 500 | 0.828 [0.792, 0.859] |
| 3 | 500 | 0.650 [0.607, 0.691] |
| 4 | 500 | 0.502 [0.458, 0.546] |
| 5 | 500 | 0.414 [0.372, 0.458] |
| 6 | 500 | 0.350 [0.309, 0.393] |
| 7 | 500 | 0.270 [0.233, 0.311] |
| 8 | 500 | 0.250 [0.214, 0.290] |
| 9 | 500 | 0.190 [0.158, 0.227] |
| 10 | 500 | 0.176 [0.145, 0.212] |
| 11 | 500 | 0.148 [0.120, 0.182] |
| 12 | 500 | 0.118 [0.093, 0.149] |

**Paired comparison with faithful filler-6** (`20261011-093413_filler_tokens_synthetic-depth-filler6-faithful`, same 5,576
items; `scripts/depth_curve.py` paired McNemar, Newcombe CI): filler 0.338 vs no-CoT 0.360, difference **-0.022 [-0.036,
-0.008]**, exact McNemar p = 0.003 (filler-only correct 744, no-CoT-only 865). Depths >= 4: -0.019 [-0.035, -0.003], p = 0.019.
**Interpretation:** Six filler positions do not help on the serial task; if anything the faithful filler model is slightly
worse (its training mix spends half the examples on CoT targets). This is the plan's H2 core-control prediction: extra
positions without feeding results forward buy nothing on depth. Accuracy decays smoothly with depth from ~0.8 at depths
1-2 to 0.12 at depth 12; the midpoint (4.5) sits well inside 1-12, so the task has the depth range H1 needs.
**Caveats:** L1 accuracy (0.79, n=76) is below L2 (0.83): L1 has only 95 distinct training problems oversampled 88x, and
the test problems are unseen ones, so L1 measures generalisation from very few examples; treat L1 as noisy. One seed;
early stopping fired at a plateau that may not be final (monitor 0.367 -> 0.315 -> 0.364).
**Next:** Coconut-repo no-CoT depth run (still training) for the main no-CoT curve; CODI/Coconut on depth in step 3.
