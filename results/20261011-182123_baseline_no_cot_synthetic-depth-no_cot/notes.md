## 2026-10-11 — Synthetic depth, no-CoT GPT-2 (Coconut repo), 25 epochs: 0.797, midpoint depth 7.8 (baseline_no_cot, run_id: 20261011-182123_baseline_no_cot_synthetic-depth-no_cot)

**Goal:** Phase A step 2: the no-CoT floor of the depth curve (how many serial multiply/add steps one forward pass of
GPT-2 small handles), next to CoT (0.999, run 20261011-105901).
**Mechanism / model:** `baseline_no_cot`, openai-community/gpt2, full fine-tune with the Coconut repo `run.py`
(@27273cb, `no_cot: True`): question -> "### answer".
**Data:** synthetic depth v1.2.0: train 100k, monitor `valid_monitor.json` 349, test 5,576 (500/level, L1 = 76).
**Training:** `configs/synthetic/gpt2_depth_no_cot.yaml`: lr 1e-4 constant, effective batch 128 (32 x 4), 25 epochs,
`save_only_improve`. Evaluated checkpoint_25 (the last and best-on-monitor). Seconds/epoch ~2,300 while sharing the GPU
with 3-4 other jobs (epochs 1-4), **535 solo** (epochs 6-25). Command: `eval_command.txt`; wrapper `scripts/pod/phase_a_step2.sh`.
**Checkpoint:** `~/Documents/Penn/CIS5980/synthetic_checkpoints/gpt2_depth_no_cot/checkpoint_25` (also checkpoint_9, the
epoch-9 best that the intended 10-epoch cap would have kept, monitor 0.524); train log alongside.
**Headline results:** test accuracy **0.797**, AUC 0.814, midpoint at depth **7.79** (floor-to-best halfway, in range).
Monitor by epoch: 0.18 0.26 0.32 0.35 0.36 0.38 0.32 0.46 0.52 0.51 0.52 0.47 0.52 0.62 0.55 0.59 0.41 0.72 0.50 0.77
0.58 0.69 0.66 0.76 **0.81** -- still rising at epoch 25, and noisy (swings of up to 0.2 between epochs).

| level | accuracy [95% CI] |
|---|---|
| 1 | 1.000 [0.952, 1.000] |
| 2 | 0.988 [0.974, 0.994] |
| 3 | 0.978 [0.961, 0.988] |
| 4 | 0.918 [0.891, 0.939] |
| 5 | 0.884 [0.853, 0.909] |
| 6 | 0.838 [0.803, 0.868] |
| 7 | 0.804 [0.767, 0.836] |
| 8 | 0.752 [0.712, 0.788] |
| 9 | 0.706 [0.665, 0.744] |
| 10 | 0.676 [0.634, 0.716] |
| 11 | 0.616 [0.573, 0.658] |
| 12 | 0.578 [0.534, 0.621] |

Paired exact McNemar, this run vs the earlier step-2 depth runs (all levels):
- vs faithful filler-6 (20261011-093413, 0.338): +0.459 [0.444, 0.473], 2708 vs 150 discordant, p ~ 0.
- vs same-stack no-CoT (20261011-133416, 0.360): +0.44, p ~ 0.

**Interpretation:** One forward pass of GPT-2 small does much more of this task than the early-stopped step-2 runs
suggested: >= 0.88 up to depth 5, 0.58 at depth 12. The step-2 depth runs that early-stopped (filler-6 after 8 epochs,
same-stack no-CoT after 5) were badly under-trained, so the step-2 filler-vs-no-CoT depth comparison (-2.2 pt) says
nothing about filler; a 25-epoch filler-6 rerun is part of step 3. Depth-range decision: the no-CoT midpoint is inside
1-12 (7.8) and CoT (0.999) stays far above it from depth ~4 on, so the range works, but the informative region for a
latent scratchpad is depth >= 6.
**Gotchas hit:** (1) This run was meant to stop at 10 epochs. `coconut_watch.py` printed "stopping" at epoch 10, but
`killpg(getpgid(pid))` got the pid of the wrapper subshell, not the setsid'd torchrun group, so it killed the job
wrapper and training ran on to `num_epochs` 25, and the wrapper's eval never ran; this eval was launched by hand on
checkpoint_25. Fixed in `scripts/pod/phase_a_step2.sh` (torchrun writes its own pid). (2) The 349-example monitor set is
too small and noisy for patience-2 early stopping.
**Caveats:** Not converged (monitor still rising at 25). One seed. The CoT comparison run trained 1 epoch (at ceiling).
**Next:** Step 3 on depth (Coconut A-lite, CODI 10 epochs) + filler-6 25-epoch rerun, compared against this run on
levels >= 6.
