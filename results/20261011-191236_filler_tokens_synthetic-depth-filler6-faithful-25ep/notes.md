## 2026-10-12 — Synthetic depth, faithful filler-6, 25 epochs: 0.669, still 12.8 pt below 25-epoch no-CoT (filler_tokens, run_id: 20261011-191236_filler_tokens_synthetic-depth-filler6-faithful-25ep)

**Goal:** Phase A step 3 follow-up to step 2: rerun depth filler-6 (faithful) with the same 25-epoch budget as the
25-epoch no-CoT run (20261011-182123, 0.797), since the step-2 filler run (20261011-093413, 0.338) early-stopped after
8 epochs and was under-trained. This is the depth half of the H2 control: does filler beat no-CoT on deep chains?
**Mechanism / model:** `filler_tokens`, gpt2, full fine-tune, 6 filler positions ('.'), faithful mode (Pfau et al.: 50%
CoT / 50% filler training mix, filler loss unmasked).
**Data:** synthetic depth v1.2.0: train 100k, monitor 349, test 5,576 (500/level, L1 = 76).
**Training:** `scripts/train_synthetic.py`, lr 1e-4 with linear decay over 25 epochs, effective batch 128, **no early
stopping**, seed 42; evaluated the best-on-monitor checkpoint (epoch 25). ~1,150 s/epoch sharing the A40 with the CODI job.
Wrapper `scripts/pod/phase_a_step3.sh` (job `filler6_long`); command in `eval_command.txt`.
**Checkpoint:** `~/Documents/Penn/CIS5980/synthetic_checkpoints/synthetic-depth-filler6-faithful-25ep/best` (+ train log).
**Headline results:** test accuracy **0.669**, AUC 0.694, midpoint at nominal depth **6.98** (collapsed depth 3.0, see
below). Monitor by epoch: 0.13 0.21 0.18 0.24 0.33 0.31 0.40 0.33 0.42 0.43 0.40 0.41 0.53 0.51 0.56 0.49 0.54 0.56 0.61 0.62 0.60 0.56 0.63 0.60 0.65.

| level | accuracy [95% CI] |
|---|---|
| 1 | 1.000 [0.952, 1.000] |
| 2 | 0.984 [0.969, 0.992] |
| 3 | 0.962 [0.941, 0.976] |
| 4 | 0.922 [0.895, 0.942] |
| 5 | 0.800 [0.763, 0.833] |
| 6 | 0.738 [0.698, 0.775] |
| 7 | 0.652 [0.609, 0.692] |
| 8 | 0.576 [0.532, 0.619] |
| 9 | 0.506 [0.462, 0.550] |
| 10 | 0.442 [0.399, 0.486] |
| 11 | 0.372 [0.331, 0.415] |
| 12 | 0.354 [0.313, 0.397] |

Paired exact McNemar vs 25-epoch no-CoT (Coconut repo, 20261011-182123):
- all levels: 0.669 vs 0.797, diff -0.128 [-0.141, -0.115], 373 vs 1088 discordant, p = 3e-81;
- nominal levels >= 6: 0.520 vs 0.710, -0.190 [-0.209, -0.171], p = 2e-81;
- collapsed depth >= 4 (n=1048): 0.379 vs 0.530, -0.151 [-0.186, -0.115], p = 4e-16;
- collapsed depth >= 5 (n=471): 0.289 vs 0.399, -0.110 [-0.161, -0.059], p = 4e-5.

**Interpretation:** With a matched epoch budget, filler-6 roughly doubles its step-2 accuracy (0.338 -> 0.669), but it is
still worse than no-CoT at every depth and on both depth axes. No sign that 6 filler positions add usable serial
compute on this task: H2 (depth) is a no-go at GPT-2 small, as it was on width (ceiling).
**Gotchas hit:** (1) First launch ran on CPU: the pod's main venv had a CUDA-13 torch (driver is 12.8); fixed by
installing torch 2.11+cu128 and calling `.venv/bin/python` (uv run re-syncs). That run was killed after 12 steps; this is
the relaunch. Torch version differs from the step-2 filler run. (2) Manifest `hardware` said "RunPod GPU" (GPU_LABEL not
exported for the relaunch); corrected to A40.
**Caveats:** Confounded comparison: filler (our trainer: linear-decay LR, faithful 50% CoT mix, seed 42) vs no-CoT
(Coconut run.py: constant LR, seed 0). The same-stack no-CoT control (20261011-133416) was only trained 5 epochs, so there
is no matched-stack 25-epoch no-CoT. Both models were still improving on the monitor. Depth level overstates serial depth
(~3x, collapsed depth; `depth_curve.py --by collapsed`); only 72 test items have collapsed depth >= 6. One seed.
**Next:** Same comparison for CODI and Coconut A-lite (step 3). For a clean H2 verdict, a 25-epoch same-stack no-CoT and a v2
depth generator that resists collapsing.
