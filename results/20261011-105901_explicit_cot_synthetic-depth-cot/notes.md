## 2026-10-11 — Synthetic depth task, explicit CoT GPT-2: 0.999 at every depth 1-12 (explicit_cot, run_id: 20261011-105901_explicit_cot_synthetic-depth-cot)

**Goal:** Phase A step 2: the CoT upper bound on the depth task (one chain of 1-12 dependent +,-,* steps, values 0..99),
and the init checkpoint for step-3 Coconut.
**Mechanism / model:** `explicit_cot`, gpt2 (openai-community/gpt2), full fine-tune.
**Data:** synthetic depth v1.2.0 (`latentreasoning/tasks`, seed 5980): train 100k (balanced, L1/L2 oversampled), test 5,576
(500/level, L1 = 76).
**Training:** Coconut repo `run.py` @27273cb, `cot: True`, config `configs/synthetic/gpt2_depth_cot.yaml` (adapted from
Dilgren & Wiegreffe `gpt2_prosqa_cot.yaml`: 1 GPU, batch 32 x grad-accum 4 = effective 128 as their 4 x 32; lr 1e-4
constant, wd 0.01, fp32, seed 0, `save_only_improve: True`, monitor set = `valid_monitor.json`). Launched with
`WANDB_MODE=disabled torchrun --nproc_per_node 1` via `scripts/pod/phase_a_step2.sh`, sharing one A40 with 2 other
training jobs, so seconds/epoch is a 3-way-shared number, not solo throughput.
**Deviation: stopped by hand after epoch 1.** The planned rule was early stop after 2 epochs without a >= 0.5 pt monitor
gain. Monitor accuracy after epoch 1 was 349/349 = 1.0, which is at the ceiling, so two more ~1 h epochs could not
improve it and would have blocked the job queue. The epoch-1 checkpoint is the one evaluated (and the best on monitor).
**Checkpoint:** `~/Documents/Penn/CIS5980/synthetic_checkpoints/gpt2_depth_cot/checkpoint_1` (Coconut state dict, 652 MB;
the step-3 Coconut init); training log next to it (`gpt2_depth_cot.log`). Pod path in `metrics.extra.checkpoint_pod_path`.
**Eval:** `scripts/eval_synthetic_coconut.py` (`eval_command.txt`): greedy, prompt = question + "\n" exactly as run.py,
answer after "###", batched decode verified identical to batch-1 on 32 examples, full test.json.
The manifest's curve fields were recomputed with the flat-curve rule (`depth_curve.py`, commit e130331) before commit:
the first eval reported a fit-noise "midpoint" at level 7.9 for depth.
**Headline results:** test accuracy **0.999**; per level 1.000 for L1-L7 and 0.996-0.998 for L8-L12. Flat, so no midpoint
(beyond range). AUC 0.999. Mean CoT length 57 tokens. Train time 3,584 s for the epoch (shared GPU).
**Interpretation:** Writing the intermediate values makes the depth task trivial for GPT-2 small at every depth, as
intended for the upper bound: whatever no-CoT loses with depth is attributable to doing the steps without a scratchpad.
**Caveats:** one epoch, one seed. **Next:** no-CoT and filler on the same task; Coconut from this checkpoint in step 3.
