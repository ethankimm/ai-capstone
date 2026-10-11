## 2026-10-11 — Synthetic width task, explicit CoT GPT-2: 0.996 at every width 2-16 (explicit_cot, run_id: 20261011-110121_explicit_cot_synthetic-width-cot)

**Goal:** Phase A step 2 width pilot: CoT upper bound on the width task (2-16 independent single-digit products, answer =
the unique largest; max sampled uniformly over achievable products, chance ~3.4%).
**Mechanism / model:** `explicit_cot`, gpt2 (openai-community/gpt2), full fine-tune.
**Data:** synthetic width v1.2.0 (data identical to v1.1.0): train 100k, test 7,500 (500/level).
**Training:** Coconut repo `run.py` @27273cb, `cot: True`, config `configs/synthetic/gpt2_width_cot.yaml` (adapted from
Dilgren & Wiegreffe `gpt2_prosqa_cot.yaml`: 1 GPU, batch 32 x grad-accum 4 = effective 128 as their 4 x 32; lr 1e-4
constant, wd 0.01, fp32, seed 0, `save_only_improve: True`, monitor set = `valid_monitor.json`). Launched with
`WANDB_MODE=disabled torchrun --nproc_per_node 1` via `scripts/pod/phase_a_step2.sh`, sharing one A40 with 2 other
training jobs, so seconds/epoch is a 3-way-shared number, not solo throughput.
**Deviation: stopped by hand after epoch 1.** The planned rule was early stop after 2 epochs without a >= 0.5 pt monitor
gain. Monitor accuracy after epoch 1 was 418/450 = 0.929, which is at the ceiling: the 30 width-16 monitor items always score 0 inside run.py, whose generation cap is 128 tokens while a width-16 CoT needs ~130, so the reachable maximum is 420/450, so two more ~1 h epochs could not
improve it and would have blocked the job queue. The epoch-1 checkpoint is the one evaluated (and the best on monitor).
**Checkpoint:** `~/Documents/Penn/CIS5980/synthetic_checkpoints/gpt2_width_cot/checkpoint_1` (Coconut state dict, 652 MB;
the step-3 Coconut init); training log next to it (`gpt2_width_cot.log`). Pod path in `metrics.extra.checkpoint_pod_path`.
**Eval:** `scripts/eval_synthetic_coconut.py` (`eval_command.txt`): greedy, prompt = question + "\n" exactly as run.py,
answer after "###", batched decode verified identical to batch-1 on 32 examples, full test.json.
The manifest's curve fields were recomputed with the flat-curve rule (`depth_curve.py`, commit e130331) before commit:
the first eval reported a fit-noise "midpoint" at level 7.9 for depth.
**Headline results:** test accuracy **0.996** (our eval sizes its token budget from the data, 146 tokens, so width 16 is
scored: 0.998). Per level 0.986-1.000; flat, no midpoint. AUC 0.996. Mean CoT length 74 tokens. Train 3,994 s for the epoch (shared GPU).
**Interpretation:** CoT solves the width task at every width; the upper bound for the H2 pilot.
**Caveats:** one epoch, one seed. The Coconut run.py monitor metric under-counts width 16 (see above); test numbers don't.
**Next:** no-CoT, filler-6 and filler-24 on the same task; McNemar filler vs no-CoT pooled over widths >= 6.
