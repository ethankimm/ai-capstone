## 2026-10-12 — CODI GPT-2 on synthetic depth, 10 epochs: 0.395, far below no-CoT; under-trained, flat curve (codi, run_id: 20261012-034825_codi_synthetic-depth-codi)

**Goal:** Phase A step 3: does CODI's latent scratchpad (6 continuous thoughts) extend the depth no-CoT reaches?
Compared against the 25-epoch no-CoT run (20261011-182123, 0.797) and CoT (20261011-105901, 0.999).
**Mechanism / model:** `codi`, gpt2 + LoRA r128/a32 + projection, 6 latents; CODI reference repo @2c23146 plus
`scripts/codi_synthetic.patch` (data_name `depth`: question + "\n", teacher CoT = steps[:-1] joined by "\n", target
"The answer is: N").
**Data:** synthetic depth v1.2.0: train 100k, test 5,576 (500/level, L1 = 76). No per-epoch monitor (CODI has none).
**Training:** `configs/synthetic/train_gpt2_depth_codi.sh` (Dilgren & Wiegreffe's ProsQA CODI script) with
**10 epochs instead of 40** (budget) and 32 x 4 instead of 64 x 2 (same 128 effective): lr 3e-3 cosine, bf16, seed 11.
~4.1 s/step, ~49 min/epoch sharing the A40 with the filler-6 job. Training loss by tenth of the run: 1.15 0.78 0.58 0.53
0.51 0.43 0.39 0.35 0.31 0.29 -- still falling when the cosine schedule ended. Wrapper `scripts/pod/phase_a_step3.sh`
(job `codi`); eval `scripts/eval_synthetic_codi.py` (greedy, 6 latent iterations, last number in output), command in
`eval_command.txt`. Mid-run monitor check (epoch 3 checkpoint, valid_monitor): 0.238.
**Checkpoint:** `~/Documents/Penn/CIS5980/synthetic_checkpoints/gpt2_depth_codi/seed_11` (final weights + checkpoint-7038/7820
without optimizer state; train log and eval log alongside).
**Headline results:** test accuracy **0.395**, AUC 0.413, midpoint at nominal depth 1.74.

| level | accuracy [95% CI] |
|---|---|
| 1 | 1.000 [0.952, 1.000] |
| 2 | 0.558 [0.514, 0.601] |
| 3 | 0.422 [0.379, 0.466] |
| 4 | 0.362 [0.321, 0.405] |
| 5 | 0.306 [0.267, 0.348] |
| 6 | 0.332 [0.292, 0.374] |
| 7 | 0.312 [0.273, 0.354] |
| 8 | 0.374 [0.333, 0.417] |
| 9 | 0.374 [0.333, 0.417] |
| 10 | 0.364 [0.323, 0.407] |
| 11 | 0.430 [0.387, 0.474] |
| 12 | 0.420 [0.378, 0.464] |

By collapsed depth 1..7 (`depth_curve.py --by collapsed`): 0.52, 0.41, 0.36, 0.37, 0.30, 0.19 (n=43), 0.32 (n=28) --
nearly flat. Paired exact McNemar vs 25-epoch no-CoT: all levels 0.395 vs 0.797, -0.402 [-0.417, -0.387], 276 vs 2518
discordant; nominal >= 6: -0.338 [-0.358, -0.317]; collapsed >= 5 (n=471): 0.289 vs 0.399, -0.110 [-0.166, -0.054],
p = 2e-4; collapsed >= 6 (n=75): 0.24 vs 0.21, +0.03 [-0.12, +0.17], p = 0.86.

**Interpretation:** This CODI model is under-trained, not a fair test of the mechanism. It fails even shallow,
long problems that no-CoT solves: on chains of a single op type (collapsed depth 1) it gets 0.56 at nominal L2 and
~0.3-0.6 beyond, vs 0.99 for no-CoT. The curve is flat by depth (no deep-problem advantage, and at collapsed >= 6 it is
level with no-CoT on 75 items). The non-monotone nominal curve (0.31 at L5, 0.42 at L12) follows the item mix:
L12 has more items with few op-type switches. Not a constant-answer artifact (prediction distributions are spread like
the gold ones; majority-class rate <= 8%).
**Gotchas hit:** None in the run. Teacher CoT drops the final step (as icot / prosqa), so for L1 items the teacher CoT is empty.
**Caveats:** 10 of the reference 40 epochs, loss still falling; one seed; compared across training stacks; depth level
overstates serial depth (~3x, see the step-2 report). A CODI verdict needs the full 40-epoch run.
**Next:** If CODI matters for step 5, run it at the reference 40 epochs (~4x this cost) or on a v2 depth generator.
