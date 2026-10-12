## 2026-10-12 — Coconut A-lite on synthetic depth (6 latents): 0.311, far below no-CoT at every depth (coconut, run_id: 20261012-064341_coconut_synthetic-depth-coconut-A-lite)

**Goal:** Phase A step 3: does Coconut's latent scratchpad (continuous thoughts) extend the depth no-CoT reaches?
Compared against 25-epoch no-CoT (20261011-182123, 0.797) and CoT (20261011-105901, 0.999, the init).
**Mechanism / model:** `coconut`, GPT-2 small, Coconut repo run.py @27273cb, `configs/synthetic/gpt2_depth_coconut_A_lite.yaml`:
option A curriculum (Coconut's GSM8K schedule: c_thought 2, max_latent_stage 3) with 1 epoch per stage (user's choice,
budget), initialized from the step-2 depth CoT checkpoint (resume 1). Epochs: stage 1 (first written step -> 2 latents),
stage 2 (2 steps -> 4), stage 3 (3 steps -> 6), then 1 epoch at the final stage (all written steps dropped, 6 latents).
Planned 3 final-stage epochs, cut to 1 after measuring speed. lr 1e-4 constant, effective batch 128, seed 0.
**Data:** synthetic depth v1.2.0: train 100k, monitor 349, test 5,576.
**Training:** solo on an A40 (pod y18ac628kz5kln) after sharing the GPU with CODI and filler slowed it to 53 s/step. Seconds per epoch by
stage: 6,198 (stage 1, 7.9 s/step), 10,761 (stage 2), 14,896 (stage 3, ~19 s/step), 6,236 (final: no written steps, shorter
sequences). Monitor accuracy by stage (run.py, at that stage's format): **1.00, 0.93, 0.72, 0.35**.
Eval: `scripts/eval_synthetic_coconut.py --method coconut` (6 latents, batch 1, greedy; `eval_command.txt`), checkpoint_5.
**Checkpoint:** LOST. The pod was removed (around 10:20 UTC 2026-10-12, not by me; billing reached $20.00 at that point) while the
checkpoint was still being synced. Only this results dir (manifest, predictions, train_log.json) survives.
**Headline results:** test accuracy **0.311**, AUC 0.344, midpoint at nominal depth 4.08 (collapsed depth 1.8).

| level | accuracy [95% CI] |
|---|---|
| 1 | 1.000 [0.952, 1.000] |
| 2 | 0.878 [0.846, 0.904] |
| 3 | 0.684 [0.642, 0.723] |
| 4 | 0.526 [0.482, 0.569] |
| 5 | 0.362 [0.321, 0.405] |
| 6 | 0.282 [0.244, 0.323] |
| 7 | 0.174 [0.143, 0.210] |
| 8 | 0.116 [0.091, 0.147] |
| 9 | 0.096 [0.073, 0.125] |
| 10 | 0.068 [0.049, 0.094] |
| 11 | 0.070 [0.051, 0.096] |
| 12 | 0.056 [0.039, 0.080] |

Paired exact McNemar vs 25-epoch no-CoT: all 0.311 vs 0.797, -0.487 [-0.500, -0.472], 91 vs 2804 discordant;
nominal >= 6: 0.123 vs 0.710, -0.587; collapsed >= 5 (n=471): 0.072 vs 0.399, -0.327 [-0.374, -0.278], p = 4e-34;
collapsed >= 6 (n=75): 0.08 vs 0.21, -0.13 [-0.24, -0.02], p = 0.03. By collapsed depth 1..7: 0.72, 0.35, 0.19, 0.14,
0.07, 0.05, 0.14 -- even collapsed depth 1 (single op-type chains) is 0.72 vs no-CoT 0.99.

**Interpretation:** The curriculum works while written steps remain (0.93 at stage 2, 0.72 at stage 3), but at the
all-latent final stage after one epoch the model is far worse than a model that answers with no scratchpad at all. Six
latents standing in for up to 12 steps, with a single epoch at that stage, did not learn usable serial computation. This
is evidence against Coconut A-lite helping, not against Coconut: the final stage got 1 epoch (reference: 3 x 3
plus long final training), and accuracy at stage boundaries typically recovers over several epochs.
**Gotchas hit:** (1) Sharing one GPU with two other trainers made Coconut ~7x slower (many small sequential passes);
moved to its own pod. (2) Option A drops all written steps after stage 3, so levels > 3 jump straight from "3 steps latent,
rest written" to "all latent" (dataset.py skip-all), and 6 latents must cover up to 12 steps. (3) Checkpoint lost (above).
**Caveats:** Under-trained final stage; one seed; different training stack from no-CoT; depth level overstates serial
depth (~3x, `depth_curve.py --by collapsed`).
**Next:** A fair Coconut test needs several final-stage epochs (and ideally option B's 1 latent per step reaching 6 stages)
on a depth task that resists collapsing (v2).
