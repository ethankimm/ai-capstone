## 2026-10-11 — Synthetic depth task, filler tokens (6 positions, faithful) GPT-2: accuracy falls from 0.99 at depth 1 to ~0.15 at depth 12, midpoint 3.5 (filler_tokens, run_id: 20261011-093413_filler_tokens_synthetic-depth-filler6-faithful)

**Goal:** Phase A step 2, H2 core control (MULTI_DATASET.md §2): filler tokens with as many positions as CODI/Coconut have
latents (6), trained on the depth task. If filler matched the latent models here, their gain would be about extra
positions, not feeding results forward. Also gives the filler depth curve against no-CoT.
**Mechanism / model:** `filler_tokens`, gpt2, full fine-tune, compute_steps=6, filler token ".".
**Data:** synthetic depth v1.2.0 (one chain of 1-12 dependent +,-,* steps, values 0..99, single tokens): train 100k
(balanced; L1/L2 enumerated and oversampled), monitor 349, test 5,576 (500/level, L1 = 76).
**Training:** `scripts/train_synthetic.py --method filler_tokens --compute-steps 6 --faithful` (Pfau et al. mode: 50/50
CoT/filler training mix, unmasked filler loss). lr 1e-4, batch 32 x grad-accum 4 = 128 (as the Coconut-repo runs), linear
decay over max 10 epochs, early stop after 2 epochs without a >= 0.5 pt monitor gain. Seed 42. Pod `ymuz0w5buqfe26` (A40
secure), GPU shared with 2-3 other jobs, so seconds/epoch (1962 s mean; 1,391-2,568 s) is a shared number.
Command: `eval_command.txt`; wrapper `scripts/pod/phase_a_step2.sh`.
**Checkpoint:** `~/Documents/Penn/CIS5980/synthetic_checkpoints/synthetic-depth-filler6-faithful/best` (HF format, epoch 6); training log next to it.
**Headline results:** test accuracy **0.338**, AUC 0.368, midpoint (halfway
between fitted floor 0.150 and best 0.987) at depth **3.46**. CoT leak
rate 0. Monitor by epoch: 0.129, 0.186, 0.206, 0.218, 0.301, 0.332, 0.332, 0.272; best epoch 6, stopped after 8.

| depth | n | accuracy [95% CI] |
|---|---|---|
| 1 | 76 | 0.987 [0.929, 0.998] |
| 2 | 500 | 0.780 [0.742, 0.814] |
| 3 | 500 | 0.600 [0.556, 0.642] |
| 4 | 500 | 0.476 [0.433, 0.520] |
| 5 | 500 | 0.370 [0.329, 0.413] |
| 6 | 500 | 0.300 [0.261, 0.342] |
| 7 | 500 | 0.250 [0.214, 0.290] |
| 8 | 500 | 0.238 [0.203, 0.277] |
| 9 | 500 | 0.202 [0.169, 0.239] |
| 10 | 500 | 0.150 [0.121, 0.184] |
| 11 | 500 | 0.112 [0.087, 0.143] |
| 12 | 500 | 0.146 [0.118, 0.180] |

**Interpretation:** Without written steps the model handles one step (0.99) and then loses accuracy with every added step;
by depth 10-12 it is near 0.11-0.15 (chance for a uniformly guessed 0..99 answer is ~0.01, but answers are not uniform).
This is the shape H1 needs: a depth range where a single forward pass with 6 extra positions is not enough while CoT stays
at 0.999 (`20261011-105901_explicit_cot_synthetic-depth-cot`). Whether the 6 filler positions help at all is the
comparison with the no-CoT runs (Coconut-repo and same-stack), done in the step-2 report with a paired test.
**Caveats:** the monitor curve was still noisy (0.30 -> 0.33 -> 0.33 -> 0.27) when early stopping fired at the 0.5-pt
rule; the no-CoT runs on the same task were still improving at that point, so "filler vs no-CoT" must be read with each
model's own stopping epoch in mind. One seed. L1 has n=76.
**Next:** paired comparison against no-CoT; CODI and Coconut on this task (step 3).
