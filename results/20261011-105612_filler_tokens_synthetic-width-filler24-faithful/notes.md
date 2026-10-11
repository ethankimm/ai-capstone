## 2026-10-11 — Synthetic width task, filler tokens (24 positions, faithful) GPT-2: 0.997, no room left for filler to help (filler_tokens, run_id: 20261011-105612_filler_tokens_synthetic-width-filler24-faithful)

**Goal:** Phase A step 2, H2 width pilot (MULTI_DATASET.md §2): does filler beat no-CoT on a task that needs width, not
depth? 24 filler positions (4x the latent count, to give filler its best chance).
**Mechanism / model:** `filler_tokens`, gpt2, full fine-tune, compute_steps=24, filler token ".".
**Data:** synthetic width v1.2.0 (2-16 single-digit products, answer = unique largest, max sampled uniformly, chance ~3.4%):
train 100k, monitor 450, test 7,500 (500/level).
**Training:** `scripts/train_synthetic.py --method filler_tokens --compute-steps 24 --faithful` (Pfau et al. paper-faithful
mode: 50/50 CoT/filler training mixture, unmasked filler-span loss; same datasets and builders as the GSM8K filler runs).
lr 1e-4, batch 32 x grad-accum 4 = 128 (matched to the Coconut-repo no-CoT/CoT runs), linear decay over max 10 epochs,
early stop after 2 epochs without a >= 0.5 pt monitor gain, best-on-monitor checkpoint evaluated. Seed 42. Pod
`ymuz0w5buqfe26` (A40 secure), sharing the GPU with 2-3 other jobs, so seconds/epoch (2219 s mean) is a shared number.
Command: `eval_command.txt`; wrapper `scripts/pod/phase_a_step2.sh`.
**Checkpoint:** `~/Documents/Penn/CIS5980/synthetic_checkpoints/synthetic-width-filler24-faithful/best` (HF format); training log next to it. Pod path in `metrics.extra.checkpoint_pod_path`.
**Headline results:** test accuracy **0.997**; widths >= 6: 1.000; per level 0.986-1.000
(lowest at widths 2-5). Monitor by epoch: 0.987, 0.998, 0.993, 0.993; best epoch 2, early-stopped after 4. CoT leak rate 0 (the model never
writes steps at test time despite the CoT half of its training mix). Curve flat: no midpoint.
**Interpretation:** The width task is solved without written steps: 0.997 at every width up to 16, about as high as
CoT (`20261011-110121_explicit_cot_synthetic-width-cot`, 0.996). So the H2 width pilot has no headroom: whether filler
beats no-CoT is decided by the no-CoT run on the same task (McNemar, widths >= 6) in the step-2 report, but with
accuracy at ceiling a 5-point gain is impossible. Finding the largest of 16 single-digit products is evidently within
one GPT-2 forward pass, which is itself consistent with the plan's premise that width does not need serial steps.
**Caveats:** one seed; GPU shared. Small dips at widths 2-5 (0.99) are where the L2 space is enumerated and train is
oversampled.
**Next:** no-CoT (Coconut repo and same-stack) on width for the paired test; a harder width task (more chains, larger
operands, or a different aggregate) if H2 is to be tested at all.
