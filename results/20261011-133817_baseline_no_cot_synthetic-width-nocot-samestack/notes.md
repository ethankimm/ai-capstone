## 2026-10-11 — Synthetic width task, no-CoT GPT-2 (same stack as filler): 0.997; filler gains +0.3 pt at widths >= 6, far below the 5-pt bar (baseline_no_cot, run_id: 20261011-133817_baseline_no_cot_synthetic-width-nocot-samestack)

**Goal:** Phase A step 2: matched no-CoT control for the H2 width pilot (filler 6 and 24 vs no-CoT, McNemar pooled over
widths >= 6; go if gain >= 5 pt with CI excluding 0).
**Mechanism / model:** `baseline_no_cot`, gpt2, full fine-tune.
**Data:** synthetic width v1.2.0: train 100k, monitor 450, test 7,500 (500/level).
**Training:** `scripts/train_synthetic.py --method filler_tokens --compute-steps 0` (no filler, not faithful: question ->
answer, i.e. no-CoT) -- the same training script, datasets/collate, lr 1e-4, effective batch 128, linear decay over max 10
epochs and early-stopping rule (2 epochs without >= 0.5 pt monitor gain) as the faithful filler runs, so filler vs this
run isolates the filler positions (and the faithful mode's 50% CoT training half) from the training stack. Seed 42.
GPU shared with other jobs. Command: `eval_command.txt`; wrapper `scripts/pod/phase_a_step2.sh`.
**Logging note:** the script logged it as `filler_tokens` (compute_steps 0); relabelled to `baseline_no_cot` before commit
as the Phase A prompt requires (`metrics.extra.relabelled_from_run_id` holds the original id).
**Checkpoint:** `~/Documents/Penn/CIS5980/synthetic_checkpoints/synthetic-width-nocot-samestack/best` (epoch 1).
**Headline results:** test accuracy **0.997**; per level 0.990-1.000; flat (no midpoint).
Monitor 1.000 at every epoch; early-stopped after 3. Mean seconds/epoch 1292 (shared GPU).
**H2 pilot, paired over widths >= 6 (n = 5,500):** filler-6 and filler-24 are both 1.000 vs no-CoT 0.9967: difference
**+0.0033 [+0.0019, +0.0052]**, exact McNemar p < 1e-5 (18 items right only with filler, 0 only without). Against the
Coconut-repo no-CoT run (`20261011-143650_baseline_no_cot_synthetic-width-no_cot`, 1.000 at widths >= 6) the difference is 0.
**Interpretation:** H2 pilot is a **no-go** by the pre-set rule: the filler gain is statistically detectable but 0.3 pt,
because no-CoT is already at ceiling. The width task is solved in one forward pass at every width up to 16, so it cannot
show whether filler helps on width; it would need a harder width task (more chains, larger values, or an aggregate that
is harder than max).
**Caveats:** one seed. The 18 filler-only items are spread over widths 11-16 (no-CoT 0.990-0.996 there).
