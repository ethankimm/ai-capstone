## 2026-10-11 — Synthetic width task, no-CoT GPT-2 (Coconut repo): 0.997 at every width (baseline_no_cot, run_id: 20261011-143650_baseline_no_cot_synthetic-width-no_cot)

**Goal:** Phase A step 2 width pilot: the no-CoT lower bound on the width task, trained with the same Coconut-repo recipe
as the CoT run.
**Mechanism / model:** `baseline_no_cot`, gpt2, full fine-tune.
**Data:** synthetic width v1.2.0: train 100k, monitor 450, test 7,500.
**Training:** Coconut repo `run.py` @27273cb, `no_cot: True`, `configs/synthetic/gpt2_width_no_cot.yaml` (1 GPU, 32 x 4 =
128 effective, lr 1e-4 constant, fp32, seed 0). **Deviation: stopped by hand after epoch 1** (monitor 449/450 = 0.998,
which cannot gain the 0.5 pt the early-stop rule requires; two more ~40-min epochs would only have slowed the depth jobs).
Epoch 1 took 2313 s on a GPU shared with 2-3 jobs.
**Checkpoint:** `~/Documents/Penn/CIS5980/synthetic_checkpoints/gpt2_width_no_cot/checkpoint_1`; training log `~/Documents/Penn/CIS5980/synthetic_checkpoints/gpt2_width_no_cot.log`.
**Eval:** `scripts/eval_synthetic_coconut.py --method no_cot` (`eval_command.txt`), greedy, prompt = question + "\n".
**Headline results:** test accuracy **0.997**; widths 2-5 0.986-0.992, widths 6-16
1.000; flat (no midpoint). AUC 0.997.
**Interpretation:** No-CoT solves the width task at every width, as well as CoT (0.996) and filler (0.997-0.998). The H2
width pilot therefore has no headroom (see `20261011-133817_baseline_no_cot_synthetic-width-nocot-samestack` for the
paired test).
**Caveats:** one epoch, one seed.
