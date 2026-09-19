## 2026-09-19 — Pilot: paper recipe on the explicit-CoT format — the loop collapses even with task headroom (recurrent_depth, run_id: 20260919-031852_recurrent_depth_cot-split4-4-4-rbar32-pilot)

**Goal:** Two questions left open by the five answer-only pilots and the step-supervised
runs. (1) Geiping et al. report GSM8K *with* chain-of-thought (8-shot CoT; 24.9% strict /
38.1% flexible at r=32, 0% at r=1): the loop adds per-token compute to a visible
rationale. Our answer-only runs were never that setting. (2) The remaining explanation
for the answer-only collapse was "no gradient pressure": on direct answers GPT-2 sits at
the guessing floor, so extra depth cannot lower the loss. The explicit-CoT task has
headroom (pilot loss 0.56, full-data 0.22, accuracy 4.5% → 34.5%), so if the collapse is
about task signal the loop should stay alive here and accuracy should rise with r. Same
architecture, same paper recipe (r ~ log-normal Poisson(32, ½), k=8), only the target
format changes to `explicit_cot`'s (rationale, then `#### answer`).
**Mechanism / model:** `recurrent_depth`, gpt2 (125,621,760 params) /
`results/20260919-031852_recurrent_depth_cot-split4-4-4-rbar32-pilot/ckpt` (`model.pt`, synced,
gitignored). `LoopedGPT2` (4, 4, 4) split, identity adapter init, state LN, `s_0 ~ N(0, 2/5)`.
Objective `cot` (`scripts/train_recurrent_depth.py --cot`): ordinary next-token loss over
` {rationale}\n#### {answer}` + EOS with the prompt masked, exactly `explicit_cot`'s
target; decoding is greedy CoT (`max_new_tokens=128`, EOS-terminated) with the whole
prefix re-looped per token (no KV cache). **Not a latent scratchpad** — the reasoning is
in the tokens; `compute_steps=32` is the training r̄ as in the answer-only runs and
`extra.cot_tokens=29.4` is the generated rationale length (explicit-CoT runs: 30–34).
**Data:** gsm8k-aug train 20,000-example subset (seed 42); eval gsm8k-aug test n=200 seed=0;
run seed=42. Sequences are ~35 tokens longer than the direct-answer format (mean ~125).
**Hyperparams:** {'objective': 'cot', 'lr': 5e-05, 'epochs': 3, 'batch_size': 16, 'train_n':
20000, 'n_prelude': 4, 'n_core': 4, 'n_coda': 4, 'mean_recurrence': 32, 'lognormal_sigma': 0.5,
'backprop_last_k': 8, 'checkpoint_iterations': False, 'init_state_std': 0.632,
'adapter_init': 'identity', 'core_init': 'pretrained', 'eval_recurrences': [1, 2, 4, 8, 16, 32,
64], 'max_new_tokens': 128}; AdamW, linear decay, no warmup, grad-clip 1.0, bf16 autocast.
Code at commit 32c7044.
**Command:** `uv run python scripts/train_recurrent_depth.py --cot --train-n 20000 --stage pilot --hardware "RunPod RTX A4500 (secure)" --slug cot-split4-4-4-rbar32-pilot`
(pod `zwdegzl32rpr4r`, EU-RO-1, $0.25/hr — no A5000 stock anywhere; 03:18–04:05 UTC; 3750
steps in 1381 s = 2.7 it/s, peak 13.8 GB; sweep eval 23 min because CoT decoding re-loops
the prefix per token (3.1 s/example at r=64); ≈$0.24 for the session. Stdout in
`pod_stdout.log`, not committed.)
**Headline results:** `final_answer_accuracy=0.050` (10/200) at r=32, `unparseable_rate=0.000`,
`train_loss=0.6982` (mean over the run; last-epoch window **0.547** vs the no-loop
explicit-CoT pilot `20260917-075729` at **0.558** on the same 20k × 3 epochs).

| r | 1 | 2 | 4 | 8 | 16 | 32 | 64 | explicit-CoT pilot (no loop) |
|---|---|---|---|---|---|---|---|---|
| accuracy | 0.055 | 0.065 | 0.060 | 0.065 | 0.055 | 0.050 | 0.060 | 0.045 |
| cot_tokens | 29.8 | 29.2 | 29.6 | 29.4 | 29.3 | 29.4 | 29.7 | 34.3 |

Correct sets: 10–13 per r, union 18, r=1 ∩ r=32 = 7, explicit-CoT pilot's 9 ∩ r=32 = 6.
Raw outputs identical between r=1 and r=32 on 67/200 examples (the loop's one active
iteration reshuffles the rationale, as it reshuffled the number guess before), between
r=32 and r=64 on 139/200 (bf16 decoding noise over ~30 greedy tokens).
Convergence at r=64, fp32, at the answer-number position given the gold rationale:
relative state change 1.0, **0.062, 0.0041, 0.00059**, 1e-4, 2e-5, … 6e-7 at 64; KL
between consecutive number distributions **4.3e-4, 2.6e-6, 4e-8**. At the first-operand
position of step 1 (prompt + ` <<`): 0.041, 0.0020, 0.00019; KL 5.0e-3, 3.1e-5, 3.5e-7.
Answer-only pilots for comparison: 0.075, 0.003, 1e-4 and KL 8e-3, 3e-5, 1e-7.
Step-supervised full run: 0.42, 0.30, 0.21 and KL 3.3, 1.5, 0.73.
Loss by epoch window (looped / explicit-CoT control): 1.225 / 1.237 → 0.745 / 0.750 →
0.608 / 0.627 → 0.547 / 0.558.
**Interpretation:**
- **Same collapse, same shape.** Iteration 2 changes the state by 6% and the number
  distribution by 4e-4 nats, iteration 3 by 0.4% / 3e-6, and the loop is at a fixed point
  to the fp32 floor by iteration ~6 — the answer-only signature, on a task where the
  loss is 0.55 nats and falling. So "no gradient pressure from a floor-level task" is
  **not** the cause of the collapse (it was the last task-side hypothesis). What the
  six paper-recipe pilots now share is only the regime: pretrained fixed-depth weights,
  whose one-pass path already fits the target, fine-tuned for ~7M tokens. The loop
  converges to the nearest solution — the pretrained one — and stays there.
- **No test-time scaling** at any r from 1 to 64 (5.0–6.5%, all within ±3 points of each
  other and of the no-loop control's 4.5%), rationale length unchanged with r. Huginn goes
  0% → 35% over the same range; that property is a from-scratch pretraining property
  (800B tokens with random r) and does not appear under fine-tuning at this budget.
- The looped model's loss is 0.01–0.02 nats below the no-loop control in every epoch
  window. Real but tiny (the two runs also differ in data order and Trainer vs plain
  loop), and it buys nothing at eval; the same 0.01-nats gap was absent in the
  answer-only comparison. Not evidence of the loop working.
- Consequence for the plan: the step-supervised loop (`20260918-214656_..._full`) remains
  the only looped GPT-2 whose iterations do anything, and it does so because its
  objective *names* a target per iteration. Paper-faithful recurrent-depth states with
  monotone scaling exist only in the released Huginn-0125 — route (f).
**Gotchas hit:**
- CoT decoding without a KV cache costs r × prefix per token: the r=64 point alone took
  10 min for 200 examples. Fine for a pilot, budget ~1 h of eval for a full run.
- The RTX A4500 (20 GB, $0.25) is a workable A5000 substitute at ~60% of its speed
  (2.7 vs 4.4 it/s; this run's sequences are also ~30% longer).
**Caveats:** pilot scale (20k of 384k); n=200 (±3 points); one seed; `train_loss` in the
manifest is the mean over the run, the window numbers above are the comparable ones;
truncated backprop k=8 as in the paper (the answer-only full-BPTT pilot showed the same
collapse, not repeated here); no few-shot prompting (fine-tuned zero-shot, like every
other run in this repo), so the number is not directly comparable to Huginn's 8-shot one.
**Next:**
- Do **not** spend the ~$1.60 on a full-scale CoT run: the loop is at a fixed point by
  iteration 3 after 3750 steps and 19× more data of the same objective is 0.3% of
  Huginn's budget. Log this as the closing paper-recipe pilot.
- Route (f): evaluate released Huginn-0125 (3.5B) on GSM8K-Aug test n=200 at r ∈ {1, 4, 8,
  16, 32, 64}, bf16, ~$0.50 — needs Ethan/Kevin's OK on a second backbone.
- Recurrent-depth entry for the Sep 25 comparison stays the step-supervised full run.
