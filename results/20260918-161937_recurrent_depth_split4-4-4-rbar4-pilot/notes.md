## 2026-09-18 — Pilot (b): small r̄ = 4 with every iteration supervised (recurrent_depth, run_id: 20260918-161937_recurrent_depth_split4-4-4-rbar4-pilot)

**Goal:** Second single-variable follow-up to the collapsed pilots. Hypothesis: with r̄ = 32 and
k = 8, most iterations are never on the gradient path and r=1…4 are never sampled, so the loop
is neither supervised nor needed at small depth. Here r is sampled from the same log-normal
Poisson with r̄ = 4 (τ ~ N(log 4 − ⅛, ½), r = Poisson(e^τ)+1 → mean ≈ 5, range ~2–12) and k = 8 ≥
almost every r, so every iteration is on-distribution and gets gradient. If a 4-iteration
loop still collapses to a fixed point in 2, depth mismatch is not the story.
**Mechanism / model:** `recurrent_depth`, gpt2 (125,621,760 params) /
`results/20260918-161937_recurrent_depth_split4-4-4-rbar4-pilot/ckpt` (`model.pt`, synced).
Same (4, 4, 4) split, identity adapter init, state LN, s_0 noise as the first pilot.
`compute_steps` = r at eval; **the manifest reports r = 4** (= r̄ for this run, not 32), so
compare its `final_answer_accuracy` with other runs' `sweep_by_r["4"]`, not their headline.
**Data:** identical to the first pilot (train 20k subset seed 42, eval test n=200 seed 0, run
seed 42).
**Hyperparams:** first pilot's except `mean_recurrence=4`: {'lr': 5e-05, 'epochs': 3,
'batch_size': 16, 'train_n': 20000, 'n_prelude': 4, 'n_core': 4, 'n_coda': 4,
'lognormal_sigma': 0.5, 'backprop_last_k': 8, 'init_state_std': 0.632, 'adapter_init':
'identity', 'core_init': 'pretrained', 'eval_recurrences': [1, 2, 4, 8, 16, 32, 64],
'max_new_tokens': 32}.
**Command:** `uv run python scripts/train_recurrent_depth.py --train-n 20000 --stage pilot --hardware "RunPod RTX A5000 (secure)" --mean-recurrence 4 --backprop-last-k 8 --slug split4-4-4-rbar4-pilot`
(pod `id24tq987niiqv`, 16:19–16:27 UTC; 3750 steps in 395 s = 9.5 it/s — 2.2× faster than
r̄ = 32; ≈$0.04 of pod time.)
**Headline results:** `final_answer_accuracy=0.025` (5/200) at r=4, `unparseable_rate=0.000`,
`train_loss=1.1638`.

| r | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| accuracy | 0.025 | 0.020 | 0.025 | 0.025 | 0.025 | 0.025 | 0.025 |

fp32 convergence at the number-predicting position (n=200; see (a)'s Gotchas for why the
manifest's in-run KL is not this): state change 1.0, 0.064, 0.0022, 1.2e-4, 7.4e-6, … 7e-7;
KL between consecutive number distributions 5.8e-3, 2.6e-5, 1.1e-7, ~0; top-1 flips 19, 1, 0,
0 (of 200); mean log p(gold) −4.448 → −4.451 → −4.451; top-1 = gold 8 → 7 → 7 of 200.
**Interpretation:** Depth / r mismatch is **not** the cause. Trained at r ≈ 2–12 with gradient
through every iteration, the loop is *indistinguishable* from the r̄ = 32 pilot's: same
contraction ratio (~1/30 per iteration), fixed point by iteration 3, same train loss to three
decimals (1.1638 vs 1.1625), flat sweep — and it generalises to r = 64 (16× the training mean)
with no change at all, which is the paper's path independence in its trivial form. Fewer
iterations to "waste" did not make any of them useful.
**Gotchas hit:** none new (see (a) for the diagnostics-position issue, which applies to this
manifest's `next_token_kl` too).
**Caveats:** as (a). Note r̄ = 4 also means ~2× less compute per step, so this run saw the same
data with the same optimiser steps but a shallower unrolled network (mean 4 + 4·5 + 4 = 28
layers vs ~136) and still matched the loss — more evidence that the extra depth was never
doing anything.
**Next:** see (c).
