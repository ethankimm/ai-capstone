## 2026-09-18 — Pilot (c): random-init core, no pretrained 12-layer shortcut (recurrent_depth, run_id: 20260918-162743_recurrent_depth_split4-4-4-rbar32-randcore-pilot)

**Goal:** Third single-variable follow-up to the collapsed pilots. Hypothesis: with pretrained
blocks 4–7 as the looped core, a 12-layer GPT-2 solution exists at r=1 and fine-tuning has no
reason to make iterations do anything. Here blocks 4–7 are re-drawn with GPT-2's own init
(`--core-init random`: Conv1D ~ N(0, 0.02), `c_proj` scaled by 1/√24, LayerNorms reset) so the
core has to be learned; it gets a 10× larger lr (`--core-lr 5e-4`) since 3750 steps at 5e-5
would barely move a fresh block. Prelude and coda stay pretrained, adapter identity init.
**Mechanism / model:** `recurrent_depth`, gpt2 prelude/coda + fresh core (125,621,760 params) /
`results/20260918-162743_recurrent_depth_split4-4-4-rbar32-randcore-pilot/ckpt` (`model.pt`,
synced). At init the fresh residual blocks are near-identity, so r=1 ≈ an 8-layer GPT-2 with
blocks 4–7 removed (CPU check: finite logits, plausible top-1). `compute_steps` = r at eval;
manifest reports r=32.
**Data:** identical to the first pilot.
**Hyperparams:** first pilot's except `core_init=random`, `core_lr=5e-4` (all other params
5e-5): {'lr': 5e-05, 'epochs': 3, 'batch_size': 16, 'train_n': 20000, 'n_prelude': 4,
'n_core': 4, 'n_coda': 4, 'mean_recurrence': 32, 'lognormal_sigma': 0.5, 'backprop_last_k': 8,
'init_state_std': 0.632, 'adapter_init': 'identity', 'eval_recurrences': [1, 2, 4, 8, 16, 32,
64], 'max_new_tokens': 32}.
**Command:** `uv run python scripts/train_recurrent_depth.py --train-n 20000 --stage pilot --hardware "RunPod RTX A5000 (secure)" --core-init random --core-lr 5e-4 --slug split4-4-4-rbar32-randcore-pilot`
(pod `id24tq987niiqv`, 16:27–16:43 UTC; 3750 steps in 862 s = 4.4 it/s; ≈$0.07 of pod time.
Whole (a)+(b)+(c) session incl. setup: pod up 16:00–16:47, ≈$0.21.)
**Headline results:** `final_answer_accuracy=0.015` (3/200) at r=32, `unparseable_rate=0.000`,
`train_loss=1.2340` — **worse** than every pretrained-core run (1.162–1.164) and than the
no-loop control (1.169).

| r | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| accuracy | 0.010 | 0.015 | 0.020 | 0.020 | 0.020 | 0.015 | 0.025 |

fp32 convergence at the number-predicting position (n=200): state change 1.0, 0.049, 0.0036,
4.4e-4, 1.6e-4, … 7.6e-5 at i=16 (slightly slower tail than the pretrained cores, still a
fixed point); KL 2.5e-3, 1.9e-5, 7.8e-7, ~5e-7; top-1 flips 11, 0, 0, 0 (of 200); mean
log p(gold) −4.513 → −4.521 → −4.521; top-1 = gold 7 → 7 → 7.
**Interpretation:** The pretrained shortcut is **not** the cause either. A freshly initialised
core with a 10× lr, which has *no* option but to learn something in the loop, learns a
contraction to a fixed point just as fast (ratio ~1/20 per iteration) and ends up fitting the
data *worse* — the model does not discover an iterative solution when the feed-forward one is
taken away; it just has 4 fewer useful layers. Taken with pilots 1–2, (a) and (b), every
specific mechanism proposed for the collapse has now been tested and eliminated at this scale:

| suspect | run | result |
|---|---|---|
| truncated backprop (k=8) | full BPTT pilot | same collapse, same loss |
| collapsed init A=[I,0] | (a) random_state | slower contraction, same collapse, same loss, loop adds +0.01 nats |
| r̄=32 too deep / r=1–4 never trained | (b) r̄=4 | identical to r̄=32 |
| pretrained 12-layer shortcut | (c) random core | same collapse, worse loss |

What all five share and what the paper does not: (i) **training budget** — 3750 steps × 16 ×
~100 tokens ≈ 6M tokens vs 800B from scratch, and converting a pretrained model into a looped
one is known to need billions of tokens of uptraining (Bae et al. 2024, Relaxed Recursive
Transformers); the adapter barely moved (‖ΔA‖_F ≈ 1.5 over 3750 steps in every run). (ii) **no
usable signal from the task** — direct numeric answers on GSM8K at GPT-2 scale sit at the
guessing floor (number-readout entropy 4.1 nats, top-1 = gold 7–12/200), the loss is the same
as the no-loop control's in every run, and (c) shows that when a feed-forward path is removed
the loss goes *up* rather than the loop stepping in: extra depth of these weights does not
lower the loss on this task at this data size, so there is nothing for the loop to learn.
Under an r-invariant loss with fresh s_0 every pass, a fast contraction to a point that equals
the feed-forward answer is then the optimum, and it is reached within a few hundred steps.
So "pretrained vs from scratch" is part of it, but via the budget and the task, not via any
of the specific mechanisms tested. What is reproduced: fixed-point convergence from random
s_0 and path independence (exact, geometric, to the fp32 floor). What is not: test-time
scaling — under a regime the paper never ran.
**Gotchas hit:**
- `gpt2._init_weights` / `block.apply(...)` silently no-ops in transformers 5.17 on modules
  loaded from a checkpoint (params carry `_is_hf_initialized`); the first attempt left the
  core pretrained (caught by a CPU equality check). `reinit_gpt2_block` spells the scheme out
  with `torch.nn.init`.
- Diagnostics position, as (a).
**Caveats:** as (a), plus: one lr for the fresh core (5e-4); a from-scratch GPT-2 lr (6e-4)
with warmup and more steps might train the core further, but the loss gap to the pretrained
core (0.07 nats) is the wrong direction for the hypothesis regardless. The paper's sandwich
norms / embedding scale were not tried — unlikely to matter given the contraction appears
with a fresh core too.
**Next:** stop attacking the collapse at pilot scale. Per `recurrent_depth.py` status/plan:
(e) step-supervised loop (r = number of rationale steps, iteration i supervised on
`intermediate_values[i]`) is the one route that creates per-iteration gradient pressure at
this data size and doubles as the Oct 9 decoding target; (f) released Huginn-0125 eval gives a
genuine recurrent-depth model with test-time scaling to probe. (d) full-scale run of this
recipe deferred — would give a quotable accuracy vs the 13.0% control but the same curve.
