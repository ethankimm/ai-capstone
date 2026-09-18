## 2026-09-18 — Pilot: recurrent depth on looped GPT-2, paper-faithful r distribution + truncated backprop (recurrent_depth, run_id: 20260918-052821_recurrent_depth_split4-4-4-rbar32-pilot)

**Goal:** First reproduction attempt of Geiping et al. (arXiv:2502.05171) at GPT-2 scale.
The paper's headline claim is *test-time scaling*: one model trained with a random
number of core-block iterations r gets better as r is raised at inference, saturating
around r≈32–64. Here pretrained `gpt2` is re-wired into their prelude / looped-core /
coda form and fine-tuned on GSM8K-Aug (direct-answer format, no visible rationale) with
their r distribution and truncated backprop, then one checkpoint is swept over
r ∈ {1,2,4,8,16,32,64}. Pilot scale (20k train examples, 3 epochs) matching the other
mechanisms' pilots so the same-recipe controls apply. Questions: (1) does accuracy rise
with r? (2) does the loop converge (their "path independence"), and how fast?
**Mechanism / model:** `recurrent_depth`, gpt2 (125,621,760 params = GPT-2's 124.4M +
a 2h→h adapter and a state LayerNorm) / `results/20260918-052821_recurrent_depth_split4-4-4-rbar32-pilot/ckpt`
(`model.pt` = `LoopedGPT2.state_dict()`, 503 MB, synced to this machine). Layer split
(prelude, core, coda) = (4, 4, 4) over GPT-2's 12 blocks, core = blocks 4–7 looped as
`s_i = core(A[e ; LN(s_{i-1})])`, `s_0 ~ N(0, 2/5)`; adapter initialised to identity-on-e /
zero-on-state so at init the model is exactly GPT-2 for every r (verified: logits match
to 0.0). Full mapping and every deviation from the paper (no sandwich norms, no
embedding scale, norm on the state path instead of the adapter output):
`latentreasoning/mechanisms/recurrent_depth.py`. `compute_steps` = r at eval; the
manifest reports r=32 (= r̄, the training mean); the whole sweep is in
`metrics.extra.sweep_by_r` and `predictions_r{r}.jsonl`.
**Data:** gsm8k-aug train, 20,000-example subset (seed 42) of the split that excludes
the 1000-example validation carve-out; eval gsm8k-aug test n=200 seed=0; run seed=42.
**Hyperparams:** {'lr': 5e-05, 'epochs': 3, 'batch_size': 16, 'train_n': 20000,
'n_prelude': 4, 'n_core': 4, 'n_coda': 4, 'mean_recurrence': 32, 'lognormal_sigma': 0.5,
'backprop_last_k': 8, 'init_state_std': 0.632, 'eval_recurrences': [1, 2, 4, 8, 16, 32, 64],
'max_new_tokens': 32}. r sampled per batch from the paper's log-normal Poisson
(τ ~ N(log 32 − ⅛, ½), r = Poisson(e^τ)+1 → mean ≈33, p5/p95 = 12/67, r=1 never
sampled); gradient only through the last k=8 iterations (earlier ones under no_grad);
AdamW, linear decay to 0, no warmup, grad-clip 1.0, bf16 autocast — i.e. the HF Trainer
defaults the other mechanisms used, in a plain loop. Greedy decoding, batched (50),
left-padded; `s_0` noise drawn once per example (seed 42) and shared across decoding steps.
**Command:** `uv run python scripts/train_recurrent_depth.py --train-n 20000 --stage pilot --hardware "RunPod RTX A5000 (secure)" --slug split4-4-4-rbar32-pilot`
(pod `y7ndb3fml97o8s`, CA-MTL-1, $0.27/hr; 05:28–05:43 UTC; 3750 steps in 854 s
= 4.4 it/s at r̄=32; ≈$0.08 of pod time. Stdout in `pod_stdout.log`, not committed.)
**Headline results:** `final_answer_accuracy=0.040` (8/200) at r=32, `unparseable_rate=0.000`,
`train_loss=1.1625` (mean over steps; windowed loss 1.72 → 1.02).
Test-time sweep on the same checkpoint (n=200 each, all `unparseable_rate=0`):

| r | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| unrolled depth | 12 | 16 | 24 | 40 | 72 | 136 | 264 |
| accuracy | 0.025 | 0.040 | 0.040 | 0.035 | 0.035 | 0.040 | 0.040 |
| correct (of 200) | 5 | 8 | 8 | 7 | 7 | 8 | 8 |
| sec/example | 0.007 | 0.009 | 0.013 | 0.020 | 0.034 | 0.065 | 0.118 |

Convergence at r=64 on the 200 eval prompts (last prompt position, batch-averaged;
`metrics.extra.state_rel_delta` / `next_token_kl`): relative state change
‖s_i − s_{i−1}‖/‖s_i‖ = 1.01, 0.055, 0.0069, 0.0064, 0.0063, … 0.0063 (i = 1, 2, 3, …, 64);
KL between consecutive iterations' next-token distributions = 30.0, then exactly 0.0
from i=2 onward. Outputs at r ∈ {2,…,64} are token-identical to r=32 on 176–181/200
examples; r=1 differs on 47/200. The 5 examples right at r=1 are all also right at r=32;
r=32 adds 3.
**Interpretation:**
- **No test-time scaling.** Accuracy is flat from r=2 to r=64 (7–8/200) and the only
  change is r=1 → r=2 (5 → 8). This is the opposite of the paper's curve, and the
  convergence diagnostic says why: the fine-tuned loop is a very fast contraction —
  after 2 iterations the state moves by 5%, after 3 by 0.7%, and the residual 0.6%
  from then on is the bf16 precision floor (2⁻⁸ ≈ 0.4%), not a drifting trajectory.
  The coda reads out an identical next-token distribution from iteration 2 on (KL
  exactly 0.0). Extra iterations have nothing left to act on; the effective network is
  ~4 + 4·2 + 4 = 20 layers regardless of r. The ~20/200 outputs that differ between
  r=32 and r=64 are wild near-tie flips (e.g. `#### 7.5` vs `#### 746`, both wrong),
  consistent with bf16 noise on an unconfident model, not with more computation.
- **Path independence, yes — trivially.** The paper's "convergence to a fixed point
  from random init" holds here, but at a speed that leaves no room for the mechanism to
  do anything. Compare the paper's trajectories, which converge slowly (some tokens
  orbit/spiral for tens of iterations) and whose fixed point encodes many iterations of
  computation. The Jacobian-lens note in PROPOSAL.md ("iterations overwrite rather than
  accumulate") is consistent with this: the state is overwritten each iteration, so
  once the map is a strong contraction the trajectory is over almost immediately.
- **Why the collapse is the path of least resistance here (hypothesis):** with r
  random and gradients only through the last k=8 iterations — which all sit at the
  fixed point — the training signal is "the fixed point should give the right answer,
  whatever r was", and the cheapest way for a pretrained network to be r-invariant is to
  make the loop contract fast. The paper trains from scratch (no pretrained solution to
  fall back to) and at 3.5B/800B tokens. Whether the truncation matters is testable
  directly: a full-BPTT variant (same run, `--backprop-last-k 128 --checkpoint-iterations`)
  is the follow-up pilot logged next.
- **Against the same-recipe controls:** 4.0% (8/200) vs 1.0% (2/200) for the no-loop
  direct-answer control (`20260917-074147_filler_tokens_budget-0-control`, identical
  prompt/target format, identical 20k×3-epoch recipe), 2.5% for 32 filler tokens, 4.5%
  for explicit CoT — all pilot scale. 8 vs 2 at n=200 is not a real gap (the filler
  pilot's 5 vs 2 vanished at full scale), and `train_loss` is the same as the control's
  (1.1625 vs 1.1693): the looped model does not fit the training set any better than
  plain GPT-2 at this scale. Note r=1 here (2.5%) is *not* the control — r=1 was never
  sampled in training, so the r=1 read-out is off-distribution for this checkpoint.
**Gotchas hit:**
- First attempt at the full-BPTT follow-up OOM'd on the 24 GB A5000 (storing every
  iteration's activations at batch 16, r up to ~90); added per-iteration activation
  checkpointing (`checkpoint_iterations`), verified gradients identical to the
  uncheckpointed path on CPU. Not used in this run (k=8 fits in 12 GB).
- rsync through the RunPod SSH proxy (`ssh.runpod.io`) fails with "unexpected tag";
  use the pod's direct SSH port.
- transformers 5.17: GPT-2 blocks are called directly with a 4-D mask from
  `create_causal_mask`; left-padded batches verified equal to unpadded ones (≤6e-4
  max logit diff, no NaNs) under both sdpa and eager.
**Caveats:**
- Pilot scale: 20k/384k examples; every same-recipe pilot sits at 1–4.5%, so the
  *accuracy* numbers here are near floor and mainly show the r-dependence, not the
  level. The convergence result (fixed point in 3 iterations) is the robust part.
- n=200 → ±~3 points 95% CI at these rates; no differences in the sweep are significant.
- One seed, one layer split (4/4/4). The paper's 2:4:2 ratio, other splits, a
  from-scratch (non-pretrained) core, and a larger state-path init are all untried.
- Compute matching: r=32 is 32 extra passes of 4 blocks over *all* positions (136-layer
  unroll), not comparable by "steps" to 32 filler positions through 12 layers; see the
  record-run skill's rule before quoting one against the other.
- bf16 autocast at eval makes the sub-1% state changes unmeasurable; a fp32 eval
  would be needed to say whether the trajectory is *exactly* stationary or slowly
  drifting (irrelevant for the accuracy conclusion, relevant for J-lens later).
**Next:**
- Full-BPTT variant (next run) to test whether truncated backprop causes the collapse.
- If the collapse persists: full-scale run (384k × 3 epochs, ≈4.6 h / ≈$1.25 on the
  A5000) to get a quotable accuracy against the 13.0% full-scale control and to see
  whether more training slows the contraction; then variants that make the loop
  matter by construction (smaller/random-init core, r̄ sweep, larger k).
- For Oct 9 decoding: `LoopedGPT2.forward(return_states=True)` exposes every s_i; with
  this checkpoint only s_1, s_2 differ meaningfully — probes on later iterations will
  be probing the same vector.
**Follow-up (added 2026-09-18):** the full-BPTT variant
(`20260918-055006_recurrent_depth_split4-4-4-rbar32-fullbptt-pilot`) shows the same
collapse (fixed point by iteration 3, flat accuracy 3.5–4.5% across r, train loss
1.1619) — the truncated backprop is not the cause.
**Correction (added 2026-09-18, after pilots a–c):** the "KL exactly 0.0 from i=2" claim above
over-reads `metrics.extra.next_token_kl`: it was measured at the bare prompt's last position,
where this model emits `" ####"` with p ≈ 1, so KL ≈ 0 there says nothing about the state (a
same-norm random perturbation of the state also gives KL ~1e-9). Re-measured in fp32 at the
position predicting the first answer-number token (n=200, s_0 seed 42): state change per
iteration 1.0, 0.075, 0.0026, 1.4e-4, 9e-6 … 7e-7 at i=16 (exact geometric fixed point, ratio
≈ 1/30); KL between consecutive number distributions 8.0e-3 (i=2), 3.0e-5, 1.3e-7, ~0; top-1
number flips 45, 3, 0, 0 (of 200); mean log p(gold first token) −4.463 → −4.473 → −4.474 →
−4.474; top-1 = gold 12/200 at every iteration. So the loop *does* act once (iteration 2
reshuffles the guess on 45/200 examples) and is frozen from iteration 4 — the fixed-point and
flat-accuracy conclusions stand; the "read-out identical from iteration 2" wording does not.
The "~20/200 flips between r=32 and r=64 are bf16 noise" remark stands (fp32: 0 flips).
Follow-ups (a)–(c) (`20260918-16*_recurrent_depth_*`) eliminated the collapsed init, r
mismatch and pretrained-shortcut hypotheses as well; see (c)'s notes for the synthesis.
