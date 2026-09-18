## 2026-09-18 — Pilot: recurrent depth, full backprop through every loop iteration (recurrent_depth, run_id: 20260918-055006_recurrent_depth_split4-4-4-rbar32-fullbptt-pilot)

**Goal:** Single-variable follow-up to `20260918-052821_recurrent_depth_split4-4-4-rbar32-pilot`,
which found the fine-tuned loop collapses to a fixed point within 3 iterations so
accuracy is flat in test-time r. Hypothesis under test: the paper's truncated backprop
(gradient only through the last k=8 of ~32 iterations, all of which sit at the fixed
point) is what rewards a fast contraction. Same run with gradient through *every*
iteration (`--backprop-last-k 128`, ≥ the largest sampled r; per-iteration activation
checkpointing to fit in 24 GB). If the collapse is a truncation artefact, this model's
state should keep changing across iterations and accuracy should depend on r.
**Mechanism / model:** `recurrent_depth`, gpt2 (125,621,760 params) /
`results/20260918-055006_recurrent_depth_split4-4-4-rbar32-fullbptt-pilot/ckpt`
(`model.pt`, 503 MB, synced to this machine). Same (4, 4, 4) split, adapter, state
norm, s₀ noise as the first pilot — see that run's notes and
`latentreasoning/mechanisms/recurrent_depth.py`. `compute_steps` = r at eval; manifest
reports r=32, sweep in `metrics.extra.sweep_by_r` / `predictions_r{r}.jsonl`.
**Data:** identical to the first pilot: gsm8k-aug train 20,000-example subset (seed
42), eval gsm8k-aug test n=200 seed=0, run seed=42 (same r sequence per batch, same
example order — the only difference is which iterations get gradient).
**Hyperparams:** as the first pilot except `backprop_last_k=128` (effectively full
BPTT: r ≤ ~90 in practice), `checkpoint_iterations=True`:
{'lr': 5e-05, 'epochs': 3, 'batch_size': 16, 'train_n': 20000, 'n_prelude': 4,
'n_core': 4, 'n_coda': 4, 'mean_recurrence': 32, 'lognormal_sigma': 0.5,
'init_state_std': 0.632, 'eval_recurrences': [1, 2, 4, 8, 16, 32, 64], 'max_new_tokens': 32}.
**Command:** `uv run python scripts/train_recurrent_depth.py --train-n 20000 --backprop-last-k 128 --checkpoint-iterations --stage pilot --hardware "RunPod RTX A5000 (secure)" --slug split4-4-4-rbar32-fullbptt-pilot`
(pod `y7ndb3fml97o8s`, 05:50–06:27 UTC; 3750 steps in 2102 s = 1.8 it/s, 2.5× slower
than k=8 as expected for backward through ~4× more blocks plus recompute; peak 7.9 GB
with checkpointing vs OOM at 24 GB without; ≈$0.17 of pod time.)
**Headline results:** `final_answer_accuracy=0.040` (8/200) at r=32, `unparseable_rate=0.000`,
`train_loss=1.1619` (first pilot: 1.1625; windowed 1.72 → 1.02 in both).

| r | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| accuracy (this run, full BPTT) | 0.035 | 0.040 | 0.040 | 0.040 | 0.040 | 0.040 | 0.045 |
| accuracy (first pilot, k=8) | 0.025 | 0.040 | 0.040 | 0.035 | 0.035 | 0.040 | 0.040 |

Convergence at r=64 (`metrics.extra`): relative state change 1.01, 0.053, 0.0071,
0.0064, 0.0063 … (bf16 floor from i=3); KL between consecutive next-token
distributions 27.3, then exactly 0.0 from i=2. Outputs identical to r=32 on 190/200
(r=2) and 181/200 (r=64) examples. The 8 correct examples at r=2…32 are *the same 8*
(test idx 55, 93, 95, 119, 121, 125, 160, 167) as in the first pilot, although the two
models' outputs agree on only 140/200 examples — those 8 are simply the questions a
direct-answer GPT-2 gets right by guessing.
**Interpretation:** Truncated backprop is **not** the cause of the collapse. With
gradient through every iteration the loop still becomes a contraction that reaches its
fixed point (to bf16 precision) by the third iteration, the read-out distribution is
identical from iteration 2 on, and accuracy is flat in r. Training loss is the same to
three decimals, so the extra gradient path buys nothing at this scale either. Together
the two pilots say: fine-tuning pretrained GPT-2 into a looped model on this recipe
produces a network that is r-invariant by being ~20 layers deep and then stationary,
rather than one that computes more with more iterations. The remaining suspects are the
pretrained starting point itself (a 12-layer solution is already available and the
zero-initialised state path makes "ignore the loop" the starting point), the tiny
fine-tuning budget relative to the paper (3750 steps vs. 800B tokens from scratch), and
the task format (a short numeric answer read out at one position).
**Gotchas hit:**
- Full BPTT at batch 16 with r up to ~90 OOMs a 24 GB card outright (first attempt,
  run dir deleted); fixed with `torch.utils.checkpoint` per iteration
  (`LoopedGPT2.forward(checkpoint_iterations=True)`), gradients verified bit-identical
  to the uncheckpointed path on CPU.
- `RunRecord.author` reads `git config user.name` on the machine that saves the record;
  it was unset on this fresh pod, so the first pilot's manifest was saved with an empty
  author and patched by hand to "Henning Lindig" (metadata only). Set it in
  `scripts/runpod_setup.sh` next time.
**Caveats:** all of the first pilot's (pilot scale, n=200, one seed, one split, bf16
eval), plus: "full BPTT" here still uses the paper's r distribution, so iterations
beyond ~90 never get gradient — immaterial given the state is stationary from i=3.
**Next:**
- Rather than the full-scale run of this exact recipe (≈4.6 h / ≈$1.25; would give a
  quotable accuracy vs. the 13.0% full-scale control but probably the same flat
  r-curve), the higher-value next pilots attack the collapse directly, each ~15 min:
  (a) random-init core (drop the pretrained blocks 4–7) so no 12-layer solution
  exists; (b) small r̄ (e.g. 4) with k ≥ r̄ so every iteration is on-distribution and
  supervised; (c) a loss that *requires* iteration, e.g. supervising intermediate
  `Example.intermediate_values` from intermediate states — which is also the Oct 9
  decoding target, so it can be set up once for both.
- Decide what "reproduced" means for this mechanism in the write-up: the paper's
  fixed-point/path-independence behaviour is reproduced; its test-time scaling is not,
  under a fine-tune-from-pretrained regime the paper never ran.
**Correction (added 2026-09-18):** same diagnostics-position issue as the first pilot (see its
correction). fp32 at the number-predicting position, n=200: state change 1.0, 0.073, 0.0026,
1.3e-4, … 7.5e-7; KL 8.3e-3, 3.1e-5, 1.0e-7, ~0; top-1 flips 30, 2, 0, 0; log p(gold)
−4.466 → −4.480 → −4.481; top-1 = gold 11 → 12 → 12 of 200. Conclusions unchanged.
