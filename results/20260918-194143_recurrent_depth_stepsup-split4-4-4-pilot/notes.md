## 2026-09-18 — Pilot (e): step-supervised loop — iteration i must read out rationale step i (recurrent_depth, run_id: 20260918-194143_recurrent_depth_stepsup-split4-4-4-pilot)

**Goal:** Plan item (e) in `latentreasoning/mechanisms/recurrent_depth.py`. After five pilots
showed that fine-tuning GPT-2 into Geiping et al.'s loop on the answer-only objective yields a
fast contraction that never uses its iterations, this run changes the *objective* (not the
architecture) so that the loop is forced to carry the rationale: r = the example's number of
`<<a op b = v>>` steps; at iteration i < n the coda + LM-head read-out at the number position
(the last `" ####"` token) is trained on the first token of `" v_i"`; at iteration n the
usual answer-span loss applies. This is the vertical analogue of CODI (which supervises its
latent tokens against a CoT teacher) and **not Geiping et al.'s method** — a new mechanism to
be named as such in the write-up. Questions: (1) is the per-step target learnable at pilot
scale? (2) does the loop now stay "alive" across iterations? (3) does the final answer
improve, and does it depend on r (a non-flat r-curve would be the first here)?
**Mechanism / model:** `recurrent_depth`, gpt2 (125,621,760 params) /
`results/20260918-194143_recurrent_depth_stepsup-split4-4-4-pilot/ckpt` (`model.pt`, synced
to this machine). Same `LoopedGPT2` as every other run: (4, 4, 4) split, identity adapter
init, state LN, `s_0 ~ N(0, 2/5)`. Objective `step_supervised`
(`scripts/train_recurrent_depth.py --step-supervised`; row builder
`build_step_supervised_row`). Loss = token mean over step tokens (weight 1) + answer-span
tokens; full backprop through every iteration (r ≤ 8, per-iteration activation
checkpointing). **`compute_steps` = 3.395, the mean per-example r of the oracle eval** (r =
each *test* example's own step count, read off its rationale) — a float on purpose; it is
not comparable to a fixed-r run's integer. Fixed-r sweep and oracle+1/+2 in
`metrics.extra.sweep_by_r`; `predictions_oracle*.jsonl`, `predictions_r{r}.jsonl`;
per-example step read-outs in `step_readouts.jsonl`.
**Data:** gsm8k-aug train 20,000-example subset (seed 42), minus 9 examples with > 8 steps
→ 19,991; eval gsm8k-aug test n=200 seed=0; run seed=42. Step-count distribution in the
subset: 1: 16%, 2: 37%, 3: 27%, 4: 12%, 5+: 7%. 84% of step values are a single GPT-2 token,
so the first-token target is the whole value for most steps (a partial value — `" 13"` for
`130000`, `" 4"` for `4.5` — for the rest).
**Hyperparams:** {'objective': 'step_supervised', 'lr': 5e-05, 'epochs': 3, 'batch_size': 16,
'train_n': 19991, 'n_prelude': 4, 'n_core': 4, 'n_coda': 4, 'step_loss_weight': 1.0,
'stepsup_max_steps': 8, 'checkpoint_iterations': True, 'init_state_std': 0.632,
'adapter_init': 'identity', 'core_init': 'pretrained', 'eval_recurrences': [1, 2, 3, 4, 6, 8],
'max_new_tokens': 32}; AdamW, linear decay, no warmup, grad-clip 1.0, bf16 autocast.
**Command:** `uv run python scripts/train_recurrent_depth.py --step-supervised --checkpoint-iterations --train-n 20000 --stage pilot --hardware "RunPod RTX A6000 (secure)" --slug stepsup-split4-4-4-pilot`
(pod `ohefhcu54q1v03`, EU-SE-1, $0.53/hr — no A5000 stock; 19:41–19:56 UTC; 3750 steps in
828 s = 4.5 it/s, peak 4.4 GB; ≈$0.13 of pod time, ≈$0.27 for the session. Stdout in
`pod_stdout.log`, not committed.)
**Headline results:** `final_answer_accuracy=0.025` (5/200) at oracle r, `unparseable_rate=0.000`,
`train_loss=1.8572` (token mean over both kinds of target; not comparable to the answer-only
runs' 1.16). Windowed losses, step 50 → 3750: step-token CE 4.74 → 3.29 nats; answer-token CE
1.77 → 1.08 (the answer-only pilots ended at ~1.02–1.05 on the same window).

| r | 1 | 2 | 3 | 4 | 6 | 8 | oracle (mean 3.4) | oracle+1 | oracle+2 |
|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.025 | 0.025 | 0.025 | 0.030 | 0.025 | 0.030 | 0.025 | 0.030 | 0.030 |

Step read-out (does the top-1 token at the number position after iteration i equal step i's
first token? 479 (example, step) pairs from the 155 eval examples with ≥ 2 steps, fp32):
**10.6%** overall (11.1% on single-token values); by iteration 1…5: 16.4%, 6.7%, 6.7%, 9.8%,
5.6% (support 189, 134, 90, 41, 18). Baselines on the same pairs: always predict the most
common step value (`" 30"`) 3.5%; "the value appears verbatim in the question" 7.5%; "value =
previous step" 1.0%. Every one of the 428 wrong read-outs is still a number, of plausible
magnitude (`' 180'` for 90, `' 14'` for 8); 53 copy a number from the question.
Iteration × step matrix (row = iteration read, col = step asked): it1 reads step 1 at 16.4%
but step 2 at 5.2%; it2 reads step 1 at 10.4% and step 2 at 6.7%; it3: 8.9% / 4.4% / 6.7% — the
state after iteration 2 still says more about step 1 than step 2.
Convergence (fp32, number position, n=200, r=8): relative state change 1.0, **0.43, 0.20,
0.070, 0.020, 0.005**, … 3e-4 at i=8; KL between consecutive number distributions **0.43,
0.14, 0.022**, 0.002, … 1e-6. Compare the answer-only pilots: 0.075, 0.003, 1e-4 and 8e-3,
3e-5, 1e-7.
**Interpretation:**
- (2) **The loop is alive now.** Iterations 2–4 each change the state by 43% / 20% / 7% and
  the number distribution by KL 0.43 / 0.14 / 0.02 — one to two orders of magnitude more than
  any answer-only run, and decaying over ~5 iterations instead of 2. The objective, not the
  architecture or init, is what determines whether the iterations do anything. (It still
  converges to a fixed point by ~iteration 8, as expected: nothing supervises iterations
  beyond the example's step count.)
- (1) **The per-step target is learnable but far from learned.** Step-token CE fell from
  4.7 to 3.3 nats and the read-out beats the trivial baselines 3× (10.6% vs 3.5%), with the
  first step easiest (16.4%). But 3.3 nats means the model's belief about each step's value
  is spread over ~25 numbers: it has learned "a number of about this size goes here", not the
  arithmetic. This is the same situation as explicit CoT at pilot scale (4.5% accuracy, vs
  34.5% with the full 384k training set): computing the intermediate values *is* the hard
  part, and 20k examples don't teach it.
- (3) **Final answer: unchanged, and still flat in r.** 5/200 at oracle r, 5–6/200 at every
  fixed r, 0 overlap with the answer-only pilots' 8 correct (i.e. a different set of lucky
  guesses, not a superset). Answer-token loss is a touch *higher* than the answer-only runs
  (1.08 vs 1.02–1.05 on the last window) — sharing the loop with the step targets costs a
  little at this scale. No r-dependence yet, because the read-outs the answer would build on
  are ~90% wrong.
- Net: the mechanism does what it was designed to do structurally (distinct, changing
  iteration states with a defined target each), and the content of those states is still
  mostly noise at pilot scale. Whether it becomes a usable vertical scratchpad is a
  data-scale question, exactly as it was for explicit CoT.
**Gotchas hit:**
- `gpt2` tokenises `" ####"` as two tokens and `" #### 42"` as `" ####"` + `" 42"`, so the
  number position is stable across answers (asserted in `build_step_supervised_row`); numbers
  ≥ 1000 and decimals split into several tokens, hence the first-token target.
- No RTX A5000 secure stock anywhere at run time; used an A6000 at 2× the price for the same
  wall time (the run is small).
**Caveats:** pilot scale (20k of 384k); n=200 (±3 points); one seed; step read-out is measured
with the model's *own* trained read-out head, so it is a training-target fit, **not** a probe
result — it can serve as a positive control for the Oct 9 decoding work, not as evidence for
RQ1; single-token-only accuracy hides that ~16% of steps get a partial-value target; oracle r
uses the test rationale's step count (an oracle) — the fixed-r rows are the honest
deployment numbers.
**Next:**
- The informative version of this run is **full scale** (384k × 3 epochs = 72k steps, ≈ 4.5 h
  ≈ $2.40 on an A6000, ≈ $1.20 on an A5000 if stock returns): explicit CoT went 4.5% → 34.5%
  with the same change in data, and the step targets here are the same arithmetic. Decide
  before spending: if the full run's step read-out stays < 30%, the vertical scratchpad is
  not learnable by GPT-2 at any data size we have and (f) Huginn is the only route.
- If it works: `LoopedGPT2.forward(return_states=True)` gives s_1…s_n with a known label each
  — the Oct 9 probes (linear / non-linear on s_i → v_i, with the trained read-out as the
  ceiling) and the Oct 16 patching (replace s_i with the state of a counterfactual v_i) are
  set up already.
- Cheaper variants if the full run is borderline: `--step-loss-weight 3` (the step tokens are
  ~25% of the loss mass now), an extra unsupervised iteration before the answer (r = n+1),
  and a distillation target from the explicit-CoT checkpoint instead of hard tokens (e′).
**Correction (added 2026-09-19):** the 479 read-out pairs come from the **189** test examples
with ≥ 2 steps (`metrics.extra.step_readout.n_examples`), not 155 as written above. Numbers
are unaffected. Full-scale result: `20260918-214656_..._stepsup-split4-4-4-full` — read-out
31.3%, criterion met.
