## 2026-09-19 — Full-scale (e): step-supervised loop on all 384k examples — the loop carries the rationale, answer = direct-answer control (recurrent_depth, run_id: 20260918-214656_recurrent_depth_stepsup-split4-4-4-full)

**Goal:** The decisive version of pilot `20260918-194143_..._stepsup-split4-4-4-pilot`, on the
full training set. Pre-registered criterion (pilot notes, "Next"): step read-out ≥ 30% →
the step-supervised loop is a usable vertical scratchpad for the Oct 9 decoding and Oct
16 causality work; < 30% → GPT-2 cannot learn it at any data size we have and (f)
Huginn-0125 is the only route to recurrent-depth states. Secondary questions: does the
final answer now depend on r, and how does it compare with the full-scale no-scratchpad
control (`20260917-163049_filler_tokens_budget-0-control-full`, 13.0%, same prompt format,
same 200 test examples)?
**Mechanism / model:** `recurrent_depth`, gpt2 (125,621,760 params) /
`results/20260918-214656_recurrent_depth_stepsup-split4-4-4-full/ckpt` (`model.pt`, 483 MB,
synced to this machine, gitignored; also the epoch-1 checkpoint was written and then
overwritten in place — `model.save` at every epoch end, commit d15567d). Same `LoopedGPT2`
as every other run: (4, 4, 4) split of pretrained GPT-2, identity adapter init, state LN,
`s_0 ~ N(0, 2/5)`. Objective `step_supervised` (`--step-supervised`, row builder
`build_step_supervised_row`): r = the example's number of `<<a op b = v>>` steps (≤ 8), at
iteration i < n the coda + LM-head read-out at the number position (last `" ####"` token)
is trained on the first token of `" v_i"`, at iteration n on the answer span; token-mean
loss, step weight 1; full backprop with per-iteration activation checkpointing. **Not
Geiping et al.'s objective** — a new mechanism (vertical analogue of CODI's supervised
latent tokens), to be named as such in the write-up. **`compute_steps` = 3.395 = mean
oracle r** (r = each test example's own step count), a float on purpose. Fixed-r sweep and
oracle+1/+2 in `metrics.extra.sweep_by_r`; `predictions_oracle*.jsonl`,
`predictions_r{r}.jsonl`; per-example step read-outs in `step_readouts.jsonl`.
**Data:** gsm8k-aug train, all 384,620 examples minus 178 with > 8 steps → 384,442
(`--train-n -1`); eval gsm8k-aug test n=200 seed=0; run seed=42. Test step counts:
1: 11, 2: 55, 3: 44, 4: 49, 5: 23, 6: 13, 7: 3, 8: 2.
**Hyperparams:** {'objective': 'step_supervised', 'lr': 5e-05, 'epochs': 3, 'batch_size': 16,
'train_n': 384442, 'n_prelude': 4, 'n_core': 4, 'n_coda': 4, 'step_loss_weight': 1.0,
'stepsup_max_steps': 8, 'checkpoint_iterations': True, 'init_state_std': 0.632,
'adapter_init': 'identity', 'core_init': 'pretrained', 'eval_recurrences': [1, 2, 3, 4, 6, 8],
'max_new_tokens': 32}; AdamW, linear decay to 0, no warmup, grad-clip 1.0, bf16 autocast.
Code at commit d15567d (the manifest says `git.commit=unknown` because the pod copy has no
`.git`; the pilot's manifest has the same gap).
**Command:** `uv run python scripts/train_recurrent_depth.py --step-supervised --checkpoint-iterations --train-n -1 --stage full_run --hardware "RunPod RTX A5000 (secure)" --slug stepsup-split4-4-4-full`
(pod `8wztldc69o575h`, CA-MTL-1, $0.27/hr; 21:44–02:15 UTC; 72,084 steps in 16,097 s =
4.5 it/s, peak 5.4 GB; ≈4.8 h of pod time ≈ **$1.30**. Stdout in `pod_stdout.log`, not
committed. Pod terminated 02:25 UTC after checksum-verified rsync.)
**Headline results:** `final_answer_accuracy=0.145` (29/200) at oracle r,
`unparseable_rate=0.000`, `train_loss=1.2602` (token mean over both target kinds).
Windowed losses (nats), step 50 → epoch 1 → 2 → 3 end: step-token CE 4.67 → 2.50 → 2.15 →
**1.93**; answer-token CE 1.78 → 0.88 → 0.78 → **0.71** (pilot ended at 3.37 / 1.07; both
still falling when the lr hit 0).

| r | 1 | 2 | 3 | 4 | 6 | 8 | oracle (mean 3.4) | oracle+1 | oracle+2 |
|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.025 | **0.110** | 0.085 | 0.055 | 0.060 | 0.065 | **0.145** | 0.090 | 0.075 |

Accuracy by test step count n (rows: r; cells: correct / n examples):

| r \ n | 2 (55) | 3 (44) | 4 (49) | 5+ (41) |
|---|---|---|---|---|
| 1 | 1 | 2 | 2 | 0 |
| 2 | **18** | 3 | 0 | 0 |
| 3 | 7 | **6** | 2 | 2 |
| 4 | 5 | 3 | 2 | 1 |
| 8 | 6 | 2 | 2 | 3 |
| oracle (r = n) | 19 | 6 | 1 | 3 |

Step read-out (top-1 token at the number position after iteration i vs step i's first
token; the same 479 (example, step) pairs from the 189 test examples with ≥ 2 steps as the
pilot, fp32): **31.3%** overall (33.0% on the 434 single-token values); by iteration 1…5:
**51.9%, 24.6%, 14.4%, 9.8%, 5.6%** (support 189, 134, 90, 41, 18; pilot: 16.4, 6.7, 6.7,
9.8, 5.6). Baselines on the same pairs, unchanged from the pilot: most common step token in
train (`" 30"`) 3.5%; value appears verbatim in the question 7.5%; value = previous step
1.0%. 323 of the 329 wrong read-outs are still numbers; 38 copy a number from the question.
Iteration × step matrix is now diagonal-dominant: iteration 1 reads step 1 at 51.9% but step
2 at 7.5%; iteration 2 reads step 2 at 24.6% but step 1 at 6.0%; iteration 3 reads step 3 at
14.4% vs steps 1–2 at 4.4% (pilot: iteration 2 still read step 1 better than step 2). All
intermediate steps correct on 40/189 examples (2-step: 27/55, 3-step: 9/44, 4-step: 4/49,
5+: 0/41).
Convergence (fp32, number position, n=200, r=8): relative state change 1.0, **0.42, 0.30,
0.21, 0.12, 0.066, 0.032, 0.014**; KL between consecutive number distributions **3.3, 1.5,
0.73, 0.18, 0.057, 0.015, 0.004**. Pilot: 0.43, 0.20, 0.07, 0.02 / 0.43, 0.14, 0.02; answer-only
runs: 0.075, 0.003 / 8e-3, 3e-5.
Post-hoc checks (script in this session, numbers reproducible from the jsonl files):
- Answer accuracy at oracle r by read-out status: all intermediate steps right **15/40 =
  37.5%**; some right 7/68 = 10.3%; none right 7/81 = 8.6%; 1-step examples 0/11.
- The fixed-r curve *is* the step read-out: at r=1 the generated answer equals v_1 on
  94/189 multi-step examples (49.7%) and equals the fp32 iteration-1 read-out on 84%; at
  r=2 it equals v_2 on 32/134 (23.9%) of examples with n > 2. At oracle+1 the answer
  changes on 95/200 examples.
- Overlap with the full-scale direct-answer control (26/200 correct): 15 shared with the
  oracle's 29. Overlap of oracle-correct with r=2-correct: 20/22; with r=1: 0/5.
- Failure mode when all intermediates are right (25/40 still wrong): the *last* operation.
  `[2, 9, 11]` → reads 2, 9 → answers 12; `[30, 48, 84]` → answers 48 (the last
  intermediate — 45/189 oracle answers equal the last mid read-out); `[120, 180, 36]` → 24.
**Interpretation:**
- **Criterion met: 31.3% ≥ 30%**, though only just, and unevenly — step 1 at 52% carries
  it; steps 2–3 are at 25% / 14%. Against the pilot's 10.6% the read-out tripled with 19×
  the data and is still improving at the end of training (step CE falling 2.15 → 1.93 in
  epoch 3 alone), so it is data/epoch-limited rather than capacity-limited, as explicit
  CoT was (4.5% → 34.5% over the same change). GPT-2 *can* be trained to carry a
  numerical rationale through its recurrent state, one step per iteration.
- **The iteration states are now distinct and step-specific.** The state after iteration
  i predicts v_i and nothing else (diagonal matrix), the state changes by 20–40% per
  iteration through iteration 4 and by ≥ 0.7 nats of KL on the number distribution, and it
  is not at a fixed point by iteration 8. This is the property the Oct 9 / Oct 16 work
  needs: s_1 … s_n with a known label each and a measurable downstream effect. (The
  earlier collapse was a property of the answer-only objective, not of the looped GPT-2.)
- **Final answer: 14.5% at oracle r = the no-scratchpad control (13.0%).** The loop does not
  add answer accuracy over a direct-answer GPT-2 (overlap 15/29, so also not the same
  examples) and is far below explicit CoT (34.5%) and CODI (41.5%) at the same scale.
  Two reasons visible in the data: (i) the per-step values are ~70% wrong, and the answer
  tracks them — 37.5% when the intermediates are right vs 8.6% when none are; (ii) even
  with correct intermediates the last operation fails 25/40 times, often by emitting the
  last intermediate itself. Reading the answer out of the same position that just read
  out v_{n-1} is a harder target than the supervision gives credit for.
- **First non-flat r-curve in any recurrent-depth run — but it is "right r", not "more
  r".** Accuracy peaks when r equals the example's step count (18/55 on 2-step examples at
  r=2 vs 1/55 at r=1 and 5–7/55 at r=3–8) and drops on either side; at r < n the model
  emits step r's value as its answer; at r > n it keeps computing and changes its answer
  half the time. This is the expected behaviour of the objective (iteration n is the only
  one trained to emit the answer) and is *not* Geiping et al.'s test-time scaling, which
  is monotone. Deploying it honestly needs a halt signal (predict n, or an "answer-ready"
  target at every iteration ≥ n); the oracle number uses the test rationale's step count.
- Where this leaves the four-mechanism comparison for Sep 25: recurrent depth on GPT-2
  now has a trained looped model whose iteration states demonstrably encode the
  arithmetic chain (read-out 31% vs 3.5% baseline), with accuracy at the direct-answer
  level. The paper-faithful recipe does not reproduce under fine-tuning (five pilots,
  `20260918-05*`/`-16*`); this step-supervised variant is the recurrent-depth entry we can
  probe. Its read-out accuracy is the *ceiling* for any probe on s_i (the trained head is
  the best decoder we have), so probes should be reported relative to it.
**Gotchas hit:**
- None new. The epoch-end checkpoint (d15567d) worked as intended (epoch-1 copy at 23:16
  UTC, then overwritten by epoch 2 and 3).
- `git.commit=unknown` in the manifest (no `.git` on the pod); code version recorded above
  by hand. Worth fixing in `runpod_setup.sh` / the rsync exclude list before the next run.
**Caveats:** n=200 (±3 points on accuracy; the 479 read-out pairs give ±2 points on the
read-out); one seed; step read-out is measured with the model's *own* trained read-out
head, so it is a training-target fit, **not** a probe result — it is the positive control
and ceiling for the Oct 9 decoding, not evidence for RQ1; the 30% criterion was set on
the overall read-out, which step 1 dominates (steps ≥ 2 are below 25%); oracle r uses the
test rationale's step count — the fixed-r rows are the honest deployment numbers and the
best of them (r=2, 11.0%) is below the direct-answer control; the first-token target is
partial for ~16% of step values (numbers ≥ 1000, decimals); the loop reaches r=8 in
training only for the 2% of examples with 8 steps, so iterations ≥ 6 are essentially
untrained (their read-outs are 0/5 and 1/2).
**Next:**
- Use this checkpoint as the recurrent-depth entry for the Oct 9 decoding step:
  `LoopedGPT2.forward(return_states=True)` gives s_1 … s_n with label v_i each; train
  linear / non-linear probes s_i → v_i (individual per iteration vs one shared across
  iterations — the "shared decoder" question inside one mechanism) and report against the
  31% / 52% ceiling of the trained head. Logit lens on s_i through the coda is the
  read-out itself.
- Oct 16 causality: patch s_i from a counterfactual example with a different v_i and
  check the read-out at i and the answer at n move as predicted — the 37.5% vs 8.6%
  conditional above says the answer *correlates* with the intermediates; patching tests
  whether it *uses* them.
- If a stronger looped model is worth ≈$1.30–2.60 more: (1) more epochs — both losses were
  still falling at lr 0; (2) `--step-loss-weight 3`; (3) an "answer-ready" target at
  iterations ≥ n so fixed-r deployment works and r > n stops hurting; (4) variant (e′),
  distilling the explicit-CoT checkpoint's step states instead of hard tokens.
- (f) Huginn-0125 eval (~$0.50) stays on the list as the only source of *paper-faithful*
  recurrent-depth states with monotone test-time scaling; needs Ethan/Kevin's OK on a
  second backbone.
