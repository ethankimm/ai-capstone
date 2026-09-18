## 2026-09-18 — Pilot (a): break the collapsed init — state channel random at init (recurrent_depth, run_id: 20260918-160346_recurrent_depth_split4-4-4-rbar32-randstate-adapter-pilot)

**Goal:** Single-variable follow-up to the two collapsed pilots
(`20260918-052821_…-rbar32-pilot`, `…-fullbptt-pilot`), which fine-tune into a loop that reaches
its fixed point within ~3 iterations so accuracy is flat in test-time r. Hypothesis under test:
the identity adapter init `A = [I, 0]` makes `s_i = R(e)` for every i at step 0 — training
*starts* at a collapsed fixed point and never leaves. Here the state half of the adapter is
drawn `N(0, 2/(5h))` at init (`--adapter-init random_state`; `e` path still identity), so
iterations differ from the first step (at init the state moves 35% → 8% → 1.7% over
iterations 2–4, CPU check). Everything else identical to the first pilot.
**Mechanism / model:** `recurrent_depth`, gpt2 (125,621,760 params) /
`results/20260918-160346_recurrent_depth_split4-4-4-rbar32-randstate-adapter-pilot/ckpt`
(`model.pt`, synced to this machine). (4, 4, 4) split, `s_i = core(A[e ; LN(s_{i-1})])`,
`s_0 ~ N(0, 2/5)`; see `latentreasoning/mechanisms/recurrent_depth.py`. `compute_steps` = r at
eval; manifest reports r=32, sweep in `metrics.extra.sweep_by_r` / `predictions_r{r}.jsonl`.
**Data:** gsm8k-aug train 20,000-example subset (seed 42, excludes the validation carve-out);
eval gsm8k-aug test n=200 seed=0; run seed=42 (same r sequence and example order as the first
pilot).
**Hyperparams:** first pilot's except `adapter_init=random_state`: {'lr': 5e-05, 'epochs': 3,
'batch_size': 16, 'train_n': 20000, 'n_prelude': 4, 'n_core': 4, 'n_coda': 4,
'mean_recurrence': 32, 'lognormal_sigma': 0.5, 'backprop_last_k': 8, 'init_state_std': 0.632,
'core_init': 'pretrained', 'core_lr': 5e-05, 'eval_recurrences': [1, 2, 4, 8, 16, 32, 64],
'max_new_tokens': 32}.
**Command:** `uv run python scripts/train_recurrent_depth.py --train-n 20000 --stage pilot --hardware "RunPod RTX A5000 (secure)" --adapter-init random_state --slug split4-4-4-rbar32-randstate-adapter-pilot`
(pod `id24tq987niiqv`, EU-SE-1, $0.27/hr; 16:03–16:19 UTC; 3750 steps in 861 s = 4.4 it/s;
≈$0.07 of pod time. Stdout in `pod_stdout.log`, not committed.)
**Headline results:** `final_answer_accuracy=0.030` (6/200) at r=32, `unparseable_rate=0.000`,
`train_loss=1.1636` (first pilot 1.1625; no-loop control 1.1693).

| r | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| accuracy (this run) | 0.035 | 0.025 | 0.025 | 0.025 | 0.025 | 0.030 | 0.025 |
| accuracy (first pilot, identity init) | 0.025 | 0.040 | 0.040 | 0.035 | 0.035 | 0.040 | 0.040 |

Convergence, re-measured after the run in fp32 on CPU at the position that predicts the first
answer-number token (prompt + `" ####"`; n=200 eval prompts, s_0 seed 42; the manifest's
`metrics.extra.state_rel_delta/next_token_kl` are the in-run bf16 numbers at the bare prompt's
last position — see Gotchas): relative state change per iteration 1.0, 0.25, 0.041, 0.0069,
0.0012, … 9e-7 at i=16 (geometric, ratio ≈ 1/6); KL between consecutive number distributions
5.7e-2 (i=2), 1.1e-3, 3.1e-5, 9.6e-7, … ~0; top-1 number flips between iterations 70, 12, 1, 1,
0 … (of 200); mean log p(gold first token) −4.461 (i=1) → −4.454 → −4.450 → −4.449 (i=16);
top-1 = gold on 9 → 10 → 10 of 200. The trained adapter's state half is essentially its init
(‖A_s‖_F 17.6 at init and after training; ‖A_e − I‖_F = 1.5). Ablating A_s := 0 on the trained
model changes 9/30 outputs on the bundled sample, accuracy unchanged.
**Interpretation:** The collapsed init is **not** the cause. Started away from the fixed point,
the loop is a slower contraction (1/6 per iteration vs 1/30 with the identity init — set by the
init, and training did not change it) but converges just the same, and the iterations it does
take don't help: the gold answer's log-probability moves by +0.01 nats over the whole
trajectory, top-1 = gold stays at 9–10/200, and the sweep is flat (5–7/200 at every r). The
second iteration reshuffles the model's guess among wrong numbers on 70/200 examples; from the
fourth iteration on nothing changes. Same train loss as the identity-init pilot and as the
no-loop control: the loop is not being used to fit the data better. Together with (b) and (c)
(logged next), the four candidate mechanisms — truncated backprop, collapsed init, r
mismatch, pretrained shortcut — are all eliminated; see the interpretation in (c)'s notes and
the status section of `recurrent_depth.py`.
**Gotchas hit:**
- The in-run convergence KL (`metrics.extra.next_token_kl`, also in the first two pilots) was
  measured at the bare prompt's last position, where the fine-tuned model emits `" ####"` with
  p ≈ 1 — KL between two near-one-hot distributions is ~0 whatever the state does (a same-norm
  *random* perturbation of the state also gave KL ~1e-9). It is uninformative and was
  over-read in the first pilot's notes ("read-out identical from iteration 2"). The
  diagnostics above are re-measured at the number-predicting position in fp32; the script now
  does that (`scripts/train_recurrent_depth.py:diagnostics`). The state-convergence numbers
  were and are fine.
- `transformers` 5 no-ops `_init_weights` on already-loaded params (needed for (c)); see
  `reinit_gpt2_block`.
**Caveats:** all of the first pilot's (pilot scale — every same-recipe pilot sits at 1–4.5%;
n=200 → nothing in the sweep is significant; one seed; one split). 6 vs 8 correct is noise.
**Next:** see (c)'s notes and the plan in `recurrent_depth.py` — (e) step-supervised loop and
(f) released Huginn-0125 are the routes to a working vertical scratchpad.
