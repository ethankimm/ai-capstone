## 2026-09-18 — CODI: authors' released GPT-2 weights, paper inference protocol (codi, run_id: 20260918-021217_codi_released-weights-6lat)

**Goal:** Step 1 of the plan for CODI — "reproduce that number first" (paper reports
43.7% on GSM8K test for GPT-2). Verify the paper's claim on our backbone/benchmark at
~zero cost by evaluating their *released* checkpoint under their *own* protocol, and
get the shared 200-slice number for cross-mechanism comparison. This is an
**inference-only verification of the released weights, not a training reproduction**
— see Caveats and `latentreasoning/mechanisms/codi.py` for what a training rerun
would cost and when it becomes necessary.
**Mechanism / model:** `codi`, gpt2 / HF `zen-E/CODI-gpt2` @ `fd641b3` (single
`pytorch_model.bin`, sha256 `fd223b14…d8417`, 406 MB; 144,499,200 params of which
20,057,088 LoRA+projection, loaded with 0 missing / 0 unexpected keys),
compute_steps=6 (continuous-thought tokens at inference, `--inf_latent_iterations 6`).
**Data:** gsm8k-aug test n=200 seed=0 (the shared slice); run seed=None (greedy
decoding, no RNG on our side; the checkpoint's training seed is 11 per the paper
script but unverifiable for released weights).
**Hyperparams:** the paper recipe, recorded in `manifest.json` from the reference
repo's `scripts/train_gpt2_gsm8k-aug.sh` @ `2c23146` (not from a `training_args.bin`
— the released weights don't ship one): lr 3e-3, 40 epochs, eff. batch 128, LoRA
r=128/α=32 on c_attn/c_proj/c_fc, 6 latents, projection 768+LN, smooth-L1 distill
÷ teacher std, α=β=γ=1. Inference: greedy, 6 latents, batch 128 in file order,
max_new_tokens 256.
**Command:**
```
# env: bash scripts/codi_setup.sh /workspace/codi   (pinned commit + their requirements.txt + codi_streaming.patch)
# weights: snapshot_download("zen-E/CODI-gpt2", revision="fd641b3d3edc59e4f534b55588e906588c9e36bb") -> /workspace/codi_released
# (a) their test.py, verbatim per scripts/test_gpt2.sh (flags below) -> "GSM8K test accuracy: 43.67%"
# (b) ours, same model loading + decode loop, scored through score_outputs:
cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/eval_codi.py \
  --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3 (sha256 fd223b14…)" \
  --slug released-weights-6lat --stage full_run --hardware "RunPod RTX A5000 (secure)" \
  --output_dir /tmp/codi_eval_unused \
  --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \
  --lora_r 128 --lora_alpha 32 --lora_init --batch_size 128 --greedy True \
  --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
  --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True
```
(full argv in `eval_command.txt`)
**Headline results:** `final_answer_accuracy=0.415` (83/200), `unparseable_rate=0.000`,
`compute_steps=6`. Full test split (1319): **0.4367 = 43.67%** vs paper's 43.7% —
and identical to what their unmodified `test.py` prints on the same weights
(`/workspace/codi_test_released.log` on the pod; `predictions_full_test.jsonl` here,
gitignored, holds all 1319 outputs). Paper's own last-number metric and ours agree on
every example (`paper_metric_accuracy_slice` = ours). Avg output length 5.3 tokens
("The answer is: N" — CODI emits no visible reasoning). `sec_per_example=0.0032`
(batched 128, A5000).
**Interpretation:** The paper's GPT-2 number reproduces exactly from the released
weights, so the claim is verified for our exact setting. The 200-slice number (41.5%)
is ~2 points under the full-split number — sampling noise on n=200 (95% CI roughly
±7 points), not a discrepancy; quote 43.67%/1319 when comparing to the paper and
41.5%/200 when comparing to our other mechanisms. Against our current full-scale
numbers on the same slice: filler_tokens compute_steps=0 (no-CoT fine-tune) 13.0%,
compute_steps=32 12.0%, explicit_cot full-scale pending (pilot 4.5%). CODI at
compute_steps=6 is far above the no-CoT control — but see Caveats before reading
that as a like-for-like mechanism comparison.
**Gotchas hit:**
- Found and fixed a **shared-scorer bug** while building this: 14/1319 test golds are
  written with thousands separators ("2,125"; 3 of them in the 200-slice), and
  `is_correct` did `float(gold)` → ValueError → fell back to a string compare a
  correct "2125" could never pass. Fixed in `latentreasoning/eval/metrics.py`
  (+ regression test). Re-scored every existing `predictions.jsonl`: **no past number
  changes** (no model had those three right). This run's 3 comma-gold examples are
  all wrong on their merits (875000 vs 1,450,000; 2600 vs 5,600; 51500 vs 43,500).
- The reference repo evaluates on `gsm8k/main` test; we generate from GSM8k-Aug's
  test file. Checked: 0 question-text and 0 answer mismatches across all 1319, so the
  two are the same benchmark.
- Their `load_dataset("zen-E/GSM8k-Aug")` in train.py hits the same pyarrow
  false-positive as our loader did → `scripts/codi_streaming.patch`. Not exercised by
  this inference-only run, but applied by `codi_setup.sh` so the training path is
  ready.
- Upstream attention-mask quirk (`fix_attn_mask=False` in the paper): left-padding is
  attended after the question step, so outputs depend on batch composition. We use
  their exact batching (128, file order) — that's *why* the numbers are identical.
**Caveats:**
- **Not a training reproduction.** We verified inference on their weights; we did not
  retrain (paper: ~36h on one A100 80GB ≈ $57 on RunPod — deferred for budget;
  `scripts/train_codi.sh` is ready, and the paper's Table A5 gives a cheaper
  verifiable target: 38.4% at 20 epochs). Retraining becomes necessary for (a) seed
  variance, (b) `compute_steps` sweeps that change `num_latent` at *training* time,
  or (c) if this verification had failed.
- **Recipe asymmetry vs our other mechanisms:** CODI = 40 epochs LoRA(r=128) on the
  full 385k at lr 3e-3; our filler_tokens / explicit_cot = 3 epochs full fine-tune at
  lr 5e-5. The paper's own No-CoT-SFT baseline (their recipe) is 19.1% vs our 13.0%
  no-CoT control — the gap between those two is the recipe, not the mechanism.
  Budget-matched comparison needs the consolidation discussion CLAUDE.md defers to
  after Sep 25; don't quote 41.5% vs 12.0% as "CODI beats filler tokens" without it.
- `compute_steps=6` here ≈ 6 extra single-token forward passes with KV cache, so it
  is FLOP-comparable to 6 filler tokens, not to recurrent-depth loops.
**Next:**
- explicit_cot full-scale (running on the same pod as of this note) to complete the
  same-recipe baseline set.
- Oct 9: this checkpoint is the CODI model for intermediate-state decoding — the 6
  latent hidden states (per layer, `output_hidden_states=True` in the decode loop)
  are the scratchpad positions. `inf_latent_iterations` < 6 on this model gives the
  `early_termination_necessity` measurement for free.
- If budget allows: `EXTRA_ARGS="--num_train_epochs 20" bash scripts/train_codi.sh`
  on an A100 80GB (~18h, ~$29) against the 38.4% Table A5 target.
