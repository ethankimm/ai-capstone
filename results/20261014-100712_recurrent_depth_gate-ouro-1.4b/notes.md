## 2026-10-14 — Ouro-1.4B step-0 gate: few-shot no-CoT GSM8K by loop count (recurrent_depth, run_id: 20261014-100712_recurrent_depth_gate-ouro-1.4b)

**Goal:** Step 0 of extending Q4 (CODI <-> Coconut linear-map transplant) to a looped model.
No GSM8K-trained Ouro exists, so: can the released base `ByteDance/Ouro-1.4B` answer the
xmech eval questions (GSM8K test, 1319) *without* CoT, at which loop count, and on enough of
the questions CODI and Coconut also get right to build ladder pairs?
**Mechanism / model:** `recurrent_depth`, released Ouro-1.4B (whole 24-layer stack looped,
trained at 4 loops), no fine-tuning. compute_steps = 4 = trained loop count; loops 1-6 swept.
**Data:** xmech eval split (= GSM8K test, 1319), shots from the xmech fit split; run seed=0.
**Hyperparams:** 8-shot `Question: ...\nAnswer: <number>`, greedy, stop at newline, max 8 new
tokens. CoT reference: same shots with the `<<..>>` rationale written out, "The answer is N",
first 200 eval questions, 4 loops, max 160 tokens. Batch size 1, eager attention, bf16.
**Command:** `bash scripts/pod/ouro_gate.sh` (transformers 4.56.2 venv); see `eval_command.txt`.

**Headline results:**

| loops | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| no-CoT acc | 0.068 | 0.152 | 0.174 | **0.183** | 0.180 | 0.180 |
| correct in Ouro & CODI & Coconut (of 363) | 42 | 113 | 132 | **139** | 137 | 137 |

CoT reference at 4 loops: 0.665 (first 200; no-CoT on the same 200: 0.24). Unparseable: 0.
**Gate (fixed before running: some loops <= 4 with acc >= 0.15 and >= 100 three-way-correct): PASS**
(loops 2, 3, 4).

**Interpretation:** The base model does the task format fine (66.5% with CoT) and gets ~18%
without it. Loops buy no-CoT accuracy steeply up to 2 and saturate at the trained count (4);
5-6 neither help nor hurt. 192 questions flip wrong->right from 1 to 4 loops, 40 right->wrong.
So there is serial computation across loops worth localizing, and enough shared-correct items
(139 three-way, 175 Ouro&CODI, 157 Ouro&Coconut) for Q4-style ladder pairs.

**Gotchas hit:**
- Ouro remote code needs transformers 4.56.x: 4.54.1 / 4.55.0 fail with
  `property 'key_cache' of 'UniversalTransformerCache' object has no setter`; 4.53.3 lacks
  `GenericForQuestionAnswering`.
- **Batched left-padded generation is broken**: only the longest (unpadded) row decodes
  correctly, padded rows produce garbage (" 1. total. total"). SDPA additionally crashes on
  padded batches at loops > 1 (`(*bias): last dimension must be contiguous`). Hence bs=1 + eager.
  An earlier batched smoke run read ~0% for this reason -- any batched Ouro number is invalid.
- Default exit rule (`early_exit_threshold=1.0` on the cumulative exit-gate pdf) can exit before
  the last loop if the bf16 cumsum rounds to 1.0; the script pins `early_exit_step = k-1`.

**Caveats:**
- Guessing: Ouro's most common no-CoT outputs are round numbers (10 x130, 100 x65, 12 x63);
  79 of the 242 loop-4 correct answers are one of its 10 most frequent outputs. Some
  "base-correct" items are lucky guesses with no computed intermediate -- step 1 should prefer
  recipients that are correct at loops 3-4 and whose answer is not a frequent guess, or at least
  report results split that way.
- One prompt, one shot seed; no prompt search. A better prompt might raise no-CoT accuracy.
- Not comparable to the CODI/Coconut accuracies (different size, pretraining, and no
  task fine-tuning); only within-Ouro curves are meaningful.

**Next:** Ouro-2.6B gate (same script, running). Then step 1: Q1-style same-problem twin
patching over (loop, layer, token position) at 4 loops to find where Ouro holds the
intermediate value, before any Q4 transplant.
