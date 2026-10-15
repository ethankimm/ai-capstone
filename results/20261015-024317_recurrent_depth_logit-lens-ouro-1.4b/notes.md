## 2026-10-15 — Ouro-1.4B step 2b: logit lens for the intermediate's leading digit per (loop, layer) (recurrent_depth, run_id: 20261015-024317_recurrent_depth_logit-lens-ouro-1.4b)

**Goal:** Follow-up to the linear probes (`20261015-022002_recurrent_depth_probe-ouro-1.4b`), which
tested log-magnitude and the *ones* digit. Ouro writes numbers one digit at a time after a `" "`,
so its unembedding at the suffix's last position can show a number's *leading* digit. Does the
first intermediate's leading digit appear there in early loops, before the answer's takes over
(a latent step-by-step signature)? The logit lens is also the readout the CODI runs here used.
**Mechanism / model:** `recurrent_depth`, released Ouro-1.4B, 4 loops, 24 layers, gate prompt.
**Data:** 1144 xmech eval questions (GSM8K test; chains >= 2 steps, integer values); 979 of them
"differ" (step1's leading digit != the answer's); 204 Ouro-correct at 4 loops. The leading-digit
probe is fit on 3200 xmech fit questions, with lambda chosen on 800.
**Method:** lens = `lm_head(model.norm(h))` on the output of layer l in loop u. Ouro applies
`model.norm` at the end of every loop, so at layer 23 this is exactly that loop's own exit
readout (sanity: it matches the model's own output digit on 100% of items). The lens is read at
the last question token (`q_last`) and the suffix's final `" "` (`sfx3`), restricted to the 10
digit tokens. **beats_donor** = P(target's leading digit) > P(donor digit), where the donor is
another item's step1 leading digit, different from this item's step1 and answer leading digits.
Chance is 0.5 regardless of how common each digit is (Benford). A ridge leading-digit probe on
the same features is also reported against the majority-class baseline.
**Command:** see `eval_command.txt` (pod, A40, ~20 min, ~$0.20). Post-hoc controls:
`python3 scripts/ouro_lens_controls.py results/<this run> xmech_questions.jsonl`.

**Headline results** (`sfx3`, differ items n=979; loops 1-indexed, layers 0-indexed):

| | loop 1 | loop 2 | loop 3 | loop 4 |
|---|---|---|---|---|
| answer beats donor (end of loop) | 0.585 | 0.68 | 0.70 | 0.705 |
| step1 beats donor (end of loop) | 0.546 | 0.545 | 0.532 | 0.513 |
| step1 beats donor (best layer in loop: l 20) | 0.60 | 0.62 | 0.60 | 0.57 |
| — step1 digit also leads a question number (n=355) | 0.63 | 0.59 | 0.58 | 0.55 |
| — step1 digit novel (n=624) | 0.50 | 0.52 | 0.51 | 0.49 |
| — novel, donor not a question-number lead (n=362) | 0.57 | 0.57 | 0.56 | 0.56 |
| copy control: a question number's leading digit (n=551, same donor restriction) | 0.54 | 0.50 | 0.52 | 0.51 |
| Ouro-correct & step1 novel (n=116): step1 / answer | 0.52 / 0.76 | 0.56 / 0.97 | 0.57 / 1.00 | 0.60 / 1.00 |

Rows marked "end of loop" and the split rows use layer 23 (per-item values saved only there);
the "best layer" row is from the manifest grids. Top-1 over digits at layer 23: answer 0.32 → 0.42 over loops 1 → 4 (majority 0.29); step1
0.27 → 0.21. Supervised leading-digit probe at
`sfx3`: answer 0.32 → 0.41 by loop 4 (majority 0.29); step1 0.21–0.27 everywhere (majority on
differ 0.24). At `q_last` nothing: lens digit mass ~0, beats_donor 0.46–0.55, probes at majority.
Standard error at n≈1000 is ±0.016; at n≈360, ±0.026.

**Interpretation:**
- **The answer's leading digit is computed across loops**: weak after loop 1 (0.585), clearly
  present from loop 2 on (0.68 → 0.70). This matches the gate's accuracy curve and step 1's
  patching effect, and it holds for answer digits that occur nowhere in the question (0.52 →
  0.66).
- **No early-intermediate signature.** The step1 digit signal is weak and comes mostly from
  items where that digit already leads a number in the question (0.63 vs. 0.50 when it doesn't).
  With the donor restricted to digits not in the question, novel step1 digits reach 0.56–0.57.
  That is a little above the copy control at loops 2–4 (0.50–0.52), but flat across loops: it
  neither peaks early nor fades as the answer forms. Where Ouro is right, the step1 digit rises
  slowly alongside the answer's (0.52 → 0.60) rather than preceding it. This looks like
  correlation with the answer's digit, not an intermediate that is computed and then consumed.
- The mid-loop bump (layer 20 of loops 1–2, ~0.60–0.62) could not be split by the copy control
  (per-item values are saved only at end of loop). Its size matches the "copyable" row, so it
  most likely has the same source.

**For Q4:** consistent with step 1 and the linear probes. The `" "` position carries the answer's
leading digit, built up over loops 2–4, and no readable intermediate. A CODI/Coconut → Ouro
transplant here would be null.

**Gotchas hit:** none new. All prompts ran uncached (as in the probe run). bs=1, eager.

**Caveats:**
- Only the leading digit is testable this way; GSM8K's first intermediate often shares it with
  the answer (165 of 1144 items excluded as "same").
- The control analysis is post hoc and at layer 23 only.
- Only 116 items are both Ouro-correct and step1-novel.
- The base model is right on only 18% of items.
- Ouro-1.4B only.

**Next:** Ouro for Q4 is done unless someone wants Ouro-2.6B or a fine-tuned Ouro (not
budgeted). Report Ouro as the layer-wise mechanism whose no-CoT computation leaves the answer
digit, but no decodable intermediate, at the answer position.
