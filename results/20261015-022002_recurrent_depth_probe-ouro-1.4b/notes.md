## 2026-10-15 — Ouro-1.4B step 2: linear probes for the intermediate value per (position, loop, layer) (recurrent_depth, run_id: 20261015-022002_recurrent_depth_probe-ouro-1.4b)

**Goal:** Step 2 of extending Q4 to a looped model. Step 1
(`20261014-163349_recurrent_depth_twin-patch-ouro-1.4b`) showed that the `\nAnswer: ` suffix
*causally* carries only the answer's leading digit. Is the intermediate step value at least
*present* (linearly decodable) at the end of the question or the suffix, and does it build up
over loops?
**Mechanism / model:** `recurrent_depth`, released Ouro-1.4B, 4 loops (trained count), 24 layers,
no fine-tuning. Same 8-shot no-CoT prompt as the gate.
**Data:** probes fit on 3200 xmech fit questions (GSM8K train, excluding the 8 shots), lambda on
another 800, reported on the 1144 xmech eval questions (GSM8K test) whose chain has >= 2 steps
and integer values. Subsets: Ouro correct at 4 loops in the gate (204) / wrong (940); step-1
value not literally in the question (1058).
**Method:** one forward pass per question; decoder-layer outputs at the last question token
(`q_last`) and the 4 suffix tokens (`sfx0`-`sfx3`, `'\n' 'Answer' ':' ' '`) for every (loop,
layer): 480 sites. Ridge on z-scored features, lambda in {1e-3..10} x n chosen per site and
target on the val split. Targets: `step1` (first intermediate), `penult` (last intermediate),
`answer`, `qlast_num` (last number in the question, a surface control), and `step1_resid` (slog
step1 with its linear dependence on slog answer and slog qlast_num removed). Regression on
signed-log1p (Pearson r; hit@5% relative error after inverting), ridge classifier on the ones
digit (accuracy vs. majority class). "Best site" = chosen on val, then reported on eval.
**Command:** see `eval_command.txt` (pod, transformers 4.56.2 venv, A40, ~20 min, ~$0.20).

**Headline results** (eval n=1144; loops 1-indexed, layers 0-indexed; sites `pos/u_l` are 0-indexed):

| target | best site | Pearson r (slog) | same pos, loop 1 layer 0 | hit@5% | ones-digit acc | majority |
|---|---|---|---|---|---|---|
| step1 | sfx3/0_16 | 0.69 | 0.41 | 0.045 | 0.45 | 0.43 |
| penult | sfx3/1_17 | 0.76 | 0.48 | 0.038 | 0.44 | 0.43 |
| answer | sfx2/3_19 | 0.90 | 0.53 | 0.052 | 0.39 | 0.38 |
| qlast_num (surface) | sfx3/0_7 | 0.76 | 0.44 | 0.036 | **0.56** | 0.35 |
| step1_resid | sfx3/1_16 | 0.46 | 0.13 | — | — | — |

hit@5% and digit accuracy are each taken at their own val-chosen best site. Best r per loop (max
over layers, `sfx3`): step1 0.70 / 0.69 / 0.67 / 0.68; step1_resid 0.45 / 0.47 / 0.46 / 0.46;
answer 0.89 / 0.89 / 0.88 / 0.89. Same picture at `q_last` (step1 0.61 → 0.58, answer 0.80 →
0.78). Ouro-correct vs wrong items: no meaningful gap (step1 r 0.69 vs 0.70; answer digit 0.51
vs 0.37, but the correct subset's majority digit (0) is already 0.46). Full 4x24 grids per
position/target in the manifest (`grids_eval_all`); all subsets, val scores and lambdas in
`per_site.json` (gitignored, local).

**Interpretation:**
- **Magnitude yes, value no.** The order of magnitude of the intermediate and the answer is
  linearly readable (r 0.69–0.90 in log space), but the exact value is not: hit@5% <= 5% for
  every target, and the ones digit of step1, penult and answer sits at the majority-class floor
  at every one of the 480 sites. The probe *can* read an exact digit when one is in the text:
  the ones digit of the last question number decodes to 0.56 vs. a 0.35 floor. So the null on
  the computed values is not a probe-capacity artifact.
- **Loops add nothing to what is decodable.** Everything that is decodable is there by the
  upper-middle layers of loop 1 and stays flat (or falls slightly) through loops 2–4. Yet the
  gate's no-CoT accuracy goes 6.8% → 18.3% over the same loops, and step 1's causal effect at the
  suffix appears only in loops 2–4. The extra loops change something the linear probe does not
  see, or they change too few items to show up in a 1144-question correlation.
- **The magnitude signal is mostly a question-number readout.** At loop 1, layer 0 (barely past
  the embeddings) step1 r is already 0.41. `step1_resid` (beyond answer and the last question
  number) still reaches 0.46, but magnitude is also predictable from the *other* question numbers,
  which this residual does not remove. Nothing here shows a computed intermediate.

**For Q4:** this confirms step 1. There is no site at the end of the question or the suffix where
Ouro holds the intermediate value in a form a linear map could carry. A CODI/Coconut → Ouro
transplant at these sites would be null by construction. Ouro joins the program as the vertical
mechanism whose no-CoT computation, at this base-model accuracy (18%), does not leave a
decodable intermediate at a post-question site. Not comparable in number to the CODI/Coconut
probe runs (`20260920-042323_codi_probe-continuous-full`: ridge r 0.11–0.18 on latent slots);
those probed slots meant to hold a step, here positions that are not.

**Gotchas hit:**
- The prefix-cache path never fired: `tok(prefix + question)` never starts with `tok(prefix)` (the
  boundary tokenizes differently when joined; not investigated), so all 5144 questions ran the
  uncached full prompt, as in the gate. Correct, just slower (~0.14 s/question at 4 loops on an A40, 702 s total).
- Features for all 480 sites (5144 x 5 x 4 x 24 x 2048, bf16) are 10.1 GB, kept on the GPU; the
  probe fits take 35 s for all sites.

**Caveats:** Linear probes only, with single-digit and log-magnitude readouts. A
non-linear probe, or a probe for "value as a digit string at the answer positions", could still
find more. The question is only 1 of 5 positions; the value could sit at the operand tokens
inside the question (not probed). The base model is right on only 18% of these items, and the
correct subset is small (204) and contains frequent guesses, so "present when Ouro gets it
right" is weakly tested. One prompt; 1.4B only.

**Next:** Recommend stopping the into-Ouro transplant and reporting Ouro as above. If more Ouro
work is wanted: run the same probe on operand-token positions inside the question, or try an
MLP probe on the best sites. Both are cheap.
