## 2026-11-08 — Teacher-side dependence: the CODI teacher's answer state depends only on the last shown CoT value on RuleChain (flip 1.00 vs .00 for every earlier value), on several values on GSM8K (.81 / .37 / .29 / .25 from the end), and on ProsQA barely at all (flip .33 at the last node, state change .02 of the between-item distance) (codi, run_id: 20261108-102233_codi_teacher-dependence-rulechain-gsm8k-prosqa)

**Goal:** Step 8 / lever 3: does the distillation target carry the intermediate values the answer cannot recompute from the question? Corrupt one CoT value given to the teacher at a time and see which ones its answer state depends on. Pre-registered (t20, 2e21ff1).
**Mechanism / model:** RuleChain start-last, relay, stabilised, SIM; GSM8K zen-E/CODI-gpt2; ProsQA grid CODI.
**Data:** 300 items each: RuleChain valid L3-L6; GSM8K-Aug test items with >= 3 steps; ProsQA test with >= 4 steps. Teacher input as in training: question + CoT without its last step + "The answer is: gold".
**Hyperparams / command:** `teacher_dependence.py <task> <model> mps <out> 300`. Corruption: value + d (d = 1..3, then -1..-3; ProsQA: another concept of the question), positional (the step's result and its next mention), same token length. flip = teacher's first answer token changes (ProsQA: preferred option, teacher-forced); move = median change of the layer-12 state at ":" / median distance between two items. Raw `codi_audit_artifacts/t20_*.json`.
**Results (k = position from the end of the shown CoT):**

| teacher | k = 1 | k = 2 | k = 3 | k = 4 | k = 5 |
|---|---|---|---|---|---|
| RuleChain start-last / relay / stab / SIM, flip | 1.00 (all four) | .00 | .00 | .00 | .00 |
| same, move | .80 / .86 / .80 / .91 | .02-.03 | .01-.02 | .01 | .01-.02 |
| GSM8K zen-E, flip (move) | .81 (.59) | .37 (.19) | .29 (.10) | .25 (.07) | .12 (.07) |
| ProsQA grid CODI, flip (move) | .33 (.02) | .00 (.02) | .00 (.02) | .00 (.02) | .00 (.01) |

Between-item scale of the answer state: RuleChain 75-78, GSM8K 71, ProsQA 13.
**Interpretation:** Predictions held for RuleChain (last value only) and GSM8K (spread over several values); ProsQA missed the predicted state movement (.02 at every k). Reading: (1) on RuleChain the distillation target holds only v_{L-1}, so CODI is trained to put one value in its latents; that fits the relay / pooled regimes and explains why only per-pass supervision (gated, SIM) yields chains. (2) On GSM8K the target depends on several step results, matching the several value slots found in the route split. (3) On ProsQA the answer state barely differs between items (scale 13 vs 71-78): at ":" the next token is the person's name, so the target carries almost no path information, which explains the constant ProsQA CODI latents.
**Caveats:** the teacher is the final CODI checkpoint (shared weights), not the teacher during training; GSM8K drops values that repeat; ProsQA flips measured on the option choice.
**Next:** same test on ProntoQA and on Kevin's GSM8K CODI teacher.
