## 2026-11-03 — RuleChain CODI-0 layerwise probes: branch bits are decodable early and stay decodable (0.96 → 0.77 over rules 2-6); the exact running value is what fails (0.51 → 0.07) (codi, run_id: 20261103-063912_codi_v2-depth2-codi0-branch-probes)

**Goal:** Test whether RuleChain's conditional (the branch decision inside each rule) is what limits CODI-0's single-pass reach (d50 4.90). Inside the one forward pass, when and how well are each rule's branch bit b_j and the values s_{j-1} (rule input) and s_j (rule output) linearly decodable? Prompted by two observations: errors on near-threshold items are 15-20 pt more frequent, and wrong answers equal a single-branch-flip walk at 2-4× chance (but only ~15% of errors).
**Mechanism / model:** `codi`, GPT-2 small + LoRA r128, CODI-0 (0 latents), `gpt2_depth2_codi0/checkpoint-29716` (epoch 38, the checkpoint behind round-1 run 20261102-060537; local copy in `rulechain-checkpoints/`).
**Data:** RuleChain (depth2, generator 2.0.0) test, levels 1-8 (4000 items, 500/level). Probe split 70/30 by sha256(item id); probes for rule j use items with level ≥ j.
**Hyperparams:** One forward per item, as eval: question (+ `<bot>`) pass, then `<eot>` "The answer is:". Hidden states (13 = embeddings + 12 blocks) at two positions: `rule_end` = the "." closing rule j's clause, `ans` = the ":" of "The answer is:". Linear softmax probes per (position, j, layer, target) on per-layer-standardised states, full-batch L-BFGS 150 iterations, L2 lambda in {1e-4, 1e-3, 1e-2} chosen on 20% of probe-train. Branch: 2-way, scored as balanced accuracy. Values: two 10-way digit heads (tens, ones), scored as exact match (both right). Baselines: majority class, and a "surface" linear probe on question features that need no execution (x0, x0 mod 2, kind one-hot + constants of rules 1..j).
**Command:** `scripts/probe_rulechain_branch_codi.py` from the CODI checkout, see `eval_command.txt` (MPS, `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5`).
**Headline results:** teacher-forced accuracy reproduces round 1 (L4 0.764, L5 0.498; overall 0.584 over L1-8). At `rule_end`, best layer per target (probe-test split):

| rule j | n test | branch b_j (bal. acc) | surface baseline | input s_{j-1} (exact) | output s_j (exact) | majority (values) |
|---|---|---|---|---|---|---|
| 1 | 1189 | 1.00 (≥0.89 from layer 1, 1.00 at layer 6) | 0.68 | 1.00 (= x0, from layer 2) | 0.99 (only from layer 10: 0.54 at L9, 0.97 at L10) | 0.02 |
| 2 | 1039 | 0.96 (L10; 0.93 at L8) | 0.64 | 0.89 (L12; 0.30 at L9) | 0.51 (L12) | 0.02 |
| 3 | 893 | 0.89 | 0.63 | 0.83 | 0.34 | 0.03 |
| 4 | 731 | 0.87 | 0.62 | 0.58 | 0.27 | 0.03 |
| 5 | 589 | 0.80 | 0.58 | 0.24 | 0.11 | 0.03 |
| 6 | 445 | 0.77 | 0.59 | 0.14 | 0.07 | 0.03 |
| 7 | 291 | 0.74 | 0.60 | 0.06 | 0.05 | 0.03 |

At `ans`, intermediates are mostly not decodable (s_j ≤ 0.18 for every j, b_j at most 0.13 above surface for j ≥ 2).

**Interpretation:**
- **The computation runs at the question tokens, in reading order.** Each rule's quantities are decodable at that rule's own "." and not at the answer position. CODI-0 executes the chain while reading the question.
- **Within rule 1 the branch comes first, then the value.** b_1 is readable from layer 1 (0.89) and perfect by layer 6; s_1 becomes exact only at layer 10. So the "condition, then arithmetic" split is real and visible in depth, with the arithmetic as the slow part (about 4 more layers).
- **Branch bits stay decodable where the exact value is gone.** At rule 4, b_4 0.87 vs exact s_4 0.27; at rule 2, b_2 reaches 0.93 by layer 8 while the input s_1 is only 0.30 exact at that layer at that position. The model carries a coarse value that is good enough to decide most branches, while the exact value degrades fast (0.99 → 0.51 → 0.34 → 0.27 → 0.11).
- **Each rule pushes the exact value further up the stack.** s_{j-1} reaches the next rule's "." only at layers 11-12 (j = 2, 3), and s_j at its own "." is best at the last layer from j = 2 on: the pass runs out of depth for exact values around rules 2-4, matching the accuracy decline.
- **Fits the error analysis.** A coarse value predicts (a) the near-threshold penalty (coarse values fail exactly where the comparison is close) and (b) most errors not being clean branch flips (the value is off, not the decision). Wrong items keep the branch bit about as well as right ones at rules 1-4 (e.g. rule 2, 0.94 vs 0.96); the separation is in the exact value at rules 5-6 (layer 12: 0.20 vs 0.08 and 0.19 vs 0.05, small n).
- **For the branch-free control (V2_TASKS_SPEC.md, "Branch-free control"):** this predicts a small gain from removing the conditional. The decision is cheap and early; exact value tracking is the bottleneck.

**Gotchas hit:** The first full run (20261103-063514, deleted, never committed) had a label-ordering bug in the batched probe loss (`y.repeat(K*L)` against logits flattened as (k, n, l)), so every probe trained on misaligned labels and sat at chance. Found because rule-1 x0 decoding was at chance. Fixed and checked on synthetic data (signal layer 0.80, others chance) before this run.
**Caveats:**
- One checkpoint, one seed; linear probes only; probe-test n falls to 291 at rule 7.
- **Binary vs 100-way is not matched difficulty.** A branch bit can be right from an approximate value, so "branch 0.87 vs value 0.27" is about exact vs coarse information, not two equally hard readouts. Tens-digit-only accuracy, which would measure coarse value directly, isn't saved separately.
- The surface baseline is one linear model and can't express the kind × comparison interaction, so it is a weak "no execution" bar (notably for j = 1, where the branch is computable from x0 and t).
- Probes pool levels: s_j is the final answer for level-j items and an intermediate otherwise.
- `rule_end` is the "." token only; computation may also continue on the following tokens ("Rule", "j+1", ":").
**Next:**
- Save tens-digit accuracy and probe the tokens after each "." to locate where s_j finishes.
- Same probes on CODI (6 latents) and on the latents themselves.
- Branch-free control C1/C2 as written in V2_TASKS_SPEC.md, now with the prediction "little gain".
