## 2026-10-14 — Ouro-2.6B step-0 gate: few-shot no-CoT GSM8K by loop count (recurrent_depth, run_id: 20261014-124731_recurrent_depth_gate-ouro-2.6b)

**Goal:** Same as the Ouro-1.4B gate (`20261014-100712_recurrent_depth_gate-ouro-1.4b`), for the
larger released model: can base `ByteDance/Ouro-2.6B` answer the xmech eval questions without
CoT, at which loop count, and on enough CODI/Coconut-correct questions for Q4 ladder pairs?
**Mechanism / model:** `recurrent_depth`, released Ouro-2.6B (48-layer stack looped, trained at 4
loops), no fine-tuning. compute_steps = 4; loops 1-6 swept.
**Data:** xmech eval split (= GSM8K test, 1319); run seed=0.
**Hyperparams:** identical to the 1.4B run (8-shot, greedy, newline stop, bs=1, eager, bf16;
CoT reference on the first 200 at 4 loops).
**Command:** `bash scripts/pod/ouro_gate.sh` (second model in the loop); see `eval_command.txt`.

**Headline results:**

| loops | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| no-CoT acc, 2.6B | 0.095 | 0.187 | 0.228 | **0.233** | 0.234 | 0.230 |
| no-CoT acc, 1.4B (for reference) | 0.068 | 0.152 | 0.174 | 0.183 | 0.180 | 0.180 |
| correct in Ouro-2.6B & CODI & Coconut (of 363) | 67 | 133 | 160 | **163** | 165 | 164 |

CoT reference at 4 loops: 0.715 (first 200; no-CoT on the same 200: 0.275). Unparseable: 0.
Ouro&CODI 206, Ouro&Coconut 183 at 4 loops. **Gate: PASS** (loops 2, 3, 4).

**Interpretation:** Same shape as 1.4B, shifted up ~5 points: steep gain from 1 to 2-3 loops,
flat from the trained count on (220 items gained, 38 lost from 1 to 4 loops). 2.6B saturates a
loop earlier than 1.4B in relative terms (loop 3 ≈ loop 4). 169 items are loop-4-correct in both
sizes. The larger model gives more three-way-correct items (163 vs 139), so it is the better
step-1 candidate if its 48 layers x loops patching grid is affordable; 1.4B is half the cost.

**Gotchas hit:** as in the 1.4B notes (transformers 4.56.2 only; batched left-padded generation
broken -> bs=1 + eager; exit step pinned to the last loop).

**Caveats:** Same guessing caveat: top outputs are round numbers (10 x130, 100 x62, 12 x57); 95 of
307 loop-4 correct answers are among its 10 most frequent outputs. One prompt / shot seed. Not
comparable to CODI/Coconut accuracies; within-Ouro curves only. Wall time ~2.5 h on an A40 at
bs=1 -- step 1 should batch by equal-length prompts (no padding) or cache the few-shot prefix.

**Next:** Step 1 localization (twin patching over loop x layer x position) on 1.4B first (cheaper),
2.6B if 1.4B localizes.
