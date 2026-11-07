## 2026-11-07 — Rule-gated start-last CODI on the canonical L1-L12 test: L1-L7 as the main record, L8-L12 at floor because the gate hides their middle rules, and the output is the answer of the visible rules on 97-99% of items (codi, run_id: 20261107-204918_codi_v2-l1to7w-startlast-gated-depth2-codi-canonical)

**Goal:** Secondary check on L8-L12 (TRAINING_PROTOCOL sections 10.3 and 12). Under the gate these levels cannot be solved by construction: six passes see rules 1-6, the answer tokens see the last rule, and rules 7 to L-1 are seen by nothing.
**Mechanism / model:** As 20261107-204814 (rule-gated start-last CODI, checkpoint-17820).
**Data:** `synthetic_data_v2_startlast`: canonical L1-L12 test, 500/level, start value after the rules. L1-L7 items are the main record's, item for item.
**Command:** `scripts/pod/codi_stab_arm.sh`; `eval_command.txt`. The visible-rules check ran on the laptop from `predictions.jsonl`: simulate each chain from the question text (0 gold mismatches), applying only rules 1-6 and the last rule.
**Headline results:**
- **Test, exact:** L1-L6 1.000, L7 0.982, L8 0.006, L9 0.104, L10 0.014, L11 0.032, L12 0.014; overall 0.596. Ungated CODI on L8-L12: .054 / .038 / .036 / .010 / .018.
- **Visible-rules answer:** applying rules 1-6 and then the last rule gives the gold answer on .004 / .104 / .014 / .032 / .014 of L8-L12 items, matching accuracy at every level. The model outputs that visible-rules answer on .988 / .986 / .982 / .968 / .982 of items.
- **Teacher (written CoT, same weights):** L8-L12 .652 / .236 / .116 / .046 / .036.

**Interpretation:**
- L9's .104 is not a leak: on about 10% of L9 items the two hidden rules (7 and 8) cancel, for example halve then double, so the visible rules give the gold answer.
- The model computes exactly the chain the gate shows it, even on levels it never trained on, extended past 6 rules only by the answer tokens' last rule. This is out-of-distribution evidence that its latents run the gated chain rule by rule.

**Gotchas hit:** None.
**Caveats:** One seed. Out of range by construction, so accuracy here says nothing about reach. The visible-rules check is behavioural, not a probe of the latents.
**Next:** None for this record.
