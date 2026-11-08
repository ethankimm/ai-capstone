## 2026-11-08 — SIM-CODI's 7th pass applies rule 6 again, not rule 7: at L7 with 7 passes the answer is r7(r6(v_6)) on .877 of items (the handoff's "rule 7 twice" .091); the pass-to-rule pointer saturates at the last trained position (codi, run_id: 20261108-013539_codi_v2-l1to7w-startlast-sim-extra-passes)

**Goal:** HANDOFF_2026-11-07 step 5, second check. SIM-CODI start-last holds L1-L6 at 1.000 with extra latent passes but falls to .02-.06 at L7 with k >= 7 (`s3_sim_passes.json`, `20261107-233522`). Untested guess from that record: the extra pass applies rule 7 and the answer tokens apply it again. (Step 5's first check, whether z_{L-1} holds v_{L-1} or the answer, is answered in `20261108-012524`: v_{L-1}.)
**Mechanism / model:** SIM-CODI start-last checkpoint-16335 (`20261107-225603`), decoder dropped, trained with 6 latents.
**Data:** valid L7 (285) and L6 (286, control), answer after k = 6, 7, 8, 9 latent passes actually run (k > 6 is outside training).
**Command:** On the laptop (MPS, float32, `run_guarded.sh`, `caffeinate`): `~/Documents/Penn/CIS5980/codi_audit_artifacts/t5_l7_passes.py`, record `record12.py`. Smoke: 40 items at k = 6 give .975, the same single error as s1's clean answers. The smoke's own floor (.98) was too tight for 40 items and was relaxed to .95.
**Candidates (fixed before the full run):** gold; last rule twice r7(r7(v_6)); three times; r7(r_j(v_6)) for j = 1..6 (the extra pass applies rule j, then the answer applies rule 7); v_6. Share = fraction of items matching; unique = matching only that candidate.

**Headline results (L7):**

| k | gold | r7(r6(v_6)) | r7(r7(v_6)) (handoff guess) | other r7(r_j(v_6)) | none | post hoc: r7(r6^(k-6)(v_6)) |
|---|---|---|---|---|---|---|
| 6 | **.996** | .004 | .000 | .000-.004 | .004 | - |
| 7 | .021 | **.877** (unique .565) | .091 | .074-.091 | .084 | .877 |
| 8 | .056 | .133 | .032 | .014-.021 | .698 | **.495** |
| 9 | .025 | .109 | .025 | .021-.046 | .733 | **.305** |

- **The 7th pass applies rule 6 again.** At k = 7 the answer is rule 7 applied to rule 6 applied to v_6 on .877 of items. The handoff's guess (rule 7 twice) matches .091, at the level of the other wrong-rule candidates, which share values with the correct one by coincidence. Unique matches: r7(r6(v_6)) .565, gold .018, every other candidate <= .007.
- **Post hoc (chosen after seeing k = 7):** each further pass applies rule 6 once more. Rule 6 twice then rule 7 matches .495 at k = 8; three times, .305 at k = 9. The share falls as errors compound.
- **L6 control:** gold .996 at every k = 6..9, answers identical across k. At L6 the 7th and later passes come after the last trained step, are idle (step 1: inert), and change nothing.

**Predictions (`t_predictions.md`):** "rule 7 twice, the majority at k = 7": **wrong** (.091). "L6 stays at gold": right.
**Interpretation:** SIM-CODI's passes map to rules by position, and the map saturates at the last position seen in training. Pass s applies rule s+1 for s = 0..5; an untrained pass 6 applies rule 6 again, the rule of the last trained pass position. At L7 the step chain has no room left: z_6 carries v_6, the extra pass applies rule 6 to it, and the answer tokens apply rule 7 to the result. At L6 and below, the pass after the last step is idle (no rule to read), so extra passes are harmless. So the rollout is a position-indexed program, not a "find the next unread rule" procedure, and its length is bounded by training. This fits the out-of-range pattern in `20261107-225709` (out of range it mostly applies rules 1-6 plus the last rule).
**Gotchas hit:** The smoke floor of .98 failed on 39/40 (a known single error); relaxed to .95 after checking the error equals s1's.
**Caveats:** One seed; valid only. The k = 8 and 9 test is post hoc. Candidates can coincide on an item, which is why unique shares are given. Only rule 6's repetition was tested, not what an extra pass does at L <= 5 (idle there by step 1).
**Next:** Handoff step 4 (rollout formation over epochs) is the only one left, optional.
