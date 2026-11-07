## 2026-11-07 — Start-last readability: the value before the last rule reads .63 / .31 / .21 on right answers at L3-L5, beats the no-latent control, and tracks the model's own errors; earlier steps do not beat the control (codi, run_id: 20261107-084635_codi_v2-l1to7w-startlast-readability-right-wrong)

**Goal:** RQ2 of the paper draft. Pooled reads sat near accuracy (L3 .46 against .52), so were values clear on right answers and absent on wrong ones? Do reads beat the matched no-latent model, and do they reflect what the model actually computed?
**Mechanism / model:** `codi`, gpt2. Start-last CODI (6 latents) and CODI-0, checkpoint-17820; lr-3e-3 CODI as the known relay. fp32, MPS.
**Data:** Valid only, L3-L5, 286 items per level. Probes fit on 8,000 train items of one level; a 4,000 fit reproduces `b20` (L3 h4 layer 2: .46). Right items: 149 / 94 / 39 at L3-L5. `predictions.jsonl` holds start-last CODI's answers and decoded values at h4 layer 2.
**Command:** `codi_audit_artifacts/b23_right_wrong.py`, tables from `a26_right_wrong.py`, under `run_guarded.sh`. Cells fixed in advance (t1, h2, h4, h6, colon at layers 0, 2, 4, 8, 12), and h4 layer 2 is the headline cell, so nothing is picked on scored items.
**Headline results:**
- **Value before the last rule, exact, h4 layer 2, L3 / L4 / L5.** All items .49 / .21 / .12. Right answers .63 / .31 / .21. Wrong answers .33 / .16 / .11. Shuffled labels .01 / .02 / .02.
- **No-latent control.** CODI-0 at the last question token .16 / .04 / .03, at the answer token .19 / .06 / .04.
- **Relay, right answers.** 1.00 / .93 / .79.
- **Raw latent input (h4 layer 0).** .22 / .14 / .09, against .98 / .74 / .44 for the relay. In start-last the value needs one transformer block to read.
- **Own errors.** Applying the last rule to the decoded value gives the model's exact answer on .23 / .21 / .16 of its wrong answers. Using another item's decoded value gives .02 / .02 / .03, and CODI-0's last-token read gives .07 / .05 / .05. A wrong answer can only match if the decoded value is itself wrong, so these are reads of the model's own mistaken value. Relay: .41 at L4, .39 at L5.
- **Earlier steps.** v_1 at h4 layer 12 reads .45 / .28 / .19, against .42 / .26 / .20 for CODI-0 at the last question token, so it does not beat the control. L4 v_2 reads .12 against .08.

**Interpretation:**
- RQ2 profile is late-only, and partial. One value is readable above control and the earlier steps are not, so there is no full rollout.
- Readability is not a null. The read is selective, beats the control, is better on right answers, and follows the model's own wrong values, which is what a monitor needs. It is weak in size: even on right answers it misses 37% at L3 and 69% at L4.
- The hoped-for pattern, clear on right answers and absent on wrong ones, is only partly there: .63 against .33 at L3.
- Used and readable agree only for the group of latents. Single latents are near copies and follow the donor .00-.01 (20261106-105634), so the draft's per-latent cells will read "readable but unused" while replacing the group moves the answer.
- Corrects my earlier statement that the results "already fit the draft's framework". Gain follows the prediction: 0 at k=1, +.02 at k=2, +.24 at k=3 = d50. Reliance does not: all latents ablated takes k=1 from 1.00 to .09 and k=2 from 1.00 to .05 (20261106-105537), though CODI-0 scores 1.00 and .98 with none. The draft's claim table lists "high already at k=1" as refuted.

**Gotchas hit:** None in this run.
**Caveats:** One seed per model. Valid only. Right items are few at L5 (39) and selected for being easy. Linear probes are lower bounds; 4,000 to 8,000 fit items added about .03 at L3. The own-errors test is correlational. The relay has no matched CODI-0.
**Next:** Update the draft's Prediction and claim table to separate need (gain) from reliance (ablation). Run the same analysis on any new start-last arm.
