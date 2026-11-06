## 2026-11-06 — Start-last position and donor tests: CODI's gain now reaches two rules before the last, not just the last rule; everything before the last rule still passes through the latents (codi, run_id: 20261106-105634_codi_v2-l1to7w-startlast-position-donor)

**Goal:** TRAINING_PROTOCOL section 11.3 position test, donor patch, mediation and same-value donors on the selected pair: does CODI's gain spread to earlier rules once prefill is closed?
**Mechanism / model:** `codi`, gpt2. Start-last CODI (6 latents) and CODI-0, checkpoint-17820, fp32, MPS.
**Data:** 200 test items per level, L3-L7 (position) and L2-L7 (donor). 15,129 threshold edits, 3,764 small edits, 10,000 constructed chains (in `predictions.jsonl`, with both models' answers). 424 different-value and 338 same-value donor pairs answered exactly right. Edited and constructed items are new questions.
**Command:** `scripts/codi_null_audit/`: `FORMAT=startlast b13_resolution`, `a25_position`, `FORMAT=startlast b16_donor`, `a21_donor`. The freeze test was skipped because rule tokens hold no running value here.
**Headline results:**
- **Position.** Right branch, CODI vs CODI-0, on items both answered within 2, by rules left after the edited rule. Margin 1: last rule .550 / .225, 1 left .234 / .138 (p .007), 2 left .455 / .359 (p .013), 3+ left .466 / .414 (p .23). Margin 2: .674 / .322, .323 / .195 (p .0003), .507 / .408 (p .024), .480 / .450 (p .35). Margin 3+: .854 / .508, .508 / .348 (p 3e-5), .600 / .501 (p .001), .561 / .533 (p .41). With the start first, round-1 CODI gained at the last rule only (.81 / .59, then .47 / .51, .59 / .61, .58 / .65).
- **Natural items** with exactly one tight decision, exact wins/losses: last rule 16/2 (p .001), 1 left 5/3, 2 left 6/6, 3+ left 16/10. These do not show a gain before the last rule.
- **Constructed chains**, exact CODI / CODI-0. Wide: L3 .51 / .25, L4 .36 / .17, L5 .20 / .11, L6 .13 / .08, L7 .10 / .06. Tight: L3 .16 / .07, L4 .06 / .03, then both near floor.
- **Mediation.** Share following the donor, through latents / direct route. Edit at the last rule 0.00 / .97. 1 rule left .97 / 0.00. 2 left .68 / 0.00. 3+ left .70 / .04. Own-latent control 1.0.
- **Same-value donors** keep the answer 1.00 / .94 / .94 / .86 at L2-L5. Single latents follow the donor .00-.01; replacing from z_1, z_2, z_3 on gives .88, .76, .49.

**Interpretation:** Section 11.3 stays "latents help", because reach difference is +0.23. But "gain at the last rule only" does not describe this model: on edited items the gain reaches one and two rules before the last and stops by three, which fits about two passes of real work (20261106-101740). The route split is the same as the working start-first models: latents carry everything before the last rule, the answer tokens apply it, and latents act through the value.
**Gotchas hit:** `a20_lasthop.py` is written for the start-first runs, so `a25_position.py` repeats its first two tables for this pair.
**Caveats:** One seed. The position split comes from the earlier post-hoc analysis, though it was pre-registered here. Edited items are conditioned on both models being within 2. Natural items do not confirm the earlier-rule gain, with few items and small counts (5/3, 6/6). Deep edits land on neither answer .26-.32 of the time through latents, unexplained.
**Next:** See 20261106-101740.
