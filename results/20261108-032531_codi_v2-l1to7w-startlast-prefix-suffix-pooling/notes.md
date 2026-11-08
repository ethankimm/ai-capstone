## 2026-11-08 — Start-last CODI's latent passes are parallel noisy copies that the answer averages: donor swaps follow the number of donor passes, not their position (1 / 2 / 3 / 4 / 5 of passes 1-5: follows .01 / .05 / .45 / .75 / .88), mixed swaps give in-between answers (.78-.86, chance .2), later passes overwrite a donor's early latents, and any single pass alone gives .61-.76 exact but .84-.95 within 2 (codi, run_id: 20261108-032531_codi_v2-l1to7w-startlast-prefix-suffix-pooling)

**Goal:** Understand how start-last CODI (`20261106-083843`, the one unforced model with a matched gain past reach) uses its latents. Earlier records established two working passes, then a fixed point (`20261106-101740`), a look-back to earlier passes that a relay lacks, and that single latents never move the answer. Open: do later passes build on earlier latents (a chain), and how does the answer combine six near-copies?
**Mechanism / model:** start-last CODI checkpoint-17820, 6 latents, ungated.
**Data:** the b16 pairs on canonical **test** (`FORMAT=startlast`, seeds 61/16, 200 items per level L2-L7). A diff pair is an item plus a copy with one taken constant changed (gold moves >= 6). Kept: both answered right, 424 diff pairs (249 with the edit 1+ rules before the last).
**Command:** On the laptop (MPS, float32, `run_guarded.sh`, `caffeinate`): `~/Documents/Penn/CIS5980/codi_audit_artifacts/t6_prefix_hide.py`, a copy of `scripts/codi_null_audit/b16_donor.py` with new conditions; tables `t6_table.py` (`t6_table.md`); record `record13.py`. The own-latent control left every answer unchanged. The `all` and `suffix_s` rows reproduce `b16_raw_startlast.json` (`20261106-105634`): .97 / .88 / .56 / .03 for suffix from z_1..z_4, one rule left.
**Conditions:** prefix_s puts the donor's z_0..z_{s-1} in and recomputes the later latents from them (a chaining test). suffix_s puts the donor's z_s..z_5 in. hide_ge_s / only_s cut the answer tokens' attention to latent passes >= s, or to all but pass s, with clean latents (which passes the answer reads). Pass t is the pass whose input is z_t.

**Headline results:**
- **Later passes overwrite a donor's early latents.** One rule left (n = 178): prefix_1, _2, _3 follow .00 / .00 / .02 and keep the recipient's answer .99 / .92 / .63. Only with four or five donor latents does the donor win (.46, .85). A chain would carry an early change forward; here each pass's output is re-derived mostly from the question.
- **The answer follows the number of donor passes, not their position.** Edits 1+ rules before the last (n = 249), donor-input passes among passes 1-5:

| Donor passes | Condition | Follows donor | Keeps own | Other |
|---|---|---|---|---|
| 1 (pass 1) | prefix_2 | .01 | .83 | .16 |
| 2 (passes 4-5) / (1-2) | suffix_4 / prefix_3 | .06 / .04 | .44 / .53 | .51 / .43 |
| 3 (3-5) / (1-3) | suffix_3 / prefix_4 | .49 / .41 | .04 / .08 | .47 / .51 |
| 4 (2-5) / (1-4) | suffix_2 / prefix_5 | .76 / .73 | .02 / .02 | .21 / .24 |
| 5 (1-5) | suffix_1 | .88 | .01 | .11 |

  Early and late passes count the same: prefix and suffix swaps with the same count agree within .08.
- **Mixed swaps give in-between answers.** When the answer is neither the donor's nor the recipient's, it lies strictly between the two .78-.86 of the time, against .18-.22 for a uniform guess. The answer averages the values the passes hold.
- **Pass 0 (input z_0 from `<bot>`) is not read; any one later pass carries most of it.** With only pass 0 visible, accuracy is .05-.11 (chance). With one pass s = 1..5 alone, recipients stay right .61-.76 at L3-L5, and within 2 .84-.95. Seeing passes 0-4 gives .95-1.00 (hide_ge_5). Each pass holds a noisy copy, and pooling makes it exact.
- **No-model re-analysis of saved runs (same session, before t6):**
  - b29's all-six donors (test, `20261107-220408`): the answer follows the donor's v_{L-1} exactly .37 / .27 / .15 at L3-L5, but within 2 .82 / .59 / .44. With donors the model gets right: exact .47 / .43, within 2 .91 / .86 at L3-L4.
  - b21's look-back block (`20261106-101740`) by level: answers changed .30 / .31 / .23 / .21 / .20 at L2-L6, accuracy .997 to .699 at L2, then net -.07, -.03, -.01, .00. Blocking only the previous latent changes .00-.11 with no accuracy cost.

**Predictions (`t_predictions.md`, written before t6 output):**
- *Prefix follows near the 'all' rate by s = 2-3:* **wrong**. Prefixes are overwritten until four or five passes are swapped.
- *hide_ge_3 near clean:* right at L2 (1.00), partly at L3-L5 (.74-.84).
- *One pass alone keeps much of the accuracy:* right (.61-.76).
- *only_0 worse:* right (chance).

**Interpretation:** Start-last CODI does not chain. Its passes 1-5 each compute a noisy estimate of the value before the last rule, mostly from the question; the look-back to earlier passes feeds into that estimate. The answer pools the passes, close to an average: swapped-in donor passes pull the answer by their number, and a split gives a value in between. This explains:
- why single swaps never move the answer (one vote in five);
- why mean-ablating all six is fatal while ablating one is not;
- why the model stops improving after two or three passes (more copies only reduce noise);
- why its errors are mostly near-misses (`20261105-074801`).

The latents add precision by averaging, on top of the one or two hops the look-back carries. For monitoring, the value is in no single latent; it is a population code across passes.
**Gotchas hit:** None in the script. hide rows count recipients once per item (diff pairs share recipients).
**Caveats:**
- One seed. Test split (the b16 pairs).
- Edits 2+ rules before the last rest on 44 and 16 pairs.
- "Averages" is a description of in-between answers and count dependence, not a fitted model of the readout.
- In prefix swaps the donor passes still attend to the clean question, so "overwritten" means the question dominates a pass's output, not that the input is ignored.
**Next:** Partial-information probes in the same session (`20261108-032531_..._partial-probes`): is the value held as a magnitude with a noisy ones digit?
