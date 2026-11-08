## 2026-11-08 — Blending two items' latents moves start-last CODI's answer monotonically through in-between values: at every blend weight, .88-.99 of answers are the last rule applied to a value between the two items' values (chance .11); the crossover is S-shaped (keeps own .52 at a = .4, follows .42 at .6), not linear (codi, run_id: 20261108-040037_codi_v2-l1to7w-startlast-latent-blend)

**Goal:** Direct test of the averaging account from `20261108-032531` (donor swaps follow the number of donor passes; mixed swaps give in-between answers; latents hold a magnitude). If a latent holds an analog magnitude, blending two items' latents should move the answer through intermediate values. A discrete code would make it jump.
**Mechanism / model:** start-last CODI checkpoint-17820 (`20261106-083843`).
**Data:** the b16 / t6 edit pairs on canonical **test** (one constant changed, gold moves >= 6, both answered right): 424 pairs, 249 with the edit 1+ rules before the last, 175 edits at the last rule (control).
**Command:** On the laptop (MPS, float32, `run_guarded.sh`, `caffeinate`): `~/Documents/Penn/CIS5980/codi_audit_artifacts/t8_blend.py` (copy of `t6_prefix_hide.py`), tables `t8_table.py` (`t8_table.md`), record `record15.py`. Every latent input becomes z_s = (1-a) z_rec + a z_don, for a = 0, .1, .., 1, on the recipient question. Checks: a = 0 leaves every answer unchanged, and a = 1 reproduces t6's `all` condition (.89 follows).

**Headline results** (edits 1+ rules before the last, n = 249):

| a | 0 | .1 | .2 | .3 | .4 | .5 | .6 | .7 | .8 | .9 | 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| keeps own | 1.00 | .94 | .81 | .72 | .52 | .21 | .03 | .02 | .02 | .02 | .01 |
| follows donor | .00 | .00 | .00 | .01 | .01 | .11 | .42 | .64 | .75 | .82 | .89 |
| strictly between the two answers | .00 | .05 | .17 | .24 | .39 | .53 | .43 | .27 | .19 | .13 | .07 |
| last rule of some value between v_rec and v_don | 1.00 | .99 | .98 | .98 | .95 | .88 | .92 | .97 | .99 | .98 | .96 |
| ... and that value within 2 of the linear blend | 1.00 | .88 | .86 | .56 | .49 | .50 | .53 | .57 | .83 | .88 | .96 |

- **Analog, bounded, monotone.** At every blend weight .88-.99 of answers are what the last rule gives for some value inside [v_rec, v_don]; a uniform guess gives .11. Per pair, the answer sequence over a is monotone in .78 of pairs and visits 3.5 distinct answers on average (mean |v_rec - v_don| = 10).
- **S-shaped, not linear.** The implied value stays near the recipient's until a = .3-.4 and near the donor's from .7-.8. In the middle it is inside the range but only half the time within 2 of the linear blend. Matching the answer to the last rule of the rounded linear blend exactly works only .05-.27 at intermediate a: the last rule's parity or threshold branch flips on small value changes.
- **Larger gaps give more in-between answers.** Edits 2+ rules before the last (gap 21): strictly between up to .68 (a = .5). Edits one rule back (gap 6): up to .47.
- **Control:** edits at the last rule leave answers unchanged at every a (keeps own .98-1.00), since the latents do not carry the last rule.

**Predictions (`t_predictions.md`, written before t8 output):**
- *Gradual, in-between answers common at a = .3-.7:* right.
- *Tracks the rounded linear blend within 2 on a majority at mid a:* **wrong** (.49-.57). The value stays in range but moves along an S-curve.
- *Crossover near a = .5:* right (between .5 and .6).

**Interpretation:** The latents carry the value before the last rule as a continuous quantity that the answer reads out monotonically. Mixing two items' latents gives a value between theirs, never an unrelated one. This supports the magnitude account (`20261108-032531_..._partial-probes`) and the pooling account (in-between answers when passes disagree). The S-shape fits a readout that is continuous but compressive near the ends, or a vote across passes with soft boundaries. These data do not separate the two.
**Gotchas hit:** Scoring against the rounded linear blend understates the analog behaviour, because the last rule is discontinuous in its input. The pre-image test (any value in range that the last rule maps to the answer) is the fair version; it was added after seeing the first table.
**Caveats:**
- One seed; test split.
- Blending in z space may leave the latent manifold at mid a; the in-range result shows the answer stays meaningful there.
- The pre-image test is lenient when the last rule maps many values to the same answer. Chance (.11) accounts for that on average.
**Next:** Attention flow in the same session (`20261108-040037_..._latent-attention-flow`).
