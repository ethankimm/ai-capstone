## 2026-11-08 — Start-last CODI's latents hold the value before the last rule as a magnitude: tens digit .97 / .95 / .87 on right answers at L3-L5, ones digit only .65 / .38 / .36; within 2 .95 / .69 / .59, against at most .40 anywhere in CODI-0. The latents add coarse magnitude the single pass lacks; parity and branch bits are computed in the question pass by both models (codi, run_id: 20261108-032531_codi_v2-l1to7w-startlast-partial-probes)

**Goal:** Exact-value probes read start-last CODI's latents weakly (.49 / .21 / .12 at L3-L5, `20261107-084635`), yet the latents are used. Swapped-in latents carry the donor's value approximately (b29 re-analysis: exact .47, within 2 .91 at L3, right donors). Does a latent hold the value as a magnitude with a noisy last digit, and does it hold anything CODI-0's single pass does not?
**Mechanism / model:** start-last CODI checkpoint-17820 (6 latents) and start-last CODI-0 checkpoint-17820 (no latents; same data, recipe, seed), `20261106-083843` / `20261106-052644`.
**Data:** `synthetic_data_v2_l1to7w_startlast` **valid** L3-L5 (286 per level); probes fit on 8,000 train items per level (seed 20), as b23.
**Command:** On the laptop (MPS, float32, `run_guarded.sh`, `caffeinate`): `~/Documents/Penn/CIS5980/codi_audit_artifacts/t7_partial_probe.py` (b23's collector, new targets), tables `t7_table.py` (`t7_table_all.md`, `t7_table_right.md`), record `record14.py`. Smoke: the start value sits at token t10 (asserted). The collector reproduces b23's right counts (CODI 149 / 94 / 39).
**Probes:** ridge on standardised features, one-hot targets, penalty chosen per target on a held-out 15% of fit items. A shuffled-label fit gives each target's floor.
- Sites: start-value token (t10), the "." after it (t9), `<bot>` (t1), latent passes h1..h6, answer colon. Layers 0, 1, 2, 3, 4, 8, 12.
- Targets per step v_j: tens digit, ones digit, parity, exact (both digits), scalar within 2. Per rule: branch taken.
- Cells below are the best over layers and sites (selected on scored items, slightly high).

**Headline results** (value before the last rule, v_{L-1}; best latent site vs best site anywhere in CODI-0):

| | L3 | L4 | L5 |
|---|---|---|---|
| CODI latents, all items: exact / tens / ones / within 2 | .50 / .92 / .52 / .85 | .27 / .77 / .30 / .56 | .13 / .60 / .21 / .36 |
| CODI latents, right answers: exact / tens / ones / within 2 | .65 / .98 / .66 / .95 | .35 / .95 / .38 / .69 | .33 / .87 / .36 / .59 |
| CODI-0, any site, all items: exact / tens / ones / within 2 | .30 / .76 / .37 / .40 | .08 / .49 / .17 / .21 | .05 / .36 / .17 / .15 |
| CODI's own question tail (t10 / t9 / t1): tens / within 2 | .72 / .52 | .44 / .26 | .34 / .14 |
| Shuffled floor (tens / ones / within 2), majority (tens / ones) | ~.2 / ~.1 / ~.08; .21 / .14 | same | same |

- **Magnitude is held, the last digit is not.** On items the model answers right, a latent gives the tens digit of v_{L-1} at .87-.98, the ones digit at .36-.66. The latents carry a coarse value.
- **The latents add magnitude the single pass lacks.** v_{L-1} within 2: CODI latents .85 / .56 / .36, CODI-0 at most .40 / .21 / .15 anywhere, CODI's own question tail .52 / .26 / .14. The value is mostly there from the second latent pass on (h2), as in `20261106-101740`.
- **The first step is computed in the question tail, more precisely by CODI-0.** v_1 ones digit at the tail: CODI-0 .80 / .60 / .55, CODI .55 / .45 / .48 at L3-L5. CODI defers work to its latents, as noted before.
- **Parity and branches come from the single pass.** Parity of v_{L-1} reads .84-.88 at L3 at every CODI site, the start-value token included, and up to .85 in CODI-0 (majority .53). Branch bits read .69-1.00 at every site in both models (majority .50-.59). Late rules' branches read better at latents and colon than at the tail (L4 rule 4: latents .93, tail .73, CODI-0 colon .80).

**Predictions (`t_predictions.md`, written before t7 output):**
- *Tens and within 2 well above exact and above the ones digit and parity:* right for tens and within 2 vs ones and exact. Parity is high, but not latent-specific.
- *CODI-0 reads coarse v_{L-1} lower than CODI's latents:* right.
- *Early branch bits readable at the tail in both:* right.

**Interpretation:** Together with `20261108-032531_..._prefix-suffix-pooling`, start-last CODI has a coherent mechanism:
- The question pass computes the first step, parity and branch decisions.
- Each latent pass from the second on computes a magnitude estimate of the value before the last rule, the part the single pass cannot reach.
- The answer averages the passes and applies the last rule.

Exact-value probes underrate what the latents hold. The right readout for this model is magnitude, which a monitor reads at .87-.98 on right answers. The exactness bottleneck is the ones digit: it is noisy in any one latent, and pooling over passes recovers it often but not always. That matches the near-miss error profile. This gives the paper's unforced case a mechanism: latents add real but analog depth, coarse magnitude carried past the single pass's reach. Forced models (gated, SIM) instead hold exact values, one per latent.
**Gotchas hit:** None.
**Caveats:**
- One seed; valid only. L5 right answers are 39 items.
- Linear probes: a ones digit held in a non-linear (for example circular) code would read low here. The earlier MLP check matched linear probes for exact values only.
- Best over sites and layers is selected on the scored items.
- Parity's high floor reflects that it is computed early, not that probes are weak.
**Next:**
- Report start-last CODI as the analog, pooled case in the readable x used story: readable as magnitude by a linear probe, used as a population, no single latent decisive.
- A direct test of the averaging account: interpolate latents between two items and check whether the answer moves monotonically. Not run.
