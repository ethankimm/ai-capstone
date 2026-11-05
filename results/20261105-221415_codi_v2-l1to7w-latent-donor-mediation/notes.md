## 2026-11-05 — Which route carries the running value to the answer? In working CODI models the latents carry everything before the last rule and the answer tokens apply the last rule; in stabilised CODI the latents carry nothing (codi, run_id: 20261105-221415_codi_v2-l1to7w-latent-donor-mediation)

**Goal:** Reference results for TRAINING_PROTOCOL section 11.3 on the three existing CODI models, before the start-last pair finishes: the paper draft's donor patch, mediation (Paul et al.) and a same-value donor (the paraphrase analogue).
**Mechanism / model:** `codi`, gpt2, 6 latents. Stabilised CODI, lr-3e-3 L1-7w CODI, round-1 CODI. fp32, MPS.
**Data:** 200 canonical test items per level, L2-7. 2,187 different-value pairs (one taken constant edited, gold moves by at least 6) and 1,200 same-value pairs (different start and earlier rules, same value after rule i, same later rules). Only pairs where both prompts are answered exactly right are scored: 986-1,252 and 602-734 per model. `predictions.jsonl` holds every pair and condition.
**Command:** `scripts/codi_null_audit/b16_donor.py`, `a21_donor.py`.
**Headline results:** Share of answers exactly equal to the donor's answer. Control: writing a run's own latents back changed no answer (1.0, all models).

| | Stabilised CODI | lr-3e-3 CODI | Round-1 CODI |
|---|---|---|---|
| Edit before the last rule, latents from donor, clean question | 0.001 | 0.946 | 0.964 |
| Edit before the last rule, edited question, clean latents | 0.899 | 0.001 | 0.000 |
| Edit at the last rule, latents from donor, clean question | 0.000 | 0.000 | 0.000 |
| Edit at the last rule, edited question, clean latents | 1.000 | 1.000 | 0.994 |

- Same-value donor keeps the answer 0.87-1.00 on lr-3e-3 CODI and 0.82-1.00 on round-1 CODI by level (0.82 at L6 is n = 22). So the latents act through the value, not item-specific detail. Stabilised keeps it 0.96-1.00, which says nothing because it ignores latents.
- Patching any single latent, later latents recomputed: 0.00 follows the donor, every site, every model. Replacing sites from s onward, edits before the last rule, lr-3e-3 / round 1: z_1 on 0.95 / 0.89, z_2 on 0.74 / 0.61, z_3 on 0.38 / 0.19, z_4 on 0.04 / 0.01, z_5 on 0.00 / 0.00. z_0 adds nothing beyond z_1 on lr-3e-3.

**Interpretation:**
- **Complete mediation, split at the last rule.** In the two working models, every edit before the last rule reaches the answer through the latents and none reaches it directly, while the last rule reaches it directly and never through the latents. The answer is a function of the latent value and the last rule's text.
- **The extra step happens at the answer tokens, not in the latents.** This corrects 20261105-185416, which said latents "finish the step themselves". It fits the training target: with `include_last_cot False`, the teacher's `:` holds the value before the last rule, and the last step is computed at the answer position.
- **The recipe picks the route.** Same objective and architecture, opposite routes: stabilised CODI answers from the question and ignores latents; lr-3e-3 and round-1 CODI answer from latents and ignore the question's chain. Accuracy is similar, so this is reliance without necessity: round-1 CODI follows donors at L2-3, where CODI-0 needs no latents.
- **A relay, not a chain.** A patched latent is overwritten by the next one, so latents re-read the state rather than build on each other. A serial computation would propagate a single patch. Most of z_1 to z_3 must change before the answer moves.
- The mean-ablation collapse of round-1 CODI looks like lost information rather than only off-distribution damage, because real donor latents move the answer exactly.

**Gotchas hit:** None in this script. The own-latent control was added after the smoke test.
**Caveats:** One seed per recipe. Only correctly answered pairs, so few at L6-7 (round 1 has no scored L7 row, 32 pairs at L6). Patches replace the input to each latent pass; KV from earlier patched latents is recomputed from them. "Direct" means every route except the latents, including the tail and answer tokens. The route labels are behavioural; which tokens hold the value in prefill is not tested here.
**Next:** The same tests on the start-last pair. `--include_last_cot True` now has a sharp prediction: the last rule should move from the direct route into the latents.
