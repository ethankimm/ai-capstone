## 2026-11-12 — RuleChain Coconut hop count (t33): latents unused, mean hops .01, F_0 >= .98 in every bin (coconut, run_id: 20261112-120347_coconut_v2-rulechain-coconut-hop-count)

**Goal:** t33 (pre-registered 2026-11-11 23:30): the hop count for our RuleChain Coconut, as the Coconut comparison to the six CODI models.
**Mechanism / model:** Coconut c = 1, M = 6, start first, GPT-2 small (rulechain-checkpoints/gpt2_depth2_coconut_c1m6/checkpoint_10), one seed.
**Data:** synthetic_data_v2/depth2 test L2-L6, minimal pairs as in t26, own-answer qualification (t31): 1,332 pairs, 1,310 with a wrong answer. Clean accuracy L2 .278, L3 .116, L4 .062, L5 .050, L6 .028.
**Command:** `QUAL=own ../codi/.venv/bin/python scripts/necessity/hops.py coconut_rc mps <out>`.
**Headline results:** mean hops .01; F_0 >= .98 in every s bin (the answer follows the donor question with the recipient's latents).
**Predictions:** F_0 >= .90 in every s bin: held.
**Interpretation:** This Coconut answers from the question pass alone; its latents carry nothing the answer reads. It gives no crossing-based monitoring point, consistent with its accuracy at or below no_thoughts (RULECHAIN_RESULTS.md).
**Gotchas hit:** right on only 22 pairs, so right-vs-wrong splits are not meaningful.
**Caveats:** one checkpoint at a weak operating point; a Coconut that beats no_thoughts on RuleChain might behave differently.
**Next:** none for this checkpoint.
