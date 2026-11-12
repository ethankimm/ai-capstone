## 2026-11-11 — GSM8K hop count, grid Coconut: hop counts rise with dependency depth but by less than one per step (depth 2 / 3 / 4: coconut 1.376 (n=178), 1.722 (n=54), 1.684 (n=19)); M_6 = donor run on every pair (run_id: 20261112-060438_coconut_gsm8k-grid-hop-count)

**Goal:** t26 / t27a (pre-registered): how many loop crossings does an answer change on GSM8K need, and does the count keep rising for longer edits?
**Mechanism / model:** gsm8k_coconut: connordilgren grid Coconut GPT-2 GSM8K (grid_checkpoints). GPT-2 small, fp32 MPS, batch 1.
**Data:** GSM8K-Aug test. A = test question; B = A with one number shifted, gold recomputed by re-executing A's gold solution (t22 candidates, 1,471; t27a long twins with >= 4 steps, 1,305). A pair is used for a model only if that model answers both right: gsm8k_coconut 220 + 42.
**Command:** `scripts/necessity/hops.py <model> mps <out>`; t27a with `CANDS=t26/gsm8k_long_twins.json`.
**Headline results:**
- t26 F_0 / F_1 / F_2: coconut 0.018 / 0.564 / 0.968.
- Mean hops by dependency depth (bins n >= 15): coconut 1.376 (n=178), 1.722 (n=54), 1.684 (n=19). Depth equals listed steps on 86-90% of pairs.
- Hop distribution at depth 4: 2 and 3+ hops roughly split for CODI / SIM (Fig 4); Coconut's depth-4 bin is small.
**Predictions:** missed: F_1 >= .9 and no rise with s (registered "slots filled from the question"). The low-confidence alternative, forward chaining across slots, held. t27a "levels off": held for CODI and SIM, Coconut missed on n = 8.
**Interpretation:** Released GSM8K latent models chain about two crossings: values are computed, stored in a slot and read back by later passes through attention, so a check patching only the recurrent input (t22: input-only <= .04) misses them. From depth 3 on, steps are merged into fewer crossings.
**Gotchas hit:** twin pairs need s >= 2, so depth 1 is empty. The neither-A-nor-B rate at M_0 is 8-12%.
**Caveats:** released checkpoints only, one each; small n at depth >= 4; hops is an upper bound on serial computation steps.
**Next:** paper figures (Fig 3b, Fig 4).
