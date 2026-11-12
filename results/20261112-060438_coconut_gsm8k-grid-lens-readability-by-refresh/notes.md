## 2026-11-11 — Vocabulary projection on grid Coconut GSM8K fails its chance control (.27-.34): the readout is not interpretable for Coconut (coconut, run_id: 20261112-060438_coconut_gsm8k-grid-lens-readability-by-refresh)

**Goal:** t30 (pre-registered): t28b's readability-by-refresh design with Dilgren & Wiegreffe's readout (vocabulary projection), applicable to models without a decoder.
**Mechanism / model:** gsm8k_coconut: connordilgren grid Coconut GPT-2 GSM8K (grid_checkpoints).
**Data:** qualified t26 + t27a GSM8K pairs; changed steps of B whose value is a single token (skipped multi-token: coconut 4).
**Command:** `scripts/necessity/gsm8k_lens.py <model> mps <out>` (queue `t26/queue_night.sh`).
**Headline results:**
- own-readable 0.945, 0.807, 0.466, 0.724, 0.308 by depth; refresh-0 reads already 0.562 / 0.488; chance 0.322, 0.305, 0.267, 0.345, 0.231.
**Predictions:** Chance <= .10 missed (.27-.34); the other Coconut predictions are not interpretable. Null result, recorded per the logging rule.
**Interpretation:** Coconut's projected latents rank many small numbers in their top 10, so hits are often luck (another pair's value is 'read' ~30% of the time). No timing claim for Coconut from this readout; a probe or decoder would be needed.
**Gotchas hit:** chance originally computed over all items while B rates are over own-readable items; recomputed on the own-readable subset (2026-11-11): unchanged for CODI / SIM (<= .07), Coconut still ~.3.
**Caveats:** single-token values only; one checkpoint per model.
**Next:** Fig 5 uses CODI from this record; Coconut omitted.
