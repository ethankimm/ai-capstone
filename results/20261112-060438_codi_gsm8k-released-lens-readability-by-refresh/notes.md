## 2026-11-11 — Vocabulary projection on released CODI and SIM-CODI: values become readable at the refresh where their crossings exist for depth <= 2 (CODI .11 -> .99 at d = 1, .37 -> .94 at d = 2), depth 3 partly early; own-run legibility low (CODI 0.37 / 0.169 / 0.137); chance <= .03 (codi, run_id: 20261112-060438_codi_gsm8k-released-lens-readability-by-refresh)

**Goal:** t30 (pre-registered): t28b's readability-by-refresh design with Dilgren & Wiegreffe's readout (vocabulary projection), applicable to models without a decoder.
**Mechanism / model:** gsm8k_codi: hf:zen-E/CODI-gpt2@fd641b3; gsm8k_sim: hf:internlm/SIM_COT-GPT2-CODI (grid_checkpoints/simcot_gpt2_codi).
**Data:** qualified t26 + t27a GSM8K pairs; changed steps of B whose value is a single token (skipped multi-token: codi 39, sim 12).
**Command:** `scripts/necessity/gsm8k_lens.py <model> mps <out>` (queue `t26/queue_night.sh`).
**Headline results:**
- codi: own-readable 0.37, 0.169, 0.137 (d = 1-3); B by refresh d1 0.106 0.988 1.0 1.0 1.0 1.0 1.0; d2 0.268 0.366 0.944 1.0 1.0 1.0 1.0; d3 0.0 0.448 0.483 0.931 1.0 1.0 1.0; stale A at d-1 0.681 / 0.465; chance max 0.028.
- sim: own-readable 0.251, 0.274, 0.155 (d = 1-3); B by refresh d1 0.068 0.874 0.961 0.971 0.981 0.99 1.0; d2 0.028 0.16 0.849 0.981 1.0 1.0 1.0; d3 0.118 0.176 0.588 0.971 0.971 1.0 1.0; stale A at d-1 0.476 / 0.491; chance max 0.027.
**Predictions:** CODI depth-1 own-readable .37 held, SIM .25 missed; switch at d held except CODI d = 2 at d-1 (.37 > .3); stale A held; chance held.
**Interpretation:** The timing result of t28b holds with a readout that needs no trained decoder: values sit in the latents from the refresh where their crossings exist, with the stale value before. Legibility under the vocabulary projection is low and falls with depth, so presence at a crossing does not imply a readable value.
**Gotchas hit:** chance originally computed over all items while B rates are over own-readable items; recomputed on the own-readable subset (2026-11-11): unchanged for CODI / SIM (<= .07), Coconut still ~.3.
**Caveats:** single-token values only; one checkpoint per model.
**Next:** Fig 5 uses CODI from this record; Coconut omitted.
