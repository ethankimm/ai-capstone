## 2026-11-11 — SIM-CODI on GSM8K chains its steps: a step's value becomes decodable exactly at the refresh where its loop crossings exist (depth 1 / 2: .12 -> .96, .20 -> .96), depth 3 arrives one refresh early (.66), and the stale value sits in the latents until then; CODI's latents are not legible to SIM's decoder (codi, run_id: 20261112-060438_codi_gsm8k-released-sim-readability-by-refresh)

**Goal:** t28 (pre-registered): are SIM-CODI's readable GSM8K steps computed in parallel from the question, or chained through the loop? t28b / t28c (pre-registered, post hoc design): readability at every refresh by crossings needed, with chance, stale-value and negative-model controls.
**Mechanism / model:** released SIM-CODI GPT-2 and its training-only step decoder; t28c decodes released CODI's latents with SIM's decoder.
**Data:** SIM's (t28c: CODI's) qualified t26 + t27a GSM8K pairs. Each changed step of B gets depth d = crossings a chain needs before its value can sit in a latent.
**Command:** `scripts/necessity/sim_parallel.py mps <out>`; `LATENT=codi` for t28c.
**Headline results:**
- t28: dependent steps read at refresh 1 / 2 / own 0.121 / 0.434 / 0.487; direct 0.828 / 0.852 / 0.85. Chain, not parallel.
- t28b, readable in own run by depth: 0.85, 0.625, 0.321, 0.275, 0.467. Given own-readable, B by refresh 0-6: d1 0.12 0.96 1.0 1.0 1.0 1.0 1.0; d2 0.069 0.2 0.959 1.0 1.0 1.0 1.0; d3 0.127 0.254 0.662 0.986 0.986 1.0 1.0; d4 0.091 0.091 0.545 0.955 1.0 1.0 1.0.
- Stale A at refresh d-1: .95 / .76 (d = 1, 2), then <= .14. Chance (another pair's value, same depth, own-readable items) <= .07.
- t28c, CODI's own-run readable share: 0.238, 0.098, 0.077, 0.068, 0.067. Refresh 6 = own: SIM 392 / 392, CODI 417 / 417.
**Predictions:** t28 parallel hypothesis refuted as registered; direct steps held; direct floor .11 missed narrowly. t28b held for d = 1, 2, missed for d = 3 (early switch). t28c held.
**Interpretation:** SIM-CODI's readable GSM8K trace is the output of the loop, not a side readout of the question: values appear when their crossings appear, matching its hop count (about 2). Beyond two dependent steps, steps merge (early arrival) and legibility drops (.32 at depth 3). Legibility is model-specific: CODI uses the same depth but its latents are not readable with SIM's decoder.
**Gotchas hit:** the first version of t28 used chain position, not dependency depth; t28b fixed depth exactly.
**Caveats:** one checkpoint; rates conditioned on own-run readability; depth 4 has few readable items.
**Next:** Fig 5 (paper_figures.py).
