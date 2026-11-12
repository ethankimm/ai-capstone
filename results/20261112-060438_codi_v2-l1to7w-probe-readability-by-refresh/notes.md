## 2026-11-11 — RuleChain readability by refresh (linear probes): in rule-gated and SIM-CODI every intermediate value becomes readable when its loop crossings exist, never before (<= .07 at refresh d-1, 1.00 at d+1), with the stale value held until then; relay hands over one value at refresh 1; start-last and stabilised latents are not linearly legible (codi, run_id: 20261112-060438_codi_v2-l1to7w-probe-readability-by-refresh)

**Goal:** t29 (pre-registered): the probe version of t28b on RuleChain, six models. Does a value appear in the latents exactly when the crossings it needs become available?
**Mechanism / model:** the six t26 RuleChain models.
**Data:** qualified t26 pairs, valid L2-L6, <= 60 per (level, edit); every intermediate value of B that differs from A's. d = crossings a chain needs (edit at rule i, value after rule k: k - max(i, 1) + 1).
**Command:** `scripts/necessity/rc_readability.py <model> mps <out>` (queue `t26/queue_night.sh`).
**Headline results** (among values read in B's own run; B by refresh 0-6):
- gated: d1 0.001 0.794 1.0 1.0 1.0 1.0 1.0; d3 0.0 0.0 0.0 0.731 1.0 1.0 1.0; stale A .98-.99 before the crossing.
- sim: d1 0.02 0.662 1.0 1.0 1.0 1.0 1.0; d3 0.008 0.008 0.024 0.747 1.0 1.0 1.0. simsf the same pattern.
- relay: one value legible per level, read at refresh 1 for every d (.93-.99).
- startlast: only v_1 at L2 legible; stab: no legible site. Chance <= .03 everywhere.
- Edit-type split (post hoc): gated at refresh d: rule edits 1.00, start edits .20 (one more crossing than d assumes); sim / simsf ~.70 at d for every edit type, 1.00 at d+1.
**Predictions:** held for gated, sim, simsf, relay, stab; startlast not testable beyond d = 1 (scored in t_predictions.md).
**Interpretation:** In the chain models each crossing carries the next rule's value, at the time the crossing exists, so their hop counts reflect computation, not copies. The relay model's value arrives after one crossing whatever its depth, which shows the readout can detect early arrival. Legibility is model-specific: start-last CODI holds values (t15, t26) in a form linear probes cannot read.
**Gotchas hit:** none in the run; the ~.70 at refresh d in the first summary was explained by the edit-type split, not a bug.
**Caveats:** one seed per model; probes trained on natural latents of train items; chance computed over all items (recomputed over own-readable items: unchanged, <= .05).
**Next:** Fig 5.
