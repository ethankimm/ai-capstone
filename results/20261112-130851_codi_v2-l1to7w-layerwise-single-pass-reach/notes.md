## 2026-11-12 — Layer-wise single-pass reach on RuleChain (t35): exact values become readable only at layers 11-12, and at most v_1 (v_2 in stabilised CODI) in one pass (codi, run_id: 20261112-130851_codi_v2-l1to7w-layerwise-single-pass-reach)

**Goal:** t35 (pre-registered 2026-11-12 00:15), thread 1b: how many RuleChain steps does one 12-layer question pass compute, measured at every layer?
**Mechanism / model:** stab0 and startlast0 (matched CODI-0 references), stab, startlast, simsf, sim.
**Data:** synthetic-depth2 valid L3-L6, design of t21c (prefill probe on the question pass), 2,000 fit items per level.
**Command:** `LAYERS=0,1,...,12 ../codi/.venv/bin/python scripts/necessity/prefill_probe.py <out> mps stab0 startlast0 stab startlast simsf sim`.
**Headline results** (best position = any rule end >= k or <bot>; max over layers; emergence = first layer >= .5):
- stab0: v_1 .58-.64 (emerges at layer 11), v_2 .39-.48 (never), v_3 .24-.28.
- stab: v_1 .65-.69 (layer 11), v_2 .52-.58 (layer 12), v_3 .27-.30.
- startlast0 v_1 <= .29, startlast v_1 <= .39; sim / simsf v_1 .44-.54 (layer 11 at L3 only); v_2 <= .13 in all four.
**Predictions:** stab0 v_1 by layer 6 missed (layer 11); stab0 v_2 emerges missed (never); v_3 never held; startlast0 v_1 by layer 12 missed; stab >= stab0 held; sim / simsf / startlast as t21c held.
**Interpretation:** Exact intermediate values appear only in the top two layers and at most one or two steps deep, yet stab0 answers L3 at .87 and t34 reads branch bits from the question pass of startlast / stab / relay at .8-.95. One pass computes more than exact-value probes show; single-pass reach should be measured by behaviour (CODI-0 accuracy, zero-hop share) or by coarser targets, not by exact-value probes.
**Gotchas hit:** none beyond the numpy warnings.
**Caveats:** linear 100-way probes; one seed each.
**Next:** reword the paper's reach argument (thread 1b) to rest on behavioural reach.
