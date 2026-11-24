## 2026-11-14 — Crossing-aware monitor vs all latents, pooled probes (t44): reading only the crossed latents loses almost everywhere (codi, run_id: 20261114-081823_codi_crossing-aware-monitor-pooled)

**Goal:** t44: does a monitor reading only the latents an item's answer crosses beat one reading all latents (gold steps T1, own answer on wrong items T2, wrong flag T3)?
**Mechanism / model:** seven RuleChain models, four released GSM8K models.
**Data:** test items with measured crossings (RuleChain valid recipients of hop pairs; GSM8K test items in t31 / t36 pairs).
**Command:** `../codi/.venv/bin/python -W ignore scripts/necessity/monitor_select.py <model> mps codi_audit_artifacts/t44/<model>.json`.
**Headline results:** T1 CROSS - ALL: gated -.11, sim -.06, simsf -.09, startlast -.02, relay -.03, stab -.04, simcoconut_rc -.11, gsm8k codi -.03, sim -.03, coconut -.07, simcoconut -.05. CROSS - random prefix of the same size: +.03 to +.06 in chaining models. T2 up to -.33 (gsm8k_simcoconut; later latents hold the answer). T3 gsm8k_codi -.10. DEPTH monitor (no minimal pairs) ~ CROSS.
**Interpretation:** Biased against CROSS: each prefix probe was fit on items of every length, where a latent's role differs. Superseded by t44b.
**Gotchas hit:** GSM8K data must load before the adapters (offline mode); fixed after the smoke test.
**Caveats:** chaining models' measured-C items are all answered right, so T2 / T3 are empty there.
**Next:** t44b matched fit.
