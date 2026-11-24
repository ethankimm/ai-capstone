## 2026-11-14 — Crossing-aware monitor, matched fit (t44b): crossed latents hold all the gold-step signal in chaining models; reading fewer latents never monitors better (codi, run_id: 20261114-081823_codi_crossing-aware-monitor-matched-fit)

**Goal:** t44b: the fair version of t44. Probes fit only on train items with the same crossing count (RuleChain: level-mean hops; GSM8K: mean hops at the item's serial depth); CROSS_m and ALL_m use the same training items.
**Mechanism / model:** seven RuleChain models, four GSM8K models.
**Data:** all eval items (RuleChain valid + test L2-L7; GSM8K test 1,319).
**Command:** `../codi/.venv/bin/python -W ignore scripts/necessity/monitor_match.py <model> mps codi_audit_artifacts/t44b/<model>.json`.
**Headline results:** T1 CROSS_m - ALL_m: gated +.001, sim .000, simsf -.009, simcoconut_rc -.016; startlast -.002, relay -.020, stab -.039; gsm8k codi -.032, sim -.032, coconut -.065, simcoconut -.050. T2: simsf -.08, gsm8k -.03 to -.28. T3: gsm8k codi -.05, sim -.09; coconut / simcoconut +.02 / +.03 where ALL_m is at chance (.50).
**Interpretation:** In chaining models the crossed latents carry all the linearly readable step information once probes are fit per crossing group (matches t38c). Restricting a monitor to them never improves it; on GSM8K, and for predicting the model's own answer, the latents beyond the crossings add signal.
**Gotchas hit:** gsm8k_simcoconut depth groups put 1 train item in c = 1 (nearest-group fallback used).
**Caveats:** linear probes; group crossings, not per-item.
**Next:** none planned; write up.
