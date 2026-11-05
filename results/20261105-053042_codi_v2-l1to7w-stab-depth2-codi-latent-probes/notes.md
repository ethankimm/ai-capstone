## 2026-11-04 — Stabilised L1-7w CODI latent probes: flat across sites, no step per latent (v_1 0.23-0.26, v_2 0.16, v_3 0.11); answer only at the readout (0.51) (codi, run_id: 20261105-053042_codi_v2-l1to7w-stab-depth2-codi-latent-probes)

**Goal:** Companion to the ablation 20261105-052233: do the latents of the stabilised CODI hold one step each?
**Mechanism / model:** `codi`, checkpoint-17820 of 20261105-044444, CPU fp32.
**Data:** probes fit on 14,000 train items, scored on 3,500 test items; majority baseline 0.02-0.03.
**Command:** `scripts/probe_latents_rulechain_codi.py` (`eval_command.txt`).
**Headline results:** Exact accuracy, h_0 / h_1..h_6 / h_ans. v_1 0.397 / 0.234-0.263 / 0.271. v_2 0.197 / 0.157-0.163 / 0.234. v_3 0.111 / 0.104-0.117 / 0.133. v_4 0.071 / 0.076-0.092 / 0.090. v_5-v_7 <= 0.096 everywhere. Answer 0.079 / 0.110-0.121 / 0.511.
**Interpretation:** No staircase across sites, as in the other RuleChain CODI models. Early values are most readable at <bot>, before any latent pass, and the answer appears only at the readout. Consistent with the ablation: the latents add little.
**Gotchas hit:** none.
**Caveats:** Linear probes on the last layer; one checkpoint, one seed.
**Next:** as the ablation record.
