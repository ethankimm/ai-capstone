## 2026-11-04 — Round-1 RuleChain CODI (40 epochs) latent probes: flat across sites, weaker than L1-7w (v_1 0.22, v_2 0.14, v_3 0.08, v_4+ <= 0.05); answer only at the readout (0.37) (codi, run_id: 20261104-070914_codi_v2-depth2-codi-r1-latent-probes)

**Goal:** Companion to the round-1 ablation (20261104-064739): does the 40-epoch RuleChain CODI hold a step per latent, like GSM8K's v_1 at z_1 / v_2 at z_3, or one shared state like L1-7w (20261104-034628)?
**Mechanism / model:** `codi`, round-1 `gpt2_depth2_codi_bs64/checkpoint-31280`, CPU fp32.
**Data:** probes fit on 2,000 canonical train items per level (24,000), scored on canonical test (6,000, L1-L12).
**Command:** `scripts/probe_latents_rulechain_codi.py` (`eval_command.txt`).
**Headline results:** exact accuracy (two digit heads), sites h_0 (<bot>) .. h_6, h_ans; majority 0.02-0.03.
v_1: 0.220 / 0.214-0.216 at h_1..h_6 / 0.153 at h_ans. v_2: 0.129 / 0.137-0.142 / 0.133. v_3: 0.066 / 0.074-0.083 / 0.107. v_4: 0.039 / 0.045-0.048 / 0.082. v_5+: at or near majority everywhere. Answer: 0.065 / 0.091-0.095 / 0.371. LM-head decode of any site is almost never a number (<= 5%); embedding decode is a number 80% of the time but equals v_j in <= 10%.
**Interpretation:** Same shape as L1-7w: every latent holds the same weak early-rule summary, nothing moves forward site by site, and the answer exists only at the readout position. Weaker than L1-7w (v_1 0.22 vs 0.39), consistent with less training on L4-7. Neither schedule nor data mix makes CODI's latents carry a step each on RuleChain.
**Gotchas hit:** none.
**Caveats:** Linear probes on the last layer (what z_s carries forward); one checkpoint, one seed.
**Next:** as the ablation record.
