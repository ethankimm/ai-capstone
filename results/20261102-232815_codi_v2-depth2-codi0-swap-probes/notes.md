## 2026-11-02 — RuleChain CODI-0 probes (reference for CODI's): v1 0.395 / v2 0.184 decodable at <bot>, answer 0.375 at the answer position (codi, run_id: 20261102-232815_codi_v2-depth2-codi0-swap-probes)

**Goal:** The no-latent reference for `20261102-232507_codi_v2-depth2-codi-swap-probes` (full story there): the same intermediate-value probes on CODI-0, at <bot> and the answer position. No swap/ablation (no latents).
**Mechanism / model:** `codi` with `num_latent 0` (CODI-0), `gpt2_depth2_codi0/checkpoint-29716` (epoch 38, behind round-1 run 20261102-060537).
**Data:** RuleChain (depth2) test, all 6000; probes trained on the same 500 train items per level as the CODI run.
**Hyperparams:** see manifest; identical probe settings to the CODI run.
**Command:** `scripts/probe_swap_codi_latents.py --num_latent 0 --inf_latent_iterations 0`, see `eval_command.txt`.
**Headline results:** teacher-forced accuracy 0.407. Best test probe accuracy over layers (majority baseline): v1 (0.015) <bot> 0.395, ans 0.101; v2 (0.023) 0.184 / 0.083; v3 (0.023) 0.104 / 0.068; v4-v6 <= 0.06; answer (0.020) <bot> 0.069, ans 0.375.
**Interpretation:** Without latents, the question pass's last position (<bot>, layer 12) already holds the first one or two intermediate values more decodably than any position of CODI (v1 0.395 vs <= 0.167). CODI's latents did not make the chain more readable; if anything CODI's representations of v1/v2 are less linearly accessible.
**Gotchas hit:** none.
**Caveats:** Probes overfit (train ~0.93-0.99 vs test): lower bounds, see the CODI run. Pooled over levels. One checkpoint, one seed.
**Next:** see the CODI run.
