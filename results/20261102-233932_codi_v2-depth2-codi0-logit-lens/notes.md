## 2026-11-02 — RuleChain CODI-0 logit lens (reference for CODI's): answer readable only at the answer position's last layer (0.407), <bot> decodes to 'The' (codi, run_id: 20261102-233932_codi_v2-depth2-codi0-logit-lens)

**Goal:** No-latent reference for `20261102-233815_codi_v2-depth2-codi-logit-lens` (full story there).
**Mechanism / model:** `codi` with `num_latent 0` (CODI-0), `gpt2_depth2_codi0/checkpoint-29716` (epoch 38, round-1 run 20261102-060537).
**Data:** RuleChain (depth2) test, all 6000.
**Hyperparams:** identical lens to the CODI run.
**Command:** `scripts/logit_lens_codi_latents.py --num_latent 0 --inf_latent_iterations 0`, see `eval_command.txt`.
**Headline results:** answer position, layer 12: answer top-1 0.407 (= model accuracy), v1-v3 0.10-0.11, any intermediate 0.095; layers 0-10 no numbers. <bot>: top-1 'The' for all 6000 items at layer 12, no numbers at any layer.
**Interpretation:** Same picture as CODI at the answer position. The intermediate values that linear probes find at CODI-0's <bot> (v1 0.395, `20261102-232815`) are not in the vocabulary basis: they are linearly present but not "written as tokens".
**Gotchas hit:** none.
**Caveats:** Logit lens only sees unembedding-aligned directions. One checkpoint, one seed.
**Next:** see the CODI run.
