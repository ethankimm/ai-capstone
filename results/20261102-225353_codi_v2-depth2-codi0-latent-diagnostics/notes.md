## 2026-11-02 — RuleChain CODI-0 diagnostics (reference for CODI's): train-test gap 0.21-0.43 at L4-L8, mid-layer distillation mismatch (codi, run_id: 20261102-225353_codi_v2-depth2-codi0-latent-diagnostics)

**Goal:** The no-latent side of the CODI latent diagnostics (`20261102-225018_codi_v2-depth2-codi-latent-diagnostics`, which has the full story): per-layer and per-level distillation loss and train vs test accuracy for CODI-0, so CODI's numbers have a matched reference.
**Mechanism / model:** `codi` with `num_latent 0` (CODI-0, round-1 run 6), `gpt2_depth2_codi0/checkpoint-29716` (epoch 38, the checkpoint behind 20261102-060537; downloaded copy in `rulechain-checkpoints/`, sha256-verified).
**Data:** RuleChain (depth2) test, all 6000 items; train check = 500 seeded-random items/level of `train.json` (same items as the CODI run).
**Hyperparams:** see manifest; same script and settings as the CODI run, `--num_latent 0 --inf_latent_iterations 0 --latent_counts 0`.
**Command:** `scripts/diagnose_codi_latents.py`, see `eval_command.txt`.
**Headline results:** `final_answer_accuracy=0.407` (teacher-forced; round 1 logged 0.405 generated). Per level: 1.000 / 0.988 / 0.964 / 0.764 / 0.498 / 0.242 / 0.130 / 0.082 / 0.052 / 0.058 / 0.056 / 0.050. Teacher 0.9965. Distillation loss 0.141 (CODI k=6: 0.088). Per layer: ~0.002-0.006 at layers 1-5, then 0.045 / 0.113 / 0.258 / 0.305 / 0.407 / 0.457 / 0.214 at layers 6-12. Train vs test: L4 0.978 / 0.764, L5 0.884 / 0.498, L6 0.676 / 0.242, L7 0.476 / 0.130, L8 0.368 / 0.082, L9-12 train 0.19-0.24 vs test 0.05-0.06.
**Interpretation:** CODI-0 matches the teacher's answer-position state in the lower layers and diverges from layer 6 up; the teacher reads the previous value from written steps, which a single pass has to compute instead. It memorises more than CODI: train-test gaps of 0.21-0.43 at L4-L8 (CODI 0.17-0.30). See the CODI run's notes for the joint reading.
**Gotchas hit:** see the CODI run (memory, CUDA-saved checkpoint).
**Caveats:** One checkpoint, one seed. Distillation loss is against CODI-0's own teacher, so comparisons with CODI's are approximate. Train sample 500/level (+-0.04).
**Next:** see the CODI run.
