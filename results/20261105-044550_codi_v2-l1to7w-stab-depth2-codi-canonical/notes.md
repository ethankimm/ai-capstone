## 2026-11-04 — RuleChain L1-7w stabilised CODI: canonical L1-L12 test, secondary out-of-range check; near floor beyond the trained levels (codi, run_id: 20261105-044550_codi_v2-l1to7w-stab-depth2-codi-canonical)

**Goal:** Secondary check from TRAINING_PROTOCOL 10.3: how far past its trained levels does the selected checkpoint reach?
**Mechanism / model:** `codi`, gpt2, num_latent 6, selected checkpoint-17820, RunPod A40.
**Data:** canonical depth2 test, 6,000 items, L1-L12. Levels 8-12 were never trained on.
**Command:** `eval_command.txt`.
**Headline results:** L1-7 as the test record; L8 0.078, L9 0.038, L10 0.046, L11 0.014, L12 0.016; overall 0.417.
**Interpretation:** Neither model extends beyond its training levels; L8-12 sit near floor. Not part of the decision rule.
**Gotchas hit:** none.
**Caveats:** One seed.
**Next:** none.
