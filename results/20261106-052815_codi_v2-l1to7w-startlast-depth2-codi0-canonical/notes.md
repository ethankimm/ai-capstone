## 2026-11-05 — RuleChain start value last, CODI-0 on the canonical L1-L12 test: 0.224, reach 2.89, secondary check (codi, run_id: 20261106-052815_codi_v2-l1to7w-startlast-depth2-codi0-canonical)

**Goal:** Secondary check on L8-L12, levels this model never trained on (TRAINING_PROTOCOL section 11.2).
**Mechanism / model:** `codi`, gpt2, num_latent 0, RunPod A40, 12 epochs, section 10.2 recipe (lr 2e-3 cosine, adam_beta2 0.98, adam_epsilon 1e-6, max_grad_norm 1.0, LoRA r128/a32, 64 x 2, bf16, seed 11).
**Data:** `synthetic_data_v2_startlast`: canonical L1-L12 test, 500/level, start value after the rules.
**Command:** `scripts/pod/codi_stab_arm.sh` via `scripts/pod/launch_stab_arm.sh` with the section 11.1 flags, `scripts/pod/pair_watcher.py` on the laptop; `eval_command.txt`.
**Headline results:** Test 0.224, d50 2.89, AUC 0.197.
**Interpretation:** Out of range for this model. Reported for continuity with the section 10 pair.
**Gotchas hit:** None.
**Caveats:** One seed. Budget-limited: the section 11.4 rule fired (valid mean L2-L7 rose 0.036 in CODI-0 and 0.076 in CODI from epoch 9 to 12), and the extension was deferred before any test record was opened. 
**Next:** None for this record.
