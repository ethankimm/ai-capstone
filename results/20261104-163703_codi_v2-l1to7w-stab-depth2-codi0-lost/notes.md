## 2026-11-04 — RuleChain L1-7w stabilised CODI0: LOST, pod terminated when the RunPod balance ran out; no checkpoint, log or eval survives (codi, run_id: 20261104-163703_codi_v2-l1to7w-stab-depth2-codi0-lost)

**Goal:** Matched CODI / CODI-0 pair under TRAINING_PROTOCOL section 10 (lr 2e-3, beta2 0.98, eps 1e-6, clip 1.0, 12 epochs).
**Mechanism / model:** `codi`, gpt2, num_latent 0, seed 11, RunPod L40S 48GB.
**Data:** `synthetic_data_v2_l1to7w` (190k train).
**Command:** `NAME=gpt2_l1to7w_stab EPOCHS=12 TRAIN_EXTRA="--learning_rate 2e-3 --adam_beta2 0.98 --adam_epsilon 1e-6 --max_grad_norm 1.0" bash scripts/pod/codi_l4to7.sh train codi0`
**Headline results:** None. Last observation 2026-11-04 07:45 UTC: step 4630, epoch 3.11, loss 0.7655, largest gradient norm after warmup 1.63, 3 readings above 1, watchdog quiet. Nothing is known after that time.
**Interpretation:** The only surviving signal is that the new flags looked stable early; CODI-0 passed the steps where the lr-3e-3 run blew up (4,090 and 4,590). That is one seed and a partial run, and no accuracy was ever measured, so it says nothing about the gain question.
**Gotchas hit:** RunPod's audit log shows actor `runpod` deleting both pods at 14:28:20 UTC; balance afterwards -$0.07. The pods used container disk only, so all data was lost. Causes on our side: the balance was never checked before launch; no sync ran during training although protocol section 9 requires one; nothing was armed to pull results or run valid evals when training ended. Spend on these pods (including two A40 pods that never got a machine): $15.79. Estimated, not observed: CODI-0 finished ~12:00 UTC and sat idle ~2.5 h (~$2.7); CODI was near step 13,500 (epoch ~9) when terminated.
**Caveats:** Times after 07:45 UTC are estimates from the observed step rate.
**Next:** Rerun both arms after topping up, with the safeguards in TRAINING_PROTOCOL section 10.5.
