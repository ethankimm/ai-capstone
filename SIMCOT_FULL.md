# SIM-CoT on the full round-1 RuleChain protocol (Coconut c1m6 + step supervision)

Status: **paused, resumable.** It was stopped by user decision at 08:13 PT on 2026-11-05 to free GPU 4.
- Stopped at final-stage epoch 1 of 3 (epoch 8/10, batch 843/3125).
- All six latent stages are done; `checkpoint_2` … `checkpoint_7` are on disk.
- Not evaluated yet: no selection, no test, so no run record (same precedent as the stopped round-1 seed-2 runs in
  `RULECHAIN_RESULTS.md`).

It ran 2026-11-04 19:30 PT to 2026-11-05 08:13 PT (12.7 h) on cci-p124, GPU 4. Branch `kevin/sim-cot`.

## Why

The k ≤ 8 pilot (`SIMCOT_PILOT.md`) was positive:
- SIM-CoT c1m6 beats `pilot_k8_no_thoughts_m6` by +0.731 at L4-6.
- It collapses exactly at k = 7, the first rule without its own latent.

The pilot used reduced data (25k items, k 1-8) and a longer schedule (3 epochs per stage). This run puts the same method
on the round-1 protocol (`TRAINING_PROTOCOL.md`), for two reasons:
- Every round-1 reference is already logged, so the comparison is fully matched with no new reference runs.
- k goes up to 12, so the "falls at k = M + 1" prediction can be tested across L7-12, not just L7-8.

## Design

Only the SIM-CoT settings are added to round-1 Coconut c1m6. Config: `configs/synthetic/gpt2_depth2_simcot_c1m6.yaml`
(= `gpt2_depth2_coconut_c1m6.yaml` + the SIM-CoT keys from `pilot_k8_simcot_c1m6.yaml`).

| | SIM-CoT pilot | **This run** |
|---|---|---|
| Train data | 25k items, k = 1-8 | **100k items, k = 1-12** (`depth2/train.json`) |
| Schedule | 3 epochs per stage, 5 final (resume 3, 26 epochs) | **1 epoch per stage, 3 final (resume 1, 10 epochs)** |
| Selection | best of checkpoint_24/25/26 on `depth2_k8/valid.json` | **best of checkpoint_8/9/10 on `depth2/valid.json`** (2,000 items, mean over levels) |
| Test | `depth2_k8/test.json`, L1-8 (4,000 items) | **`depth2/test.json`, L1-12** (6,000 items; L1-8 are the pilot's items) |
| Latents / step loss / code | c1 × M6; SIM-CoT fork @ `cd3291d` + `scripts/simcot_explain_loss.patch` | unchanged |
| Init / lr / batch / seed | CoT `gpt2_depth2_cot/checkpoint_3`; 1e-4, wd 0.01; 32 × 4 = 128, fp32; seed 0 | unchanged |

Latent-to-step mapping (SIM-CoT's code):
- Stage s: latent j decodes step j.
- Final stage: latents 1-5 decode steps 1-5, and latent 6 decodes steps 6..k joined (here up to 7 steps).
- For items with k < 6, the spare latents decode `### N`.

All 100k train items parse into exactly one `<<…>>` group per step. The fork's target builder has no length cap.

**Matched references (already logged, no new runs):**
- round-1 `gpt2_depth2_no_thoughts_m6`: seed 0 `20261101-153922`, seed 1 `20261102-190130`
- round-1 plain `gpt2_depth2_coconut_c1m6`: `20261102-022150`
- round-1 `gpt2_depth2_codi0`: `20261102-060537`, the best no-latent model

**Decision rule**, the same as the pilot's:
- **Positive:** above `no_thoughts_m6` by exact McNemar pooled over L4-6.
- **Effect of the step loss:** compare against plain c1m6.
- **Expected fall:** accuracy stays high to k = 6, then collapses at k = 7 and stays down through L12.

## Progress at the stop

Each epoch took longer than the last as stages added latents and longer step targets:

| Epoch (stage) | 2 (1) | 3 (2) | 4 (3) | 5 (4) | 6 (5) | 7 (6) | 8 (final 1, stopped at 27%) |
|---|---|---|---|---|---|---|---|
| Duration | 44 min | 79 min | 109 min | 136 min | 160 min | 186 min | ~3 h projected |
| Checkpoint written (PT) | ckpt_2 20:14 | ckpt_3 21:33 | ckpt_4 23:22 | ckpt_5 01:38 | ckpt_6 04:18 | ckpt_7 07:24 | — |

**Step loss** (summed over step tokens, as in SIM-CoT):
- It fell from 68.6 at the first batch to about 0.13 by the end of stage 1.
- It jumped to 37.5 when the final stage started (latent 6's new target, steps 6..k joined), then fell back to about
  1.5 within 600 batches.

**Monitor accuracy** (360 `valid_monitor` items, k 1-12, each stage scored in its own format):

| | St. 1 | St. 2 | St. 3 | St. 4 | St. 5 | St. 6 |
|---|---|---|---|---|---|---|
| **SIM-CoT, full** | 0.144 | 0.122 | 0.158 | 0.183 | 0.247 | 0.208 |
| round-1 `no_thoughts_m6` | 1.000 | 0.297 | 0.194 | 0.164 | 0.147 | 0.142 |
| round-1 plain c1m6 | 0.978 | 0.294 | 0.164 | 0.161 | 0.133 | 0.131 |

**Reading (provisional, not a result):**
- From stage 4 on, the monitor is slightly above both round-1 models.
- It is far below the pilot, which reached 0.717 on k ≤ 8 by the end of stage 6.
- A perfect one-rule-per-latent model would score about 0.50 on this monitor: L1-6 right, L7-12 wrong.
- The likely cause is less training per stage: the pilot's monitor rose only in the third epoch of stage 1
  (0.05 → 0.11 → 0.56).
- The final stage's 3 epochs may recover part of the gap; the pilot's final stage started at 0.60.
- Whatever the outcome, data size and schedule change together, so a weak result cannot say which one caused it.

## How to resume

**Resume is lossless apart from data order.**
- `run.py` resumes from the newest checkpoint when relaunched under the same name: it ignores the config's `resume`,
  loads `checkpoint_7`, and starts at epoch 7, the first final-stage epoch.
- The optimizer is reset at every epoch anyway (`reset_optimizer: True`, `run.py:347`), so no optimizer state is lost.
- `checkpoint_*` files hold the base model and the auxiliary decoder (1.46 GB each).
- Only the partial epoch 8 (843 of 3,125 batches) is redone.

Steps (get GPU approval first; the queue takes the GPU index, default 4):

```bash
cd /local-ssd/ktn48/lrm_work
nvidia-smi -i 4 --query-gpu=memory.used,utilization.gpu --format=csv,noheader   # confirm the GPU is free
setsid nohup bash queue_simcot_full_gpu4.sh 4 > logs/queue_simcot_full_gpu4.out 2>&1 < /dev/null &
# expect in logs/gpt2_depth2_simcot_c1m6.log:
#   "NOTE: ... already has checkpoints -> run.py will RESUME" and "Loading from previous run epoch_7!"
```

**What the queue does** (`lrm_work/queue_simcot_full_gpu4.sh`, not in git):
1. Sources `lrm_work/pilot_k8_lib.sh` and overrides `VALID_JSON` / `TEST_JSON` to `depth2/valid.json` and
   `depth2/test.json`.
2. Sets `COCONUT_DIR=lrm_work/simcot/Coconut`.
3. Runs `run_pilot coconut gpt2_depth2_simcot_c1m6 coconut configs/synthetic/gpt2_depth2_simcot_c1m6.yaml 32 "8 9 10"`.
   - Training: `scripts/pod/coconut_v2.sh coconut`.
   - Selection: best of checkpoint_8/9/10 on `depth2/valid.json`.
   - Then one logged test eval (`RunRecord` written by `scripts/eval_synthetic_coconut.py`).
4. Logs to `lrm_work/logs/queue_simcot_full.log` (PT timestamps).

**Gotchas:**
- The stopped run's log was moved to `logs/gpt2_depth2_simcot_c1m6.stopped-ep8.log`. It holds stages 1-6 and the
  monitor lines above.
  - It ends in the SIGTERM traceback, and `finish_pilot` refuses to select when the log contains `Traceback`.
  - The relaunch therefore writes a fresh `gpt2_depth2_simcot_c1m6.log`. Don't move the old one back.
- Stop a resumed run by killing the queue's process group first, then the training process group in
  `logs/gpt2_depth2_simcot_c1m6.pid`. Otherwise `finish_pilot` sees the exit and starts selection on missing
  checkpoints.

**ETA after resuming:**
- Training: 3 final epochs at about 3 h each = about 9 h.
- Selection and test: about 30 min (3 × 2,000 valid items, then 6,000 test items).
- Total: about 9.5 h from launch.

**When it finishes:**
1. Fill in the run record's `notes.md` (record-run skill).
2. Run the paired tests against the references above. `scripts/rulechain_gain_analysis.py` already handles the round-1
   models on `depth2/test.json`, but needs an entry for this run.
3. `scripts/rebuild_index.py`, then commit.

## If it comes out weak at L4-6

The clean follow-up is the full data on the pilot's schedule:
- 3 epochs per stage, 5 final epochs: about 42 h.
- Plus a matched `no_thoughts_m6` on the same schedule (about 6-7 h, on another GPU).

That separates "more data" from "less training per stage". CODI + SIM-CoT is on hold (user decision 2026-11-05). The
fork's `CODI/` would first need our CODI patches ported (synthetic loader, `fix_attn_mask` + `pad_invariant`,
zero-loss logging) and a check that its step loss is actually computed.

## Where things are

| What | Path |
|---|---|
| Config (git) | `configs/synthetic/gpt2_depth2_simcot_c1m6.yaml` |
| Rendered config | `lrm_work/logs/configs/gpt2_depth2_simcot_c1m6.yaml` |
| Checkpoints | `lrm_work/coconut_runs/gpt2_depth2_simcot_c1m6/checkpoint_2` … `_7` |
| Training log (stopped) | `lrm_work/logs/gpt2_depth2_simcot_c1m6.stopped-ep8.log` |
| Queue / queue log | `lrm_work/queue_simcot_full_gpu4.sh` / `lrm_work/logs/queue_simcot_full.log` |
| SIM-CoT code | `lrm_work/simcot/Coconut` (InternLM/SIM-CoT @ `cd3291d`, `scripts/simcot_explain_loss.patch` applied) |
