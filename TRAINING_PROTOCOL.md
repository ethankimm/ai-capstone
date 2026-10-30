# RuleChain training protocol: Coconut, CODI and their references

Status: **agreed 2026-10-29.** This is how we train and evaluate the main latent models on RuleChain (`depth2`).
This file is the runbook. If you change a value, change it here.

Earlier plans (`c_thought 2 x max_latent_stage 3`) are superseded where they disagree with this file.

---

## 1. What we train

Every model uses GPT-2 small (`gpt2`, 124M) on the same data files. Five models in total:

| # | Run name | Code | What it is | Depends on |
|---|---|---|---|---|
| 1 | `gpt2_depth2_cot` | Coconut `run.py` | CoT model, used to initialize 2 and 3 | data |
| 2 | `gpt2_depth2_coconut_c1m6` | Coconut `run.py` | Coconut, 6 latents (c = 1, M = 6) | 1 passes gate G1 |
| 3 | `gpt2_depth2_no_thoughts_m6` | Coconut `run.py` | Coconut reference: same curriculum, no latent slots | 1 passes gate G1 |
| 4 | `gpt2_depth2_codi` | CODI `train.py` | CODI, 6 latents | data |
| 5 | `gpt2_depth2_codi0` | CODI `train.py` + CODI-0 patch | CODI reference: same losses, 0 latents | data, CODI-0 patch |

Runs 2 and 3 share the same CoT checkpoint. Runs 4 and 5 are independent of the Coconut stack and can start right away.
Use **one job per GPU** (sharing a GPU changes timings and has crashed runs before).

Seeds for the first round: Coconut-stack runs use seed 0, CODI runs use seed 11 (CODI's published seed).
Whether we run a second seed is decided after the first round's results are in.

## 2. Before anyone launches: code changes still needed

None of these are done yet. Each one blocks the runs listed next to it.

- [ ] **Coconut config** (blocks 2): in `configs/synthetic/gpt2_depth2_coconut_A.yaml` (or a new copy named `gpt2_depth2_coconut_c1m6.yaml`), set `c_thought: 1`, `max_latent_stage: 6`, `num_epochs: 10`, and update the header comment. Keep `pad_latent_to_max: True`, `epochs_per_stage: 1`, `resume: 1`, `uniform_prob: 0`, `reset_optimizer: True`, `save_only_improve: False`, lr 1e-4, weight decay 0.01.
- [ ] **no_thoughts config** (blocks 3): same three changes in `gpt2_depth2_no_thoughts.yaml`, so its schedule matches run 2 exactly.
- [ ] **`scripts/pod/coconut_v2.sh`** (blocks 2, 3): set `FINAL_STAGE_CKPTS="8 9 10"`, point the `coconut` job at the new config, give the new runs new names (otherwise `run.py` resumes old checkpoints), and change `select` to rank by mean valid accuracy over **all 12 levels** (it currently uses L1–5).
- [ ] **CODI launch script** `configs/synthetic/train_gpt2_depth_codi.sh` (blocks 4, 5): add `--fix_attn_mask True`, change `--save_total_limit 1` to `3` (we select among the last 3 epochs), and let `NUM_LATENT` be set from the environment (default 6).
- [ ] **CODI-0 patch** (blocks 5): upstream CODI skips the student branch entirely when `num_latent` is 0 (`src/model.py:331`). Patch it so that with 0 latents the student still runs: question + `<bot>`, then directly `<eot>` + "The answer is: N", with student CE and distillation as usual. Teacher CE stays. The projection is unused. Add it to `scripts/codi_synthetic.patch` (or a separate `codi_zero_latent.patch`), and check on a 16-item smoke run that the student loss is non-zero.

`eval_synthetic_coconut.py` already supports `--method no_thoughts` and reads the latent count from the config (c × M = 6). `eval_synthetic_codi.py` already passes `--inf_latent_iterations`, and 0 works (the loop is skipped).

## 3. Data

RuleChain v2, generator 2.0.0, seed 5980: 100,000 train items, 2,000 valid, 500 test items per level, levels k = 1–12.

```bash
uv run python -m latentreasoning.tasks.synthetic_v2 --task depth2 --out /workspace/synthetic_data_v2
python3 scripts/check_data_hashes.py /workspace/synthetic_data_v2 depth2
```

The hash check must pass (hashes in `configs/synthetic/v2_data_sha256.json`). Every launch script re-checks it. Do not regenerate with other arguments.

- **Selection** of checkpoints uses `depth2/valid.json`.
- **Test** (`depth2/test.json`) is touched once per model, for the selected checkpoint only.

## 4. Coconut stack (runs 1–3)

### 4.1 Setup

```bash
bash scripts/pod/coconut_v2.sh setup      # clones Coconut @ 27273cb, builds .venv_coconut, checks data hashes
```

The effective batch is always 128. The script sets micro-batch 16 × accumulation 8 by default; use `MICRO_BS=32` on 48 GB cards.

### 4.2 CoT model and gate G1

```bash
bash scripts/pod/coconut_v2.sh cot        # 3 epochs
bash scripts/pod/coconut_v2.sh gate       # G1 on valid.json, uncapped decoding
```

**G1:** CoT accuracy ≥ 0.97 at **every** level on valid. If it fails, resume to 5 epochs under the same name and re-check. If it fails a second time, stop and tell the team.

### 4.3 Timing test

```bash
bash scripts/pod/coconut_v2.sh timing     # stages 1-4 on 5k items: seconds/step and peak memory per stage
```

Stage time grows with the number of latents (each latent is a sequential forward pass). With M = 6 there are 6 curriculum stages plus 3 final epochs, so the old Coconut A time estimate is too low. **Project the full run from the per-stage timings before launching.**

### 4.4 What the Coconut run does

`resume 1` skips stage 0 (the CoT checkpoint stands in for it), so `num_epochs 10` runs epochs 1–9:

| Epoch | Stage | Training input | Checkpoint |
|---|---|---|---|
| 1–6 | 1–6 | First s steps replaced by s latents (one per rule); later steps stay written | `checkpoint_2` … `checkpoint_7` |
| 7–9 | final | All steps dropped; 6 latents for every item (`pad_latent_to_max`), then `### answer` | `checkpoint_8`, `_9`, `_10` |

The `no_thoughts` reference runs the identical schedule with no latent tokens: steps are dropped stage by stage, and the question goes straight to `### answer`.

### 4.5 Launch, select, test

```bash
bash scripts/pod/coconut_v2.sh coconut gpt2_depth2_coconut_c1m6 0
bash scripts/pod/coconut_v2.sh ref no_thoughts                 # separate GPU

# after training
bash scripts/pod/coconut_v2.sh select gpt2_depth2_coconut_c1m6 coconut
bash scripts/pod/coconut_v2.sh select gpt2_depth2_no_thoughts_m6 no_thoughts
bash scripts/pod/coconut_v2.sh test gpt2_depth2_coconut_c1m6 coconut <SELECTED_CKPT>
bash scripts/pod/coconut_v2.sh test gpt2_depth2_no_thoughts_m6 no_thoughts <SELECTED_CKPT>
```

- **Checkpoint rule:** the best of the 3 final-stage checkpoints by mean valid accuracy over all levels (ties go to the later one).
- **Under-trained flag:** if valid accuracy still rises between the last two final-stage epochs, label the run "under-trained" in `notes.md`.
- **Inference** always uses the trained count of 6 latents. Do not evaluate at other latent counts and report it as a smaller model.
- **Crash recovery:** relaunch under the **same** name (`run.py` resumes from the newest checkpoint). A new run needs a **new** name.

## 5. CODI stack (runs 4–5)

### 5.1 Setup

```bash
bash scripts/codi_setup.sh /workspace/codi                     # CODI @ 2c23146, own venv
cd /workspace/codi && git apply /workspace/ai-capstone/scripts/codi_synthetic.patch
# run 5 only: also apply the CODI-0 patch (section 2)
```

### 5.2 Settings

Everything else is CODI's published GPT-2 recipe, kept as in the launch script.

| Setting | Value |
|---|---|
| Latents | `--num_latent 6` at training, `--inf_latent_iterations 6` at test (run 5: 0 and 0) |
| Teacher CoT | `INCLUDE_LAST_COT=False`: the step that computes the answer is dropped, so the teacher applies the last rule itself |
| Attention mask | `--fix_attn_mask True` |
| Distillation | At the token after "The answer is:"; smooth L1 divided by teacher std (`--distill_loss_div_std True`); all loss weights 1 |
| Projection | `--use_prj True --prj_dim 768`, LayerNorm on |
| Optimization | LoRA r 128 / α 32, lr 3e-3, cosine with 3% warmup, weight decay 0.1, 40 epochs, bf16, 64 × 2 = 128 effective batch, max grad norm 2.0 |
| Checkpoints | Saved each epoch, last 3 kept (`--save_total_limit 3`) |

### 5.3 Launch

```bash
TASK=depth2 INCLUDE_LAST_COT=False EXP_NAME=gpt2_depth2_codi \
  SAVE_DIR=/workspace/codi_runs CODI_SYNTHETIC_DATA_DIR=/workspace/synthetic_data_v2 \
  bash configs/synthetic/train_gpt2_depth_codi.sh

# run 5, separate GPU, CODI-0 patch applied
TASK=depth2 INCLUDE_LAST_COT=False EXP_NAME=gpt2_depth2_codi0 NUM_LATENT=0 \
  SAVE_DIR=/workspace/codi_runs CODI_SYNTHETIC_DATA_DIR=/workspace/synthetic_data_v2 \
  bash configs/synthetic/train_gpt2_depth_codi.sh
```

Before the full run, do a smoke run with `EXTRA_ARGS="--exp_mode True --exp_data_num 16 --max_steps 2 --per_device_train_batch_size 4 --gradient_accumulation_steps 1"`. Also run a short timing test before the 40-epoch run.

### 5.4 Select and test

Evaluate each of the 3 kept checkpoints on `valid.json` with `eval_synthetic_codi.py --no_log True`. Pick the best by mean accuracy over all levels (ties go to the later one), then run one logged test eval on it (command template in the script's docstring).

**Evaluate CODI at batch size 1 (`--batch_size 1`).** CODI's inference loop runs the latent and answer passes with no attention mask, so in a padded batch they attend to the question's left padding. That padding varies by 1–8 tokens within a level. Batch 1 has no padding, which also makes inference match training with `fix_attn_mask True`.

For run 5 use `--num_latent 0 --inf_latent_iterations 0`.

## 6. What must be identical across all runs

| Matched | Not matched (each method keeps its published recipe) |
|---|---|
| Data files and hashes, test set, answer scoring | Optimizer, learning rate, schedule |
| 6 latents for both latent methods, at training and test | LoRA (CODI) vs. full fine-tuning (Coconut) |
| GPT-2 small, greedy decoding | Number of epochs |
| Checkpoint rule: best of last 3 on valid, mean over all levels | |
| d50 measured from each method's own reference (run 3 for Coconut, run 5 for CODI), not from the older no-CoT run | |

Cross-method statements always go through each method's own reference. A raw Coconut-vs-CODI accuracy gap is reported as description, not as a finding.

## 7. Logging and safety

- **Log every real run with `record-run`** (`RunRecord` → `notes.md` → `scripts/rebuild_index.py` → commit), including failed runs. Write the run name, commit, config, selected checkpoint and the under-trained flag in `notes.md`.
- **Copy checkpoints off the training machine as they appear** and checksum them: `bash scripts/pod/coconut_v2.sh manifest` covers all run roots, including Coconut's extensionless `checkpoint_N`.

## 8. Expected results (so nobody is surprised)

- Coconut should hold up through about k = 6 and then fall near k ≈ M + d50 − 1 ≈ 8, because rules beyond 6 never got a latent and must be done in one forward pass. That fall is a check that the setup works, not a finding about latent capacity.
- Latents beyond the rule count (j > k) have no rule to stand for. Necessity analyses use only latents j ≤ k and report j > k separately.
