# RuleChain training protocol: Coconut, CODI and their references

Status: **agreed 2026-10-30** (first version 2026-10-29). This is how we train and evaluate the main latent models on
RuleChain. It is the runbook: if you change a value, change it here. Earlier plans
(`c_thought 2 x max_latent_stage 3`) are superseded where they disagree with this file.

**Naming.** The paper calls the task **RuleChain**. In code it keeps the task ID `depth2` (generator 2.0.0): the data
folder `synthetic_data_v2/depth2/`, the keys in `configs/synthetic/v2_data_sha256.json`, run names (`gpt2_depth2_*`), and
CODI's `--data_name depth2`. Renaming it would change data paths and hashes and break the link to logged runs, so
`depth2` stays. When writing, RuleChain = `depth2`.

---

## 1. Key parameter decisions

The decisions that shape every result. Section 6 and 7 list the remaining settings.

| Decision | Value | Why |
|---|---|---|
| Backbone | GPT-2 small (124M), pretrained, for every model | Shared backbone; d50 ≈ 3 for GPT-2 small no-CoT, so the task spans below and above single-pass reach |
| Latent budget | **6 latents** for Coconut and CODI, at training and test, for every k | Same total budget for both methods; fixed per item, as in every published Coconut/CODI configuration |
| Coconut latents per rule (`c_thought`) | **1** | One forward pass already handles about 3 rules, so one latent per rule is enough, and latent j can be probed for step s_j (Coconut also uses 1 on ProsQA/ProntoQA) |
| Coconut rules replaced (`max_latent_stage`, M) | **6** | Covers single-pass reach with room above it; rules 7–12 never get a latent, so accuracy should fall near k ≈ M + d50 (a check of the setup, not a finding) |
| `pad_latent_to_max` | **True** | Fixed budget (6 latents even when k < 6); a null result below d50 is then stronger, since unused latent compute was available |
| Coconut schedule | 1 epoch per stage, 3 final-stage epochs, start from the CoT checkpoint | About as many items per stage and final-stage items as Coconut's ProsQA schedule |
| CODI teacher CoT (`include_last_cot`) | **False** | Keeping the answer-computing step lets the teacher copy the answer into the distillation position (CODI's ablation); dropping it makes the teacher apply the last rule itself, as in Liang & Pan |
| CODI padding | **`fix_attn_mask True` + `pad_invariant True`** (our patch) | Upstream's latent passes attend to left padding (up to ~160 tokens when k = 1 and k = 12 share a batch) and get shifted positions; that varies with k, our analysis axis. Liang & Pan mask the latent passes the same way; we also fix positions so batch size cannot change results (tested) |
| CODI recipe | Published GPT-2 recipe: LoRA r 128/α 32, lr 3e-3, cosine, 40 epochs | Each method keeps its published recipe; comparisons go through each method's own reference |
| Matched references | `no_thoughts` (Coconut), CODI-0 (CODI) | Same code, data and schedule as the latent model, minus the latents |
| Single-pass reach (d50) | **Envelope**: per level, the best no-latent model (section 2) | A weakly trained reference would put d50 too low and inflate latent gains below it |
| Primary ablation | **Resample swap within level**; mean (train split), zero and truncation secondary | The latent models never trained without latents; a swap keeps inputs in-distribution and shows direction (follows donor vs keeps own answer) |
| Checkpoint rule | Best of the last 3 checkpoints on `valid.json` by mean accuracy over all 12 levels; test touched once | One rule for every model |
| Answer format | Each method's own: Coconut `### N`, CODI `The answer is: N` | Changing either edits the method (CODI distills at the token after "The answer is:"); both are a marker before one answer token |
| Seeds (round 1) | Coconut stack 0, CODI 11 (CODI's published seed); one seed per model | Supports existence claims only; any claim about where onset happens needs a second seed of the four latent/reference runs |

## 2. Reference models

All references start from pretrained GPT-2 small (or from the CoT model trained from it) and see the same 100,000
training items.

| Reference | How it is trained | Role | What we expect to show |
|---|---|---|---|
| **`no_thoughts`** (Coconut stack) | Coconut `run.py` with `no_thoughts: True`: starts from the CoT checkpoint, runs Coconut's exact curriculum (stage s drops the first s steps with nothing in their place), then 3 final epochs of question → `### N`. Same schedule, seed and checkpoint rule as the Coconut run | **Matched reference for Coconut.** Gain G(k) = Coconut − `no_thoughts` | Close to the envelope at small k. Coconut's gain should be near zero below d50 and positive above it. Large-k items get only the 3 final epochs, so it may sit below the envelope there |
| **Plain no-CoT, to plateau** (Coconut stack) | Coconut `run.py` with `no_cot: True`: pretrained GPT-2, question → `### N` from epoch 1, no curriculum, full fine-tune, lr 1e-4. Stopped when monitor accuracy gains < 0.5 pt over 3 epochs (cap 60). Keeps the last 3 checkpoints and `checkpoint_12` | **Single-pass reach** (feeds the envelope). `checkpoint_12` = **data-matched** point: 12 passes over the train set, as many as CoT (3) + Coconut (9) | The best direct-answer accuracy GPT-2 small reaches here; d50 near or a little above 3. The gap between `checkpoint_12` and the plateau checkpoint shows how much extra training moves d50 |
| **CODI-0** (CODI stack) | CODI `train.py` with `num_latent 0` (our patch): student goes from question + `<bot>` straight to `<eot>` "The answer is: N". Keeps teacher CE (writes the CoT) and distillation toward the teacher's hidden state. Same LoRA recipe, 40 epochs | **Matched reference for CODI.** Gain G(k) = CODI − CODI-0. Also the intermediate-supervision control: it gets the teacher's step supervision but no inference-time steps | Close to student-only and the envelope. If it falls clearly below them, distillation toward an unreachable target is hurting it, and CODI's gain is also reported against student-only and the envelope |
| **Student-only CODI** (CODI stack) | CODI-0 with `ref_loss_factor 0` and `distill_loss_factor 0`: only question → "The answer is: N" CE. A plain no-CoT in CODI's code, same LoRA recipe. Existing flags, no new code | **Health check for CODI-0**, and an envelope member (LoRA no-CoT) | ≈ CODI-0. A clear gap means CODI-0's extra losses help or hurt direct answering. First run to drop if budget is short |
| **Envelope** (derived, not trained) | Per level, the best valid accuracy among plain no-CoT (plateau), `no_thoughts`, CODI-0 and student-only | **Defines d50** for the onset prediction | Reach is at least what any no-latent model of this size shows; this is the conservative threshold |

Older runs (25-epoch `train_synthetic.py` no-CoT, d50 ≈ 3.0) are context only, not references.

**Health checks (pass before reading gains):** each matched reference is within a few points of the envelope at every
level. A reference far below it is flagged as handicapped, and its method's gain is also reported against the envelope.

## 3. Analysis rules fixed before training

- **d50**: from a logistic fit of accuracy against k with a bootstrap interval over items, for every reference and for the
  envelope. The onset prediction uses the envelope's d50; the others are sensitivity checks.
- **Gain**: report G(k) and the normalized gain G(k) / (1 − acc_ref(k)), so the ceiling at k = 1 (reference ≈ 0.96) does
  not decide the result.
- **Necessity**: primary = resample swap. Replace the latents with those of another item at the same level (excluding
  donors with the same answer), scored follows-donor / keeps-own / other. Secondary = mean ablation with the mean taken
  over the train split, zero ablation, truncation; claim only when they agree in direction. The prediction is a rise in
  the swap effect from about 0 below the envelope d50 to clearly positive above it. Ablated accuracy is also compared with
  the reference: below the reference means dependence, not computation.
- **Latents beyond the rule count** (j > k at small k) have no rule to stand for; per-position analyses use j ≤ k and
  report j > k separately.
- **Coconut's expected fall**: Coconut's own d50 should land in about [8, 10] (M = 6 latents plus about 3 rules in the
  final pass). A miss suggests the latents are not one per rule.
- **Scoring**: same answer comparison (`latentreasoning.eval.metrics.is_correct`) for all, with each method's own
  extraction: Coconut takes the text after the last `###` (none = unparseable), CODI takes the last number. Report the
  unparseable rate for every model.

## 4. Runs

| # | Run name | Code | What it is | Depends on |
|---|---|---|---|---|
| 1 | `gpt2_depth2_cot` | Coconut | CoT model, initializes 2 and 3 (retrained, section 6.2) | data |
| 2 | `gpt2_depth2_coconut_c1m6` | Coconut | Coconut, 6 latents | 1 passes G1 |
| 3 | `gpt2_depth2_no_thoughts_m6` | Coconut | Matched reference for Coconut | 1 passes G1 |
| 4 | `gpt2_depth2_no_cot_plateau` | Coconut | Plain no-CoT to plateau | data |
| 5 | `gpt2_depth2_codi` | CODI | CODI, 6 latents | data, padding check |
| 6 | `gpt2_depth2_codi0` | CODI | Matched reference for CODI | data, padding check |
| 7 | `gpt2_depth2_codi_student` | CODI | Student-only health check | data, padding check |

Use **one job per GPU** (sharing a GPU changes timings and has crashed runs before). Runs 4–7 can start immediately; the Coconut chain (1 → timing → 2) is the critical path.

## 5. Code changes for this protocol (2026-10-29/30)

- **Configs**: `gpt2_depth2_coconut_c1m6.yaml`, `gpt2_depth2_no_thoughts_m6.yaml`, `gpt2_depth2_no_cot_plateau.yaml`
  (`configs/synthetic/`). The older `coconut_A`, `no_thoughts`, `no_cot` configs stay for provenance.
- **`scripts/pod/coconut_v2.sh`**: new configs and run names; final-stage checkpoints 8/9/10; `select` ranks by mean
  valid accuracy over all levels and verifies batched decoding for no-latent models; hash check covers `TASKS` (default
  depth2 only); `COT_EPOCHS` override; `timing` runs stages 1–6 plus one final epoch; `ref no_cot` trains to plateau
  with the watcher and keeps `checkpoint_12`; the manifest now includes CODI optimizer/scheduler state.
- **`scripts/coconut_timing.py`**: projects any M (`--max-latent-stage`), times every stage, warns when one is missing.
- **`scripts/codi_model_fixes.patch`** (replaces the earlier `codi_zero_latent.patch`; apply after
  `codi_streaming.patch` and `codi_synthetic.patch`): (a) `num_latent 0` trains the student (upstream skipped the
  student branch and crashed); (b) `--pad_invariant`: latent passes get the running attention mask and all student passes
  get mask-derived position ids. With 6 latents and `pad_invariant` off, the computation equals upstream.
- **`configs/synthetic/train_gpt2_depth_codi.sh`**: `--fix_attn_mask True --pad_invariant True --save_total_limit 3`;
  `NUM_LATENT`, `REF_LOSS_FACTOR`, `DISTILL_LOSS_FACTOR` from the environment; data hash check; refuses a smoke run
  without a `_smoke` name; checks the patch is applied; per-run logging dir.
- **`scripts/eval_codi.py`**: shared `tokenize_questions` / `encode_and_think` / `PadInvariant` helpers; batch-size-1
  crash fixed (`argmax(...).squeeze(-1)`); device follows the model; `pad_invariant`-aware decoding.
- **`scripts/eval_synthetic_codi.py`**: fails if eval flags differ from the checkpoint's `training_args.bin` or weights
  are missing; adds teacher-forced student accuracy at the distillation position, teacher accuracy per level, accuracy
  by answer value; records the data sha256 and the split.
- **`scripts/check_codi_padding.py`** (new): the padding test (section 7.2).
- **`scripts/pod/pull_artifacts.sh`, `sync_loop.sh`**: optimizer state pulled; one local folder per machine; retries with
  an alert after 3 failures; crash/finish alerts from the training logs (`WATCH_LOGS`).

## 6. Coconut stack (runs 1–4)

### 6.1 Data and setup

```bash
uv run python -m latentreasoning.tasks.synthetic_v2 --task depth2 --out /workspace/synthetic_data_v2
python3 scripts/check_data_hashes.py /workspace/synthetic_data_v2 depth2      # must pass; do not regenerate with other args
bash scripts/pod/coconut_v2.sh setup      # Coconut @ 27273cb, .venv_coconut, hash check
```

RuleChain v2, generator 2.0.0, seed 5980: 100,000 train, 2,000 valid, 500 test per level, k = 1–12. Selection uses
`depth2/valid.json`; `depth2/test.json` is evaluated once per model, for the selected checkpoint. Effective batch is
always 128 (micro-batch 16 × accumulation 8; `MICRO_BS=32` on 48 GB cards, though it will not speed up Coconut).

### 6.2 CoT model and gate G1

**The CoT model is retrained from scratch for this protocol.** The CoT model from 2026-10-15 (run record
`results/20261015-065640_explicit_cot_v2-depth2-coconutstack-cot`) used the same config and passed G1 (all levels 1.000
except L11 0.994). It shows what to expect; do not reuse its checkpoint. Start from an empty
`/workspace/coconut_runs/gpt2_depth2_cot/`: `run.py` resumes from any checkpoint it finds there.

```bash
bash scripts/pod/coconut_v2.sh cot        # COT_EPOCHS=3
bash scripts/pod/coconut_v2.sh gate       # G1 on valid.json, uncapped decoding
```

**G1:** CoT accuracy ≥ 0.97 at every level on valid. On failure: `COT_EPOCHS=5 bash scripts/pod/coconut_v2.sh cot` under
the same name (resumes), re-gate; a second failure stops Coconut.

### 6.3 Timing test

```bash
bash scripts/pod/coconut_v2.sh timing     # stages 1-6 + one final epoch on 5k items; s/step and peak memory per stage
```

Run it on the GPU that will run Coconut. The projection excludes per-epoch validation (add about 10%).

### 6.4 What the Coconut run does

`resume 1` skips stage 0, so `num_epochs 10` runs epochs 1–9:

| Epoch | Stage | Training input | Checkpoint |
|---|---|---|---|
| 1–6 | 1–6 | First s steps replaced by s latents (one per rule); later steps written | `checkpoint_2` … `_7` |
| 7–9 | final | All steps dropped; 6 latents for every item, then `### N` | `checkpoint_8`, `_9`, `_10` |

### 6.5 Launch, select, test

```bash
bash scripts/pod/coconut_v2.sh coconut gpt2_depth2_coconut_c1m6 0
bash scripts/pod/coconut_v2.sh ref no_thoughts                 # separate GPU
bash scripts/pod/coconut_v2.sh ref no_cot                      # separate GPU; plateau watcher starts with it

bash scripts/pod/coconut_v2.sh select gpt2_depth2_coconut_c1m6 coconut
bash scripts/pod/coconut_v2.sh select gpt2_depth2_no_thoughts_m6 no_thoughts
bash scripts/pod/coconut_v2.sh select gpt2_depth2_no_cot_plateau no_cot     # last 3 checkpoints of the plateau run
bash scripts/pod/coconut_v2.sh test <NAME> <METHOD> <SELECTED_CKPT>
bash scripts/pod/coconut_v2.sh test gpt2_depth2_no_cot_plateau no_cot <...>/checkpoint_12 v2-depth2-no-cot-datamatched
```

- **Under-trained flag:** valid accuracy still rising between the last two final-stage epochs → label "under-trained".
- **Inference** uses the trained 6 latents only.
- **Never attach `coconut_watch.py` to runs 2 or 3**: its patience rule would stop them at a curriculum stage switch. It
  is only for run 4.
- **Crash recovery:** relaunch under the same name (`run.py` resumes from the newest checkpoint, losing up to one epoch).
  A new run needs a new name.

## 7. CODI stack (runs 5–7)

### 7.1 Setup

```bash
bash scripts/codi_setup.sh /workspace/codi           # CODI @ 2c23146, own venv, applies codi_streaming.patch
cd /workspace/codi
git apply /workspace/ai-capstone/scripts/codi_synthetic.patch
git apply /workspace/ai-capstone/scripts/codi_model_fixes.patch
```

### 7.2 Padding check (required, CPU is fine)

```bash
cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/check_codi_padding.py \
    --data_json /workspace/synthetic_data_v2/depth2/valid.json
```

Must print `RESULT: OK`. It compares items of levels 1/4/8/12 alone vs in one padded batch: training forward, eval
logits at the distillation position, greedy generation, and training vs eval. On 2026-10-30 (CPU, random LoRA) the
`pad_invariant` path matched to ≤ 2e-4, while upstream's path differed by about 100 in logits and changed 3 of 4
generations.

**Rule for every CODI script on depth2** (eval, probes, patching, logit lens): run the question and latent passes
through `eval_codi.encode_and_think` / `PadInvariant`, or reproduce the same mask and position rule. The older GSM8K
scripts (`decode_patch_codi.py`, `necessity_codi.py`, …) do not; port them before using them on these checkpoints.

### 7.3 Settings

| Setting | Value |
|---|---|
| Latents | `--num_latent 6` training, `--inf_latent_iterations 6` test (CODI-0, student-only: 0 and 0) |
| Teacher CoT | `INCLUDE_LAST_COT=False` |
| Padding | `--fix_attn_mask True --pad_invariant True` |
| Distillation | Token after "The answer is:"; smooth L1 / teacher std; weights 1 (student-only: teacher and distillation weights 0) |
| Projection | `--use_prj True --prj_dim 768`, LayerNorm |
| Optimization | LoRA r 128 / α 32, lr 3e-3, cosine, 3% warmup, wd 0.1, 40 epochs, bf16, 64 × 2 = 128, grad norm 2.0 |
| Checkpoints | Each epoch, last 3 kept, in `$SAVE_DIR/$EXP_NAME/gpt2/ep_40/lr_0.003/seed_11/checkpoint-<step>/` |

### 7.4 Smoke run, then launch

```bash
COMMON="TASK=depth2 INCLUDE_LAST_COT=False SAVE_DIR=/workspace/codi_runs CODI_SYNTHETIC_DATA_DIR=/workspace/synthetic_data_v2"
# smoke (own name, required by the script)
env $COMMON EXP_NAME=gpt2_depth2_codi_smoke \
  EXTRA_ARGS="--exp_mode True --exp_data_num 16 --max_steps 2 --per_device_train_batch_size 4 --gradient_accumulation_steps 1" \
  bash configs/synthetic/train_gpt2_depth_codi.sh

env $COMMON EXP_NAME=gpt2_depth2_codi bash configs/synthetic/train_gpt2_depth_codi.sh                      # run 5
env $COMMON EXP_NAME=gpt2_depth2_codi0 NUM_LATENT=0 bash configs/synthetic/train_gpt2_depth_codi.sh         # run 6
env $COMMON EXP_NAME=gpt2_depth2_codi_student NUM_LATENT=0 REF_LOSS_FACTOR=0 DISTILL_LOSS_FACTOR=0 \
  bash configs/synthetic/train_gpt2_depth_codi.sh                                                        # run 7
```

Each on its own GPU. Micro-batch 64 has not been tested on a 24 GB card: watch `nvidia-smi` for the first 50 steps; if it
does not fit, use 32 × 4 (`EXTRA_ARGS`) and note the deviation. Resume after a crash with
`RESUME_FROM=<...>/checkpoint-<step>` (needs `optimizer.pt` and `scheduler.pt`, which the sync now keeps).

### 7.5 Select and test

Evaluate the 3 kept checkpoints on `valid.json` with `eval_synthetic_codi.py --no_log True` (command in its docstring;
the flags must match training or it stops). Pick the best by mean accuracy over all levels (ties → later), then one
logged test eval. Batched eval is fine with `pad_invariant`. Do a 100-item run first (`--smoke_n 100`) and check s/item
and how many outputs run to 256 tokens. Each eval also reports teacher-forced accuracy at the distillation position,
teacher accuracy per level (the teacher's CoT is empty at k = 1, so its k = 1 result is a no-CoT result) and accuracy by
answer value.

## 8. Matched vs not matched across methods

| Matched | Not matched (each method keeps its published recipe) |
|---|---|
| Data files and hashes, test set, answer comparison (`is_correct`) | Answer extraction (Coconut `###`, CODI last number) |
| 6 latents for both latent methods, at training and test | Optimizer, learning rate, schedule |
| GPT-2 small, greedy decoding | LoRA (CODI) vs full fine-tuning (Coconut) |
| Checkpoint rule: best of last 3 on valid, mean over all levels | Number of epochs |
| d50 from the envelope; gains from each method's own reference | |

A raw Coconut-vs-CODI accuracy gap is description, not a finding.

## 9. Logging and operations

- **Log every real run with `record-run`**, failed runs included: run name, commit, config, selected checkpoint,
  under-trained flag, and any deviation (e.g. a micro-batch change).
- **Sync each training machine** while it runs:
  `WATCH_LOGS="/workspace/logs/<run>.log" bash scripts/pod/sync_loop.sh <ip> <port>` (own folder per machine, alerts on
  crash, finish or 3 failed syncs). Do not delete a run's files until `pull_artifacts.sh` verifies.

## 10. Amendment (2026-11-04): CODI stack on RuleChain L1-7w, stabilised optimizer

Applies to every CODI-stack run on RuleChain from now on (CODI, CODI-0, and the planned SIM-CoT-decoder arm). The
round-1 settings in section 7.3 stay as they were for the round-1 records.

### 10.1 Why the change

At the published lr 3e-3, 3 of 6 CODI-stack runs on RuleChain diverged (round-1 CODI 32 x 4 at epoch 6.1, student-only
at 7.3 twice, the L1-7w CODI-0 `20261103-174232` at epoch 2.76), and the survivors had loss spikes. A review of the
L1-7w log histories (2026-11-04) found:
- Every burst is led by the teacher-CoT CE and happens while lr >= 1.7e-3. In the surviving CODI run there are none
  after step 8,500, although the teacher CE is then 10-30x lower than before any spike. The CODI-0 run's fatal second
  blow-up started with the teacher CE at 1.45. So the trigger is the learning rate, not a near-zero teacher CE.
- Ruled out: bf16 overflow, the distillation term's division by the teacher std (flat 0.13-0.22 through every event),
  weight decay.
- Most likely mechanism: lr at the edge of stability in the teacher-CE direction, with Adam's slow second moment
  (beta2 0.999) amplifying each burst; CODI-0 died when a second burst hit before it had recovered.
- `adam_epsilon` 1e-6 is mostly cosmetic here (~5% of a typical sqrt(v)); kept only because it is harmless.

### 10.2 Settings (both arms identical; only `num_latent` differs)

| | Value |
|---|---|
| Data | `synthetic_data_v2_l1to7w` (`scripts/make_rulechain_levels.py --levels 1-7 --train-per-level "1:10000,2:10000,3:10000,4:40000,5:40000,6:40000,7:40000"`), hashes `configs/synthetic/v2_l1to7w_data_sha256.json` |
| Schedule | 12 epochs = 17,820 steps, cosine, 3% warmup |
| **Optimizer** | **lr 2e-3, adam_beta2 0.98, adam_epsilon 1e-6, max_grad_norm 1.0** (rest as 7.3: wd 0.1, 64 x 2, bf16, LoRA r128/a32, seed 11) |
| Model flags | `fix_attn_mask` + `pad_invariant`, `include_last_cot False`, projection 768 + LayerNorm; 6 latents (CODI-0: 0) |
| Checkpoints | every epoch, all 12 kept (`--save_total_limit 12`) |
| Command | `DATA=... HASH_FILE=$REPO/configs/synthetic/v2_l1to7w_data_sha256.json NAME=gpt2_l1to7w_stab EPOCHS=12 TRAIN_EXTRA="--learning_rate 2e-3 --adam_beta2 0.98 --adam_epsilon 1e-6 --max_grad_norm 1.0" bash scripts/pod/codi_l4to7.sh train codi|codi0` (later flags override the recipe's) |
| Watchdog | `scripts/pod/divergence_watchdog.py <log>` on every run: kills on non-finite loss/grad, grad_norm > 50, loss > 20x or > 3x (50 logs) the running median of the Trainer's 10-step losses. Replayed on the 2026-11-03 logs: never kills the surviving CODI, kills the diverged CODI-0 at its second blow-up (step ~4590) |

Check after launch that the logged learning rate matches 2e-3 (first logged value ~3.4e-5 at step 10) and that
`training_args.bin` of the first checkpoint shows the four optimizer values.

### 10.3 Selection, test and decision rule (fixed before looking at test)

- Valid eval of every epoch checkpoint for both arms (after training on the CODI pod; mid-run evals OOM on an A40).
  Selection: best of the last 3 on valid, unweighted mean over levels. One test eval on canonical L1-L7 (500/level);
  L8-L12 canonical test items as a secondary out-of-range check.
- **Primary:** pooled L5-L7 paired gain CODI - CODI-0, exact McNemar. Secondary: per level L4-L7, Holm.
- **Gain:** pooled L5-7 gain >= 0.05 and p < 0.05. **Null:** 95% CI inside +-0.05. **Otherwise ambiguous:** second seed
  pair (possibly 20 epochs).
- Reported as a gain at a matched 12-epoch budget, with both arms' per-epoch valid curves.
- **If either arm diverges again** (watchdog fires): rerun both arms at lr 1e-3.
- Power: minimum detectable paired gain ~0.05 per level (n = 500), ~0.03 pooled L5-7.
- After selection, rerun `scripts/ablate_latents_rulechain_codi.py` and `scripts/probe_latents_rulechain_codi.py` on
  the selected CODI, so "uses its latents" and "gains from its latents" describe the same model.

### 10.4 Notes

- Single seed (11): any gain under ~0.05 is not interpretable without a second seed.
- Hardware: both arms of the completed pair ran on A40s (bf16). The earlier lost attempt used an A40 and an L40S.
- The test split cannot be enlarged by raising `test_per_level`: `build_task` draws test, valid, train from one stream
  per level with a shared duplicate-rejection set, so the train data would change.

### 10.5 Operations safeguards (added 2026-11-04, after the first stabilised pair was lost)

The first section-10 pair was lost when the RunPod balance ran out and RunPod terminated both pods (records
`*_v2-l1to7w-stab-depth2-*-lost`). Container disk does not survive termination. For every RunPod training run:
- Before launch, run `runpodctl user` and confirm the balance covers the full estimated cost plus 50%.
- Sync checkpoints and logs off the pod during training (section 9), at least every epoch, without `optimizer.pt`.
- Chain the valid evals and an end-of-run marker onto the training command, so a pod does not sit idle when training ends.

## 11. Amendment (2026-11-05): start value last

**Question.** Can CODI's latents do serial work once prefill cannot run the chain? Run `20261105-185416` found that the state passes
rule by rule through the rule tokens, which come before the latents and cannot see them, so latents add about one step at the end.
Putting the start value after the rules means no running value can exist at the rule tokens.

### 11.1 Change

Only the question text changes: `Rule 1: ... Rule n: ... Start with 57. What is the final number?`. Chains, ids, levels, steps, answers
and splits are those of section 10, so results pair item by item with the section 10 pair. All other settings are section 10.2,
including seed 11 and the 12-epoch budget. Data: `synthetic_data_v2_l1to7w_startlast` and `synthetic_data_v2_startlast` (canonical
L1-L12 test), made by `scripts/make_rulechain_startlast.py`, hashes `configs/synthetic/v2_l1to7w_startlast_data_sha256.json` and
`v2_startlast_data_sha256.json`.

Training stays on L1-7: it changes one thing against section 10, six latents cannot plausibly exceed about seven steps, and round 1
trained through L12 yet scored 0.08-0.09 at L8 against 0.07-0.08 for the section 10 pair, which never saw L8. L8-L12 stay measured
on the canonical test.

Launch: `SRC_DATA=synthetic_data_v2_l1to7w_startlast SRC_CANON=synthetic_data_v2_startlast HASH_CFG=v2_l1to7w_startlast_data_sha256.json
CANON_HASH_CFG=v2_startlast_data_sha256.json ARM_NAME=gpt2_l1to7w_startlast ARM_SLUG=v2-l1to7w-startlast-depth2 bash
scripts/pod/launch_stab_arm.sh codi|codi0 <pod> <rate> "RunPod A40 48GB"`, with `ARM_NAME` also set for `pair_watcher.py`. Section 10.5
applies. Estimate from the section 10 pair: about 20 pod-hours, $12.30 at $0.60/h, so the balance must be at least $18.50.

### 11.2 Selection and test

As section 10.3: best of the last 3 epochs on valid, one test eval on L1-L7 (500 per level), canonical L1-L12 as the secondary check.
If either arm diverges, rerun both at lr 1e-3. If valid is still climbing at epoch 12, report the result as budget-limited; do not
extend without a new amendment.

### 11.3 Analysis, fixed before training

Reach is the `midpoint` in the run record (section 10 pair: CODI-0 4.86, CODI 5.25). The primary levels move to L2-L7 because CODI-0's
reach is expected to fall, which would put L5-L7 at the floor.

- **Manipulation check:** start-last CODI-0 reach against the section 10 CODI-0, paired by item.
- **Primary:** pooled L2-L7 paired gain, CODI - CODI-0, exact McNemar (n = 3000).
- **Position test:** tight thresholds at each rule, as `a20_lasthop.py`, on the selected pair.
- Rerun `scripts/ablate_latents_rulechain_codi.py` on the selected CODI.

| Outcome | Rule | Reading | Next |
|---|---|---|---|
| Prefill not closed | CODI-0 reach falls by less than 1.0 level | The CODI comparison cannot answer the question | Rerun Tests 2-4 to find how the chain still runs |
| Latents take over | Gain >= 0.05, p < 0.05, reach difference >= +1.0, and a CODI gain before the last rule | Latents can do serial work when prefill cannot | Write up, then latents between rules |
| Latents help | Gain >= 0.05, p < 0.05, but reach difference below +1.0 or gain at the last rule only | Latents add less than one hop even here | Step supervision (SIM-CODI) |
| Null | 95% CI inside +-0.05 | CODI's objective does not move the chain into latents | Step supervision (SIM-CODI) |
| Otherwise | Ambiguous | | Second seed pair |

One seed, as section 10.4: a gain under about 0.05 is not interpretable without a second.

