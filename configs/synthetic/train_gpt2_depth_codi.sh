#!/usr/bin/env bash
# CODI on GPT-2 / synthetic tasks (main RuleChain runs: TRAINING_PROTOCOL.md). Adapted from Dilgren &
# Wiegreffe's args_codi/train_gpt2_prosqa_codi.sh (are-lrms-easily-interpretable), run the
# way scripts/train_codi.sh runs CODI (reference repo @ 2c23146, its own venv from
# scripts/codi_setup.sh). Needs scripts/codi_synthetic.patch applied on top of
# scripts/codi_streaming.patch:
#   cd /workspace/codi && git apply /workspace/ai-capstone/scripts/codi_synthetic.patch
# The patch adds data_name `depth` / `width`: reads $CODI_SYNTHETIC_DATA_DIR/<task>/train.json,
# question.strip()+"\n", steps joined by "\n" (prosqa / Coconut format), last step dropped
# unless --include_last_cot, answer template "The answer is: <answer>" (the token after
# "The answer is:" is CODI's distillation position -- same template as icot / prontoqa / prosqa).
#
# Differences from the prosqa script, all deliberate:
#   1. --data_name depth, EXP_NAME gpt2_depth_codi, SAVE_DIR / data dir via env.
#   2. Their script `cp`s a helper .sh that isn't in the repo (dead line) -- dropped.
#   3. PYTHONBREAKPOINT=0 (train.py calls breakpoint() if "The answer is:" isn't found).
#   4. EXTRA_ARGS appended last (later flags win), e.g. smoke:
#        EXTRA_ARGS="--exp_mode True --exp_data_num 16 --max_steps 2 --per_device_train_batch_size 4 --gradient_accumulation_steps 1"
#   5. include_last_cot: left at the default False, i.e. the final step "<<c+9=41>>" (which
#      computes the answer) is NOT in the teacher CoT -- same as icot (GSM8K) and prosqa.
#      modal (v2) also keeps False: its last step <<m+0=m>> is the answer step by design
#      (synthetic_v2.py), so True would put the answer in the teacher CoT before "The answer is:".
#      Only width-style tasks whose last step is NOT an answer step (width, paircount, distinct)
#      need --include_last_cot True; say so in notes.md.
# Kept from the prosqa script: lr 3e-3, 40 epochs, 64 x accum 2 = 128 effective on 1 GPU,
# LoRA r128/a32, bf16, cosine + 3% warmup, wd 0.1, 6 latents, projection 768, seed 11,
# model_max_length 1024, save each epoch.
# Added for the RuleChain main runs (TRAINING_PROTOCOL.md, 2026-10-29), not in the prosqa script:
#   6. --fix_attn_mask True: the answer pass masks the question's left padding (upstream default False attends to it).
#   7. --save_total_limit 3: the evaluated checkpoint is the best of the last 3 epochs on valid.json.
#   8. NUM_LATENT env (default 6); NUM_LATENT=0 = the CODI-0 reference. Needs scripts/codi_model_fixes.patch.
#   9. --pad_invariant True (codi_model_fixes.patch): the latent passes get the running attention mask (as in Liang & Pan's
#      code) and every student pass gets mask-derived position ids, so a left-padded question computes the same as alone.
#      scripts/eval_codi.py applies the same rule at inference; scripts/check_codi_padding.py tests it.
#  10. REF_LOSS_FACTOR / DISTILL_LOSS_FACTOR env (default 1.0); both 0 with NUM_LATENT=0 = the student-only run.
#  11. Data hash check, smoke-name guard, per-run logging dir.
set -euo pipefail

CODI_DIR="${CODI_DIR:-/workspace/codi}"
SAVE_DIR="${SAVE_DIR:-/workspace/codi_runs}"
export CODI_SYNTHETIC_DATA_DIR="${CODI_SYNTHETIC_DATA_DIR:-/workspace/synthetic_data}"
# TASK = depth | width | depth2 | modal | paircount | distinct (data dir name under CODI_SYNTHETIC_DATA_DIR).
# INCLUDE_LAST_COT=True for width-style tasks whose last step is not an answer step (paircount, distinct,
# width), so the teacher must see it. Leave False for depth / depth2 / modal (modal's last step IS the answer).
TASK="${TASK:-depth}"
INCLUDE_LAST_COT="${INCLUDE_LAST_COT:-False}"
# NUM_LATENT=0 trains the CODI-0 reference (TRAINING_PROTOCOL.md); needs scripts/codi_model_fixes.patch applied.
NUM_LATENT="${NUM_LATENT:-6}"
# REF_LOSS_FACTOR=0 DISTILL_LOSS_FACTOR=0 with NUM_LATENT=0 = the student-only run (plain no-CoT in CODI's code).
REF_LOSS_FACTOR="${REF_LOSS_FACTOR:-1.0}"
DISTILL_LOSS_FACTOR="${DISTILL_LOSS_FACTOR:-1.0}"
EXP_NAME="${EXP_NAME:-gpt2_${TASK}_codi}"
# SEED (default 11, the published seed); the checkpoint dir is .../ep_40/lr_0.003/seed_$SEED. Second-seed runs: SEED=12.
SEED="${SEED:-11}"
REPO="${REPO:-/workspace/ai-capstone}"
mkdir -p "$SAVE_DIR"

# A smoke run (--max_steps) writes into $SAVE_DIR/$EXP_NAME/gpt2/ep_40/... exactly like the real run (--max_steps does not
# change the ep_40 folder), so it must have its own name.
if [[ "${EXTRA_ARGS:-}" == *--max_steps* && "$EXP_NAME" != *_smoke ]]; then
    echo "ERROR: smoke run (--max_steps) needs EXP_NAME ending in _smoke, got $EXP_NAME" >&2; exit 1
fi
# Data hashes (configs/synthetic/v2_data_sha256.json) before any training.
python3 "$REPO/scripts/check_data_hashes.py" "$CODI_SYNTHETIC_DATA_DIR" "$TASK" || { echo "ERROR: data hash check failed" >&2; exit 1; }
# codi_model_fixes.patch must be applied: it adds --pad_invariant (and makes NUM_LATENT=0 train the student).
grep -q "pad_invariant" "$CODI_DIR/src/model.py" || { echo "ERROR: scripts/codi_model_fixes.patch not applied in $CODI_DIR" >&2; exit 1; }
cd "$CODI_DIR"

export PYTHONBREAKPOINT=0
RESUME_ARGS=()
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARGS=(--resume_from_checkpoint "$RESUME_FROM")
fi
# shellcheck disable=SC2206
EXTRA=(${EXTRA_ARGS:-})

echo "codi commit: $(git rev-parse --short HEAD) | data: $CODI_SYNTHETIC_DATA_DIR/$TASK | include_last_cot=$INCLUDE_LAST_COT | num_latent=$NUM_LATENT | ref/distill factors=$REF_LOSS_FACTOR/$DISTILL_LOSS_FACTOR | seed=$SEED | SAVE_DIR=$SAVE_DIR | EXTRA_ARGS=${EXTRA_ARGS:-<none>}"
echo "start: $(date -u +%FT%TZ)"

.venv/bin/python train.py \
	--output_dir "$SAVE_DIR" \
	--expt_name "$EXP_NAME" \
	--logging_dir "$SAVE_DIR/logs/$EXP_NAME" \
	--logging_steps 10 \
	--model_name_or_path gpt2 \
	--data_name "$TASK" \
	--include_last_cot "$INCLUDE_LAST_COT" \
	--seed "$SEED" \
	--model_max_length 1024 \
	--per_device_train_batch_size 64 \
	--gradient_accumulation_steps 2 \
	--bf16 \
	--num_train_epochs 40 \
	--learning_rate 3e-3 \
	--max_grad_norm 2.0 \
	--use_lora True \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--save_strategy "epoch" \
	--save_safetensors False \
	--save_total_limit 3 \
	--weight_decay 0.1 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--do_train \
	--report_to tensorboard \
	--num_latent "$NUM_LATENT" \
	--fix_attn_mask True \
	--pad_invariant True \
	--ref_loss_factor "$REF_LOSS_FACTOR" \
	--distill_loss_factor "$DISTILL_LOSS_FACTOR" \
	--logging_strategy "steps" \
	--use_prj True \
	--prj_dim 768 \
	--prj_dropout 0.0 \
	--distill_loss_div_std True \
	--exp_mode False \
	--exp_data_num 2000 \
	--remove_eos True \
	--print_ref_model_stats True \
	${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
	${EXTRA[@]+"${EXTRA[@]}"}

echo "end: $(date -u +%FT%TZ)"
echo "CODI_TRAIN_DONE"
