#!/usr/bin/env bash
# DRAFT -- not launched. CODI on GPT-2 / synthetic depth task. Adapted from Dilgren &
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
#      For the WIDTH task the last step is just the last product, not an answer step: a
#      width copy of this script should pass --include_last_cot True (otherwise the teacher
#      never sees the last product), and say so in notes.md.
# Kept from the prosqa script: lr 3e-3, 40 epochs, 64 x accum 2 = 128 effective on 1 GPU,
# LoRA r128/a32, bf16, cosine + 3% warmup, wd 0.1, 6 latents, projection 768, seed 11,
# model_max_length 1024, save each epoch (save_total_limit 1).
set -euo pipefail

CODI_DIR="${CODI_DIR:-/workspace/codi}"
SAVE_DIR="${SAVE_DIR:-/workspace/codi_runs}"
export CODI_SYNTHETIC_DATA_DIR="${CODI_SYNTHETIC_DATA_DIR:-/workspace/synthetic_data}"
EXP_NAME=gpt2_depth_codi
mkdir -p "$SAVE_DIR"
cd "$CODI_DIR"

export PYTHONBREAKPOINT=0
RESUME_ARGS=()
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARGS=(--resume_from_checkpoint "$RESUME_FROM")
fi
# shellcheck disable=SC2206
EXTRA=(${EXTRA_ARGS:-})

echo "codi commit: $(git rev-parse --short HEAD) | data: $CODI_SYNTHETIC_DATA_DIR/depth | SAVE_DIR=$SAVE_DIR | EXTRA_ARGS=${EXTRA_ARGS:-<none>}"
echo "start: $(date -u +%FT%TZ)"

.venv/bin/python train.py \
	--output_dir "$SAVE_DIR" \
	--expt_name "$EXP_NAME" \
	--logging_dir "$SAVE_DIR/logs" \
	--logging_steps 10 \
	--model_name_or_path gpt2 \
	--data_name depth \
	--seed 11 \
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
	--save_total_limit 1 \
	--weight_decay 0.1 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--do_train \
	--report_to tensorboard \
	--num_latent 6 \
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
