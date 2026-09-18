#!/usr/bin/env bash
# CODI on GPT-2 / GSM8k-Aug -- the authors' `scripts/train_gpt2_gsm8k-aug.sh` (commit
# 2c23146 of github.com/zhenyi4/codi), argument-for-argument, run inside the venv that
# scripts/codi_setup.sh builds. Paper (arXiv:2502.21074, Appendix A): lr 3e-3, 40
# epochs, effective batch 128, LoRA r=128/alpha=32, bf16, cosine + 3% warmup, 6 latent
# tokens, seed 11; 43.7% on GSM8K test, ~36h on one A100 80GB. Table A5: 38.4% at 20
# epochs (gamma=1) -- the cheaper, still paper-reported target.
#
# The COMPLETE list of differences from their script:
#   1. SAVE_DIR -> ours (env `SAVE_DIR`, default /workspace/codi_run). Their script also
#      `cp`s a helper .sh that doesn't exist in the repo (dead line) -- dropped.
#   2. --save_strategy steps --save_steps 3000 --save_total_limit 2 instead of
#      --save_strategy no --save_total_limit 1: their script writes weights ONLY at the
#      end of the 36h run; a pod restart would lose everything. Periodic checkpoints
#      change nothing about the optimization. Resume with RESUME_FROM=<checkpoint dir>
#      (train.py is patched by scripts/codi_streaming.patch to honor it).
#   3. PYTHONBREAKPOINT=0: train.py calls breakpoint() when it can't find the answer
#      prompt in a tokenized example, which would leave a detached job stuck in pdb;
#      as a no-op the failure surfaces as an exception instead.
#   4. Anything in EXTRA_ARGS is appended last (later flags win in argparse) -- use it
#      for smoke tests / epoch changes and record what you passed in notes.md, e.g.
#        EXTRA_ARGS="--num_train_epochs 20"                        # Table A5 setting
#        EXTRA_ARGS="--exp_mode True --exp_data_num 64 --max_steps 4 --per_device_train_batch_size 4 --save_steps 2"
#
# Final weights land in $SAVE_DIR/gsm8k_llama1b_latent_baseline/gpt2/ep_<E>/lr_0.003/seed_11/
# (the expt_name is their misnomer, kept verbatim -- it's only a path component):
# pytorch_model.bin + trainer_state.json (log_history) + training_args.bin.
# Then: scripts/eval_codi.py --ckpt_dir <that dir> ...
set -euo pipefail

CODI_DIR="${CODI_DIR:-/workspace/codi}"
SAVE_DIR="${SAVE_DIR:-/workspace/codi_run}"
mkdir -p "$SAVE_DIR"
cd "$CODI_DIR"

export PYTHONBREAKPOINT=0
RESUME_ARGS=()
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARGS=(--resume_from_checkpoint "$RESUME_FROM")
fi
# shellcheck disable=SC2206
EXTRA=(${EXTRA_ARGS:-})

echo "codi commit: $(git rev-parse --short HEAD) | SAVE_DIR=$SAVE_DIR | EXTRA_ARGS=${EXTRA_ARGS:-<none>} | RESUME_FROM=${RESUME_FROM:-<none>}"
echo "start: $(date -u +%FT%TZ)"

.venv/bin/python train.py \
	--output_dir "$SAVE_DIR" \
	--expt_name gsm8k_llama1b_latent_baseline \
	--logging_dir "$SAVE_DIR/logs" \
	--logging_steps 10 \
	--model_name_or_path gpt2 \
	--data_name icot \
	--seed 11 \
	--model_max_length 512 \
	--per_device_train_batch_size 64 \
	--gradient_accumulation_steps 2 \
	--bf16 \
	--num_train_epochs 40 \
	--learning_rate 3e-3 \
	--max_grad_norm 2.0 \
	--use_lora True \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--save_strategy "steps" \
	--save_steps 3000 \
	--save_safetensors False \
	--save_total_limit 2 \
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
	"${RESUME_ARGS[@]}" \
	"${EXTRA[@]}"

echo "end: $(date -u +%FT%TZ)"
echo "CODI_TRAIN_DONE"
