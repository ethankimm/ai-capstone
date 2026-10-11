#!/usr/bin/env bash
# Phase A step 1 on one pod: GSM8K twin transplants over a wide change set, CODI and Coconut
# (GPT-2, released checkpoints), run in parallel. Expects the repo rsync'd to
# /workspace/ai-capstone and the D&W data file at /workspace/coconut_data/.
#   bash scripts/pod/phase_a_step1.sh setup     # envs + checkpoints
#   bash scripts/pod/phase_a_step1.sh smoke     # n=8, no logging
#   bash scripts/pod/phase_a_step1.sh run       # full runs, logs to results/
set -euo pipefail
REPO=/workspace/ai-capstone
CHANGES="1,-1,2,-2,3,-3,5,-5,10,-10,20,-20,x0.5,x1.25,x1.5,x2,x3"
export PATH="$HOME/.local/bin:$PATH"
cd $REPO

setup() {
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  git config --global user.name >/dev/null || git config --global user.name "Henning Lindig"
  bash scripts/codi_setup.sh /workspace/codi
  uv venv -q --python 3.11 .venv_coconut
  uv pip install -q --python .venv_coconut/bin/python torch==2.5.1 transformers==4.46.2 datasets pyyaml tqdm numpy "huggingface_hub[cli]"
  .venv_coconut/bin/huggingface-cli download zen-E/CODI-gpt2 --local-dir /workspace/codi_released
  .venv_coconut/bin/huggingface-cli download connordilgren/gpt2-gsm8k-coconut checkpoint_33 --local-dir /workspace/coconut_checkpoints
  ls -la /workspace/codi_released /workspace/coconut_checkpoints /workspace/coconut_data
}

codi() {  # $@: extra args
  (cd /workspace/codi && .venv/bin/python $REPO/scripts/patch_minimal_pair_codi.py \
    --ckpt_dir /workspace/codi_released --checkpoint_label hf:zen-E/CODI-gpt2@fd641b3 \
    --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 --lora_r 128 --lora_alpha 32 \
    --lora_init --greedy True --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
    --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True --output_dir /tmp/o \
    --eval_n 0 --change_set "$CHANGES" --conditions all_slot --n_pairs 100000 --n_nonresponsive 300 "$@")
}

coconut() {
  .venv_coconut/bin/python scripts/patch_minimal_pair_coconut.py \
    --checkpoint_path /workspace/coconut_checkpoints/checkpoint_33 --data_dir /workspace/coconut_data \
    --eval_n 0 --change_set "$CHANGES" --conditions all_slot --n_pairs 100000 --n_nonresponsive 300 "$@"
}

case "${1:-}" in
  setup) setup ;;
  smoke) codi --smoke_n 8 2>&1 | tail -12; coconut --smoke_n 8 2>&1 | tail -12 ;;
  run)
    HW="RunPod ${GPU_LABEL:-GPU} (secure)"
    nohup bash -c "$(declare -f codi); REPO=$REPO CHANGES='$CHANGES'; codi --slug twin-magnitude-gsm8k --stage full_run --hardware '$HW'" > /workspace/step1_codi.log 2>&1 &
    nohup bash -c "$(declare -f coconut); cd $REPO; CHANGES='$CHANGES'; coconut --slug twin-magnitude-gsm8k --stage full_run --hardware '$HW'" > /workspace/step1_coconut.log 2>&1 &
    echo launched ;;
  *) echo "usage: $0 setup|smoke|run"; exit 1 ;;
esac
