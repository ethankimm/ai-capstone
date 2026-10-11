#!/usr/bin/env bash
# Phase A step 3 (+ step-2 follow-up) on one pod, GPT-2 small, synthetic depth task (1-12), three
# jobs in parallel, no early stopping (step 2 showed patience 2 on the 349-example monitor set stops
# runs far too early: depth no-CoT went 0.52 -> 0.81 monitor between epochs 10 and 25):
#   coconut_A_lite  Coconut repo, configs/synthetic/gpt2_depth_coconut_A_lite.yaml (6 epochs from the
#                   step-2 depth CoT checkpoint), then per-level test eval (--method coconut)
#   codi            CODI repo + scripts/codi_synthetic.patch, configs/synthetic/train_gpt2_depth_codi.sh
#                   with 10 epochs instead of 40 (budget) and 32 x 4 instead of 64 x 2 (same 128
#                   effective; three jobs share one GPU), then per-level test eval
#   filler6_long    train_synthetic.py faithful filler, 6 positions, 25 epochs, no early stop
#                   (same epoch budget as the 25-epoch depth no-CoT run), eval on best-on-monitor
# Expects /workspace/ai-capstone, /workspace/synthetic_data, and the step-2 depth CoT checkpoint at
# /workspace/coconut_runs/gpt2_depth_cot/checkpoint_1.
#   bash scripts/pod/phase_a_step3.sh setup | smoke | run | status | codi_monitor
set -euo pipefail
REPO=/workspace/ai-capstone
DATA=/workspace/synthetic_data
LOGS=/workspace/logs
HW="RunPod ${GPU_LABEL:-GPU} (secure)"
CODI_EPOCHS=${CODI_EPOCHS:-10}
FILLER_EPOCHS=${FILLER_EPOCHS:-25}
CODI_FLAGS="--model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 --lora_r 128 --lora_alpha 32 \
  --lora_init --batch_size 256 --greedy True --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False \
  --prj_dropout 0.0 --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p $LOGS
cd $REPO

setup() {
  bash scripts/pod/phase_a_step2.sh setup
  bash scripts/codi_setup.sh /workspace/codi
  (cd /workspace/codi && { git apply --reverse --check $REPO/scripts/codi_synthetic.patch 2>/dev/null \
      || git apply $REPO/scripts/codi_synthetic.patch; })
}

codi_ckpt() { find /workspace/codi_runs -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -1; }
codi_final() { dirname "$(find /workspace/codi_runs -name pytorch_model.bin -not -path '*checkpoint-*' | head -1)"; }

codi_eval() {  # ckpt_dir json extra...
  local ck=$1 js=$2; shift 2
  (cd /workspace/codi && .venv/bin/python $REPO/scripts/eval_synthetic_codi.py --ckpt_dir $ck --test_json $js \
      --output_dir /tmp/codi_eval_unused $CODI_FLAGS --hardware "$HW" "$@")
}

job() {
  case $1 in
    coconut_A_lite)
      local name=gpt2_depth_coconut_A_lite
      (cd /workspace/coconut && WANDB_MODE=disabled $REPO/.venv_coconut/bin/torchrun --nnodes 1 --nproc_per_node 1 \
          --master_port 29511 run.py $REPO/configs/synthetic/$name.yaml) > $LOGS/$name.log 2>&1 || true
      local ckpt=$(ls -v /workspace/coconut_runs/$name/checkpoint_* | tail -1)
      .venv_coconut/bin/python scripts/eval_synthetic_coconut.py --method coconut --checkpoint_path $ckpt \
          --checkpoint_label "$ckpt" --test_json $DATA/depth/test.json --train_log $LOGS/$name.log \
          --config_path configs/synthetic/$name.yaml --slug synthetic-depth-coconut-A-lite --stage pilot \
          --hardware "$HW" > $LOGS/$name.eval 2>&1 ;;
    codi)
      SAVE_DIR=/workspace/codi_runs CODI_SYNTHETIC_DATA_DIR=$DATA \
        EXTRA_ARGS="--num_train_epochs $CODI_EPOCHS --per_device_train_batch_size 32 --gradient_accumulation_steps 4 --save_total_limit 2" \
        bash configs/synthetic/train_gpt2_depth_codi.sh > $LOGS/gpt2_depth_codi.log 2>&1 || true
      local ck=$(codi_final); [ "$ck" = . ] && ck=$(codi_ckpt)
      codi_eval $ck $DATA/depth/test.json --checkpoint_label "$ck" --train_log_from $(codi_ckpt)/trainer_state.json \
          --slug synthetic-depth-codi --stage pilot > $LOGS/gpt2_depth_codi.eval 2>&1 ;;
    filler6_long)
      uv run python scripts/train_synthetic.py --method filler_tokens --compute-steps 6 --faithful \
          --train-json $DATA/depth/train.json --val-json $DATA/depth/valid_monitor.json \
          --test-json $DATA/depth/test.json --epochs $FILLER_EPOCHS --batch-size 32 --grad-accum 4 --lr 1e-4 \
          --early-stop-patience 0 --output-dir /workspace/synthetic_runs/synthetic-depth-filler6-faithful-25ep \
          --verify-batched-n 32 --slug synthetic-depth-filler6-faithful-25ep --stage pilot --hardware "$HW" \
          > $LOGS/synthetic-depth-filler6-faithful-25ep.log 2>&1 ;;
  esac
  echo "$(date -u +%FT%TZ) done $1" >> $LOGS/step3.done
}

case "${1:-}" in
  setup) setup ;;
  smoke)
    # Coconut eval path on the CoT checkpoint (no latent weights trained yet -- checks the code runs)
    .venv_coconut/bin/python scripts/eval_synthetic_coconut.py --method coconut \
        --checkpoint_path /workspace/coconut_runs/gpt2_depth_cot/checkpoint_1 --test_json $DATA/depth/test.json \
        --config_path configs/synthetic/gpt2_depth_coconut_A_lite.yaml --smoke_n 4 2>&1 | tail -4
    (cd /workspace/codi && SAVE_DIR=/workspace/codi_smoke CODI_SYNTHETIC_DATA_DIR=$DATA \
        EXTRA_ARGS="--exp_mode True --exp_data_num 16 --max_steps 2 --per_device_train_batch_size 4 --gradient_accumulation_steps 1" \
        bash $REPO/configs/synthetic/train_gpt2_depth_codi.sh 2>&1 | tail -3)
    find /workspace/codi_smoke -name 'pytorch_model.bin' -o -name 'model.safetensors'
    ;;
  run)
    export -f job codi_ckpt codi_final codi_eval
    export REPO DATA LOGS HW CODI_EPOCHS FILLER_EPOCHS CODI_FLAGS PATH
    for j in coconut_A_lite codi filler6_long; do
      nohup setsid bash -c "job $j" > $LOGS/step3_$j.out 2>&1 &
      sleep 60
    done
    echo launched ;;
  status)
    date -u; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    cat $LOGS/step3.done 2>/dev/null || true
    echo "== coconut A-lite monitor acc:"; grep -a "Accuracy on validation" $LOGS/gpt2_depth_coconut_A_lite.log 2>/dev/null || true
    tail -c 300 $LOGS/gpt2_depth_coconut_A_lite.log 2>/dev/null | tr '\r' '\n' | tail -1; echo
    echo "== codi:"; grep -a -o "'loss': [0-9.]*\|'epoch': [0-9.]*" $LOGS/gpt2_depth_codi.log 2>/dev/null | tail -2 | tr '\n' ' '
    tail -c 300 $LOGS/gpt2_depth_codi.log 2>/dev/null | tr '\r' '\n' | tail -1; echo
    echo "== filler6 25ep:"; grep -a "monitor" $LOGS/synthetic-depth-filler6-faithful-25ep.log 2>/dev/null | tail -30 || true
    ;;
  codi_monitor)  # monitor-set accuracy of the newest CODI checkpoint (no RunRecord)
    ck=$(codi_ckpt); echo "$ck"
    codi_eval $ck $DATA/depth/valid_monitor.json --no_log True 2>&1 | grep -E "accuracy=|per-level"
    ;;
  *) echo "usage: $0 setup|smoke|run|status|codi_monitor"; exit 1 ;;
esac
