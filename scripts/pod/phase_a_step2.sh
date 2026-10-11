#!/usr/bin/env bash
# Phase A step 2 on one pod: GPT-2 small on the synthetic depth (1-12) and width (2-16) tasks.
#   Coconut repo (configs/synthetic/): no-CoT and CoT, depth and width.
#   train_synthetic.py: filler tokens, faithful mode, 6 positions (depth, width) and 24 (width);
#     plus a same-stack no-CoT control per task (filler script, 0 positions, not faithful) so the
#     filler-vs-no-CoT McNemar compares models from one training stack.
# All jobs: lr 1e-4, effective batch 128 (32 x 4), early stop after PATIENCE epochs without a
# >= 0.5 pt monitor-accuracy gain, at most MAX_EPOCHS epochs. Every job evaluates on test.json
# per level and logs a RunRecord.
# Expects the repo at /workspace/ai-capstone and the data at /workspace/synthetic_data.
#   bash scripts/pod/phase_a_step2.sh setup | smoke | run
set -euo pipefail
REPO=/workspace/ai-capstone
DATA=/workspace/synthetic_data
RUNS=/workspace/synthetic_runs
LOGS=/workspace/logs
COCONUT_COMMIT=27273cb8cca4bb763c041a63b036d0c3b7cbbb48
PATIENCE=${PATIENCE:-2}
MAX_EPOCHS=${MAX_EPOCHS:-10}
SLOTS=${SLOTS:-5}
HW="RunPod ${GPU_LABEL:-GPU} (secure)"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p $RUNS $LOGS
cd $REPO

setup() {
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  git config --global user.name >/dev/null || git config --global user.name "Henning Lindig"
  uv sync --extra train --extra dev
  [ -d /workspace/coconut/.git ] || git clone -q https://github.com/facebookresearch/coconut.git /workspace/coconut
  git -C /workspace/coconut checkout -q $COCONUT_COMMIT
  [ -x .venv_coconut/bin/python ] || uv venv -q --python 3.11 .venv_coconut
  uv pip install -q --python .venv_coconut/bin/python torch==2.5.1 transformers==4.46.2 wandb==0.18.7 \
      datasets pyyaml tqdm numpy
  ls $DATA/depth $DATA/width
}

coconut_job() {  # name (= config basename), method (cot|no_cot), task
  local name=$1 method=$2 task=$3 port=$4
  local log=$LOGS/$name.log
  (cd /workspace/coconut && WANDB_MODE=offline setsid $REPO/.venv_coconut/bin/torchrun --nnodes 1 \
      --nproc_per_node 1 --master_port $port run.py $REPO/configs/synthetic/$name.yaml > $log 2>&1) &
  local pid=$!
  sleep 5
  .venv_coconut/bin/python scripts/pod/coconut_watch.py --log $log --pid $pid \
      --patience $PATIENCE --max-epochs $MAX_EPOCHS >> $LOGS/$name.watch 2>&1 || true
  wait $pid || true
  local ckpt=$(ls -v /workspace/coconut_runs/$name/checkpoint_* | tail -1)
  .venv_coconut/bin/python scripts/eval_synthetic_coconut.py --method $method --checkpoint_path $ckpt \
      --checkpoint_label "$ckpt" --test_json $DATA/$task/test.json --train_log $log \
      --config_path configs/synthetic/$name.yaml --verify_batched_n 32 \
      --slug synthetic-$task-$method --stage pilot --hardware "$HW" > $LOGS/$name.eval 2>&1
}

ours_job() {  # name task compute_steps faithful(0|1)
  local name=$1 task=$2 k=$3 faithful=$4
  local extra=""; [ "$faithful" = 1 ] && extra="--faithful"
  uv run python scripts/train_synthetic.py --method filler_tokens --compute-steps $k $extra \
      --train-json $DATA/$task/train.json --val-json $DATA/$task/valid_monitor.json \
      --test-json $DATA/$task/test.json --epochs $MAX_EPOCHS --batch-size 32 --grad-accum 4 --lr 1e-4 \
      --early-stop-patience $PATIENCE --output-dir $RUNS/$name --verify-batched-n 32 \
      --slug $name --stage pilot --hardware "$HW" > $LOGS/$name.log 2>&1
}

job() {
  case $1 in
    depth_cot)          coconut_job gpt2_depth_cot cot depth 29501 ;;
    width_cot)          coconut_job gpt2_width_cot cot width 29502 ;;
    depth_no_cot)       coconut_job gpt2_depth_no_cot no_cot depth 29503 ;;
    width_no_cot)       coconut_job gpt2_width_no_cot no_cot width 29504 ;;
    depth_filler6)      ours_job synthetic-depth-filler6-faithful depth 6 1 ;;
    width_filler6)      ours_job synthetic-width-filler6-faithful width 6 1 ;;
    width_filler24)     ours_job synthetic-width-filler24-faithful width 24 1 ;;
    depth_nocot_same)   ours_job synthetic-depth-nocot-samestack depth 0 0 ;;
    width_nocot_same)   ours_job synthetic-width-nocot-samestack width 0 0 ;;
  esac
  echo "$(date -u +%FT%TZ) done $1" >> $LOGS/jobs.done
}

JOBS="depth_cot width_cot width_filler24 depth_filler6 width_filler6 depth_no_cot width_no_cot depth_nocot_same width_nocot_same"

case "${1:-}" in
  setup) setup ;;
  smoke)
    uv run python scripts/train_synthetic.py --method filler_tokens --compute-steps 6 --faithful \
        --train-json $DATA/width/train.json --val-json $DATA/width/valid_monitor.json \
        --test-json $DATA/width/test.json --epochs 1 --smoke-n 8 --verify-batched-n 8 2>&1 | tail -5
    ;;
  run)
    export -f job coconut_job ours_job
    export REPO DATA RUNS LOGS PATIENCE MAX_EPOCHS HW PATH
    nohup bash -c "echo $JOBS | tr ' ' '\n' | xargs -P $SLOTS -I{} bash -c 'job {}'" > $LOGS/step2.log 2>&1 &
    echo launched ;;
  *) echo "usage: $0 setup|smoke|run"; exit 1 ;;
esac
