#!/usr/bin/env bash
# v2 task pilot on one pod: GPT-2 small, no-CoT and explicit CoT on depth2 / paircount / distinct
# (generator 2.0.0, see V2_TASKS_SPEC.md). Purpose: read the accuracy-by-level curves and decide
# whether v2 passes the spec's gates before any CODI/Coconut spend.
#   no-CoT   = train_synthetic.py --method filler_tokens --compute-steps 0 (same stack as CoT), fixed
#              12 epochs, no early stopping (v1: patience-2 stopping cut no-CoT at 0.36 vs 0.80 at 25 epochs)
#   CoT      = train_synthetic.py --method explicit_cot, fixed 3 epochs
# All jobs: lr 1e-4, effective batch 128 (32 x 4), eval on best-on-monitor checkpoint, per-level test
# eval, one RunRecord each (results/<run_id>/ under the repo on the pod).
# Expects /workspace/ai-capstone and the data at /workspace/synthetic_data_v2.
#   bash scripts/pod/pilot_v2.sh setup | smoke | run | queue JOB... | status
set -euo pipefail
REPO=/workspace/ai-capstone
DATA=/workspace/synthetic_data_v2
RUNS=/workspace/synthetic_runs
LOGS=/workspace/logs
HW="RunPod ${GPU_LABEL:-A40} (secure)"
NOCOT_EPOCHS=${NOCOT_EPOCHS:-12}
COT_EPOCHS=${COT_EPOCHS:-3}
SLOTS=${SLOTS:-4}
export PATH="$HOME/.local/bin:$PATH"
mkdir -p $RUNS $LOGS
cd $REPO

setup() {
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  git config --global user.name >/dev/null 2>&1 || git config --global user.name "Henning Lindig"
  uv sync --extra train --extra dev
  # driver is CUDA 12.8: uv's default torch is cu130 and silently runs on CPU
  uv pip install --python .venv/bin/python --reinstall torch --index-url https://download.pytorch.org/whl/cu128
  .venv/bin/python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ls $DATA
}

job() {  # <task>_<cot|nocot>
  local task=${1%_*} kind=${1##*_} name=pilot-$1
  local common="--train-json $DATA/$task/train.json --val-json $DATA/$task/valid_monitor.json \
      --test-json $DATA/$task/test.json --batch-size 32 --grad-accum 4 --lr 1e-4 --early-stop-patience 0 \
      --output-dir $RUNS/$name --verify-batched-n 32 --stage pilot --hardware \"$HW\""
  case $kind in
    cot)   eval .venv/bin/python scripts/train_synthetic.py --method explicit_cot --epochs $COT_EPOCHS \
               $common --slug v2-$task-cot > $LOGS/$name.log 2>&1 ;;
    nocot) eval .venv/bin/python scripts/train_synthetic.py --method filler_tokens --compute-steps 0 \
               --epochs $NOCOT_EPOCHS $common --slug v2-$task-nocot > $LOGS/$name.log 2>&1 ;;
  esac
  echo "$(date -u +%FT%TZ) done $1 (exit $?)" >> $LOGS/jobs.done
}

JOBS="depth2_nocot depth2_cot paircount_nocot distinct_nocot paircount_cot distinct_cot"

case "${1:-}" in
  setup) setup ;;
  smoke)
    .venv/bin/python scripts/train_synthetic.py --method explicit_cot --train-json $DATA/depth2/train.json \
        --val-json $DATA/depth2/valid_monitor.json --test-json $DATA/depth2/test.json --epochs 1 \
        --smoke-n 8 --verify-batched-n 8 2>&1 | tail -5 ;;
  run)
    export -f job
    export REPO DATA RUNS LOGS HW NOCOT_EPOCHS COT_EPOCHS PATH
    nohup bash -c "echo $JOBS | tr ' ' '\n' | xargs -P $SLOTS -I{} bash -c 'job {}'" > $LOGS/pilot.log 2>&1 &
    echo launched ;;
  queue)  # bash pilot_v2.sh queue job1 job2 ...: start each job once QUEUE_MIN_FREE_MB (default 11 GB) of GPU memory is free
    shift
    export -f job
    export REPO DATA RUNS LOGS HW NOCOT_EPOCHS COT_EPOCHS PATH
    nohup bash -c '
      for j in '"$*"'; do
        until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -ge ${QUEUE_MIN_FREE_MB:-11000} ]; do sleep 60; done
        echo "$(date -u +%FT%TZ) start $j"; (job $j &) ; sleep 240
      done' > $LOGS/queue.log 2>&1 &
    echo queued ;;
  status)
    cat $LOGS/jobs.done 2>/dev/null || echo "no jobs done yet"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    for f in $LOGS/pilot-*.log; do echo "== $f"; tail -c 300 $f | tr '\r' '\n' | tail -2; done ;;
  *) echo "usage: $0 setup|smoke|run|status"; exit 1 ;;
esac
