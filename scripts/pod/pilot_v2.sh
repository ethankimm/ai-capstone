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
#   bash scripts/pod/pilot_v2.sh setup | smoke | run | queue JOB... | status | manifest
# MONITOR=valid.json (2k items) instead of the default valid_monitor.json (30/level) gives a less noisy
# best-checkpoint pick (review 2026-10-13); both are disjoint from test.
# Checkpoints live only under $RUNS on the pod. Before terminating the pod: `manifest`, then pull with
# scripts/pod/pull_artifacts.sh from the laptop (rsync + sha256 check). Never delete the pod before that passes.
set -euo pipefail
REPO=/workspace/ai-capstone
DATA=/workspace/synthetic_data_v2
RUNS=/workspace/synthetic_runs
LOGS=/workspace/logs
HW="RunPod ${GPU_LABEL:-A40} (secure)"
MONITOR=${MONITOR:-valid_monitor.json}
NOCOT_EPOCHS=${NOCOT_EPOCHS:-12}
COT_EPOCHS=${COT_EPOCHS:-3}
FILLER_EPOCHS=${FILLER_EPOCHS:-25}
SUF=${SLUG_SUFFIX:-}   # appended to the RunRecord slug, e.g. -25ep, so reruns are distinguishable in the index
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
  local common="--train-json $DATA/$task/train.json --val-json $DATA/$task/$MONITOR \
      --test-json $DATA/$task/test.json --batch-size 32 --grad-accum 4 --lr 1e-4 --early-stop-patience 0 \
      --output-dir $RUNS/$name --verify-batched-n 32 --on-mismatch ${ON_MISMATCH:-batch1} --stage pilot --hardware \"$HW\""
  local rc=0
  case $kind in
    cot)   eval .venv/bin/python scripts/train_synthetic.py --method explicit_cot --epochs $COT_EPOCHS \
               $common --slug v2-$task-cot$SUF > $LOGS/$name.log 2>&1 || rc=$? ;;
    nocot) eval .venv/bin/python scripts/train_synthetic.py --method filler_tokens --compute-steps 0 \
               --epochs $NOCOT_EPOCHS $common --slug v2-$task-nocot$SUF > $LOGS/$name.log 2>&1 || rc=$? ;;
    # faithful filler (50/50 CoT/filler mix, unmasked filler loss) = the v1 filler-6/24 recipe; same stack as nocot
    # control: filler-N trained answer-only with the filler span masked from the loss (NO CoT supervision)
    nf6|nf24) eval .venv/bin/python scripts/train_synthetic.py --method filler_tokens \
               --compute-steps ${kind#nf} --epochs $FILLER_EPOCHS $common --slug v2-$task-$kind$SUF > $LOGS/$name.log 2>&1 || rc=$? ;;
    filler6|filler24) eval .venv/bin/python scripts/train_synthetic.py --method filler_tokens --faithful \
               --compute-steps ${kind#filler} --epochs $FILLER_EPOCHS $common --slug v2-$task-$kind$SUF > $LOGS/$name.log 2>&1 || rc=$? ;;
  esac
  echo "$(date -u +%FT%TZ) done $1 (exit $rc)" >> $LOGS/jobs.done
}

JOBS=${JOBS:-"depth2_nocot depth2_cot modal_nocot modal_cot"}

case "${1:-}" in
  setup) setup ;;
  smoke)
    .venv/bin/python scripts/train_synthetic.py --method explicit_cot --train-json $DATA/depth2/train.json \
        --val-json $DATA/depth2/valid_monitor.json --test-json $DATA/depth2/test.json --epochs 1 \
        --smoke-n 8 --verify-batched-n 8 2>&1 | tail -5 ;;
  run)
    export -f job
    export REPO DATA RUNS LOGS HW MONITOR ON_MISMATCH NOCOT_EPOCHS COT_EPOCHS FILLER_EPOCHS SUF PATH
    nohup bash -c "echo $JOBS | tr ' ' '\n' | xargs -P $SLOTS -I{} bash -c 'job {}'" > $LOGS/pilot.log 2>&1 &
    echo launched ;;
  queue)  # bash pilot_v2.sh queue job1 job2 ...: start each job once QUEUE_MIN_FREE_MB (default 11 GB) of GPU memory is free
    shift
    export -f job
    export REPO DATA RUNS LOGS HW MONITOR ON_MISMATCH NOCOT_EPOCHS COT_EPOCHS FILLER_EPOCHS SUF PATH
    nohup bash -c '
      for j in '"$*"'; do
        until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -ge ${QUEUE_MIN_FREE_MB:-11000} ]; do sleep 60; done
        echo "$(date -u +%FT%TZ) start $j"; (job $j &) ; sleep 240
      done' > $LOGS/queue.log 2>&1 &
    echo queued ;;
  manifest)  # sha256 of every checkpoint weight file + the RunRecords, for pull_artifacts.sh to verify
    (cd $RUNS && find . -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \) -print0 | xargs -0 sha256sum) > $RUNS/SHA256SUMS
    (cd $REPO && find results -type f \( -name manifest.json -o -name predictions.jsonl -o -name train_log.json -o -name notes.md \) -print0 | xargs -0 sha256sum) > $RUNS/SHA256SUMS_results
    wc -l $RUNS/SHA256SUMS $RUNS/SHA256SUMS_results ;;
  status)
    cat $LOGS/jobs.done 2>/dev/null || echo "no jobs done yet"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    for f in $LOGS/pilot-*.log; do echo "== $f"; tail -c 300 $f | tr '\r' '\n' | tail -2; done ;;
  *) echo "usage: $0 setup|smoke|run|queue|status|manifest"; exit 1 ;;
esac
