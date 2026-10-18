#!/usr/bin/env bash
# depth2 layer-count scan on T5-efficient-base nl2..nl48 (scripts/t5_depth2.py), full fine-tune per depth.
# Container disk only (see scripts/pod/depth_prompt_check.sh for why); the image's torch is reused.
# Expects /root/ai-capstone and /root/depth2/{train,test}.json.
#   bash scripts/pod/t5_depth2.sh setup | smoke | run
set -euo pipefail
cd /root/ai-capstone
export HF_HOME=/root/hf PYTHONUNBUFFERED=1
PY=/root/venv/bin/python
ARGS=(--train-json /root/depth2/train.json --test-json /root/depth2/test.json --hardware "RunPod ${GPU_LABEL:-A40} (secure)")
case "${1:-}" in
  setup)
    command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync)
    [ -d /root/venv ] || python -m venv --system-site-packages /root/venv  # image python is PEP 668-managed
    $PY -m pip install -q "transformers==4.56.2" sentencepiece accelerate pyyaml
    $PY -m pip install -q --no-deps -e .
    git config --global user.name >/dev/null 2>&1 || git config --global user.name "Henning Lindig"
    $PY -c "import torch, transformers; print('cuda', torch.cuda.is_available(), torch.__version__, transformers.__version__)"
    ;;
  # throughput check at the deepest model before committing to the full scan
  smoke) $PY scripts/t5_depth2.py "${ARGS[@]}" --depths 48 --train-n 1280 --epochs 1 --per-level 20 --smoke ;;
  run) $PY scripts/t5_depth2.py "${ARGS[@]}" "${@:2}" 2>&1 | tee -a /root/run.out ;;
  *) echo "usage: $0 setup | smoke | run [extra args]"; exit 1 ;;
esac
