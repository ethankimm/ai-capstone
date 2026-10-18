#!/usr/bin/env bash
# depth2 LoRA data-efficiency test (scripts/lora_depth2.py) on one pod: Qwen3-1.7B-Base, sizes 0/200/1k/5k/20k.
# Container disk only (see scripts/pod/depth_prompt_check.sh for why); the image's torch is reused.
# Expects /root/ai-capstone and /root/depth2/{train,test}.json.
#   bash scripts/pod/lora_depth2.sh setup | smoke | run
set -euo pipefail
cd /root/ai-capstone
export HF_HOME=/root/hf PYTHONUNBUFFERED=1
PY=/root/venv/bin/python
ARGS=(--train-json /root/depth2/train.json --test-json /root/depth2/test.json --hardware "RunPod ${GPU_LABEL:-A40} (secure)")
case "${1:-}" in
  setup)
    command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync)
    [ -d /root/venv ] || python -m venv --system-site-packages /root/venv  # image python is PEP 668-managed
    $PY -m pip install -q "transformers==4.56.2" "peft==0.17.1" accelerate pyyaml
    $PY -m pip install -q --no-deps -e .
    git config --global user.name >/dev/null 2>&1 || git config --global user.name "Henning Lindig"
    $PY -c "import torch, peft; print('cuda', torch.cuda.is_available(), torch.__version__, 'peft', peft.__version__)"
    ;;
  smoke) $PY scripts/lora_depth2.py "${ARGS[@]}" --sizes 0,64 --per-level 8 --min-passes 256 --smoke ;;
  run) $PY scripts/lora_depth2.py "${ARGS[@]}" 2>&1 | tee /root/run.out ;;
  *) echo "usage: $0 setup | smoke | run"; exit 1 ;;
esac
