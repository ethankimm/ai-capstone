#!/usr/bin/env bash
# Follow-up to the depth2 layer scan (scripts/pod/depth_scan.sh): is the one-rule no-CoT ceiling a prompting artifact?
# Qwen only, levels 1-4, 100 items per level: base models with 24 shots instead of 6, and instruct models with an
# explicit "apply each rule in order, answer only the number" system message (0 and 6 shots, Qwen3 thinking off),
# decoding constrained to digits (unconstrained, the instruct models start writing CoT from 2 rules on).
# Everything on the container disk (no network volume: the HF lock hang and slow installs of the first scan).
# Expects /root/ai-capstone and /root/depth2/{train,test}.json.
#   bash scripts/pod/depth_prompt_check.sh setup | smoke | run
set -euo pipefail
cd /root/ai-capstone
export HF_HOME=/root/hf PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
PY=/root/vllm-venv/bin/python
HW="RunPod ${GPU_LABEL:-A40} (secure)"
COMMON=(--test-json /root/depth2/test.json --train-json /root/depth2/train.json --levels 1,2,3,4 --per-level 100
        --cot-per-level 0 --max-len 8192 --hardware "$HW")
mkdir -p /root/logs

case "${1:-}" in
  setup)
    command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync)
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d /root/vllm-venv ] || uv venv -q -p 3.12 /root/vllm-venv
    uv pip install -q -p $PY "vllm==0.10.2" "transformers==4.56.2" pyyaml
    uv pip install -q -p $PY --no-deps -e .
    git config --global user.name >/dev/null 2>&1 || git config --global user.name "Henning Lindig"
    $PY -c "import torch, vllm; print('vllm', vllm.__version__, 'cuda', torch.cuda.is_available())"
    ;;
  smoke)
    $PY scripts/depth_scaling_fewshot.py --model Qwen/Qwen3-8B --chat --force-number --n-shots 6 --max-new 12 --smoke-n 3 "${COMMON[@]}"
    ;;
  run)
    {
      $PY scripts/depth_scaling_fewshot.py --model Qwen/Qwen2.5-7B --n-shots 24 --slug-suffix=-24shot "${COMMON[@]}"
      $PY scripts/depth_scaling_fewshot.py --model Qwen/Qwen3-8B-Base --n-shots 24 --slug-suffix=-24shot "${COMMON[@]}"
      for m in Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-8B; do
        for k in 0 6; do
          $PY scripts/depth_scaling_fewshot.py --model $m --chat --force-number --n-shots $k --max-new 12 --slug-suffix=-instr-${k}shot "${COMMON[@]}"
        done
      done
      echo DONE
    } 2>&1 | tee /root/logs/run.out
    ;;
  *) echo "usage: $0 setup | smoke | run"; exit 1 ;;
esac
