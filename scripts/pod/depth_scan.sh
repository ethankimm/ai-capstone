#!/usr/bin/env bash
# depth2 depth-vs-layers scan of pretrained (not fine-tuned) models: scripts/depth_scaling_fewshot.py over a
# model list, one RunRecord per model under results/. Two venvs: vLLM for ordinary models, transformers 4.56.2
# for Ouro (its remote code needs 4.56.x and HF bs=1, see scripts/ouro_gate.py). HF cache is deleted after
# each model so the 200 GB volume holds one model at a time.
# Expects /workspace/ai-capstone and /workspace/synthetic_data_v2/depth2/{train,test}.json.
#   bash scripts/pod/depth_scan.sh setup | smoke | run [MODEL...]
set -euo pipefail
REPO=/workspace/ai-capstone
DATA=/workspace/synthetic_data_v2/depth2
LOGS=/workspace/logs
HW="RunPod ${GPU_LABEL:-A100 SXM 80GB} (secure)"
export HF_HOME=/workspace/hf PATH="$HOME/.local/bin:$PATH"
mkdir -p $LOGS
# /workspace is a network FS (MooseFS) on which huggingface_hub's file locks hang forever (2026-10-17): keep the
# cache there (32B does not fit the container disk) but put the .locks dir on local disk.
[ -e $HF_HOME/hub/.locks ] || { mkdir -p $HF_HOME/hub /root/hf_locks && ln -s /root/hf_locks $HF_HOME/hub/.locks; }
cd $REPO
VLLM_PY=/workspace/vllm-venv/bin/python
OURO_PY=/workspace/ouro-venv/bin/python
COMMON=(--test-json $DATA/test.json --train-json $DATA/train.json --hardware "$HW")

MODELS=(
  EleutherAI/pythia-160m-deduped EleutherAI/pythia-410m-deduped EleutherAI/pythia-1b-deduped
  EleutherAI/pythia-1.4b-deduped EleutherAI/pythia-2.8b-deduped EleutherAI/pythia-6.9b-deduped
  EleutherAI/pythia-12b-deduped
  Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B Qwen/Qwen2.5-32B
  Qwen/Qwen3-0.6B-Base Qwen/Qwen3-1.7B-Base Qwen/Qwen3-4B-Base Qwen/Qwen3-8B-Base Qwen/Qwen3-14B-Base
  ByteDance/Ouro-1.4B ByteDance/Ouro-2.6B
)

run_one() {
  local m=$1; shift
  local name; name=$(basename "$m")
  if [[ $m == ByteDance/Ouro* ]]; then
    $OURO_PY scripts/depth_scaling_fewshot.py --model "$m" --loops 1,2,3,4 --levels 1,2,3,4 --cot-per-level 20 "${COMMON[@]}" "$@"
  else
    local len=4096; [[ $m == EleutherAI/* ]] && len=2048
    $VLLM_PY scripts/depth_scaling_fewshot.py --model "$m" --max-len $len "${COMMON[@]}" "$@"
  fi 2>&1 | tee "$LOGS/$name.log"
}

case "${1:-}" in
  setup)
    command -v rsync >/dev/null && command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync git)
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d /workspace/vllm-venv ] || uv venv -q -p 3.12 /workspace/vllm-venv
    uv pip install -q -p $VLLM_PY "vllm==0.10.2" "transformers==4.56.2" pyyaml
    uv pip install -q -p $VLLM_PY --no-deps -e .
    [ -d /workspace/ouro-venv ] || python -m venv --system-site-packages /workspace/ouro-venv
    $OURO_PY -m pip install -q "transformers==4.56.2" accelerate pyyaml
    $OURO_PY -m pip install -q --no-deps -e .
    git config --global user.name >/dev/null 2>&1 || git config --global user.name "Henning Lindig"
    $VLLM_PY -c "import torch, vllm; print('vllm', vllm.__version__, 'cuda', torch.cuda.is_available())"
    $OURO_PY -c "import torch, transformers; print('ouro venv', transformers.__version__, 'cuda', torch.cuda.is_available())"
    ;;
  smoke)
    run_one Qwen/Qwen2.5-0.5B --smoke-n 4
    run_one ByteDance/Ouro-1.4B --smoke-n 2 --loops 1,4
    ;;
  run)
    shift
    for m in "${@:-${MODELS[@]}}"; do
      run_one "$m" --stage pilot || echo "FAILED $m" | tee -a $LOGS/failed.txt
      rm -rf "$HF_HOME/hub/models--${m//\//--}"
    done
    echo DONE | tee $LOGS/done.txt
    ;;
  *) echo "usage: $0 setup | smoke | run [MODEL...]"; exit 1 ;;
esac
