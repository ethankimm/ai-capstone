#!/usr/bin/env bash
# Ouro step 0 gate on a pod. Separate venv (Ouro's remote code needs transformers 4.56.x: 4.54/4.55 break its cache class, 4.53 lacks an import),
# reusing the image's CUDA torch via --system-site-packages.
# Expects in /workspace/ouro_in: xmech_questions.jsonl, codi_correct.json, coconut_correct.json
set -euo pipefail
cd /workspace/ai-capstone
[ -d /workspace/ouro-venv ] || python -m venv --system-site-packages /workspace/ouro-venv
PY=/workspace/ouro-venv/bin/python
$PY -m pip install -q "transformers==4.56.2" accelerate pyyaml
$PY -m pip install -q --no-deps -e .
$PY -c "import torch; print('cuda', torch.cuda.is_available(), torch.__version__)"
git config --global user.name >/dev/null 2>&1 || git config --global user.name "Henning Lindig"
IN=/workspace/ouro_in
for M in ByteDance/Ouro-1.4B ByteDance/Ouro-2.6B; do
    $PY scripts/ouro_gate.py --model "$M" --questions $IN/xmech_questions.jsonl \
        --codi $IN/codi_correct.json --coconut $IN/coconut_correct.json \
        --hardware "RunPod A40 x1" "$@"
done
