#!/usr/bin/env bash
# Set up the CODI reference implementation (github.com/zhenyi4/codi) next to this repo
# on a CUDA box, in its OWN venv pinned to the authors' requirements.txt -- the pins
# (torch 2.7.1 / transformers 4.52.4 / peft 0.15.2) differ from this repo's `train`
# extra, so the two environments are kept apart on purpose.
#
# Usage: bash scripts/codi_setup.sh [CODI_DIR]      (default CODI_DIR=/workspace/codi)
#
# What it does, and why each step exists:
#   1. clone at CODI_COMMIT (pinned so the reproduction is against a fixed upstream)
#   2. apply scripts/codi_streaming.patch -- the only source change to their code:
#      `load_dataset("zen-E/GSM8k-Aug")` hits a datasets/pyarrow false-positive
#      ArrowInvalid on the train file; `streaming=True` sidesteps it (same bug and
#      fix as latentreasoning/data/gsm8k_aug.py). Training semantics are unchanged.
#   3. python 3.12 venv (their README: `conda create --name codi python=3.12`) with
#      `pip install -r requirements.txt` equivalent via uv (every line is `==`-pinned,
#      so the resolved set is identical).
#   4. install this repo into that venv (base deps only, no torch) so
#      scripts/eval_codi.py can use the shared loader / scorer / RunRecord.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CODI_DIR="${1:-/workspace/codi}"
CODI_REPO="https://github.com/zhenyi4/codi.git"
CODI_COMMIT="2c2314662c63e9f482ebc46614ffe9af17a241e5"   # 2025-12-15 "Add acceptance note for EMNLP 2025"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "$CODI_DIR/.git" ]; then
    git clone "$CODI_REPO" "$CODI_DIR"
fi
cd "$CODI_DIR"
git fetch -q origin
git checkout -q "$CODI_COMMIT"

if git apply --check "$REPO_ROOT/scripts/codi_streaming.patch" 2>/dev/null; then
    git apply "$REPO_ROOT/scripts/codi_streaming.patch"
    echo "applied codi_streaming.patch"
elif git apply --reverse --check "$REPO_ROOT/scripts/codi_streaming.patch" 2>/dev/null; then
    echo "codi_streaming.patch already applied"
else
    echo "ERROR: codi_streaming.patch neither applies nor is applied -- upstream changed?" >&2
    exit 1
fi

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-deps -e "$REPO_ROOT"
# base deps of this repo that aren't already in CODI's pins (datasets/numpy/tqdm are)
uv pip install --python .venv/bin/python "pyyaml>=6.0"

echo
echo "CODI env ready at $CODI_DIR (commit $(git rev-parse --short HEAD), patch applied)."
.venv/bin/python - <<'EOF'
import torch, transformers, peft, datasets
print(f"torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__} | datasets {datasets.__version__}")
print("cuda available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
import latentreasoning; print("latentreasoning importable from", latentreasoning.__file__)
EOF
