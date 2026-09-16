#!/usr/bin/env bash
# Bootstrap a fresh RunPod pod (or any CUDA box) to run real training/eval in this repo.
# Usage: bash scripts/runpod_setup.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Base deps + the `train` extra (torch/transformers/accelerate) -- not installed by
# default locally since local dev machines here don't have CUDA.
uv sync --extra train --extra dev

echo
echo "Done. Sanity-check GPU visibility:"
uv run python -c "import torch; print('cuda available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo
echo "Next: uv run pytest && uv run python scripts/run_baseline.py   (no GPU; validates the pipeline)"
echo "Then: pick a mechanism config under configs/mechanisms/ and start training."
