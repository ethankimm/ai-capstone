#!/usr/bin/env bash
# CODI "do latents work when they are needed?" runs on RuleChain restricted to levels 4-7 (above single-pass reach
# ~2-5, every level within the 6-latent budget). Data: scripts/make_rulechain_levels.py --levels 4-7 --train-total 200000
# (test = the canonical L4-L7 test items). Recipe = configs/synthetic/train_gpt2_depth_codi.sh (round 1) except the
# schedule (EPOCHS, default 4: 200k fresh items x 4 = ~6.2k steps, each item seen 4x instead of 40x).
#   bash scripts/pod/codi_l4to7.sh setup              # CODI @ 2c23146 + patches, data hash check
#   bash scripts/pod/codi_l4to7.sh timing             # 100 steps each of CODI (6 latents) and CODI-0; s/step, memory
#   bash scripts/pod/codi_l4to7.sh train codi|codi0   # full run
# Other level mixes (e.g. levels 1-7 with 4-7 up-weighted): override DATA, HASH_FILE, NAME (run prefix) and EPOCHS, e.g.
#   DATA=/workspace/synthetic_data_v2_l1to7w HASH_FILE=$REPO/configs/synthetic/v2_l1to7w_data_sha256.json \
#   NAME=gpt2_l1to7w EPOCHS=12 bash scripts/pod/codi_l4to7.sh train codi
# Optimizer variants: TRAIN_EXTRA="--adam_epsilon 1e-6" NAME=gpt2_l1to7w_eps6 ... (same flags for codi and codi0)
set -euo pipefail
REPO=${REPO:-/workspace/ai-capstone}
CODI_DIR=${CODI_DIR:-/workspace/codi}
DATA=${DATA:-/workspace/synthetic_data_v2_l4to7}
RUNS=${RUNS:-/workspace/codi_runs}
LOGS=${LOGS:-/workspace/logs}
EPOCHS=${EPOCHS:-4}
HASH_FILE=${HASH_FILE:-$REPO/configs/synthetic/v2_l4to7_data_sha256.json}
NAME=${NAME:-gpt2_l4to7}
TRAIN_EXTRA=${TRAIN_EXTRA:-}  # extra train.py flags for both kinds, e.g. "--adam_epsilon 1e-6"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$RUNS" "$LOGS"

common_env() {
  export TASK=depth2 INCLUDE_LAST_COT=False SAVE_DIR=$RUNS CODI_SYNTHETIC_DATA_DIR=$DATA CODI_DIR REPO \
         HASH_FILE
}

case "${1:?setup|timing|train}" in
  setup)
    bash "$REPO/scripts/codi_setup.sh" "$CODI_DIR"
    cd "$CODI_DIR"
    for p in codi_synthetic codi_model_fixes codi_zero_loss_log; do
      if git apply --check "$REPO/scripts/$p.patch" 2>/dev/null; then git apply "$REPO/scripts/$p.patch"; echo "applied $p"
      else echo "skip $p (already applied?)"; fi
    done
    .venv/bin/python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.__version__)"
    HASH_FILE=$HASH_FILE python3 "$REPO/scripts/check_data_hashes.py" "$DATA" depth2
    ;;
  timing)
    common_env
    for nl in 6 0; do
      name=${NAME}_timing_nl${nl}_smoke
      NUM_LATENT=$nl EXP_NAME=$name EXTRA_ARGS="--max_steps 100 --save_strategy no --num_train_epochs $EPOCHS" \
        bash "$REPO/configs/synthetic/train_gpt2_depth_codi.sh" > "$LOGS/$name.log" 2>&1 &
      pid=$!
      peak=0
      while kill -0 $pid 2>/dev/null; do
        m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$m" -gt "$peak" ] && peak=$m
        sleep 5
      done
      wait $pid || echo "timing nl=$nl exited non-zero"
      echo "nl=$nl peak_mem_MiB=$peak"
      grep -E "train_runtime|train_steps_per_second|'loss'" "$LOGS/$name.log" | tail -3
    done
    ;;
  train)
    common_env
    kind=${2:?codi|codi0}
    nl=6; [ "$kind" = codi0 ] && nl=0
    NUM_LATENT=$nl EXP_NAME=${NAME}_${kind} EXTRA_ARGS="--num_train_epochs $EPOCHS --save_total_limit $EPOCHS $TRAIN_EXTRA" \
      bash "$REPO/configs/synthetic/train_gpt2_depth_codi.sh" > "$LOGS/${NAME}_${kind}.log" 2>&1
    ;;
esac
