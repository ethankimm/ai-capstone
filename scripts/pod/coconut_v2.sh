#!/usr/bin/env bash
# v2 Coconut-stack jobs on depth2 (TRAINING_PROTOCOL.md): CoT gate, no_thoughts / no_cot
# references, Coconut c1m6 (c_thought 1 x max_latent_stage 6), a per-stage timing test, valid.json checkpoint selection, final test eval, and checksums over ALL roots.
# Replaces phase_a_step3.sh (v1 data, 10-epoch CODI, no TASK) for these runs. One job per GPU: start each of
# `cot`, `ref no_thoughts`, `ref no_cot`, `coconut` on its own pod unless you are deliberately sharing.
#
# Expects /workspace/ai-capstone (repo rsynced; .git may be absent) and /workspace/synthetic_data_v2/{depth2,modal}.
#   bash scripts/pod/coconut_v2.sh setup
#   bash scripts/pod/coconut_v2.sh cot                      # train the Coconut-stack CoT (COT_EPOCHS, default 3), background
#   bash scripts/pod/coconut_v2.sh gate                     # G1: valid.json CoT >= 0.97 at EVERY level, uncapped
#   bash scripts/pod/coconut_v2.sh timing                   # debug run of Coconut c1m6 stages 1-6 + final (5k examples), s/step + peak mem
#   bash scripts/pod/coconut_v2.sh coconut [NAME] [SEED]    # Coconut c1m6, background (NAME/SEED default gpt2_depth2_coconut_c1m6 / 0)
#   bash scripts/pod/coconut_v2.sh ref no_thoughts|no_cot   # no_thoughts (Coconut schedule) | no_cot trained to plateau, background
#   bash scripts/pod/coconut_v2.sh select NAME METHOD       # best final-stage checkpoint on valid.json (no RunRecord)
#   bash scripts/pod/coconut_v2.sh test NAME METHOD CKPT [SLUG]   # logged test eval of the selected checkpoint
#   bash scripts/pod/coconut_v2.sh manifest | status | prune NAME
# Crash recovery: relaunch the SAME job under the SAME name (run.py resumes from the newest checkpoint on purpose).
# A NEW run (new seed, new schedule) needs a NEW name -- otherwise run.py silently resumes the old one.
set -euo pipefail
REPO=${REPO:-/workspace/ai-capstone}
DATA=${DATA:-/workspace/synthetic_data_v2}
LOGS=${LOGS:-/workspace/logs}
CRUNS=${CRUNS:-/workspace/coconut_runs}
COCONUT_COMMIT=27273cb8cca4bb763c041a63b036d0c3b7cbbb48
HW="RunPod ${GPU_LABEL:-GPU} (secure)"
COT_NAME=${COT_NAME:-gpt2_depth2_cot}
COCONUT_CFG=${COCONUT_CFG:-configs/synthetic/gpt2_depth2_coconut_c1m6.yaml}
NO_THOUGHTS_CFG=${NO_THOUGHTS_CFG:-configs/synthetic/gpt2_depth2_no_thoughts_m6.yaml}
FINAL_STAGE_CKPTS=${FINAL_STAGE_CKPTS:-"8 9 10"}   # num_epochs 10, resume 1, max_latent_stage 6: epochs 7-9 = final stage -> checkpoint_8..10
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$LOGS" "$CRUNS" "$LOGS/configs"
cd "$REPO"
PY=.venv_coconut/bin/python

# TASKS: data dirs whose hashes must check out (TRAINING_PROTOCOL.md generates depth2 only; add modal for modal runs).
TASKS=${TASKS:-depth2}
hashes_ok() { python3 scripts/check_data_hashes.py "$DATA" $TASKS; }

setup() {
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  [ -d /workspace/coconut/.git ] || git clone -q https://github.com/facebookresearch/coconut.git /workspace/coconut
  git -C /workspace/coconut checkout -q "$COCONUT_COMMIT"
  [ -x $PY ] || uv venv -q --python 3.11 .venv_coconut
  uv pip install -q --python $PY torch==2.5.1 transformers==4.46.2 wandb==0.18.7 datasets pyyaml tqdm numpy scipy
  $PY -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.version.cuda)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  hashes_ok
}

# Newest checkpoint_N of a run (the CoT stage uses save_only_improve True, so newest = best-on-monitor).
newest_ckpt() { ls -v "$CRUNS/$1"/checkpoint_* 2>/dev/null | tail -1; }

# Micro-batch: the configs say 32 x 4 (= 128 effective, the reference's 4 GPUs x 32), which OOMs a 24 GB GPU on depth2 CoT
# (2026-10-15, 4090: logits of 32 x ~230 x 50257 fp32 + activations > 22 GB). MICRO_BS (default 16 -> accum 8) keeps the
# effective batch 128 exactly; use MICRO_BS=32 on 48 GB cards.
MICRO_BS=${MICRO_BS:-16}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# render_cfg SRC_YAML NAME SEED LOADPATH -> path of a per-run copy under $LOGS/configs (kept for provenance)
render_cfg() {
  local src=$1 name=$2 seed=$3 load=$4 out="$LOGS/configs/$2.yaml"
  sed -e "s|^name: .*|name: $name|" -e "s|^seed: .*|seed: $seed|" \
      -e "s|^load_model_path: .*|load_model_path: $load|" \
      -e "s|^batch_size_training: .*|batch_size_training: $MICRO_BS|" \
      -e "s|^gradient_accumulation_steps: .*|gradient_accumulation_steps: $((128 / MICRO_BS))|" "$src" > "$out"
  echo "$out"
}

# Launch run.py in its own session; pid file = process group id (killpg-safe; see phase_a_step2.sh).
launch() {  # CFG NAME PORT
  local cfg=$1 name=$2 port=$3
  [ -d "$CRUNS/$name" ] && ls "$CRUNS/$name"/checkpoint_* >/dev/null 2>&1 \
    && echo "NOTE: $CRUNS/$name already has checkpoints -> run.py will RESUME (intended only for crash recovery)"
  hashes_ok
  (cd /workspace/coconut && WANDB_MODE=disabled setsid bash -c 'echo $$ > '"$LOGS/$name.pid"'; exec "$@"' _ \
      "$REPO/$PY" -m torch.distributed.run --nnodes 1 --nproc_per_node 1 --master_port "$port" run.py "$cfg" \
      >> "$LOGS/$name.log" 2>&1) &
  sleep 5
  echo "launched $name (pid group $(cat "$LOGS/$name.pid")), log $LOGS/$name.log"
}

# Keep the newest 3 checkpoints (checkpoint selection = best of the last 3 on valid) plus checkpoint_$KEEP_CKPT (the
# data-matched no_cot checkpoint, TRAINING_PROTOCOL.md section 4.6). Used for the no_cot run, which saves every epoch.
KEEP_CKPT=${KEEP_CKPT:-12}
prune_loop() {
  local name=$1
  nohup bash -c "while true; do
      ls -v $CRUNS/$name/checkpoint_* 2>/dev/null | head -n -3 | grep -v '/checkpoint_${KEEP_CKPT}\$' | xargs -r rm -f; sleep 300; done" \
      > "$LOGS/$name.prune" 2>&1 &
}

parse_valid_levels() { python3 scripts/parse_levels.py; }

case "${1:-}" in
  setup) setup ;;

  cot)  # COT_EPOCHS (default 3): after a G1 failure, rerun with COT_EPOCHS=5 under the SAME name -> run.py resumes
    cfg=$(render_cfg configs/synthetic/gpt2_depth2_cot.yaml "$COT_NAME" 0 None)
    sed -i "s|^num_epochs: .*|num_epochs: ${COT_EPOCHS:-3}|" "$cfg"
    launch "$cfg" "$COT_NAME" 29521 ;;

  gate)  # G1 on valid.json, uncapped (max_new_tokens = longest gold CoT + 16), no RunRecord
    ck=$(newest_ckpt "$COT_NAME"); echo "CoT checkpoint: $ck"
    out=$($PY scripts/eval_synthetic_coconut.py --method cot --checkpoint_path "$ck" --test_json "$DATA/depth2/valid.json" \
        --no_log --batch_size 64 --verify_batched_n 32 2>&1 | tee "$LOGS/gate_$COT_NAME.txt")
    echo "$out" | grep -E "accuracy=|verify batched"
    echo "$out" | parse_valid_levels | tee -a "$LOGS/gate_$COT_NAME.txt"
    python3 - "$LOGS/gate_$COT_NAME.txt" <<'PY'
import json, re, sys
t = open(sys.argv[1]).read()
pl = {int(k): v for k, v in json.loads(re.search(r"per-level: (\{.*\})", t).group(1)).items()}
bad = {k: v for k, v in pl.items() if v < 0.97}
print("G1", "PASS" if not bad else f"FAIL levels<0.97: {bad}")
sys.exit(1 if bad else 0)
PY
    ;;

  timing)  # Coconut c1m6 in debug mode (5k train examples, no saves), after CoT training finished. num_epochs 8 with
           # resume 1 = epochs 1-7 = stages 1-6 + one final-stage epoch, so every stage cost is measured (none extrapolated)
    ck=$(newest_ckpt "$COT_NAME"); [ -n "$ck" ] || { echo "no CoT checkpoint"; exit 1; }
    name=timing_scratch_$(date +%H%M%S)
    cfg=$(render_cfg "$COCONUT_CFG" "$name" 0 "$ck")
    sed -i -e 's|^debug: .*|debug: True|' -e 's|^num_epochs: .*|num_epochs: 8|' "$cfg"
    ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$LOGS/$name.mem"; sleep 10; done ) &
    memloop=$!
    ( cd /workspace/coconut && WANDB_MODE=disabled "$REPO/$PY" -m torch.distributed.run --nnodes 1 --nproc_per_node 1 \
        --master_port 29530 run.py "$cfg" > "$LOGS/$name.log" 2>&1 ) || true
    kill $memloop 2>/dev/null || true
    grep -a "missing_keys\|_IncompatibleKeys\|unexpected" "$LOGS/$name.log" | head -3 || true
    $PY scripts/coconut_timing.py --log "$LOGS/$name.log" --mem "$LOGS/$name.mem" --train-n 100000 --debug-n 5000 \
        --rate "${RATE:-0.27}" --max-latent-stage 6 --final-epochs 3 ;;

  coconut)
    name=${2:-gpt2_depth2_coconut_c1m6}; seed=${3:-0}
    ck=$(newest_ckpt "$COT_NAME"); [ -n "$ck" ] || { echo "no CoT checkpoint at $CRUNS/$COT_NAME"; exit 1; }
    cfg=$(render_cfg "$COCONUT_CFG" "$name" "$seed" "$ck")
    launch "$cfg" "$name" 29522 ;;

  ref)
    kind=${2:?no_thoughts|no_cot}
    case $kind in
      no_thoughts)
        ck=$(newest_ckpt "$COT_NAME"); [ -n "$ck" ] || { echo "no CoT checkpoint"; exit 1; }
        name=gpt2_depth2_no_thoughts_m6
        cfg=$(render_cfg "$NO_THOUGHTS_CFG" "$name" 0 "$ck")
        launch "$cfg" "$name" 29523 ;;
      no_cot)  # trained to plateau: stop when monitor accuracy gains < 0.5 pt over 3 epochs, cap 60 (coconut_watch.py)
        name=gpt2_depth2_no_cot_plateau
        cfg=$(render_cfg configs/synthetic/gpt2_depth2_no_cot_plateau.yaml "$name" 0 None)
        launch "$cfg" "$name" 29524; prune_loop "$name"
        nohup python3 scripts/pod/coconut_watch.py --log "$LOGS/$name.log" --pid "$(cat "$LOGS/$name.pid")" \
            --patience 3 --min-delta 0.005 --max-epochs 60 > "$LOGS/$name.watch" 2>&1 &
        echo "plateau watcher started, log $LOGS/$name.watch" ;;
    esac ;;

  select)  # NAME METHOD -> evaluate each final-stage checkpoint on valid.json; pick best mean accuracy over all levels (ties -> later)
           # no_cot: the last 3 checkpoints of the plateau run (CKPTS="..." overrides; checkpoint_12 is evaluated separately)
    name=${2:?run name}; method=${3:?coconut|no_thoughts|no_cot}
    best=""; bestv=-1
    ckpts=${CKPTS:-$FINAL_STAGE_CKPTS}
    if [ "$method" = no_cot ] && [ -z "${CKPTS:-}" ]; then
      ckpts=$(ls -v "$CRUNS/$name"/checkpoint_* | grep -v "/checkpoint_${KEEP_CKPT}\$" | tail -3 | sed 's|.*checkpoint_||' | tr '\n' ' ')
    fi
    for n in $ckpts; do
      ck="$CRUNS/$name/checkpoint_$n"; [ -f "$ck" ] || { echo "missing $ck"; continue; }
      extra=""; [ "$method" != coconut ] && extra="--verify_batched_n 32"   # batched no-latent eval: check vs batch 1
      out=$($PY scripts/eval_synthetic_coconut.py --method "$method" --checkpoint_path "$ck" \
          --config_path "$LOGS/configs/$name.yaml" --test_json "$DATA/depth2/valid.json" --no_log $extra 2>&1)
      line=$(echo "$out" | parse_valid_levels)
      echo "checkpoint_$n: $line" | tee -a "$LOGS/select_$name.txt"
      v=$(echo "$line" | awk '{print $2}')
      if python3 -c "import sys; sys.exit(0 if float('$v') >= float('$bestv') else 1)"; then best=$ck; bestv=$v; fi
    done
    echo "SELECTED $best (mean over all levels on valid = $bestv)" | tee -a "$LOGS/select_$name.txt" ;;

  test)  # NAME METHOD CKPT [SLUG] -> the one logged test eval (RunRecord) of the selected checkpoint
    name=${2:?}; method=${3:?}; ck=${4:?}; slug=${5:-v2-depth2-$name}
    extra=""; [ "$method" != coconut ] && extra="--verify_batched_n 32"
    $PY scripts/eval_synthetic_coconut.py --method "$method" --checkpoint_path "$ck" --checkpoint_label "$ck" \
        --test_json "$DATA/depth2/test.json" --train_log "$LOGS/$name.log" --config_path "$LOGS/configs/$name.yaml" \
        --slug "$slug" --stage full_run --hardware "$HW" $extra 2>&1 | tee "$LOGS/test_$name.txt" ;;

  prune) prune_loop "${2:?run name}" ;;

  manifest)  # sha256 of every checkpoint under EVERY run root (incl. extensionless Coconut checkpoint_N) older than 2 min
    for root in /workspace/synthetic_runs "$CRUNS" /workspace/codi_runs; do
      [ -d "$root" ] || continue
      # includes CODI's optimizer.pt / scheduler.pt / rng_state.pth: without them a CODI resume restarts the cosine schedule
      (cd "$root" && find . -type f -mmin +2 \( -name 'checkpoint_*' -o -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' \) \
          -print0 | xargs -0 -r sha256sum) > "$root/SHA256SUMS"
      wc -l "$root/SHA256SUMS"
    done
    (cd "$REPO" && find results -type f \( -name manifest.json -o -name predictions.jsonl -o -name train_log.json -o -name notes.md \) \
        -print0 | xargs -0 -r sha256sum) > "$LOGS/SHA256SUMS_results"
    cp "$LOGS/SHA256SUMS_results" "$CRUNS/SHA256SUMS_results"
    wc -l "$LOGS/SHA256SUMS_results" ;;

  status)
    date -u; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    for f in "$LOGS"/gpt2_depth2_*.log; do [ -f "$f" ] || continue; echo "== $f"
      grep -a "Accuracy on validation set" "$f" | tail -3; tail -c 250 "$f" | tr '\r' '\n' | tail -1; echo; done
    ls "$CRUNS"/*/ 2>/dev/null | head -40 ;;

  *) sed -n 2,22p "$0"; exit 1 ;;
esac
