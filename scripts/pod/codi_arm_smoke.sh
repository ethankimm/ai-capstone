#!/usr/bin/env bash
# TRAINING_PROTOCOL section 13.2: a short real run of the arm's train, save and eval path, on the pod, before the arm starts. A crash before
# the first checkpoint would otherwise leave the pod billing with nothing for pair_watcher.py to verify. Exit 1 on any failure.
#   ARM_DATA=/workspace/<data> ARM_HASH=<hash json> ARM_EXTRA="--step_decoder True" bash scripts/pod/codi_arm_smoke.sh
set -uo pipefail
DATA=${ARM_DATA:?}; HASH=${ARM_HASH:?}; EXTRA=${ARM_EXTRA:-}
REPO=/workspace/ai-capstone; LOG=/workspace/logs/arm_smoke.log; RUNS=/workspace/codi_smoke; MEM=/workspace/logs/arm_smoke_mem.log
C=$RUNS/arm_smoke/gpt2/ep_12/lr_0.002/seed_11/checkpoint-20
FLAGS="--output_dir /tmp/u --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 --lora_r 128 --lora_alpha 32 --lora_init \
 --batch_size 64 --greedy True --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
 --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \
 --fix_attn_mask True --pad_invariant True --include_last_cot False $EXTRA"
bad() { echo "SMOKE_FAILED: $1"; tail -15 "$LOG" | cut -c1-200; exit 1; }
mkdir -p "$RUNS" /workspace/logs
(while sleep 2; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; done) > "$MEM" &
W=$!
TASK=depth2 INCLUDE_LAST_COT=False SAVE_DIR=$RUNS CODI_SYNTHETIC_DATA_DIR=$DATA CODI_DIR=/workspace/codi REPO=$REPO HASH_FILE=$HASH \
NUM_LATENT=6 EXP_NAME=arm_smoke EXTRA_ARGS="--exp_mode True --exp_data_num 4000 --max_steps 20 --num_train_epochs 12 --save_strategy steps --save_steps 20 \
 --save_total_limit 1 --learning_rate 2e-3 --adam_beta2 0.98 --adam_epsilon 1e-6 --max_grad_norm 1.0 $EXTRA" \
  bash "$REPO/configs/synthetic/train_gpt2_depth_codi.sh" > "$LOG" 2>&1
rc=$?; kill $W 2>/dev/null
[ $rc -eq 0 ] || bad "train exit $rc"
grep -q CODI_TRAIN_DONE "$LOG" || bad "no CODI_TRAIN_DONE"
[ -f "$C/pytorch_model.bin" ] || bad "no checkpoint at $C"
if [[ "$EXTRA" == *--step_decoder* ]]; then
  grep "optimizer groups" "$LOG" | head -1 | grep -q "0.0001" || bad "no 1e-4 decoder group"
  grep -q "'explain_z5'" "$LOG" || bad "no step loss at latent 5"
  python3 - "$LOG" <<'P' || bad "step loss not above 0"
import re, sys
t = open(sys.argv[1]).read().replace("\r", "\n")
v = [float(x) for x in re.findall(r"'explain_loss': ([0-9.e+-]+)", t)]
print("explain_loss first", v[:1], "last", v[-1:]); sys.exit(0 if v and min(v) > 0 else 1)
P
fi
grep -E "optimizer groups" "$LOG" | head -1 | cut -c1-160
cd /workspace/codi
HASH_FILE=$HASH .venv/bin/python "$REPO/scripts/eval_synthetic_codi.py" --ckpt_dir "$C" --test_json "$DATA/depth2/valid.json" --no_log True $FLAGS > /workspace/logs/arm_smoke_eval.log 2>&1 || { tail -8 /workspace/logs/arm_smoke_eval.log | cut -c1-200; bad "eval failed"; }
grep -E "^accuracy|^per-level" /workspace/logs/arm_smoke_eval.log | cut -c1-200 | grep -q "^accuracy" || bad "no eval accuracy"
grep -E "^accuracy" /workspace/logs/arm_smoke_eval.log | cut -c1-120
peak=$(sort -n "$MEM" | tail -1); echo "peak_mem_MiB=$peak"; [ "${peak:-0}" -lt 46000 ] || bad "peak memory $peak MiB too close to 48 GB"
tr '\r' '\n' < "$LOG" | grep -E 'it\]$|s/it\]' | tail -1 | cut -c1-100
du -sh "$C" | cut -f1 | sed 's/^/checkpoint size /'
rm -rf "$RUNS"; echo SMOKE_OK
