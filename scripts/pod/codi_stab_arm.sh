#!/usr/bin/env bash
# One arm of the TRAINING_PROTOCOL section 10 pair, unattended on the pod: train + divergence watchdog, valid eval of
# every epoch, selection (best of last 3, unweighted mean over levels), one logged test eval on L1-L7, a logged canonical
# L1-L12 eval as the secondary check. Ends by writing /workspace/ARM_DONE or /workspace/ARM_FAILED, which
# scripts/pod/pair_watcher.py reads to pull the results and terminate the pod.
#   setsid nohup bash scripts/pod/codi_stab_arm.sh codi|codi0 "RunPod A40 48GB" > /workspace/logs/arm.log 2>&1 &
set -uo pipefail
K=${1:?codi|codi0}; HW=${2:?hardware label}
# ARM_NAME, ARM_DATA, ARM_CANON, ARM_HASH, ARM_CANON_HASH, ARM_SLUG select another dataset (section 11); the defaults are the section 10 pair.
# ARM_EPOCHS and ARM_RESUME (a checkpoint folder on the pod) extend a finished run (section 11.4).
EP=${ARM_EPOCHS:-12}
N=${ARM_NAME:-gpt2_l1to7w_stab}; nl=6; [ "$K" = codi0 ] && nl=0
REPO=/workspace/ai-capstone; LOG=/workspace/logs/${N}_$K.log; VAL=/workspace/logs/valid_$K.log
D=/workspace/codi_runs/${N}_$K/gpt2/ep_$EP/lr_0.002/seed_11
L17=${ARM_HASH:-$REPO/configs/synthetic/v2_l1to7w_data_sha256.json}
DATA=${ARM_DATA:-/workspace/synthetic_data_v2_l1to7w}; CANON=${ARM_CANON:-/workspace/synthetic_data_v2/depth2/test.json}
SLUG=${ARM_SLUG:-v2-l1to7w-stab-depth2}
EXTRA=${ARM_EXTRA:-}  # extra model flags for training and every eval, e.g. "--rule_gate True" (section 12)
FLAGS="--output_dir /tmp/u --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 --lora_r 128 --lora_alpha 32 --lora_init \
 --batch_size 64 --greedy True --num_latent $nl --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
 --inf_latent_iterations $nl --inf_num_iterations 1 --remove_eos True --use_lora True \
 --fix_attn_mask True --pad_invariant True --include_last_cot False $EXTRA"
fail() { echo "$1" > /workspace/ARM_FAILED; echo "ARM_FAILED: $1"; exit 1; }

cd "$REPO"
(while [ ! -f "$LOG" ]; do sleep 5; done; exec python3 scripts/pod/divergence_watchdog.py "$LOG") > /workspace/logs/watchdog_$K.log 2>&1 &
RESUME_FROM=${ARM_RESUME:-} DATA=$DATA HASH_FILE=$L17 NAME=$N EPOCHS=$EP \
  TRAIN_EXTRA="--learning_rate 2e-3 --adam_beta2 0.98 --adam_epsilon 1e-6 --max_grad_norm 1.0 $EXTRA" \
  bash scripts/pod/codi_l4to7.sh train "$K"
rc=$?
[ -f "$LOG.watchdog" ] && fail "diverged: $(tail -1 "$LOG.watchdog")"
[ $rc -eq 0 ] || fail "train exit $rc"
ls -d "$D"/checkpoint-* >/dev/null 2>&1 || fail "no checkpoints in $D"

cd /workspace/codi
for c in $(ls -d "$D"/checkpoint-* | sort -V); do
  echo "=== $K checkpoint-${c##*-}" >> "$VAL"
  HASH_FILE=$L17 .venv/bin/python "$REPO/scripts/eval_synthetic_codi.py" --ckpt_dir "$c" \
    --test_json $DATA/depth2/valid.json --no_log True $FLAGS 2>&1 \
    | grep -E "^accuracy|^per-level|Traceback|Error" >> "$VAL"
done

SEL=$(python3 - "$VAL" <<'EOF'
import json, re, sys
cur, score = None, {}
for line in open(sys.argv[1]):
    m = re.match(r"=== \S+ checkpoint-(\d+)", line)
    if m:
        cur = int(m.group(1))
    elif line.startswith("per-level:") and cur:
        v = json.loads(line.split(":", 1)[1])
        score[cur] = sum(v.values()) / len(v)
last3 = sorted(score)[-3:]
print(max(last3, key=lambda s: score[s]))
EOF
)
[ -n "$SEL" ] || fail "selection failed, see $VAL"
echo "$SEL" > /workspace/SELECTED
C=$D/checkpoint-$SEL
HASH_FILE=$L17 .venv/bin/python "$REPO/scripts/eval_synthetic_codi.py" --ckpt_dir "$C" \
  --test_json $DATA/depth2/test.json --slug "$SLUG-$K" --stage full_run \
  --hardware "$HW" --checkpoint_label "checkpoint-$SEL" --train_log_from "$C/trainer_state.json" $FLAGS 2>&1 \
  | grep -vE "it/s\]|s/it\]|^batch " | tail -12 > /workspace/logs/test_$K.log
grep -q "^run_id=" /workspace/logs/test_$K.log || fail "test eval failed, see test_$K.log"
# Secondary, out of range: canonical L1-L12 test; a failure here does not fail the arm.
HASH_FILE=${ARM_CANON_HASH:-} .venv/bin/python "$REPO/scripts/eval_synthetic_codi.py" --ckpt_dir "$C" \
  --test_json $CANON --slug "$SLUG-$K-canonical" --stage full_run \
  --hardware "$HW" --checkpoint_label "checkpoint-$SEL" $FLAGS 2>&1 \
  | grep -vE "it/s\]|s/it\]|^batch " | tail -12 > /workspace/logs/canonical_$K.log
echo "$SEL" > /workspace/ARM_DONE
echo ARM_DONE
