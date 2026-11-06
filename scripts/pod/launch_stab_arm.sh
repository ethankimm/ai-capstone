#!/usr/bin/env bash
# Laptop-side: wait for a pod's SSH, sync repo + data, run CODI setup, start scripts/pod/codi_stab_arm.sh detached.
# Prints "ARM kind:host:port:pod_id:rate" on success, the spec scripts/pod/pair_watcher.py takes.
#   bash scripts/pod/launch_stab_arm.sh codi|codi0 <pod_id> <rate $/h> "RunPod A40 48GB"
set -uo pipefail
K=${1:?kind}; POD=${2:?pod id}; RATE=${3:?rate}; HW=${4:?hardware label}
cd "$(dirname "$0")/../.."
# Another dataset (section 11): SRC_DATA, SRC_CANON (folders beside the repo), HASH_CFG, CANON_HASH_CFG, ARM_NAME, ARM_SLUG. Defaults are the section 10 pair.
SRC_DATA=${SRC_DATA:-synthetic_data_v2_l1to7w}; SRC_CANON=${SRC_CANON:-synthetic_data_v2}; HASH_CFG=${HASH_CFG:-v2_l1to7w_data_sha256.json}
ARM_NAME=${ARM_NAME:-gpt2_l1to7w_stab}; ARM_SLUG=${ARM_SLUG:-v2-l1to7w-stab-depth2}
# Extension (section 11.4): RESUME_SRC (a synced checkpoint folder) and ARM_EPOCHS.
ARM_EPOCHS=${ARM_EPOCHS:-12}; ARM_RESUME=""
CANON_HASH=""; [ -n "${CANON_HASH_CFG:-}" ] && CANON_HASH=/workspace/ai-capstone/configs/synthetic/$CANON_HASH_CFG
for i in $(seq 1 80); do
  hp=$(runpodctl pod get "$POD" 2>/dev/null | python3 -c '
import json, re, sys
s = json.load(sys.stdin).get("ssh", {})
if s.get("error"):
    sys.exit(1)
t = json.dumps(s)
ip = re.search(r"\b(\d+\.\d+\.\d+\.\d+)\b", t)
port = s.get("port") or (re.search(r"-p (\d+)", t) or [None, None])[1]
print(ip.group(1), port) if ip and port else sys.exit(1)') && break
  hp=""; sleep 15
done
[ -n "$hp" ] || { echo "ERROR no ssh for $POD: $(runpodctl pod get "$POD" 2>&1 | grep -A3 '"ssh"')"; exit 1; }
H=${hp% *}; P=${hp#* }; echo "$K $H $P"
for i in $(seq 1 30); do ssh -n -p "$P" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "root@$H" true 2>/dev/null && break; sleep 10; done
E="ssh -p $P -o StrictHostKeyChecking=accept-new"
rsync -az -e "$E" --exclude .venv --exclude .git --exclude 'rulechain-checkpoints*' --exclude results --exclude __pycache__ ./ "root@$H:/workspace/ai-capstone/" || exit 1
rsync -az -e "$E" "../$SRC_DATA" "root@$H:/workspace/" || exit 1
ssh -n -p "$P" "root@$H" "mkdir -p /workspace/$SRC_CANON/depth2 /workspace/logs" || exit 1
rsync -az -e "$E" "../$SRC_CANON/depth2/test.json" "root@$H:/workspace/$SRC_CANON/depth2/" || exit 1
if [ -n "${RESUME_SRC:-}" ]; then
  ARM_RESUME=/workspace/resume/$(basename "$RESUME_SRC")
  ssh -n -p "$P" "root@$H" "mkdir -p $ARM_RESUME" || exit 1
  rsync -az -e "$E" "$RESUME_SRC/" "root@$H:$ARM_RESUME/" || exit 1
fi
ssh -n -p "$P" "root@$H" "cd /workspace/ai-capstone && DATA=/workspace/$SRC_DATA HASH_FILE=/workspace/ai-capstone/configs/synthetic/$HASH_CFG bash scripts/pod/codi_l4to7.sh setup 2>&1 | tail -6" || exit 1
ssh -n -p "$P" "root@$H" "cd /workspace/ai-capstone && ARM_NAME=$ARM_NAME ARM_SLUG=$ARM_SLUG ARM_DATA=/workspace/$SRC_DATA ARM_CANON=/workspace/$SRC_CANON/depth2/test.json ARM_HASH=/workspace/ai-capstone/configs/synthetic/$HASH_CFG ARM_CANON_HASH=$CANON_HASH ARM_EPOCHS=$ARM_EPOCHS ARM_RESUME=$ARM_RESUME setsid nohup bash scripts/pod/codi_stab_arm.sh $K '$HW' > /workspace/logs/arm_$K.log 2>&1 < /dev/null & disown; sleep 2; pgrep -f '[c]odi_stab_arm' >/dev/null && echo arm_started" < /dev/null > >(cat) &
sleep 20; kill %1 2>/dev/null  # the ssh above can hang after detaching; it has done its job by now
echo "ARM $K:$H:$P:$POD:$RATE"
