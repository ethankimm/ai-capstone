#!/usr/bin/env bash
# Laptop-side: wait for a pod's SSH, sync repo + data, run CODI setup, start scripts/pod/codi_stab_arm.sh detached.
# Prints "ARM kind:host:port:pod_id:rate" on success, the spec scripts/pod/pair_watcher.py takes.
#   bash scripts/pod/launch_stab_arm.sh codi|codi0 <pod_id> <rate $/h> "RunPod A40 48GB"
set -uo pipefail
K=${1:?kind}; POD=${2:?pod id}; RATE=${3:?rate}; HW=${4:?hardware label}
cd "$(dirname "$0")/../.."
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
rsync -az -e "$E" ../synthetic_data_v2_l1to7w "root@$H:/workspace/" || exit 1
ssh -n -p "$P" "root@$H" 'mkdir -p /workspace/synthetic_data_v2/depth2 /workspace/logs' || exit 1
rsync -az -e "$E" ../synthetic_data_v2/depth2/test.json "root@$H:/workspace/synthetic_data_v2/depth2/" || exit 1
ssh -n -p "$P" "root@$H" 'cd /workspace/ai-capstone && DATA=/workspace/synthetic_data_v2_l1to7w HASH_FILE=/workspace/ai-capstone/configs/synthetic/v2_l1to7w_data_sha256.json bash scripts/pod/codi_l4to7.sh setup 2>&1 | tail -6' || exit 1
ssh -n -p "$P" "root@$H" "cd /workspace/ai-capstone && setsid nohup bash scripts/pod/codi_stab_arm.sh $K '$HW' > /workspace/logs/arm_$K.log 2>&1 < /dev/null & disown; sleep 2; pgrep -f '[c]odi_stab_arm' >/dev/null && echo arm_started" < /dev/null > >(cat) &
sleep 20; kill %1 2>/dev/null  # the ssh above can hang after detaching; it has done its job by now
echo "ARM $K:$H:$P:$POD:$RATE"
