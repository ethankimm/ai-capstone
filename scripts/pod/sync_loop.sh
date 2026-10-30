#!/usr/bin/env bash
# Laptop-side: while a long job runs, every INTERVAL seconds ask the pod to write fresh checkpoint checksums
# (coconut_v2.sh manifest) and pull + verify everything, so a vanished pod costs at most one interval of work
# (checkpoints are synced as they appear; the v1 Coconut A-lite checkpoint was lost with its pod).
#   WATCH_LOGS="/workspace/logs/gpt2_depth2_coconut_c1m6.log" bash scripts/pod/sync_loop.sh <pod_ip> <ssh_port> [interval_s=900]
#   stop: touch /tmp/stop_sync_<ip>
# Each pod syncs into its own folder, $DEST_ROOT/<ip>_<port> (one SHA256SUMS list per pod). A failed sync is retried next
# interval; after 3 failures in a row it alerts loudly (and keeps trying) instead of exiting, so protection never silently
# stops. WATCH_LOGS (pod paths, space-separated): alert once when a log shows a crash (Traceback / out of memory / CUDA
# error) or a finish marker (CODI_TRAIN_DONE, or the plateau watcher stopping), so a dead or finished
# pod is not left billing overnight. Alerts print and, on macOS, raise a notification. Run it with `run_in_background`.
set -uo pipefail
HOST=${1:?pod ip}; PORT=${2:?ssh port}; INTERVAL=${3:-900}
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST_ROOT=${DEST_ROOT:-$HOME/Documents/Penn/CIS5980/synthetic_checkpoints/pod_pull}
DEST="$DEST_ROOT/${HOST}_${PORT}"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
WATCH_LOGS=${WATCH_LOGS:-}
fails=0
MARK=/tmp/sync_alerts_${HOST}_${PORT}; mkdir -p "$MARK"   # one marker file per alert already raised (bash 3.2: no assoc arrays)
alert() {
  echo "!!!! ALERT $(date -u +%FT%TZ) [$HOST]: $*"
  command -v osascript >/dev/null && osascript -e "display notification \"$*\" with title \"pod $HOST\"" >/dev/null 2>&1 || true
}
while [ ! -e "/tmp/stop_sync_$HOST" ]; do
  echo "== $(date -u +%FT%TZ) sync $HOST:$PORT -> $DEST"
  $SSH "root@$HOST" "cd /workspace/ai-capstone && bash scripts/pod/coconut_v2.sh manifest" || echo "manifest failed (pod busy/down?)"
  if bash "$HERE/pull_artifacts.sh" "$HOST" "$PORT" "$DEST"; then
    fails=0
  else
    fails=$((fails + 1)); echo "sync/verify failed ($fails in a row)"
    [ $fails -ge 3 ] && alert "sync failed $fails times in a row -- checkpoints are NOT being protected"
  fi
  for log in $WATCH_LOGS; do
    tail_txt=$($SSH "root@$HOST" "tail -c 20000 $log 2>/dev/null" || true)
    if echo "$tail_txt" | grep -qE "Traceback|out of memory|CUDA error"; then
      m="$MARK/$(basename "$log").crash"; [ -e "$m" ] || { alert "CRASH in $log"; touch "$m"; }
    elif echo "$tail_txt" | grep -qE "CODI_TRAIN_DONE|watch: stopping|watch: process exited"; then
      m="$MARK/$(basename "$log").done"; [ -e "$m" ] || { alert "FINISHED: $log -- run select/test, then stop the pod"; touch "$m"; }
    fi
  done
  sleep "$INTERVAL"
done
echo "stopped"
