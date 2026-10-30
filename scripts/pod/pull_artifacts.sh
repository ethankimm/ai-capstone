#!/usr/bin/env bash
# Pull checkpoints + RunRecords off a RunPod pod and verify them. Run on the LAPTOP, after the pod-side manifest
# (`bash scripts/pod/coconut_v2.sh manifest`, or `pilot_v2.sh manifest` for synthetic_runs only). Do NOT delete the pod
# until this exits 0 (the v1 Coconut A-lite checkpoint was lost to exactly that).
#   bash scripts/pod/pull_artifacts.sh <pod_ip> <ssh_port> [dest=~/Documents/Penn/CIS5980/synthetic_checkpoints/pod_pull]
# Use ONE dest per pod (sync_loop.sh does: pod_pull/<ip>_<port>): each root's SHA256SUMS is per pod, so two pods syncing
# into the same folder overwrite each other's lists.
# Roots pulled and verified (each against its OWN SHA256SUMS written on the pod, which lists every checkpoint file --
# including extensionless Coconut checkpoint_N -- older than 2 min): ROOTS="/workspace/synthetic_runs /workspace/coconut_runs
# /workspace/codi_runs" by default; set ROOTS to override. A root that does not exist on the pod is skipped; a root that exists
# but has no SHA256SUMS FAILS (run the manifest first). Use the DIRECT SSH ip/port from the RunPod console, not the proxy.
# results/ is pulled with --ignore-existing, so a locally edited notes.md (or any local file) is never overwritten.
set -euo pipefail
HOST=${1:?pod ip}; PORT=${2:?ssh port}
DEST=${3:-$HOME/Documents/Penn/CIS5980/synthetic_checkpoints/pod_pull}
ROOTS=${ROOTS:-"/workspace/synthetic_runs /workspace/coconut_runs /workspace/codi_runs"}
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$DEST"
fail=0
# verify_sums <sums_file> : GNU sha256sum on Linux; the macOS sha256sum has no --quiet, so fall back to shasum. An empty
# list (nothing checkpointed yet) verifies trivially.
verify_sums() {
  [ -s "$1" ] || return 0
  if sha256sum -c --quiet /dev/null >/dev/null 2>&1; then sha256sum -c --quiet "$1"; else shasum -a 256 -c -q "$1"; fi
}
for r in $ROOTS; do
  if ! $SSH "root@$HOST" "[ -d $r ]"; then echo "skip $r (not on pod)"; continue; fi
  local_dir="$DEST/$(basename "$r")"
  # optimizer.pt / scheduler.pt are pulled too (CODI): without them a resume from this copy restarts the cosine schedule
  rsync -a --partial -e "$SSH" "root@$HOST:$r/" "$local_dir/"
  if [ ! -f "$local_dir/SHA256SUMS" ]; then echo "FAIL $r: no SHA256SUMS (run the pod-side manifest first)"; fail=1; continue; fi
  if (cd "$local_dir" && verify_sums SHA256SUMS); then echo "checkpoints OK: $r ($(wc -l < "$local_dir/SHA256SUMS") files)"; else echo "FAIL $r: checksum mismatch"; fail=1; fi
done
rsync -a --ignore-existing -e "$SSH" "root@$HOST:/workspace/ai-capstone/results/" "$HERE/results/"
for f in SHA256SUMS_results; do
  src=$(for r in $ROOTS; do echo "$DEST/$(basename "$r")/$f"; done | while read -r p; do [ -f "$p" ] && echo "$p" && break; done || true)
  [ -n "${src:-}" ] || { $SSH "root@$HOST" "ls /workspace/logs/SHA256SUMS_results" >/dev/null 2>&1 && rsync -a -e "$SSH" "root@$HOST:/workspace/logs/SHA256SUMS_results" "$DEST/" && src="$DEST/$f"; }
  if [ -n "${src:-}" ]; then
    # notes.md is edited locally on purpose; everything else under results/ must match the pod
    grep -v 'notes.md$' "$src" > "$DEST/.results_sums" || true
    if (cd "$HERE" && verify_sums "$DEST/.results_sums"); then echo "results OK"; else echo "FAIL results checksum"; fail=1; fi
  else echo "WARN no SHA256SUMS_results found (results not verified)"; fi
done
[ $fail -eq 0 ] || { echo "NOT verified -- do NOT terminate the pod"; exit 1; }
echo "verified -- now: rebuild index (scripts/rebuild_index.py), commit results/, then the pod may be terminated."
