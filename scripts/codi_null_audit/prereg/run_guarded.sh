#!/bin/sh
# usage: run_guarded.sh <log> <python args...>; one job, MPS capped, killed if swap passes the limit
LOG=$1; shift
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 OMP_NUM_THREADS=6 PYTORCH_ENABLE_MPS_FALLBACK=1
cd /Users/henninglindig/Documents/Penn/CIS5980/codi
nice -n 5 .venv/bin/python "$@" > $LOG 2>&1 &
P=$!
while kill -0 $P 2>/dev/null; do
  u=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9]+).*/\1/')
  if [ "$u" -gt 14000 ]; then echo "swap ${u}M: stopping" >> $LOG; kill $P; exit 9; fi
  sleep 3
done
wait $P; rc=$?; echo "rc=$rc" >> $LOG; exit $rc
