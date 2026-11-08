#!/bin/sh
# t22/t24 queue: one model job at a time; smoke before each full job; stop on any failure
A=/Users/henninglindig/Documents/Penn/CIS5980/codi_audit_artifacts; R=/Users/henninglindig/Documents/Penn/CIS5980/ai-capstone; S=$R/scripts/necessity/sim_gsm8k.py
until grep -q '^rc=' $A/t22/sweep_sim.log; do sleep 20; done
grep -q '^rc=0' $A/t22/sweep_sim.log || { echo QUEUE_FAIL sweep_sim; exit 1; }
echo STEP sweep_sim done
sh $A/run_guarded.sh $A/t22/sweep_codi.log $S $R sweep codi mps $A/t22 || { echo QUEUE_FAIL sweep_codi; exit 1; }; echo STEP sweep_codi done
SMOKE=1 sh $A/run_guarded.sh $A/t22/route_sim_smoke.log $S $R route sim mps $A/t22 || { echo QUEUE_FAIL route_sim_smoke; exit 1; }; echo STEP route_sim_smoke done
sh $A/run_guarded.sh $A/t22/route_sim.log $S $R route sim mps $A/t22 || { echo QUEUE_FAIL route_sim; exit 1; }; echo STEP route_sim done
sh $A/run_guarded.sh $A/t22/route_codi.log $S $R route codi mps $A/t22 || { echo QUEUE_FAIL route_codi; exit 1; }; echo STEP route_codi done
for M in r1 lr3; do
  MODEL=$M SMOKE=1 sh $A/run_guarded.sh $A/t24_${M}_smoke.log $A/t24_samelevel_mean.py $R/scripts/codi_null_audit mps $A || { echo QUEUE_FAIL t24_${M}_smoke; exit 1; }
  MODEL=$M sh $A/run_guarded.sh $A/t24_${M}.log $A/t24_samelevel_mean.py $R/scripts/codi_null_audit mps $A || { echo QUEUE_FAIL t24_$M; exit 1; }; echo STEP t24_$M done
done
echo QUEUE_DONE
