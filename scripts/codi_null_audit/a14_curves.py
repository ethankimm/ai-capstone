import json,sys,collections
import numpy as np
S=sys.argv[1]
def series(lh):
    out=collections.defaultdict(list)
    for r in lh:
        for k in ('ce_loss','distill_loss','ref_ce_loss','loss'):
            if k in r and 'step' in r: out[k].append((r['step'],r[k]))
    return out
stu=json.load(open(S+'/student_log.json'))
stu=stu['log_history'] if isinstance(stu,dict) else stu
c0=json.load(open('rulechain-checkpoints/gpt2_depth2_codi0/checkpoint-29716/trainer_state.json'))['log_history']
cd=json.load(open('rulechain-checkpoints/gpt2_depth2_codi_bs64/checkpoint-31280/trainer_state.json'))['log_history']
print('keys',{k:len(v) for k,v in series(stu).items()},{k:len(v) for k,v in series(c0).items()})
A,B,C=series(stu),series(c0),series(cd)
spe=782
print('Round 1, same stack, lr 3e-3, 782 steps/epoch. Student CE by epoch (mean over logged steps)')
print('ep  student-only  CODI-0   CODI | CODI-0 teacher CE  distill')
for e in range(0,10):
    lo,hi=e*spe,(e+1)*spe
    m=lambda s:(np.mean([x for t,x in s if lo<t<=hi]) if any(lo<t<=hi for t,x in s) else float('nan'))
    print(f'{e+1:2d}  {m(A["ce_loss"]):.4f}   {m(B["ce_loss"]):.4f}  {m(C["ce_loss"]):.4f} | {m(B["ref_ce_loss"]):.4f}  {m(B["distill_loss"]):.4f}')
