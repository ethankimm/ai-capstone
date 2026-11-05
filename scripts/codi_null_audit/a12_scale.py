import json, sys
sys.path.insert(0,sys.argv[1])
import numpy as np
from rc import *
R='results/'
B=[json.loads(l) for l in open(R+'20261105-024111_codi_v2-l1to7w-stab-depth2-codi0/predictions.jsonl')]
A=[json.loads(l) for l in open(R+'20261105-044444_codi_v2-l1to7w-stab-depth2-codi/predictions.jsonl')]
def info(r):
    x0,rules=parse(r['question']); v,b=walk(x0,rules)
    dbl=[j for j,(ru,br) in enumerate(zip(rules,b)) if ru[0]=='D' and br]
    hal=[j for j,(ru,br) in enumerate(zip(rules,b)) if ru[0]=='P' and br]
    return len(dbl),len(hal),(dbl[-1] if dbl else -1),len(rules)
print('If errors were a value offset scaled by later doubling, |e|=2,4 would need a doubling. Among L5-7 near-misses (|e| in 1,2,4):')
for name,rows in (('CODI-0',B),('CODI',A)):
    for key,lab in ((0,'no doubling taken'),(1,'>=1 doubling taken')):
        rr=[r for r in rows if 5<=r['level']<=7 and not r['correct'] and (info(r)[0]>0)==bool(key)]
        e=np.array([abs(int(r['predicted_answer'])-int(r['gold_answer'])) for r in rr])
        print(f' {name:7s} {lab:20s} n={len(rr)} |e|=1 {np.mean(e==1):.3f} |e|=2 {np.mean(e==2):.3f} |e|=4 {np.mean(e==4):.3f} |e|=3 {np.mean(e==3):.3f}')
print('Gold parity / last-op split among near-misses: does the error depend on the LAST rule taken?')
for name,rows in (('CODI-0',B),('CODI',A)):
    for kind in 'PTD':
        rr=[r for r in rows if 5<=r['level']<=7 and parse(r['question'])[1][-1][0]==kind]
        e=np.array([abs(int(r['predicted_answer'])-int(r['gold_answer'])) for r in rr])
        print(f' {name:7s} last rule {kind} n={len(rr)} exact {np.mean(e==0):.3f} close {np.mean(e<=2):.3f}')
