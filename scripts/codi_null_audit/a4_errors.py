import json, sys, random, collections
sys.path.insert(0,sys.argv[1])
import numpy as np
from rc import *
R='results/'
A=R+'20261105-044444_codi_v2-l1to7w-stab-depth2-codi/'
B=R+'20261105-024111_codi_v2-l1to7w-stab-depth2-codi0/'
C=R+'20261103-193425_codi_v2-l1to7w-depth2-codi/'
rd=lambda p:[json.loads(l) for l in open(p+'predictions.jsonl')]
arms={'CODI':rd(A),'CODI0':rd(B),'CODI3e-3':rd(C)}
# sanity: rule parser reproduces gold on all items
bad=0
for x in arms['CODI']:
    x0,rules=parse(x['question']); v,_=walk(x0,rules)
    bad+= str(v[-1])!=x['gold_answer']
print('parser mismatches',bad)
random.seed(0)
def rate(rows,fn,shuf=False):
    wrong=[r for r in rows if not r['correct']]
    preds=[r['predicted_answer'] for r in wrong]
    if shuf: random.shuffle(preds)
    hit=0
    for r,p in zip(wrong,preds):
        x0,rules=parse(r['question'])
        try: p=int(p)
        except: continue
        hit+= p in set(fn(x0,rules))
    return hit/max(1,len(wrong)),len(wrong)
print('share of WRONG answers explained (obs / chance-by-shuffling-within-level)')
for name,rows in arms.items():
    for L in (4,5,6,7):
        rr=[r for r in rows if r['level']==L]
        out=[]
        for fn in (flip_outcomes,path_flip_outcomes,skip_outcomes):
            o,n=rate(rr,fn); c=np.mean([rate(rr,fn,True)[0] for _ in range(20)])
            out.append(f'{o:.3f}/{c:.3f}')
        # abs error distribution
        err=[abs(int(r['predicted_answer'])-int(r['gold_answer'])) for r in rr if not r['correct'] and r['predicted_answer'].isdigit()]
        print(f'{name:9s} L{L} wrong={n:3d} flip {out[0]} pathflip {out[1]} skip {out[2]} | median|err|={np.median(err):.0f} within5={np.mean(np.array(err)<=5):.2f}')
# per-rule location of the flip
print('which rule flip explains the wrong answer (fraction of wrong items, first match)')
for name,rows in arms.items():
    for L in (5,7):
        cnt=collections.Counter(); wrong=[r for r in rows if r['level']==L and not r['correct']]
        for r in wrong:
            x0,rules=parse(r['question'])
            p=r['predicted_answer']
            if not p.isdigit(): continue
            for j,o in enumerate(flip_outcomes(x0,rules)):
                if o==int(p): cnt[j+1]+=1
        print(name,'L',L,[round(cnt[j]/len(wrong),3) for j in range(1,L+1)])
