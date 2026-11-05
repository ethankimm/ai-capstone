import json, sys, collections
sys.path.insert(0,sys.argv[1])
import numpy as np
from rc import *
R='results/'
rows={k:[json.loads(l) for l in open(R+v+'/predictions.jsonl')] for k,v in {'CODI':'20261105-044444_codi_v2-l1to7w-stab-depth2-codi','CODI0':'20261105-024111_codi_v2-l1to7w-stab-depth2-codi0'}.items()}
for name,rr in rows.items():
    for L in (3,5,7):
        c=collections.Counter()
        for r in rr:
            if r['level']==L and not r['correct']:
                c[int(r['predicted_answer'])-int(r['gold_answer'])]+=1
        tot=sum(c.values())
        print(name,L,'n',tot,sorted(c.items(),key=lambda kv:-kv[1])[:14])
print()
k=0
for r in rows['CODI']:
    if r['level']==3 and not r['correct']:
        x0,rules=parse(r['question']); v,b=walk(x0,rules)
        print(r['question'][:190]); print('   walk',v,'branch',b,'pred',r['predicted_answer'],'kinds',[x[0] for x in rules])
        k+=1
        if k>=14: break
