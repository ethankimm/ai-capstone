import json, re, collections, sys
import numpy as np
base='/Users/henninglindig/Documents/Penn/CIS5980/'
runs={
 'stab CODI lr2e-3': 'ai-capstone/results/20261105-044444_codi_v2-l1to7w-stab-depth2-codi/train_log.json',
 'stab CODI-0 lr2e-3': 'ai-capstone/results/20261105-024111_codi_v2-l1to7w-stab-depth2-codi0/train_log.json',
 'lr3e-3 CODI (L1-7w)': 'ai-capstone/results/20261103-193425_codi_v2-l1to7w-depth2-codi/train_log.json',
}
import os
for k,p in runs.items():
    p=base+p
    if not os.path.exists(p): print('missing',k); continue
    lh=json.load(open(p))
    ser=collections.defaultdict(list)
    for r in lh:
        for key in ('ce_loss','distill_loss','ref_ce_loss','loss','grad_norm'):
            if key in r and 'step' in r: ser[key].append((r['step'],r[key]))
    print('##',k,{kk:len(v) for kk,v in ser.items()})
    print('epoch |  ce    distill  ref_ce  | grad')
    for e in range(12):
        lo,hi=e*1485,(e+1)*1485
        row=[]
        for key in ('ce_loss','distill_loss','ref_ce_loss','grad_norm'):
            v=[x for s,x in ser[key] if lo<s<=hi]
            row.append(np.mean(v) if v else float('nan'))
        print(f'{e+1:3d}  | {row[0]:.4f} {row[1]:.4f} {row[2]:.5f} | {row[3]:.3f}')
