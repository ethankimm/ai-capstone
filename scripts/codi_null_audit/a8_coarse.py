import json, sys, glob, os
sys.path.insert(0,sys.argv[1]); S=sys.argv[1]
import numpy as np
from rc import *
R='results/'
files={
 'CODI stab':R+'20261105-044550_codi_v2-l1to7w-stab-depth2-codi-canonical/predictions.jsonl',
 'CODI-0 stab':R+'20261105-024204_codi_v2-l1to7w-stab-depth2-codi0-canonical/predictions.jsonl',
 'CODI 3e-3 L1-7w':R+'20261103-193425_codi_v2-l1to7w-depth2-codi/predictions.jsonl',
 'CODI r1':S+'/preds/20261102-000204_codi_v2-depth2-gpt2_depth2_codi_bs64-sel.jsonl',
 'CODI-0 r1':S+'/preds/20261102-060537_codi_v2-depth2-gpt2_depth2_codi0.jsonl',
 'no-CoT plateau':S+'/preds/20261101-154024_baseline_no_cot_v2-depth2-gpt2_depth2_no_cot_plateau.jsonl',
 'no_thoughts':S+'/preds/20261101-153922_baseline_no_cot_v2-depth2-gpt2_depth2_no_thoughts_m6.jsonl',
 'Coconut c1m6':S+'/preds/20261102-022150_coconut_v2-depth2-gpt2_depth2_coconut_c1m6.jsonl',
 'SIM-Coconut':S+'/preds/20261104-144235_coconut_v2-depth2-pilot_k8_simcot_c1m6.jsonl',
}
rng=np.random.default_rng(0)
def num(x):
    try: return int(str(x).strip())
    except: return None
print('Per level: exact acc / P(|pred-gold|<=2) / chance(<=2, shuffled within level) / P(<=5) / chance(<=5)')
tab={}
for name,p in files.items():
    if not os.path.exists(p): print('missing',name); continue
    rows=[json.loads(l) for l in open(p)]
    Ls=sorted({r['level'] for r in rows})
    print('##',name,'n',len(rows))
    for L in Ls:
        rr=[r for r in rows if r['level']==L]
        pr=np.array([num(r['predicted_answer']) if num(r['predicted_answer']) is not None else -999 for r in rr]); go=np.array([int(r['gold_answer']) for r in rr])
        e=np.abs(pr-go)
        ch2=np.mean([np.mean(np.abs(rng.permutation(pr)-go)<=2) for _ in range(30)])
        ch5=np.mean([np.mean(np.abs(rng.permutation(pr)-go)<=5) for _ in range(30)])
        print(f'  L{L:2d} exact {np.mean(e==0):.3f} | <=2 {np.mean(e<=2):.3f} (ch {ch2:.3f}) | <=5 {np.mean(e<=5):.3f} (ch {ch5:.3f}) | near-miss share of wrong within2 {np.mean(e[e>0]<=2) if (e>0).any() else 0:.2f}')
