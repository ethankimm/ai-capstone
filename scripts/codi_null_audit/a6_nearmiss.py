import json, sys, random, glob, os
import numpy as np
R='results/'
runs={
 'CODI stab':'20261105-044444_codi_v2-l1to7w-stab-depth2-codi',
 'CODI-0 stab':'20261105-024111_codi_v2-l1to7w-stab-depth2-codi0',
 'CODI lr3e-3 L1-7w':'20261103-193425_codi_v2-l1to7w-depth2-codi',
}
for name,pat in [('CODI r1','*codi_bs64-sel'),('CODI-0 r1','*v2-depth2-gpt2_depth2_codi0'),('no-CoT plateau','*no_cot_plateau'),('no_thoughts','*gpt2_depth2_no_thoughts_m6'),('Coconut c1m6','*gpt2_depth2_coconut_c1m6')]:
    g=[x for x in glob.glob(R+pat) if os.path.exists(x+'/predictions.jsonl')]
    if g: runs[name]=os.path.basename(sorted(g)[-1])
rng=np.random.default_rng(0)
print(f"{'run':20s} L  acc  | wrong: |e|<=1 <=2 <=5 <=10  med | shuffled-chance <=2 <=5 med | tens-ok units-ok parity-ok(chance .5)")
for name,d in runs.items():
    p=R+d+'/predictions.jsonl'
    if not os.path.exists(p): print('missing',name,d); continue
    rows=[json.loads(l) for l in open(p)]
    for L in (3,4,5,6,7):
        rr=[r for r in rows if r['level']==L]
        if not rr: continue
        acc=np.mean([r['correct'] for r in rr])
        w=[(int(r['predicted_answer']),int(r['gold_answer'])) for r in rr if not r['correct'] and str(r['predicted_answer']).lstrip('-').isdigit()]
        if not w: continue
        pr=np.array([a for a,b in w]); go=np.array([b for a,b in w]); e=np.abs(pr-go)
        sh=[]
        allg=np.array([int(r['gold_answer']) for r in rr])
        for _ in range(50):
            q=rng.permutation(pr); ee=np.abs(q-go); ee=ee[ee>0]; sh.append((np.mean(ee<=2),np.mean(ee<=5),np.median(ee)))
        sh=np.mean(sh,0)
        print(f"{name:20s} {L} {acc:.3f} | {np.mean(e<=1):.2f} {np.mean(e<=2):.2f} {np.mean(e<=5):.2f} {np.mean(e<=10):.2f} {np.median(e):4.1f} | {sh[0]:.2f} {sh[1]:.2f} {sh[2]:4.1f} | {np.mean(pr//10==go//10):.2f} {np.mean(pr%10==go%10):.2f} {np.mean(pr%2==go%2):.2f}")
