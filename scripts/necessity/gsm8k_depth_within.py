import os, sys, json, numpy as np
from collections import defaultdict
sys.argv=['x','/dev/null']
src=open('scripts/necessity/gsm8k_depth.py').read().split("dep = np.array")[0]
for MODE in ('strict','loose'):
    os.environ['MODE']=MODE; ns={}; exec(src, ns)
    INFO, BYQ, A, HOPF, qtext, down = ns['INFO'], ns['BYQ'], ns['A'], None, ns['qtext'], ns['down']
    HOPF=dict(codi=['t31/gsm8k_codi.json','t31/long_gsm8k_codi.json'], sim=['t31/gsm8k_sim.json','t31/long_gsm8k_sim.json'],
              coconut=['t31/gsm8k_coconut.json','t31/long_gsm8k_coconut.json'], simcoconut=['t36/gsm8k_simcoconut.json','t36/long_gsm8k_simcoconut.json'])
    rng=np.random.default_rng(0); out={}
    for m,fs in HOPF.items():
        by=defaultdict(list)
        for f in fs:
            for p in json.load(open(A/f))['pairs']:
                if not p.get('qualified') or p.get('hops') is None: continue
                i=BYQ.get(qtext(p['R']['question']));
                if i is None: continue
                sd=down(INFO[i][MODE], int(p['edit'][4:]))
                if sd is not None: by[i].append((sd, int(p['s']), float(p['hops'])))
        G=[np.array(v) for v in by.values() if len({x[0] for x in v})>1]
        def slope(Gs, col):
            x=np.concatenate([g[:,col]-g[:,col].mean() for g in Gs]); y=np.concatenate([g[:,2]-g[:,2].mean() for g in Gs]); return (x*y).sum()/(x*x).sum()
        def ci(col):
            bs=[slope([G[j] for j in rng.integers(0,len(G),len(G))], col) for _ in range(1000)]; return np.percentile(bs,[2.5,97.5])
        b=slope(G,0); lo,hi=ci(0)
        # serial vs index, both within item: residual of serial after index
        Gi=[np.array(v) for v in by.values() if len({x[1] for x in v})>1]
        print(f"{MODE:6s} {m:10s} items with >1 serial distance {len(G):4d} | within-item hops per serial step {b:+.3f} [{lo:+.3f}, {hi:+.3f}] | per index step {slope(Gi,1):+.3f}")
