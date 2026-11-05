"""Does CODI's precision gain sit at the end of the chain, and how many levels of reach is it worth? Usage: a20_lasthop.py <this dir> <artifacts dir>  (from repo root)"""
import json, sys, collections
sys.path.insert(0,sys.argv[1]); W=sys.argv[2]
import numpy as np
from rc import *
from scipy.stats import binomtest
D=json.load(open(W+'/b13_raw.json')); Q=D['Q']; ZS=D['ZS']; ST=D['store']
gold={r['id']:int(r['answer']) for r in D['base']}
print('1. THRESHOLD EDITS by rules left after the edited rule (0 = it is the last rule). Right branch, items BOTH models answered within 2.')
print('   wins = CODI right and CODI-0 not. p = sign test over items (net wins per item), since queries share items.')
for a,b,lab in (('stab CODI','stab CODI-0','stabilised'),('r1 CODI','r1 CODI-0','round 1')):
    print(' ##',lab)
    for ms,ml in (((1,),'margin 1'),((2,),'margin 2'),((3,4,6),'margin 3+')):
        for d,dl in ((0,'last rule'),(1,'1 rule left'),(2,'2 rules left'),(3,'3+ left')):
            w=l=0;ra=[];rb=[];net=collections.Counter()
            for q,pa,pb in zip(Q,ST[a]['q'],ST[b]['q']):
                if q['kind']!='psy' or q['m'] not in ms: continue
                dd=q['level']-q['j']
                if (dd!=d if d<3 else dd<3): continue
                i=q['item']
                if abs(ST[a]['base'][i]-gold[i])>2 or abs(ST[b]['base'][i]-gold[i])>2: continue
                right=q['hi'] if q['side']=='hi' else q['lo']
                x=abs(pa-right)<=2; y=abs(pb-right)<=2; ra.append(x); rb.append(y)
                if x and not y: w+=1; net[i]+=1
                if y and not x: l+=1; net[i]-=1
            up=sum(v>0 for v in net.values()); dn=sum(v<0 for v in net.values()); p=binomtest(up,up+dn).pvalue if up+dn else 1
            print(f'   {ml:10s} {dl:13s} n={len(ra):4d} CODI {np.mean(ra):.3f} CODI-0 {np.mean(rb):.3f} d {np.mean(ra)-np.mean(rb):+.3f} ({w}/{l}; items {up}/{dn} p={p:.4f})')
print('\n2. NATURAL TEST ITEMS with exactly one tight decision (margin <= 2), by rules left after it, L3-7. Exact wins/losses')
R='results/'; PR=W+'/preds/'
P={'stabilised':(R+'20261105-044550_codi_v2-l1to7w-stab-depth2-codi-canonical/predictions.jsonl',R+'20261105-024204_codi_v2-l1to7w-stab-depth2-codi0-canonical/predictions.jsonl'),
   'round 1':(PR+'20261102-000204_codi_v2-depth2-gpt2_depth2_codi_bs64-sel.jsonl',PR+'20261102-060537_codi_v2-depth2-gpt2_depth2_codi0.jsonl')}
def num(x):
    try: return int(str(x).strip())
    except: return -999
for lab,(pa,pb) in P.items():
    A={r['id']:r for r in map(json.loads,open(pa))}; B={r['id']:r for r in map(json.loads,open(pb))}; t=collections.defaultdict(lambda:[0,0,0])
    for i,r in A.items():
        L=int(r['level'])
        if not 3<=L<=7: continue
        x0,ru=parse(r['question']); v,br=walk(x0,ru); ti=[]
        for j,u in enumerate(ru):
            if u[0]!='P':
                s=v[j]
                if ((s-u[1]+1) if s>=u[1] else (u[1]-s))<=2: ti.append(j)
        if len(ti)!=1: continue
        d=min(L-1-ti[0],3); g=int(r['gold_answer']); ca=num(r['predicted_answer'])==g; cb=num(B[i]['predicted_answer'])==g
        t[d][0]+=1; t[d][1]+=ca and not cb; t[d][2]+=cb and not ca
    print(' ##',lab,{('last rule' if d==0 else f'{d}{"+" if d==3 else ""} left'):tuple(t[d]) for d in sorted(t)})
print('\n3. REACH on constructed chains: level where exact accuracy crosses 0.5 (interpolated). A full extra hop would be about +1.0')
def d50(acc):
    Ls=sorted(acc)
    if acc[Ls[0]]<.5: return None
    for a,b in zip(Ls,Ls[1:]):
        if acc[a]>=.5>acc[b]: return a+(acc[a]-.5)/(acc[a]-acc[b])
    return None
conds=[(False,False,'wide, no P after halving'),(False,True,'wide, P after halving'),(True,False,'tight, no P after halving'),(True,True,'tight, P after halving')]
for tg,hv,lab in conds:
    out={}
    for n in ('stab CODI-0','stab CODI','r1 CODI-0','r1 CODI'):
        acc={L:float(np.mean([p==z['gold'] for p,z in zip(ST[n]['z'],ZS) if z['level']==L and z['tight']==tg and z['halve']==hv])) for L in range(3,8)}
        out[n]=d50(acc)
    f=lambda x:'-' if x is None else f'{x:.2f}'
    dd=lambda a,b:'-' if out[a] is None or out[b] is None else f'{out[a]-out[b]:+.2f}'
    print(f"   {lab:28s} stab CODI-0 {f(out['stab CODI-0'])} CODI {f(out['stab CODI'])} ({dd('stab CODI','stab CODI-0')}) | r1 CODI-0 {f(out['r1 CODI-0'])} CODI {f(out['r1 CODI'])} ({dd('r1 CODI','r1 CODI-0')})")
