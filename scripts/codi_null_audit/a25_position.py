"""Position test on the start-last pair (TRAINING_PROTOCOL section 11.3), as a20_lasthop.py. Usage: a25_position.py <this dir> <artifacts dir>  (from the repo root)"""
import json, sys, collections, math
sys.path.insert(0,sys.argv[1]); W=sys.argv[2]
import numpy as np
from rc import *
def bp(k,n):
    if not n: return 1.0
    k=min(k,n-k); return min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/2**n)
D=json.load(open(W+'/b13_raw_startlast.json')); Q=D['Q']; ZS=D['ZS']; ST=D['store']; a,b='startlast CODI','startlast CODI-0'
gold={r['id']:int(r['answer']) for r in D['base']}
print('1. THRESHOLD EDITS by rules left after the edited rule. Right branch, items BOTH models answered within 2. wins = CODI right and CODI-0 not; p = sign test over items.')
for ms,ml in (((1,),'margin 1'),((2,),'margin 2'),((3,4,6),'margin 3+')):
    for d,dl in ((0,'last rule'),(1,'1 rule left'),(2,'2 rules left'),(3,'3+ left')):
        w=l=0;ra=[];rb=[];net=collections.Counter()
        for q,pa,pb in zip(Q,ST[a]['q'],ST[b]['q']):
            if q['kind']!='psy' or q['m'] not in ms: continue
            dd=q['level']-q['j']
            if (dd!=d if d<3 else dd<3): continue
            i=q['item']
            if abs(ST[a]['base'][i]-gold[i])>2 or abs(ST[b]['base'][i]-gold[i])>2: continue
            right=q['hi'] if q['side']=='hi' else q['lo']; x=abs(pa-right)<=2; y=abs(pb-right)<=2; ra.append(x); rb.append(y)
            if x and not y: w+=1; net[i]+=1
            if y and not x: l+=1; net[i]-=1
        up=sum(v>0 for v in net.values()); dn=sum(v<0 for v in net.values())
        if ra: print(f'   {ml:10s} {dl:13s} n={len(ra):4d} CODI {np.mean(ra):.3f} CODI-0 {np.mean(rb):.3f} d {np.mean(ra)-np.mean(rb):+.3f} ({w}/{l}; items {up}/{dn} p={bp(up,up+dn):.3g})')
print('\n2. NATURAL TEST ITEMS with exactly one tight decision (margin <= 2), by rules left after it, L3-7. Exact wins/losses')
R='results/'; pa=R+'20261106-083843_codi_v2-l1to7w-startlast-depth2-codi/predictions.jsonl'; pb=R+'20261106-052644_codi_v2-l1to7w-startlast-depth2-codi0/predictions.jsonl'
num=lambda x:int(str(x).strip()) if str(x).strip().lstrip('-').isdigit() else -999
Aa={r['id']:r for r in map(json.loads,open(pa))}; Bb={r['id']:r for r in map(json.loads,open(pb))}; t=collections.defaultdict(lambda:[0,0,0])
for i,r in Aa.items():
    L=int(r['level'])
    if not 3<=L<=7: continue
    x0,ru=parse(r['question']); v,br=walk(x0,ru); ti=[]
    for j,u in enumerate(ru):
        if u[0]!='P':
            s=v[j]
            if ((s-u[1]+1) if s>=u[1] else (u[1]-s))<=2: ti.append(j)
    if len(ti)!=1: continue
    d=min(L-1-ti[0],3); g=int(r['gold_answer']); ca=num(r['predicted_answer'])==g; cb=num(Bb[i]['predicted_answer'])==g
    t[d][0]+=1; t[d][1]+=ca and not cb; t[d][2]+=cb and not ca
print('  ',{('last rule' if d==0 else f'{d}{"+" if d==3 else ""} left'):tuple(t[d])+(round(bp(t[d][1],t[d][1]+t[d][2]),3),) for d in sorted(t)})
print('\n3. CONSTRUCTED CHAINS, exact CODI/CODI-0 by level (no P after halving)')
for tg,lab in ((False,'wide'),(True,'tight')):
    r=[]
    for L in range(3,8):
        ii=[k for k,z in enumerate(ZS) if z['level']==L and z['tight']==tg and not z['halve']]
        r.append(f"L{L} {np.mean([ST[a]['z'][k]==ZS[k]['gold'] for k in ii]):.2f}/{np.mean([ST[b]['z'][k]==ZS[k]['gold'] for k in ii]):.2f}")
    print(f'   {lab:6s}',' '.join(r))
