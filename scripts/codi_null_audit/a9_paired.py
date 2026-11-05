import json, sys, collections
sys.path.insert(0,sys.argv[1]); S=sys.argv[1]
import numpy as np
from scipy import stats
from rc import *
R='results/'
A=[json.loads(l) for l in open(R+'20261105-044444_codi_v2-l1to7w-stab-depth2-codi/predictions.jsonl')]
B=[json.loads(l) for l in open(R+'20261105-024111_codi_v2-l1to7w-stab-depth2-codi0/predictions.jsonl')]
C=[json.loads(l) for l in open(R+'20261103-193425_codi_v2-l1to7w-depth2-codi/predictions.jsonl')]
SIM={json.loads(l)['id']:json.loads(l) for l in open(S+'/preds/20261104-144235_coconut_v2-depth2-pilot_k8_simcot_c1m6.jsonl')}
assert [x['id'] for x in A]==[x['id'] for x in B]
def mc(a,b):
    n10=int(np.sum(a&~b)); n01=int(np.sum(b&~a)); n=n10+n01
    return n10,n01,(1.0 if n==0 else min(1.,2*stats.binom.cdf(min(n10,n01),n,.5)))
def e(r): return abs(int(r['predicted_answer'])-int(r['gold_answer']))
print('== paired CODI vs CODI-0 by metric (EXPLORATORY for the non-exact rows)')
for lo,hi in ((5,7),(5,6),(7,7),(3,4)):
    for name,th in (('exact',0),('<=2',2),('<=5',5),('<=10',10)):
        a=np.array([e(x)<=th for x in A if lo<=x['level']<=hi]); b=np.array([e(x)<=th for x in B if lo<=x['level']<=hi])
        n10,n01,p=mc(a,b); n=len(a); d=(n10-n01)/n; se=np.sqrt((n10+n01)/n**2-d*d/n)
        print(f'L{lo}-{hi} {name:5s} CODI {a.mean():.3f} CODI-0 {b.mean():.3f} G {d:+.3f} [{d-1.96*se:+.3f},{d+1.96*se:+.3f}] {n10}/{n01} p={p:.3f}')
print('\n== decomposition: exact = P(close) * P(exact|close); CODI-0, CODI, SIM-Coconut (same items)')
for L in (3,4,5,6,7):
    ids=[x['id'] for x in A if x['level']==L]
    for name,rows in (('CODI-0',{x['id']:x for x in B}),('CODI',{x['id']:x for x in A}),('CODI3e-3',{x['id']:x for x in C}),('SIM',SIM)):
        rr=[rows[i] for i in ids if i in rows]
        err=np.array([abs((int(r['predicted_answer']) if str(r['predicted_answer']).lstrip('-').isdigit() else -999)-int(r.get('gold_answer',r.get('answer')))) for r in rr])
        close=np.mean(err<=2); ex=np.mean(err==0)
        print(f'L{L} {name:9s} n={len(rr)} exact {ex:.3f} close(<=2) {close:.3f} exact|close {ex/close:.3f} far {1-close:.3f}')
print('\n== residue structure of WRONG answers, L5-7 (chance: mod2 .5, mod5 .2 excluding exact gives .19ish)')
for name,rows in (('CODI',A),('CODI-0',B),('CODI3e-3',C)):
    w=[r for r in rows if 5<=r['level']<=7 and not r['correct']]
    p=np.array([int(r['predicted_answer']) for r in w]); g=np.array([int(r['gold_answer']) for r in w])
    d=p-g
    print(name,'n',len(w),f'mod2 {np.mean(d%2==0):.3f} mod4 {np.mean(d%4==0):.3f} mod5 {np.mean(d%5==0):.3f} mod10 {np.mean(d%10==0):.3f} |d|in{{1,2,4,8}} {np.mean(np.isin(np.abs(d),[1,2,4,8])):.3f}')
    near=[r for r in w if abs(int(r['predicted_answer'])-int(r['gold_answer']))<=2]
    fl=0
    for r in near:
        x0,rules=parse(r['question'])
        fl+= int(r['predicted_answer']) in flip_outcomes(x0,rules)
    print(f'   near-miss (<=2) n={len(near)}; explained by one branch flip {fl/max(1,len(near)):.3f}')
print('\n== what drives failure? L5-7 logistic-style table: exact & close by # of taken op type (CODI-0 / CODI)')
def feats(r):
    x0,rules=parse(r['question']); v,b=walk(x0,rules)
    half=sum(1 for ru,br in zip(rules,b) if ru[0]=='P' and br)
    dbl=sum(1 for ru,br in zip(rules,b) if ru[0]=='D' and br)
    nP=sum(ru[0]=='P' for ru in rules)
    # min margin to a threshold over T/D rules
    mar=[abs(v[j]-ru[1]) for j,ru in enumerate(rules) if ru[0] in 'TD']
    return dict(half=half,dbl=dbl,nP=nP,margin=min(mar) if mar else 99,odd_halfable=0)
for key in ('nP','half','dbl'):
    print(key)
    for val in range(0,6):
        out=[]
        for name,rows in (('CODI-0',B),('CODI',A)):
            rr=[r for r in rows if 5<=r['level']<=7 and feats(r)[key]==val]
            if len(rr)<30: out.append(f'{name} n<30'); continue
            er=np.array([e(r) for r in rr]); out.append(f'{name} n={len(rr)} exact {np.mean(er==0):.3f} close {np.mean(er<=2):.3f}')
        print('  ',val,' | '.join(out))
print('margin (min |s - threshold| over T/D rules)')
for lo,hi in ((0,1),(2,3),(4,7),(8,15),(16,99)):
    out=[]
    for name,rows in (('CODI-0',B),('CODI',A)):
        rr=[r for r in rows if 5<=r['level']<=7 and lo<=feats(r)['margin']<=hi]
        er=np.array([e(r) for r in rr]); out.append(f'{name} n={len(rr)} exact {np.mean(er==0):.3f} close {np.mean(er<=2):.3f}')
    print(f'  [{lo},{hi}]',' | '.join(out))
