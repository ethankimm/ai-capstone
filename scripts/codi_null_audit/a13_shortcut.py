"""How far can you get on RuleChain WITHOUT following the chain? Exact solvers that are denied information, compared with the models."""
import json, sys, collections, random
sys.path.insert(0,sys.argv[1]); S=sys.argv[1]
import numpy as np
from rc import *
BASE='/Users/henninglindig/Documents/Penn/CIS5980/'
tr=json.load(open(BASE+'synthetic_data_v2_l1to7w/depth2/train.json'))
R='results/'
rd=lambda p:[json.loads(l) for l in open(p)]
A=rd(R+'20261105-044444_codi_v2-l1to7w-stab-depth2-codi/predictions.jsonl'); B=rd(R+'20261105-024111_codi_v2-l1to7w-stab-depth2-codi0/predictions.jsonl')
# value marginals by (level, position), branch priors by rule kind and threshold bucket
marg=collections.defaultdict(lambda:np.zeros(100)); prior=collections.defaultdict(lambda:[0,0]); pos_prior=collections.defaultdict(lambda:[0,0])
seen3=set(); seen_suf=collections.Counter()
rng=random.Random(0)
for r in tr:
    x0,rules=parse(r['question']); v,b=walk(x0,rules); n=len(rules)
    for j in range(n+1): marg[(n,j)][v[j]]+=1
    for j,(ru,br) in enumerate(zip(rules,b)):
        k=(ru[0],ru[1]//5); prior[k][int(br)]+=1
    seen3.add((x0,tuple(rules[:3]))) if n>=3 else None
    seen_suf[tuple(rules[-3:])]+=1
def pb(ru):
    c=prior[(ru[0],ru[1]//5)]; return c[1]>=c[0]
print('rule kinds: P(first branch)',{k:round(sum(prior[x][1] for x in prior if x[0]==k)/sum(sum(prior[x]) for x in prior if x[0]==k),3) for k in 'PTD'})
pa=[]
for k,c in prior.items(): pa.append((max(c),sum(c)))
print('branch predictable from kind+threshold alone: %.3f'%(sum(a for a,b in pa)/sum(b for a,b in pa)))
def nt_solver(x0,rules):
    s=x0
    for ru in rules: s=apply(ru,s,pb(ru))
    return s
def final_from(s,rules):
    for ru in rules: s=apply(ru,s)
    return s
def best_guess(rules,w):
    """Bayes-optimal guess of the answer knowing only `rules` and the start-value distribution w: returns exact-mode and +-2-window centre."""
    mass=collections.Counter()
    for s in np.nonzero(w)[0]:
        f=final_from(int(s),rules); mass[f]+=w[s]
    mode=max(mass,key=mass.get)
    win={c:sum(mass.get(c+d,0) for d in range(-2,3)) for c in range(-5,205)}
    centre=max(win,key=win.get)
    return mode,centre,mass[mode]/sum(mass.values()),win[centre]/sum(mass.values())
print('\nLevel | CODI-0 exact/close | no-tracking solver (branch from surface prior, exact arithmetic) exact/close | Bayes-optimal given ONLY the last k rules, no start: exact/close for k=1,2,3,4,all')
items=[(a,b)+parse(a['question']) for a,b in zip(A,B)]
pred_store={}
for L in range(2,8):
    rr=[i for i in items if i[0]['level']==L]
    g=np.array([int(i[0]['gold_answer']) for i in rr]); p0=np.array([int(i[1]['predicted_answer']) for i in rr])
    nt=np.array([nt_solver(i[2],i[3]) for i in rr])
    row=[]
    for k in [1,2,3,4,L]:
        if k>L: row.append('  -  '); continue
        ex=[];cl=[];conf=[]
        for i,gg in zip(rr,g):
            rules=i[3]; w=marg[(L,L-k)]; w=w/w.sum()
            mode,c,mc,wc=best_guess(rules[L-k:],w)
            ex.append(mode==gg); cl.append(abs(c-gg)<=2); conf.append(wc)
        row.append(f'{np.mean(ex):.3f}/{np.mean(cl):.3f}')
        if k==3: pred_store[L]=np.array(conf)
    print(f'L{L} | {np.mean(p0==g):.3f}/{np.mean(np.abs(p0-g)<=2):.3f} | {np.mean(nt==g):.3f}/{np.mean(np.abs(nt-g)<=2):.3f} | '+' '.join(row))
print('\nDoes model success concentrate where the last 3 rules alone pin the answer? accuracy by tercile of last-3 predictability (exact/close)')
for L in (5,6,7):
    rr=[i for i in items if i[0]['level']==L]; c=pred_store[L]; q=np.quantile(c,[1/3,2/3])
    g=np.array([int(i[0]['gold_answer']) for i in rr]); p0=np.array([int(i[1]['predicted_answer']) for i in rr]); p1=np.array([int(i[0]['predicted_answer']) for i in rr])
    out=[]
    for lo,hi in ((-1,q[0]),(q[0],q[1]),(q[1],9)):
        m=(c>lo)&(c<=hi); out.append(f'[{c[m].mean():.2f}] CODI-0 {np.mean(p0[m]==g[m]):.3f}/{np.mean(np.abs(p0[m]-g[m])<=2):.3f}')
    print(f' L{L}',' | '.join(out))
print('\nAgainst-prior branches: accuracy by number of branches that go AGAINST the surface prior (L5-7)')
for nsur in range(0,6):
    sel=[i for i in items if i[0]['level']>=5 and sum(pb(ru)!=br for ru,br in zip(i[3],walk(i[2],i[3])[1]))==nsur]
    if len(sel)<25: continue
    g=np.array([int(i[0]['gold_answer']) for i in sel]); p0=np.array([int(i[1]['predicted_answer']) for i in sel]); nt=np.array([nt_solver(i[2],i[3]) for i in sel])
    print(f' {nsur} against-prior n={len(sel)} CODI-0 exact {np.mean(p0==g):.3f} close {np.mean(np.abs(p0-g)<=2):.3f} | no-tracking solver close {np.mean(np.abs(nt-g)<=2):.3f}')
print('\nTrain overlap: test items whose (x0, rules 1-3) appear in train; whose last 3 rules appear in train')
for L in (5,6,7):
    rr=[i for i in items if i[0]['level']==L]
    print(f' L{L} prefix {np.mean([(i[2],tuple(i[3][:3])) in seen3 for i in rr]):.3f} suffix {np.mean([seen_suf[tuple(i[3][-3:])]>0 for i in rr]):.3f}')
