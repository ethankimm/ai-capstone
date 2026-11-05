import json, math, re, collections
from scipy import stats
R='results/'
A=R+'20261105-044444_codi_v2-l1to7w-stab-depth2-codi/'
B=R+'20261105-024111_codi_v2-l1to7w-stab-depth2-codi0/'
def rd(p): return [json.loads(l) for l in open(p+'predictions.jsonl')]
a,b=rd(A),rd(B)
assert [x['id'] for x in a]==[x['id'] for x in b]; print('same items in same order', len(a))
def mc(n10,n01):
    n=n10+n01
    return 1.0 if n==0 else min(1.0, 2*stats.binom.cdf(min(n10,n01),n,.5))
print('lvl  codi  codi0  G      n10 n01  p   | discord  expected-if-indep  phi')
pool=collections.Counter()
for L in range(1,8):
    x=[(p['correct'],q['correct']) for p,q in zip(a,b) if p['level']==L]
    n=len(x); n11=sum(i and j for i,j in x); n10=sum(i and not j for i,j in x); n01=sum(j and not i for i,j in x); n00=n-n11-n10-n01
    pa,pb=(n11+n10)/n,(n11+n01)/n
    exp=n*(pa*(1-pb)+pb*(1-pa))
    den=math.sqrt(max(pa*(1-pa)*pb*(1-pb),1e-12))
    phi=(n11/n-pa*pb)/den
    print(f'L{L} {pa:.3f} {pb:.3f} {pa-pb:+.3f} {n10:4d} {n01:4d} {mc(n10,n01):.3f} | {n10+n01:4d} {exp:6.1f} phi={phi:.3f}')
    if L>=5: pool['n']+=n; pool['n10']+=n10; pool['n01']+=n01
n,n10,n01=pool['n'],pool['n10'],pool['n01']
d=(n10-n01)/n
# Wald paired CI + Newcombe-ish simple
se=math.sqrt((n10+n01)/n**2-d*d/n)
print(f'pooled L5-7 n={n} G={d:+.4f} n10={n10} n01={n01} p={mc(n10,n01):.3f} waldCI=[{d-1.96*se:+.4f},{d+1.96*se:+.4f}]')
# item-level agreement on *wrong* items: do they give the same wrong answer?
for L in range(3,8):
    w=[(p,q) for p,q in zip(a,b) if p['level']==L and not p['correct'] and not q['correct']]
    same=sum(p['predicted_answer']==q['predicted_answer'] for p,q in w)
    print(f'L{L} both wrong {len(w)} same wrong answer {same} ({same/max(1,len(w)):.2f})')
