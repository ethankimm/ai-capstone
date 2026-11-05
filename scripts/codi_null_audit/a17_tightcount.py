"""Reads b13_raw.json. Is difficulty the number of rules, or the number of decisions that need an exact value? Usage: a17_tightcount.py <this dir> <artifacts dir>"""
import json, sys, collections
sys.path.insert(0,sys.argv[1]); W=sys.argv[2]
import numpy as np
from rc import *
from scipy.stats import binomtest
D=json.load(open(W+'/b13_raw.json')); ZS=D['ZS']; ST=D['store']
def feat(z):
    x0,ru=parse(z['q']); v,b=walk(x0,ru); tight=0; h=False; pa=0
    for j,(r,br) in enumerate(zip(ru,b)):
        if r[0]=='P':
            pa+=h; h=h or br
        else:
            s=v[j]; m=(s-r[1]+1) if s>=r[1] else (r[1]-s); tight+=m<=2
    return tight,pa
F=[feat(z) for z in ZS]; tc=np.array([f[0] for f in F]); pa=np.array([f[1] for f in F]); L=np.array([z['level'] for z in ZS]); tg=np.array([z['tight'] for z in ZS])
print('constructed chains: exact / within 2 by number of tight decisions (rows) and level (cols); n in brackets for first model')
for n in ('stab CODI-0','stab CODI','r1 CODI-0','r1 CODI'):
    e=np.array([abs(p-z['gold']) for p,z in zip(ST[n]['z'],ZS)]); print(' ##',n)
    for c in range(0,6):
        row=''
        for l in range(3,8):
            ii=(tc==c)&(L==l)&(pa==0)
            row+=(f'{np.mean(e[ii]==0):.2f}/{np.mean(e[ii]<=2):.2f} ({ii.sum()})'.rjust(19) if ii.sum()>=20 else '-'.rjust(19))
        print(f'   tight={c}',row)
print('\npaired exact and within 2 by number of tight decisions, no P after halving excluded neither: all constructed chains')
for pa_,pb_,lab in (('stab CODI','stab CODI-0','stabilised'),('r1 CODI','r1 CODI-0','round 1')):
    a=np.array([abs(p-z['gold']) for p,z in zip(ST[pa_]['z'],ZS)]); b=np.array([abs(p-z['gold']) for p,z in zip(ST[pb_]['z'],ZS)]); print(' ##',lab)
    for c,cl in ((0,'0'),(1,'1'),(2,'2'),(3,'3'),(4,'4+')):
        ii=(tc==c) if c<4 else (tc>=4)
        w=int(((a[ii]==0)&(b[ii]!=0)).sum()); l=int(((a[ii]!=0)&(b[ii]==0)).sum()); w2=int(((a[ii]<=2)&(b[ii]>2)).sum()); l2=int(((a[ii]>2)&(b[ii]<=2)).sum())
        print(f'   tight={cl:3s} n={ii.sum():5d} exact {np.mean(a[ii]==0):.3f} vs {np.mean(b[ii]==0):.3f} ({w}/{l} p={binomtest(w,w+l).pvalue:.4f}) | close {np.mean(a[ii]<=2):.3f} vs {np.mean(b[ii]<=2):.3f} ({w2}/{l2} p={binomtest(w2,w2+l2).pvalue:.4f})')
