"""Reads b13_raw.json and b15_raw.json. Usage: a18_route.py <this dir> <artifacts dir>"""
import json, sys, collections
sys.path.insert(0,sys.argv[1]); W=sys.argv[2]
import numpy as np
from rc import *
from scipy.stats import binomtest
D=json.load(open(W+'/b13_raw.json')); ZS=D['ZS']; ST=D['store']
E=json.load(open(W+'/b15_raw.json')); FZ=E['FZ']; S=E['store']
def tcount(z):
    x0,ru=parse(z['q']); v,b=walk(x0,ru); c=0
    for j,r in enumerate(ru):
        if r[0]!='P':
            s=v[j]; c+=((s-r[1]+1) if s>=r[1] else (r[1]-s))<=2
    return c
tc=np.array([tcount(z) for z in ZS]); L=np.array([z['level'] for z in ZS])
gold=np.array([z['gold'] for z in ZS])
print('1. LATENT ABLATION on constructed chains (exact). intact vs all latents mean-ablated vs the matched CODI-0')
for c,o in (('stab CODI','stab CODI-0'),('r1 CODI','r1 CODI-0'),('lr3 CODI',None)):
    I=np.array(ST[c]['z']); A=np.array(S[c]['abl']); chk=np.mean(np.array(S[c]['intact_check'])==I[:200])
    Z=np.array(ST[o]['z']) if o else None
    print(' ##',c,f'(this script vs b13 agreement on 200 items: {chk:.3f})')
    for lab,ii in (('0 tight',tc==0),('1-3 tight, L3-4',(tc>=1)&(tc<=3)&(L<=4)),('1-3 tight, L5-7',(tc>=1)&(tc<=3)&(L>=5)),('4+ tight',tc>=4)):
        a=(I[ii]==gold[ii]); b=(A[ii]==gold[ii]); w=int((a&~b).sum()); l=int((~a&b).sum()); p=binomtest(w,w+l).pvalue if w+l else 1
        s=f'   {lab:17s} n={ii.sum():5d} intact {a.mean():.3f} ablated {b.mean():.3f} drop {a.mean()-b.mean():+.3f} ({w}/{l} p={p:.4f})'
        if o:
            d=(Z[ii]==gold[ii]); w2=int((b&~d).sum()); l2=int((~b&d).sum()); p2=binomtest(w2,w2+l2).pvalue if w2+l2 else 1
            s+=f' | CODI-0 {d.mean():.3f}; ablated CODI minus CODI-0 {b.mean()-d.mean():+.3f} ({w2}/{l2} p={p2:.4f})'
        print(s)
print('\n2. ONE RULE BLOCKED OR SUPPLIED (constant +-3 edits with at least 2 later rules)')
print('   rule by rule predicts: block next 0, block last 0, supply next 1, supply last 1. A parallel read at the last rule: 1, 0, 0, 1')
def gain(rows,key):
    dt=np.array([q['new']-q['gold'] for q,r in rows],float); dm=np.array([r[key]-r['clean'] for q,r in rows],float)
    return float((dm*dt).sum()/(dt*dt).sum())
good={}
for n,s in S.items():
    g=[(q,r) for q,r in zip(FZ,s['route']) if abs(r['clean']-q['gold'])<=2 and abs(r['edit']-q['new'])<=2]; good[n]=g
    print(' ##',n)
    for lab,sel in (('2 later rules',lambda q:q['dist']==2),('3+ later rules',lambda q:q['dist']>=3)):
        c=[(q,r) for q,r in g if q['kind']=='const' and sel(q) and all(abs(r[k]-r['clean'])<=30 for k in ('Bnext','Blast','Anext','Alast'))]
        if len(c)<10: continue
        print(f"   constant, {lab:15s} n={len(c):4d} full {gain(c,'edit'):.2f} | block next {gain(c,'Bnext'):.2f} block last {gain(c,'Blast'):.2f} | supply next {gain(c,'Anext'):.2f} supply last {gain(c,'Alast'):.2f}")
    c=[(q,r) for q,r in g if q['kind']=='flip']
    def cl(k):
        t=collections.Counter()
        for q,r in c:
            p=r[k]; t['new' if abs(p-q['new'])<=2 else 'old' if abs(p-q['gold'])<=2 else 'other']+=1
        return ' '.join(f'{x} {t[x]/len(c):.2f}' for x in ('new','old','other'))
    print(f"   branch flip n={len(c)} | block next: {cl('Bnext')} | block last: {cl('Blast')} | supply next: {cl('Anext')} | supply last: {cl('Alast')}")
print('\n3. WHEN IS THE STATE NEEDED? Patch rule j+k at one block output from the edited run. Gain by layer; constant edits.')
for n,g in good.items():
    print(' ##',n); print('      layer '+''.join(f'{l+1:6d}' for l in range(12))+'   last layer with gain>=0.5')
    for k in (1,2,3,4):
        c=[(q,r) for q,r in g if q['kind']=='const' and q['dist']>=k+0 and f'{k}|0' in r['grid']]
        if len(c)<15: continue
        dt=np.array([q['new']-q['gold'] for q,r in c],float); row=[]
        for l in range(12):
            dm=np.array([r['grid'][f'{k}|{l}']-r['clean'] for q,r in c],float); dm=np.clip(dm,-30,30); row.append(float((dm*dt).sum()/(dt*dt).sum()))
        last=max([l+1 for l,x in enumerate(row) if x>=.5],default=0)
        print(f'   rule j+{k} (n={len(c):3d})'+''.join(f'{x:6.2f}' for x in row)+f'   {last}')
    # split by whether rule j+k is the last rule, so "last rule" timing is not mixed with mid-chain timing
    for k in (1,2,3):
        c=[(q,r) for q,r in g if q['kind']=='const' and q['dist']>k and f'{k}|0' in r['grid']]
        if len(c)<15: continue
        dt=np.array([q['new']-q['gold'] for q,r in c],float); row=[]
        for l in range(12):
            dm=np.array([r['grid'][f'{k}|{l}']-r['clean'] for q,r in c],float); dm=np.clip(dm,-30,30); row.append(float((dm*dt).sum()/(dt*dt).sum()))
        last=max([l+1 for l,x in enumerate(row) if x>=.5],default=0)
        print(f'   mid j+{k}  (n={len(c):3d})'+''.join(f'{x:6.2f}' for x in row)+f'   {last}')
