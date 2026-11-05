"""Reads b13_raw.json. Usage: a16_resolution.py <this dir> <artifacts dir>"""
import json, sys, collections
sys.path.insert(0,sys.argv[1]); W=sys.argv[2]
import numpy as np
from scipy.stats import binomtest
D=json.load(open(W+'/b13_raw.json')); Q=D['Q']; FZ=D['FZ']; ZS=D['ZS']; ST=D['store']
print('queries',collections.Counter(q['kind'] for q in Q),'freeze',collections.Counter(q['kind'] for q in FZ),'sets',len(ZS))
gold={r['id']:int(r['answer']) for r in D['base']}; lev={r['id']:r['level'] for r in D['base']}
print('\n0. base accuracy on the sampled items (exact / within 2), L3-7')
for n,s in ST.items():
    print(f'  {n:12s}',' '.join(f"L{L} {np.mean([s['base'][i]==gold[i] for i in gold if lev[i]==L]):.2f}/{np.mean([abs(s['base'][i]-gold[i])<=2 for i in gold if lev[i]==L]):.2f}" for L in range(3,8)))
def close(s,it): return abs(s['base'][it]-gold[it])<=2
JB=lambda j:'r1' if j==1 else 'r2' if j==2 else 'r3' if j==3 else 'r4' if j==4 else 'r5+'
print('\n1. PSYCHOMETRIC: threshold placed m away from the true running value; items the model answered within 2.')
print('   right-branch share = right/(right+wrong) among answers near one of the two outcomes; n in brackets; other = near neither')
for n,s in ST.items():
    t=collections.defaultdict(lambda:[0,0,0])
    for q,p in zip(Q,s['q']):
        if q['kind']!='psy' or not close(s,q['item']): continue
        right,wrong=(q['hi'],q['lo']) if q['side']=='hi' else (q['lo'],q['hi'])
        c=t[(JB(q['j']),q['m'])]
        if abs(p-right)<=2: c[0]+=1
        elif abs(p-wrong)<=2: c[1]+=1
        else: c[2]+=1
    print(' ##',n); print('      '+''.join(f'm={m}'.rjust(16) for m in (1,2,3,4,6)))
    for jb in ('r1','r2','r3','r4','r5+'):
        row=''
        for m in (1,2,3,4,6):
            a,b,c=t[(jb,m)]
            row+=(f'{a/max(1,a+b):.2f} ({a+b}) o{c/max(1,a+b+c):.2f}').rjust(16)
        print('  ',jb.ljust(4),row)
print('\n2. SMALL EDITS: +-1..4 to x0 or a taken constant, first effect a later branch flip; items answered within 2.')
print('   follow = answer near the new true path; kept = near the old decisions with shifted value; other = neither')
def bitrows(s):
    t=collections.defaultdict(lambda:[0,0,0])
    for q,p in zip(Q,s['q']):
        if q['kind']!='bit' or not close(s,q['item']): continue
        if q['cls']=='P':
            if q['e']==q['k']-1 and q['e']==0: key='P at rule 1, read off x0'
            elif q['nh']>0: key='P, halving between edit and rule'
            elif q['nhall']>0: key='P, halving before edit only'
            else: key=f"P, no halving, {min(q['dist'],3)}{'+' if q['dist']>=3 else ''} rules on"
        else:
            key=f"T/D crossed, new margin {min(q['m'],3)}{'+' if q['m']>=3 else ''}"
        c=t[key]
        if abs(p-q['new'])<=2: c[0]+=1
        elif abs(p-q['kept'])<=2: c[1]+=1
        else: c[2]+=1
    return t
for n,s in ST.items():
    t=bitrows(s); print(' ##',n)
    for k in sorted(t):
        a,b,c=t[k]; m=a+b+c
        print(f'   {k:36s} n={m:4d} follow {a/m:.3f} kept {b/m:.3f} other {c/m:.3f}')
print('\n3. FREEZE: edit rule j. A = clean run with LATER rule tokens taken from the edited run. B = edited run with later rule tokens frozen clean.')
print('   serial carrier: A moves like the full edit, B does not. Direct read from rule j: the reverse.')
for n,s in ST.items():
    f=s['fz']; ok=[(q,r) for q,r in zip(FZ,f) if r['aligned']]
    print(' ##',n,f"controls: own-state patch unchanged {np.mean([r['I']==r['clean'] for q,r in ok]):.3f}; all positions from edited = edited {np.mean([r['F']==r['edit'] for q,r in ok]):.3f}")
    def gain(rows,key):
        dt=np.array([q['new']-q['gold'] for q,r in rows],float); dm=np.array([r[key]-r['clean'] for q,r in rows],float)
        return float((dm*dt).sum()/(dt*dt).sum())
    good=[(q,r) for q,r in ok if abs(r['clean']-q['gold'])<=2 and abs(r['edit']-q['new'])<=2]
    for lab,sel in (('1 later rule',lambda q:q['dist']==1),('2 later',lambda q:q['dist']==2),('3 later',lambda q:q['dist']==3),('4+ later',lambda q:q['dist']>=4)):
        cc=[(q,r) for q,r in good if q['kind']=='const' and sel(q) and abs(r['A']-r['clean'])<=30 and abs(r['B']-r['clean'])<=30]
        if len(cc)<10: continue
        print(f"   constant +-3, {lab:13s} n={len(cc):4d} gain: full {gain(cc,'edit'):.2f}  A later-tokens-only {gain(cc,'A'):.2f}  B rule-j-only {gain(cc,'B'):.2f}")
    for lab,sel in (('1 later rule',lambda q:q['dist']==1),('2 later',lambda q:q['dist']==2),('3+ later',lambda q:q['dist']>=3)):
        cc=[(q,r) for q,r in good if q['kind']=='flip' and sel(q)]
        if len(cc)<10: continue
        def cl(key):
            c=collections.Counter()
            for q,r in cc:
                p=r[key]; c['new' if abs(p-q['new'])<=2 else 'keptdec' if abs(p-q['kept'])<=2 else 'old' if abs(p-q['gold'])<=2 else 'other']+=1
            return ' '.join(f"{k} {c[k]/len(cc):.2f}" for k in ('new','keptdec','old','other'))
        print(f"   branch flip,  {lab:13s} n={len(cc):4d} | A: {cl('A')} | B: {cl('B')}")
print('\n4. ZERO-SHOT SETS (same grammar): exact / within 2, 500 items per cell')
conds=[(False,False,'wide, no P after halving'),(False,True,'wide, P after halving'),(True,False,'tight, no P after halving'),(True,True,'tight, P after halving')]
for n,s in ST.items():
    print(' ##',n); e=np.array([abs(p-z['gold']) for p,z in zip(s['z'],ZS)])
    for tg,hv,lab in conds:
        print(f'   {lab:28s}',' '.join(f"L{L} {np.mean(e[[i for i,z in enumerate(ZS) if z['level']==L and z['tight']==tg and z['halve']==hv]]==0):.2f}/{np.mean(e[[i for i,z in enumerate(ZS) if z['level']==L and z['tight']==tg and z['halve']==hv]]<=2):.2f}" for L in range(3,8)))
print('\n5. PAIRED CODI vs CODI-0 on the sets (exact). wins = CODI right, CODI-0 wrong')
for pa,pb,lab in (('stab CODI','stab CODI-0','stabilised'),('r1 CODI','r1 CODI-0','round 1')):
    a=np.array([abs(p-z['gold']) for p,z in zip(ST[pa]['z'],ZS)]); b=np.array([abs(p-z['gold']) for p,z in zip(ST[pb]['z'],ZS)])
    print(' ##',lab)
    for tg,hv,cl in conds:
        for lo,hi in ((3,4),(5,7)):
            ii=[i for i,z in enumerate(ZS) if z['tight']==tg and z['halve']==hv and lo<=z['level']<=hi]
            w=int(((a[ii]==0)&(b[ii]!=0)).sum()); l=int(((a[ii]!=0)&(b[ii]==0)).sum()); pv=binomtest(w,w+l).pvalue if w+l else 1
            w2=int(((a[ii]<=2)&(b[ii]>2)).sum()); l2=int(((a[ii]>2)&(b[ii]<=2)).sum()); pv2=binomtest(w2,w2+l2).pvalue if w2+l2 else 1
            print(f'   {cl:28s} L{lo}-{hi} n={len(ii)} exact {np.mean(a[ii]==0):.3f} vs {np.mean(b[ii]==0):.3f} d {np.mean(a[ii]==0)-np.mean(b[ii]==0):+.3f} ({w}/{l} p={pv:.4f}) | close {np.mean(a[ii]<=2):.3f} vs {np.mean(b[ii]<=2):.3f} d {np.mean(a[ii]<=2)-np.mean(b[ii]<=2):+.3f} ({w2}/{l2} p={pv2:.4f})')
