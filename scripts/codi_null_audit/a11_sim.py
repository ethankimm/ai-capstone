"""Noisy-analog-tracker simulator. Fit ONLY on per-level exact/close curves (L2-L7) of CODI-0;
then check held-out predictions: margin dependence, far-rate, signed error histogram, and CODI (no refit)."""
import json, sys, itertools
sys.path.insert(0,sys.argv[1])
import numpy as np
from rc import *
R='results/'
A=[json.loads(l) for l in open(R+'20261105-044444_codi_v2-l1to7w-stab-depth2-codi/predictions.jsonl')]
B=[json.loads(l) for l in open(R+'20261105-024111_codi_v2-l1to7w-stab-depth2-codi0/predictions.jsonl')]
items=[(r['level'],)+parse(r['question'])+(int(r['gold_answer']),) for r in B]
def margin(x0,rules):
    v,_=walk(x0,rules); m=[abs(v[j]-ru[1]) for j,ru in enumerate(rules) if ru[0] in 'TD']
    return min(m) if m else 99
MARG=np.array([margin(i[1],i[2]) for i in items]); LEV=np.array([i[0] for i in items]); GOLD=np.array([i[3] for i in items])
def sim(sigma,pi,rho,reps=12,seed=0):
    """sigma: value noise added per rule (units). pi: per-parity-rule error rate. rho: prob. final answer snaps to the
    tracked parity (the exact parity walk) rather than the plain nearest integer."""
    rng=np.random.default_rng(seed); P=np.zeros((reps,len(items)))
    for k,(L,x0,rules,g) in enumerate(items):
        n=reps
        s=np.full(n,float(x0)); par=np.full(n,x0%2)   # analog value + separately tracked parity bit
        for ru in rules:
            kind,t,a,b=ru
            if kind=='P':
                br=(par==0)
                wrong=rng.random(n)<pi
                br=np.where(wrong,~br,br)
                new=np.where(br,s/2,s+a)
            elif kind=='T':
                br=s>=t; new=np.where(br,s-b,s+a)
            else:
                br=s<t; new=np.where(br,2*s,s-b)
            s=new+rng.normal(0,sigma,n)
            par=np.round(s).astype(int)%2 if False else par  # placeholder, replaced below
            # exact parity bit follows the value it would have with no noise on this branch path: use the noisy-rounded value
            par=np.mod(np.round(new).astype(int),2)
        pred=np.round(s)
        snap=rng.random(n)<rho
        tgt=g%2  # parity retained at readout
        adj=np.where(np.mod(pred,2)==tgt,pred,pred+rng.choice([-1,1],n))
        P[:,k]=np.where(snap,adj,pred)
    return P
def curves(P):
    e=np.abs(P-GOLD[None]); out={}
    for L in range(2,8):
        m=LEV==L; out[L]=(float(np.mean(e[:,m]==0)),float(np.mean(e[:,m]<=2)))
    return out
def data_curves(rows):
    p=np.array([int(r['predicted_answer']) for r in rows]); e=np.abs(p-GOLD)
    return {L:(float(np.mean(e[LEV==L]==0)),float(np.mean(e[LEV==L]<=2))) for L in range(2,8)}, p
d0,p0=data_curves(B); d1,p1=data_curves(A)
best=None
for sigma,pi,rho in itertools.product((.2,.3,.4,.5,.6,.8,1.0,1.3),(0,.02,.05,.08,.12),(0,.4,.7,.9)):
    c=curves(sim(sigma,pi,rho,reps=4))
    loss=sum((c[L][0]-d0[L][0])**2+(c[L][1]-d0[L][1])**2 for L in c)
    if best is None or loss<best[0]: best=(loss,sigma,pi,rho)
print('best fit (sum sq err over 12 numbers)',best)
_,sg,pi,rho=best
P=sim(sg,pi,rho,reps=40,seed=1); c=curves(P)
print('level  CODI-0 exact/close | sim exact/close | CODI exact/close')
for L in c: print(L,f'{d0[L][0]:.3f}/{d0[L][1]:.3f} | {c[L][0]:.3f}/{c[L][1]:.3f} | {d1[L][0]:.3f}/{d1[L][1]:.3f}')
print('\nHELD OUT 1: margin dependence at L5-7 (never fit)')
hard=LEV>=5
for lo,hi in ((0,1),(2,3),(4,7),(8,15),(16,99)):
    m=hard&(MARG>=lo)&(MARG<=hi)
    e0=np.abs(p0[m]-GOLD[m]); es=np.abs(P[:,m]-GOLD[m][None])
    print(f' margin [{lo},{hi}] n={m.sum()} data exact {np.mean(e0==0):.3f} close {np.mean(e0<=2):.3f} | sim exact {np.mean(es==0):.3f} close {np.mean(es<=2):.3f}')
print('\nHELD OUT 2: signed |error| shares among wrong L5-7 answers')
w=hard&(p0!=GOLD); d=np.abs(p0[w]-GOLD[w])
ws=np.abs(P[:,hard]-GOLD[hard][None]); ws=ws[ws>0]
for k in (1,2,3,4,5,6,8):
    print(f' |e|={k}: data {np.mean(d==k):.3f} sim {np.mean(ws==k):.3f}')
print(f' |e|>10: data {np.mean(d>10):.3f} sim {np.mean(ws>10):.3f}')
# how much do individual item outcomes agree: does the simulator's item-level failure predict the model's?
acc_sim=np.mean(P==GOLD[None],0)
for L in (5,6,7):
    m=LEV==L; c0=(p0[m]==GOLD[m]).astype(float); c1=(p1[m]==GOLD[m]).astype(float)
    print(f'L{L} corr(sim P(correct), CODI-0 correct) {np.corrcoef(acc_sim[m],c0)[0,1]:.3f}; corr(sim, CODI) {np.corrcoef(acc_sim[m],c1)[0,1]:.3f}; corr(CODI, CODI-0) {np.corrcoef(c0,c1)[0,1]:.3f}')
