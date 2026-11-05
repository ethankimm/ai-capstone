"""What does the distillation target contain? Teacher hidden state at ':' per layer on train items.
(a) held-out R^2 from answer identity alone; (b) what the residual still encodes; (c) the distill penalty of being off by delta."""
import sys, json, time; sys.path.insert(0,sys.argv[1])
from harness import *
import numpy as np
from rc import *
DEV=sys.argv[2]
torch.manual_seed(0); rng=np.random.default_rng(0)
D=BASE+'/synthetic_data_v2_l1to7w/depth2/'
tr=rows(D+'train.json',per_level=1500,seed=11,levels=[2,3,4,5,6,7])   # 9000
m,tok,ta=load(CK['stab']%17820,6,DEV); m.eval()
H=[];meta=[]
t0=time.time()
with torch.no_grad():
    rs=sorted(tr,key=lambda r:len(r['question']))
    for i in range(0,len(rs),24):
        bt=rs[i:i+24]; b=encode(bt,tok,m,DEV); t=teacher(m,b)
        rp=(b['ref_answer_position']-1)
        hs=torch.stack([x.gather(1,rp[:,None,None].expand(-1,-1,x.size(-1)))[:,0] for x in t.hidden_states],1)  # B,13,d
        H.append(hs.float().cpu()); meta+=bt
        del t,hs,b
        if DEV=='mps': torch.mps.empty_cache()
H=torch.cat(H).numpy(); print('teacher states',H.shape,f'{time.time()-t0:.0f}s',flush=True)
ans=np.array([int(r['answer']) for r in meta]); lev=np.array([int(r['level']) for r in meta])
vals=[walk(*parse(r['question']))[0] for r in meta]
N=len(meta); idx=rng.permutation(N); fit,ho=idx[:int(.8*N)],idx[int(.8*N):]
out={'r2_answer':[], 'probe':{}, 'penalty':{}}
std=H.std(axis=(0,2))
print('layer: R2(answer identity, held-out) | R2(answer+level)')
for l in range(13):
    X=H[:,l]; mu=X[fit].mean(0); tot=((X[ho]-mu)**2).sum()
    cm=np.zeros((100,X.shape[1]))+mu; 
    for a in range(100):
        s=fit[ans[fit]==a]
        if len(s)>=3: cm[a]=X[s].mean(0)
    res=((X[ho]-cm[ans[ho]])**2).sum()
    r2=1-res/tot
    out['r2_answer'].append(float(r2)); print(l,round(float(r2),3),flush=True)
    # penalty of being off by delta: SmoothL1(class mean a, class mean a+delta)/std of layer
    pen={}
    for dl in (1,2,3,4,5,8,10,20,30):
        v=[]
        for a in range(100-dl):
            if (ans[fit]==a).sum()>=20 and (ans[fit]==a+dl).sum()>=20:
                d=np.abs(cm[a]-cm[a+dl]); sl=np.where(d<1,.5*d*d,d-.5).mean()/std[l]; v.append(sl)
        pen[dl]=float(np.mean(v))
    out['penalty'][l]=pen
print('distill penalty by |delta| per layer, relative to delta=30')
for l in (4,6,8,9,10,11,12):
    p=out['penalty'][l]; print(l,{k:round(v/p[30],3) for k,v in p.items()})
print('mean over layers (the actual loss averages all 13):', {k:round(float(np.mean([out['penalty'][l][k] for l in range(13)])),4) for k in (1,2,3,4,5,8,10,20,30)})
# does the target hold the intermediate values beyond the answer? digit probes on PCA'd state with and without the answer's class mean removed
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
print('linear probe, exact value of v_(n-j) at the teacher ":" (items level>=j+1), layer 9 and 12')
for l in (6,9,12):
    X=H[:,l]; mu=X[fit].mean(0)
    cm=np.zeros((100,X.shape[1]))+mu
    for a in range(100):
        s=fit[ans[fit]==a]
        if len(s)>=3: cm[a]=X[s].mean(0)
    R=X-cm[ans]
    pca=PCA(128,random_state=0).fit(X[fit]); pr=PCA(128,random_state=0).fit(R[fit])
    A,Bm=pca.transform(X),pr.transform(R)
    for back in (0,1,2,3):
        tgt=np.array([v[-1-back] if len(v)>back else -1 for v in vals]); ok=(lev>=back+1)&(tgt>=0)
        f=fit[ok[fit]]; h=ho[ok[ho]]
        def acc(F_):
            c=LogisticRegression(max_iter=300,C=1.0).fit(F_[f],tgt[f]); return float((c.predict(F_[h])==tgt[h]).mean())
        maj=float(np.mean(tgt[h]==np.bincount(tgt[f]).argmax()))
        a1=acc(A); a2=acc(Bm)
        out['probe'][f'L{l}_back{back}']=(a1,a2,maj)
        print(f' layer {l} value {back} rules before the end: raw {a1:.3f} residual-after-answer {a2:.3f} majority {maj:.3f}',flush=True)
json.dump(out,open(sys.argv[1]+'/b5_teacher.json','w'),indent=1)
print('DONE')
