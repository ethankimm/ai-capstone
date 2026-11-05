"""What is readable at the rule tokens? Usage: b14_probes.py <this dir> <device> <output dir>
Linear probes for the running value after rule j, on items where it is still an intermediate (level > j):
exact (tens and ones heads), within 2 (ridge), parity, second bit, split by earlier halvings. Sites: the '.' closing rule j, and the
':' of the next rule. Question pass only, so latents are not involved."""
import sys, json, time, hashlib; sys.path.insert(0,sys.argv[1])
from harness import *
import numpy as np
from rc import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
DEV=sys.argv[2]; OUT=sys.argv[3]
TE=BASE+'/synthetic_data_v2_l1to7w/depth2/test.json'
base=rows(TE,per_level=500,seed=0,levels=[2,3,4,5,6,7,8])
LAYERS=(4,8,10,12); JS=(1,2,3,4,5,6)
def sites(tok,q):
    ids=tok(q.strip()+'\n').input_ids
    st=[i for i,t in enumerate(ids) if tok.decode([t]).strip()=='Rule']
    return st,len(ids)+1
@torch.no_grad()
def collect(m,tok):
    X={}; meta=[]; body=body_of(m)
    idx=sorted(range(len(base)),key=lambda i:len(base[i]['question']))
    for i in range(0,len(idx),32):
        rr=[base[j] for j in idx[i:i+32]]
        b=encode([dict(question=r['question'],steps=['x'],answer='0') for r in rr],tok,m,DEV)
        mask=b['encoder_attention_mask'].long(); pos=(mask.cumsum(-1)-1).clamp(min=0); T=mask.size(1)
        hs=body(input_ids=b['encoder_input_ids'],attention_mask=mask,position_ids=pos,output_hidden_states=True).hidden_states
        for r_i,r in enumerate(rr):
            x0,rules=parse(r['question']); v,br=walk(x0,rules); n=len(rules); st,tot=sites(tok,r['question']); off=T-tot
            for j in JS:
                if n<=j: continue
                dot=off+st[j]-1; nxt=off+st[j]+2
                nh=sum(1 for i2 in range(j) if rules[i2][0]=='P' and br[i2])
                meta.append((r['id'],j,v[j],nh))
                for l in LAYERS:
                    X.setdefault(('dot',j,l),[]).append(hs[l][r_i,dot].float().cpu().numpy())
                    X.setdefault(('next',j,l),[]).append(hs[l][r_i,nxt].float().cpu().numpy())
        del hs,b
        if DEV=='mps': torch.mps.empty_cache()
    return X,meta
def fit(Xtr,ytr,Xte,yte):
    c=LogisticRegression(max_iter=400,C=0.1).fit(Xtr,ytr); return c.predict(Xte)
res={}
for name,ck,nl in (('stab CODI-0',CK['stab0']%17820,0),('stab CODI',CK['stab']%17820,6),('r1 CODI-0',CK['r10'],0)):
    t0=time.time(); m,tok,ta=load(ck,nl,DEV); m.eval(); X,meta=collect(m,tok); del m
    if DEV=='mps': torch.mps.empty_cache()
    R={}
    for j in JS:
        mj=[x for x in meta if x[1]==j]; ids=[x[0] for x in mj]; y=np.array([x[2] for x in mj]); nh=np.array([x[3] for x in mj])
        te=np.array([int(hashlib.sha256(i.encode()).hexdigest(),16)%10<3 for i in ids]); tr=~te
        for site in ('dot','next'):
            for l in LAYERS:
                A=np.stack(X[(site,j,l)]); sc=StandardScaler().fit(A[tr]); pc=PCA(128,random_state=0).fit(sc.transform(A[tr])); Z=pc.transform(sc.transform(A))
                par=fit(Z[tr],y[tr]%2,Z[te],y[te]%2)==(y[te]%2)
                b1=fit(Z[tr],(y[tr]>>1)%2,Z[te],(y[te]>>1)%2)==((y[te]>>1)%2)
                tn=fit(Z[tr],y[tr]//10,Z[te],y[te]//10); on=fit(Z[tr],y[tr]%10,Z[te],y[te]%10)
                rg=Ridge(alpha=30).fit(Z[tr],y[tr]).predict(Z[te])
                h0=nh[te]==0
                R[f'{site}|{j}|{l}']=dict(n=int(te.sum()),exact=float(np.mean((tn==y[te]//10)&(on==y[te]%10))),tens=float(np.mean(tn==y[te]//10)),within2=float(np.mean(np.abs(rg-y[te])<=2.5)),mae=float(np.mean(np.abs(rg-y[te]))),
                    parity=float(par.mean()),parity_h0=float(par[h0].mean()) if h0.any() else None,parity_h1=float(par[~h0].mean()) if (~h0).any() else None,n_h1=int((~h0).sum()),bit1=float(b1.mean()))
    res[name]=R; print('##',name,f'{time.time()-t0:.0f}s',flush=True)
    json.dump(res,open(OUT+'/b14_probes.json','w'),indent=0)
print('DONE')
