"""Do the latents hold END-RELATIVE chain values (v_{n-1}, the distillation target's content) rather than forward-indexed v_j?
Sites: h_0 (<bot>), h_1..h_6 (latent pass outputs, last layer). Fit on train items, scored on held-out valid."""
import sys, json, time, warnings; sys.path.insert(0,sys.argv[1])
warnings.filterwarnings('ignore')
from harness import *
import numpy as np
from rc import *
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
DEV='cpu'; torch.manual_seed(0)
D=BASE+'/synthetic_data_v2_l1to7w/depth2/'
fit=rows(D+'train.json',per_level=1000,seed=21,levels=[3,4,5,6,7])
ev=[r for r in json.load(open(D+'valid.json')) if int(r['level'])>=3]
SITES=(0,1,2,3,6)
@torch.no_grad()
def states(m,tok,rs):
    body=body_of(m); H=[];order=[]
    idx=sorted(range(len(rs)),key=lambda i:len(rs[i]['question']))
    for i in range(0,len(idx),32):
        ii=idx[i:i+32]; b=encode([rs[j] for j in ii],tok,m,DEV)
        mask=b['encoder_attention_mask'].long(); pos=(mask.cumsum(-1)-1).clamp(min=0)
        o=body(input_ids=b['encoder_input_ids'],use_cache=True,attention_mask=mask,position_ids=pos)
        past=o.past_key_values; npos=mask.sum(-1,keepdim=True); h=[o.last_hidden_state[:,-1]]; z=m.prj(o.last_hidden_state[:,-1:,:])
        for s in range(6):
            mask=torch.cat((mask,torch.ones(mask.size(0),1,dtype=mask.dtype)),1)
            o=body(inputs_embeds=z,use_cache=True,past_key_values=past,attention_mask=mask,position_ids=npos)
            past=o.past_key_values; npos=npos+1; h.append(o.last_hidden_state[:,-1]); z=m.prj(o.last_hidden_state[:,-1:,:])
        H.append(torch.stack([h[s] for s in SITES],1)); order+=ii
    return torch.cat(H).numpy()[np.argsort(order)]
def tg(rs):
    v=[walk(*parse(r['question']))[0] for r in rs]
    return {'v_(n-1)':np.array([x[-2] for x in v]),'v_(n-2)':np.array([x[-3] for x in v]),'v_2 (forward)':np.array([x[2] for x in v]),'answer':np.array([x[-1] for x in v])}
tf,te=tg(fit),tg(ev); lev=np.array([int(r['level']) for r in ev]); hard=lev>=5
print('majority baseline, hard items:',{k:round(float(np.mean(te[k][hard]==np.bincount(tf[k]).argmax())),3) for k in tf},flush=True)
res={}
for name,ck in (('CODI lr3e-3',CK['lr3']),('CODI r1 (40ep)',CK['r1']),('CODI stab',CK['stab']%17820)):
    t0=time.time(); m,tok,ta=load(ck,6,DEV); m.eval()
    Hf=states(m,tok,fit); He=states(m,tok,ev); del m
    r={}
    for si,s in enumerate(SITES):
        sc=StandardScaler().fit(Hf[:,si]); pca=PCA(128,random_state=0).fit(sc.transform(Hf[:,si]))
        A,B=pca.transform(sc.transform(Hf[:,si])),pca.transform(sc.transform(He[:,si]))
        row={}
        for k in tf:
            p=LogisticRegression(max_iter=200,C=.5).fit(A,tf[k]).predict(B)
            row[k]=round(float(np.mean(p[hard]==te[k][hard])),3)
            if k=='v_(n-1)':
                row['v_(n-1) by level L3..7']=[round(float(np.mean(p[lev==L]==te[k][lev==L])),3) for L in range(3,8)]
                row['close v_(n-1) hard']=round(float(np.mean(np.abs(p[hard]-te[k][hard])<=2)),3)
        r[f'h_{s}']=row
        print(name,f'h_{s}',row,flush=True)
    res[name]=r; print(f'  ({time.time()-t0:.0f}s)',flush=True)
json.dump(res,open(sys.argv[1]+'/b10_endrel.json','w'),indent=1)
print('DONE')
