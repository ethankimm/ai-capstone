"""Does the student hold the distillation target's reasoning content? Probe v_{n-1} and the answer at the student ':' per layer,
fit on train items, scored on held-out valid. Then: does the model's answer follow its own decoded v_{n-1}?"""
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
LAYERS=(4,6,8,9,10,12)
NUM={}
@torch.no_grad()
def states(m,tok,rs,nl,mode):
    if not NUM:
        for n in range(100): NUM[tok.encode(f"The answer is: {n}")[-1]]=n
    body=body_of(m); dec=torch.tensor([m.eot_id]+tok.encode('The answer is:'))
    H=[];P=[];order=[]
    idx=sorted(range(len(rs)),key=lambda i:len(rs[i]['question']))
    for i in range(0,len(idx),32):
        ii=idx[i:i+32]; bt=[rs[j] for j in ii]; b=encode(bt,tok,m,DEV)
        if mode=='teacher':
            t=teacher(m,b); rp=b['ref_answer_position']-1
            hs=torch.stack([t.hidden_states[l].gather(1,rp[:,None,None].expand(-1,-1,768))[:,0] for l in LAYERS],1)
            H.append(hs); order+=ii; continue
        mask=b['encoder_attention_mask'].long(); pos=(mask.cumsum(-1)-1).clamp(min=0)
        o=body(input_ids=b['encoder_input_ids'],use_cache=True,attention_mask=mask,position_ids=pos)
        past=o.past_key_values; npos=mask.sum(-1,keepdim=True); z=m.prj(o.last_hidden_state[:,-1:,:]) if nl else None
        for s in range(nl):
            mask=torch.cat((mask,torch.ones(mask.size(0),1,dtype=mask.dtype)),1)
            o=body(inputs_embeds=z,use_cache=True,past_key_values=past,attention_mask=mask,position_ids=npos)
            past=o.past_key_values; npos=npos+1; z=m.prj(o.last_hidden_state[:,-1:,:])
        emb=m.get_embd(m.codi,m.model_name)(dec)[None].expand(mask.size(0),-1,-1); T=emb.size(1)
        mask=torch.cat((mask,torch.ones(mask.size(0),T,dtype=mask.dtype)),1)
        o=body(inputs_embeds=emb,use_cache=True,output_hidden_states=True,past_key_values=past,attention_mask=mask,position_ids=npos+torch.arange(T)[None])
        H.append(torch.stack([o.hidden_states[l][:,-1] for l in LAYERS],1))
        lg=m.codi.get_base_model().lm_head(o.last_hidden_state[:,-1])[:,:50257]
        P+=[NUM.get(t,-999) for t in lg.argmax(-1).tolist()]; order+=ii
    Hc=torch.cat(H).numpy(); inv=np.argsort(order)
    return Hc[inv],(np.array(P)[inv] if P else None)
def tgt(rs):
    v=[walk(*parse(r['question']))[0] for r in rs]
    return np.array([x[-2] for x in v]),np.array([x[-1] for x in v])
vf,af=tgt(fit); ve,ae=tgt(ev); rules=[parse(r['question'])[1][-1] for r in ev]; lev=np.array([int(r['level']) for r in ev])
res={}
for name,ck,nl,mode in (('teacher (stab CODI weights)',CK['stab']%17820,6,'teacher'),('CODI stab',CK['stab']%17820,6,'student'),('CODI-0 stab',CK['stab0']%17820,0,'student'),('CODI lr3e-3',CK['lr3'],6,'student')):
    t0=time.time(); m,tok,ta=load(ck,nl,DEV); m.eval()
    Hf,_=states(m,tok,fit,nl,mode); He,pe=states(m,tok,ev,nl,mode); del m
    r={}
    for li,l in enumerate(LAYERS):
        pp=make=None
        sc=StandardScaler().fit(Hf[:,li]); pca=PCA(128,random_state=0).fit(sc.transform(Hf[:,li]))
        A,B=pca.transform(sc.transform(Hf[:,li])),pca.transform(sc.transform(He[:,li]))
        cv=LogisticRegression(max_iter=200,C=.5).fit(A,vf); pv=cv.predict(B)
        ca=LogisticRegression(max_iter=200,C=.5).fit(A,af); pa=ca.predict(B)
        hard=lev>=5
        row=dict(v_exact=float(np.mean(pv==ve)),v_exact_hard=float(np.mean(pv[hard]==ve[hard])),v_close_hard=float(np.mean(np.abs(pv[hard]-ve[hard])<=2)),ans_exact_hard=float(np.mean(pa[hard]==ae[hard])))
        if pe is not None:
            own=np.array([apply(ru,int(x)) for ru,x in zip(rules,pv)])
            row['model_acc_hard']=float(np.mean(pe[hard]==ae[hard]))
            row['acc_given_v_right']=float(np.mean(pe[hard&(pv==ve)]==ae[hard&(pv==ve)]))
            row['acc_given_v_wrong']=float(np.mean(pe[hard&(pv!=ve)]==ae[hard&(pv!=ve)]))
            w=hard&(pe!=ae)
            row['wrong_answer_follows_decoded_v']=float(np.mean(pe[w]==own[w]))
        r[l]={k:round(v,3) for k,v in row.items()}
        print(name,'layer',l,r[l],flush=True)
    res[name]=r; print(f'  ({time.time()-t0:.0f}s)',flush=True)
json.dump(res,open(sys.argv[1]+'/b9_student.json','w'),indent=1)
print('DONE')
