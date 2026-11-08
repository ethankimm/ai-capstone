"""Are values readable when the model is right? Usage: b23_right_wrong.py <codi_null_audit dir> <device> <output dir>. SMOKE=1: a few items. Valid only.
Fixed cells, chosen before looking (peak cell from b20, plus the raw latent input and the last layer), so no best-of selection on the scored items.
Per item: decoded value of every step, the model's own answer. Also shuffled-label probes (selectivity). Fit sizes 4000 (should reproduce b20) and 8000."""
import sys, json, time, os; sys.path.insert(0,sys.argv[1])
from harness import *
import numpy as np
from rc import *
DEV=sys.argv[2]; OUT=sys.argv[3]; SMOKE=int(os.environ.get('SMOKE','0'))
FIT=300 if SMOKE else 8000; LV=[4] if SMOKE else [3,4,5]
SL=BASE+'/codi_runs_l1to7w_startlast'; DL=BASE+'/synthetic_data_v2_l1to7w_startlast/depth2/'; DF=BASE+'/synthetic_data_v2_l1to7w/depth2/'
MODELS=[('startlast CODI',f'{SL}/codi/run/checkpoint-17820',6,True),('startlast CODI-0',f'{SL}/codi0/run/checkpoint-17820',0,True),('lr3 relay',CK['lr3'],6,False)]
if SMOKE: MODELS=MODELS[:1]
NT=13; DECN=['eot','The','answer','is','colon']; LAY=[0,2,4,8,12]
def names(nl): return [f't{NT-i}' for i in range(NT)]+[f'h{s+1}' for s in range(nl)]+DECN
def keep(nl): return ['t1','h2','h4','h6','colon'] if nl else ['t1','colon']
@torch.no_grad()
def collect(m,tok,rows_,nl,dec,num):
    N=names(nl); K=[N.index(k) for k in keep(nl)]; body=body_of(m); X=np.zeros((len(rows_),len(K),len(LAY),768),np.float16); A=np.zeros(len(rows_),int)-999
    order=sorted(range(len(rows_)),key=lambda i:len(rows_[i]['question']))
    for i in range(0,len(order),32):
        ii=order[i:i+32]; b=encode([dict(question=rows_[j]['question'].strip(),steps=['x'],answer='0') for j in ii],tok,m,DEV)
        mask=b['encoder_attention_mask'].long(); pos=(mask.cumsum(-1)-1).clamp(min=0)
        o=body(input_ids=b['encoder_input_ids'],use_cache=True,attention_mask=mask,position_ids=pos,output_hidden_states=True)
        cols=[torch.stack(o.hidden_states,2)[:,-NT:]]; past=o.past_key_values; npos=mask.sum(-1,keepdim=True); z=m.prj(o.last_hidden_state[:,-1:,:]) if nl else None
        for s in range(nl):
            mask=torch.cat((mask,torch.ones(mask.size(0),1,dtype=mask.dtype,device=mask.device)),1)
            o=body(inputs_embeds=z,use_cache=True,past_key_values=past,attention_mask=mask,position_ids=npos,output_hidden_states=True)
            cols.append(torch.stack(o.hidden_states,2)); past=o.past_key_values; npos=npos+1; z=m.prj(o.last_hidden_state[:,-1:,:])
        emb=m.get_embd(m.codi,m.model_name)(dec)[None].expand(mask.size(0),-1,-1); T=emb.size(1)
        mask=torch.cat((mask,torch.ones(mask.size(0),T,dtype=mask.dtype,device=mask.device)),1)
        o=body(inputs_embeds=emb,use_cache=True,past_key_values=past,attention_mask=mask,position_ids=npos+torch.arange(T,device=mask.device)[None],output_hidden_states=True)
        cols.append(torch.stack(o.hidden_states,2)); C=torch.cat(cols,1)[:,K][:,:,LAY]; X[ii]=C.to('cpu',torch.float16).numpy()
        A[ii]=[num.get(t,-999) for t in m.codi.get_base_model().lm_head(o.last_hidden_state[:,-1,:])[:,:50257].argmax(-1).tolist()]
        del b,o,cols,C
        if DEV=='mps': torch.mps.empty_cache()
    return X,A
def targets(rows_):
    V=np.array([walk(*parse(r['question']))[0] for r in rows_]); assert V.min()>=0 and V.max()<=99
    n,T=V.shape; Y=np.zeros((n,20*T),np.float32)
    for j in range(T): Y[np.arange(n),20*j+V[:,j]//10]=1; Y[np.arange(n),20*j+10+V[:,j]%10]=1
    return V,Y
def dec_(P,T): return np.stack([10*P[:,20*j:20*j+10].argmax(1)+P[:,20*j+10:20*j+20].argmax(1) for j in range(T)],1)
def probe(Xa,Ya,Va,Xb):
    Xa=Xa.astype(np.float64); Xb=Xb.astype(np.float64); mu,sd=Xa.mean(0),Xa.std(0)+1e-6; Xa=(Xa-mu)/sd; Xb=(Xb-mu)/sd; ym=Ya.mean(0); Yc=(Ya-ym).astype(np.float64); c=int(len(Xa)*.85); T=Va.shape[1]
    def fit(X,Y):
        w,V=np.linalg.eigh(X.T@X); XY=V.T@(X.T@Y); return lambda l:V@(XY/(np.maximum(w,0)+l)[:,None])
    f=fit(Xa[:c],Yc[:c]); best=max((1e1,1e2,1e3,1e4,1e5),key=lambda l:float(np.mean(dec_(Xa[c:]@f(l)+ym,T)==Va[c:])))
    return dec_(Xb@fit(Xa,Yc)(best)+ym,T)
def measure(ck,nl,last):
    D=DL if last else DF; m,tok,_=load(ck,nl,DEV); m.eval(); dec=torch.tensor([m.eot_id]+tok.encode('The answer is:'),device=DEV)
    num={tok.encode(f"The answer is: {n}")[-1]:n for n in range(100)}; va=json.load(open(D+'valid.json')); out={}
    for L in LV:
        fit=rows(D+'train.json',per_level=FIT,seed=20,levels=[L]); ev=[r for r in va if int(r['level'])==L]; ev=ev[:40] if SMOKE else ev
        Xa,_=collect(m,tok,fit,nl,dec,num); Xb,Ab=collect(m,tok,ev,nl,dec,num); Va,Ya=targets(fit); Vb,_=targets(ev)
        rng=np.random.default_rng(0); pm=rng.permutation(len(Xa)); o=dict(gold=Vb[:,-1].tolist(),ans=Ab.tolist(),V=Vb.tolist(),n_fit=len(Xa),last_rule=[list(parse(r['question'])[1][-1]) for r in ev],cells={})
        h=min(4000,len(Xa))
        for si,s in enumerate(keep(nl)):
            for li,l in enumerate(LAY):
                a=Xa[:,si,li]; b=Xb[:,si,li]
                o['cells'][f'{s}|{l}']=dict(full=probe(a,Ya,Va,b).tolist(),half=probe(a[:h],Ya[:h],Va[:h],b).tolist(),shuf=probe(a,Ya[pm],Va[pm],b).tolist())
        out[L]=o; del Xa,Xb; print(time.strftime('%H:%M'),f'L{L} done right={int((Ab==Vb[:,-1]).sum())}/{len(ev)}',flush=True)
    del m
    if DEV=='mps': torch.mps.empty_cache()
    return out
PATH=OUT+('/b23_smoke.json' if SMOKE else '/b23_right_wrong.json'); RES=json.load(open(PATH)) if os.path.exists(PATH) and not SMOKE else {}
for name,ck,nl,last in MODELS:
    if name in RES: continue
    t0=time.time(); RES[name]=measure(ck,nl,last); json.dump(RES,open(PATH,'w')); print('##',name,f'{time.time()-t0:.0f}s',flush=True)
print('DONE')
