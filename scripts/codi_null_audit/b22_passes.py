"""Do latents reach a fixed point, and do extra passes help? Usage: b22_passes.py <this dir> <device> <output dir>. Valid only.
Answer after k latent passes, k = 1..12, by truly stopping or continuing (not mean ablation; k != 6 is outside training), and how much each latent moves."""
import sys, json, time, os; sys.path.insert(0,sys.argv[1])
from harness import *
import numpy as np
from rc import *
DEV=sys.argv[2]; OUT=sys.argv[3]; K=12
SL=BASE+'/codi_runs_l1to7w_startlast'; DL=BASE+'/synthetic_data_v2_l1to7w_startlast/depth2/'; DF=BASE+'/synthetic_data_v2_l1to7w/depth2/'
MODELS=[('startlast CODI',f'{SL}/codi/run/checkpoint-17820',DL),('lr3 relay',CK['lr3'],DF)]
RES={}
for name,ck,D in MODELS:
    m,tok,_=load(ck,6,DEV); m.eval(); num={tok.encode(f"The answer is: {n}")[-1]:n for n in range(100)}; body=body_of(m)
    dec=torch.tensor([m.eot_id]+tok.encode('The answer is:'),device=DEV); va=json.load(open(D+'valid.json')); va.sort(key=lambda r:len(r['question']))
    A=np.zeros((len(va),K),int); Z=np.zeros((len(va),K+1,768),np.float32)
    with torch.no_grad():
        for i in range(0,len(va),32):
            rs=va[i:i+32]; b=encode([dict(question=r['question'].strip(),steps=['x'],answer='0') for r in rs],tok,m,DEV)
            mask=b['encoder_attention_mask'].long(); pos=(mask.cumsum(-1)-1).clamp(min=0)
            o=body(input_ids=b['encoder_input_ids'],use_cache=True,attention_mask=mask,position_ids=pos); past=o.past_key_values; npos=mask.sum(-1,keepdim=True); z=m.prj(o.last_hidden_state[:,-1:,:])
            Z[i:i+len(rs),0]=z[:,0].cpu().numpy()
            for s in range(K):
                mask=torch.cat((mask,torch.ones(mask.size(0),1,dtype=mask.dtype,device=mask.device)),1)
                o=body(inputs_embeds=z,use_cache=True,past_key_values=past,attention_mask=mask,position_ids=npos); past=o.past_key_values; npos=npos+1; z=m.prj(o.last_hidden_state[:,-1:,:])
                Z[i:i+len(rs),s+1]=z[:,0].cpu().numpy()
                emb=m.get_embd(m.codi,m.model_name)(dec)[None].expand(mask.size(0),-1,-1); T=emb.size(1)
                fm=torch.cat((mask,torch.ones(mask.size(0),T,dtype=mask.dtype,device=mask.device)),1)
                o2=body(inputs_embeds=emb,use_cache=True,past_key_values=clone_cache(past),attention_mask=fm,position_ids=npos+torch.arange(T,device=mask.device)[None])
                A[i:i+len(rs),s]=[num.get(t,-999) for t in m.codi.get_base_model().lm_head(o2.last_hidden_state[:,-1,:])[:,:50257].argmax(-1).tolist()]
            del b,o,o2,past
            if DEV=='mps': torch.mps.empty_cache()
    g=np.array([int(r['answer']) for r in va]); lv=np.array([int(r['level']) for r in va])
    Zc=Z-Z.mean(0,keepdims=True); cos=[float(np.mean(np.sum(Zc[:,s]*Zc[:,s+1],1)/(np.linalg.norm(Zc[:,s],axis=1)*np.linalg.norm(Zc[:,s+1],axis=1)+1e-9))) for s in range(K)]
    move=[float(np.mean(np.linalg.norm(Z[:,s+1]-Z[:,s],axis=1)/(np.linalg.norm(Z[:,s],axis=1)+1e-9))) for s in range(K)]
    RES[name]=dict(exact={int(L):[float(np.mean(A[lv==L,k]==g[lv==L])) for k in range(K)] for L in sorted(set(lv))},within={int(L):[float(np.mean(np.abs(A[lv==L,k]-g[lv==L])<=2)) for k in range(K)] for L in sorted(set(lv))},cos=cos,move=move)
    json.dump(RES,open(OUT+'/b22_passes.json','w')); print('##',name,flush=True); del m
    if DEV=='mps': torch.mps.empty_cache()
print('DONE')
