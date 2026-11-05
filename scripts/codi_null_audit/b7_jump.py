"""Is the epoch-12 jump in gradient through the latents real? Epochs 8..12, three item samples, per-level-balanced, same code as b1."""
import sys, json, time; sys.path.insert(0,sys.argv[1])
from harness import *
torch.manual_seed(0)
DEV='cpu'
T1=BASE+'/synthetic_data_v2_l1to7w/depth2/train.json'
samples={s:rows(T1,per_level=6,seed=s,levels=[3,4,5,6,7]) for s in (2,3,4)}
out={}
for ep in (8,9,10,11,12):
    m,tok,ta=load(CK['stab']%(1485*ep),6,DEV); m.distill_loss_div_std=True
    for p in m.parameters(): p.requires_grad_(False)
    lora=[p for n,p in m.named_parameters() if 'lora_' in n]+list(m.prj.parameters())
    for p in lora: p.requires_grad_(True)
    nprj=len(list(m.prj.parameters()))
    for sd,tr in samples.items():
        b=encode(tr,tok,m,DEV)
        o1,mask,pos=question(m,b); past=o1.past_key_values
        z=m.prj(o1.last_hidden_state[:,-1:,:]); zs=[z]
        for s in range(6):
            h,z2,past,mask,pos=latent_step(m,zs[-1],past,mask,pos); zs.append(z2)
        o=decode(m,b,past,mask,pos)
        with torch.no_grad(): t=teacher(m,b)
        ce,dist,_=losses(m,b,o,t)
        g_full=torch.autograd.grad(ce+dist,lora,retain_graph=True,allow_unused=True)
        gz=torch.autograd.grad(ce+dist,zs[:6])
        o2,mask2,pos2=question(m,b); pq=o2.past_key_values
        with torch.no_grad():
            c=pq; zz=m.prj(o2.last_hidden_state[:,-1:,:]); mk,ps=mask2,pos2
            for s in range(6): h,zz,c,mk,ps=latent_step(m,zz,c,mk,ps)
        Lq=pq[0][0].size(2)
        pc=tuple((torch.cat([pq[i][0],c[i][0][:,:,Lq:].detach()],2),torch.cat([pq[i][1],c[i][1][:,:,Lq:].detach()],2)) for i in range(len(pq)))
        o3=decode(m,b,pc,mk,ps); ce2,dist2,_=losses(m,b,o3,t)
        g_q=torch.autograd.grad(ce2+dist2,lora,allow_unused=True)
        fl=lambda g:torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1) for x,p in zip(g,lora)])
        f,q=fl(g_full),fl(g_q); lat=f-q
        per=[float(g[:,0].norm(dim=-1).max()/g[:,0].norm(dim=-1).median()) for g in gz]
        r=dict(ce=round(float(ce),3),via=round(float(lat.norm()),4),direct=round(float(q.norm()),4),share=round(float(lat.norm()/(lat.norm()+q.norm())),3),prj=round(float(torch.cat([x.reshape(-1) for x in g_full[-nprj:]]).norm()),4),max_over_median_item_grad=round(max(per),1))
        out[f'e{ep}_s{sd}']=r
        print(f'e{ep} sample{sd}',r,flush=True)
    del m
json.dump(out,open(sys.argv[1]+'/b7_jump.json','w'),indent=1)
