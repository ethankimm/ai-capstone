import sys, json, time; sys.path.insert(0,sys.argv[1])
from harness import *
torch.manual_seed(0)
DEV='cpu'
T1=BASE+'/synthetic_data_v2_l1to7w/depth2/train.json'
tr=rows(T1,per_level=6,seed=2,levels=[3,4,5,6,7])   # 30 items at the levels that matter
targets=[('stab e1',CK['stab']%1485),('stab e3',CK['stab']%4455),('stab e6',CK['stab']%8910),('stab e9',CK['stab']%13365),('stab e12',CK['stab']%17820),('lr3e-3 e12',CK['lr3']),('round1 e40',CK['r1'])]
if os.environ.get('STARTLAST'):   # section 11: same measurement on the start-last checkpoints
    S=BASE+'/codi_runs_l1to7w_startlast/codi/run/checkpoint-%d'; T1=BASE+'/synthetic_data_v2_l1to7w_startlast/depth2/train.json'
    tr=rows(T1,per_level=6,seed=2,levels=[3,4,5,6,7]); targets=[(f'startlast e{e}',S%(1485*e)) for e in (3,4,5,6,9,12)]
res={}
for name,ck in targets:
    t0=time.time()
    m,tok,ta=load(ck,6,DEV); m.distill_loss_div_std=True
    for p in m.parameters(): p.requires_grad_(False)
    lora=[p for n,p in m.named_parameters() if 'lora_' in n]+list(m.prj.parameters())
    for p in lora: p.requires_grad_(True)
    b=encode(tr,tok,m,DEV)
    # --- full graph
    out,mask,pos=question(m,b); past=out.past_key_values
    z=m.prj(out.last_hidden_state[:,-1:,:]); z.retain_grad()
    zs=[z]; 
    for s in range(6):
        h,z2,past,mask,pos=latent_step(m,zs[-1],past,mask,pos)
        z2.retain_grad(); zs.append(z2)
    o=decode(m,b,past,mask,pos)
    with torch.no_grad(): t=teacher(m,b)
    ce,dist,_=losses(m,b,o,t)
    loss=ce+dist
    g_full=torch.autograd.grad(loss,lora,retain_graph=True,allow_unused=True)
    gz=torch.autograd.grad(loss,zs[:6],retain_graph=True)
    gz_ce=torch.autograd.grad(ce,zs[:6],retain_graph=True)
    gq=None
    # --- question-only graph: latent passes run on constants (their K/V and inputs detached), question KV keeps grad
    out2,mask2,pos2=question(m,b); pastq=out2.past_key_values
    kq=[p[0] for p in pastq]; vq=[p[1] for p in pastq]
    with torch.no_grad():
        c=clone_cache(pastq); zz=m.prj(out2.last_hidden_state[:,-1:,:]); mk,ps=mask2,pos2
        for s in range(6):
            h,zz,c,mk,ps=latent_step(m,zz,c,mk,ps)
    Lq=kq[0].size(2)
    pc=tuple((torch.cat([kq[i],c[i][0][:,:,Lq:].detach()],2),torch.cat([vq[i],c[i][1][:,:,Lq:].detach()],2)) for i in range(len(kq)))
    o2=decode(m,b,pc,mk,ps)
    ce2,dist2,_=losses(m,b,o2,t)
    g_q=torch.autograd.grad(ce2+dist2,lora,allow_unused=True)
    def flat(g): return torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1) for x,p in zip(g,lora)])
    f,q=flat(g_full),flat(g_q); lat=f-q
    r={'ce':float(ce),'dist':float(dist),'g_full':float(f.norm()),'g_direct':float(q.norm()),'g_via_latents':float(lat.norm()),
       'share_via_latents':float(lat.norm()/(lat.norm()+q.norm())),'cos(direct,latent)':float(F.cosine_similarity(q,lat,dim=0)),
       'prj_grad':float(torch.cat([x.reshape(-1) for x in g_full[-len(list(m.prj.parameters())):]]).norm()),
       'dL/dz_s (rel = ||g||*||z||, per item)':[round(float((g[:,0].norm(dim=-1)*zs[i][:,0].norm(dim=-1)).mean()),4) for i,g in enumerate(gz)],
       'dCE/dz_s':[round(float((g[:,0].norm(dim=-1)*zs[i][:,0].norm(dim=-1)).mean()),4) for i,g in enumerate(gz_ce)]}
    res[name]=r
    print(name,json.dumps(r),f'{time.time()-t0:.0f}s',flush=True)
    del m
json.dump(res,open(os.environ.get('OUT',sys.argv[1])+'/b1_grad'+('_startlast' if os.environ.get('STARTLAST') else '')+'.json','w'),indent=1)
