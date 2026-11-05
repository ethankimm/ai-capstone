"""Does latent s+1 depend on latent s? (recurrence inertness), and contraction of the map z_s -> z_{s+1}."""
import sys, json, time; sys.path.insert(0,sys.argv[1])
from harness import *
torch.manual_seed(0)
DEV=sys.argv[2] if len(sys.argv)>2 else 'cpu'
TE=BASE+'/synthetic_data_v2_l1to7w/depth2/test.json'
te=rows(TE,per_level=16,seed=3,levels=[3,4,5,6,7])  # 80 items
targets=[('stab e12',CK['stab']%17820),('lr3e-3 e12',CK['lr3']),('round1 e40',CK['r1'])]
res={}
@torch.no_grad()
def chain(m,b):
    out,mask,pos=question(m,b); past=out.past_key_values
    z=m.prj(out.last_hidden_state[:,-1:,:])
    zs=[z]; pasts=[];st=[]
    for s in range(6):
        st.append((clone_cache(past),mask,pos))
        h,z,past,mask,pos=latent_step(m,z,past,mask,pos)
        zs.append(z)
    return zs,st
for name,ck in targets:
    t0=time.time()
    m,tok,ta=load(ck,6,DEV)
    for p in m.parameters(): p.requires_grad_(False)
    b=encode(te,tok,m,DEV)
    zs,st=chain(m,b)
    N=zs[0].size(0)
    perm=torch.roll(torch.arange(N,device=DEV),17)
    rows_=[]
    for s in range(6):
        past,mask,pos=st[s]
        with torch.no_grad():
            # own context, own z_s  -> z_{s+1}
            c=clone_cache(past); _,zo,_,_,_=latent_step(m,zs[s],c,mask,pos)
            # own context, other item's z_s
            c=clone_cache(past); _,zx,_,_,_=latent_step(m,zs[s][perm],c,mask,pos)
            # own context, population-mean z_s
            c=clone_cache(past); _,zm,_,_,_=latent_step(m,zs[s].mean(0,keepdim=True).expand_as(zs[s]),c,mask,pos)
        mu=zo.mean(0,keepdim=True)
        spread=(zo-mu).norm(dim=-1).mean()
        d_swap=(zo-zx).norm(dim=-1).mean()
        d_mean=(zo-zm).norm(dim=-1).mean()
        dep=float(d_swap/ (spread*2**.5))        # 1.0 ~ output as different as two unrelated items; 0 ~ z_s has no influence
        # Jacobian top singular value of z_s -> z_{s+1} (power iteration through jvp-by-finite-difference + vjp)
        zin=zs[s].detach().clone().requires_grad_(True)
        c=clone_cache(past)
        _,zout,_,_,_=latent_step(m,zin,c,mask,pos)
        u=torch.randn_like(zin); u=u/u.norm(dim=-1,keepdim=True)
        sig=None
        for it in range(12):
            with torch.no_grad():
                eps=1e-2
                c=clone_cache(past); _,zp,_,_,_=latent_step(m,zin+eps*u,c,mask,pos)
                c=clone_cache(past); _,zn,_,_,_=latent_step(m,zin-eps*u,c,mask,pos)
                v=(zp-zn)/(2*eps)
            sig=v.norm(dim=-1)  # ||J u||
            w=v/ (sig[...,None]+1e-9)
            g=torch.autograd.grad(zout,zin,grad_outputs=w,retain_graph=True)[0]
            u=g/(g.norm(dim=-1,keepdim=True)+1e-9)
        rows_.append({'s':s,'dependence_on_prev':round(dep,4),'|z_s|':round(float(zs[s].norm(dim=-1).mean()),2),'spread_next':round(float(spread),3),'top_sv_median':round(float(sig.median()),4),'top_sv_p90':round(float(sig.quantile(.9)),4)})
    res[name]=rows_
    print(name,f'{time.time()-t0:.0f}s',flush=True)
    for r in rows_: print('  ',r,flush=True)
    del m
json.dump(res,open(sys.argv[1]+'/b2_recur.json','w'),indent=1)
