"""Reads b13_raw.json. Usage: b15_route.py <this dir> <device> <artifacts dir>
abl   : mean-ablate every latent on the constructed chains. Does CODI's gain come from the latents or from its weights?
block : edit rule j, then block or supply ONE later rule's tokens. Rule-by-rule passing and a parallel read at the last rule predict different things.
grid  : patch one later rule's tokens at one layer from the edited run. Where the patch works gives the layer the state is needed at each rule."""
import sys, json, time, os; sys.path.insert(0,sys.argv[1])
from harness import *
import numpy as np
DEV=sys.argv[2]; OUT=sys.argv[3]; SMOKE=int(os.environ.get('SMOKE','0'))
D=json.load(open(OUT+'/b13_raw.json')); FZ=[q for q in D['FZ'] if q['dist']>=2]; ZS=D['ZS']; base=D['base']
if SMOKE: FZ=FZ[:20]; ZS=ZS[::500]; base=base[:32]
print('pairs',len(FZ),'sets',len(ZS),flush=True)
NUM={}
def setup(tok):
    if not NUM:
        for n in range(100): NUM[tok.encode(f"The answer is: {n}")[-1]]=n
def batch_of(qs,tok,m): return encode([dict(question=q,steps=['x'],answer='0') for q in qs],tok,m,DEV)
def finish(m,o,mask,nl,dec):
    body=body_of(m); past=o.past_key_values; npos=mask.sum(-1,keepdim=True)
    z=m.prj(o.last_hidden_state[:,-1:,:]) if nl else None
    for s in range(nl):
        mask=torch.cat((mask,torch.ones(mask.size(0),1,dtype=mask.dtype,device=mask.device)),1)
        o=body(inputs_embeds=z,use_cache=True,past_key_values=past,attention_mask=mask,position_ids=npos)
        past=o.past_key_values; npos=npos+1; z=m.prj(o.last_hidden_state[:,-1:,:])
    emb=m.get_embd(m.codi,m.model_name)(dec)[None].expand(mask.size(0),-1,-1); T=emb.size(1)
    mask=torch.cat((mask,torch.ones(mask.size(0),T,dtype=mask.dtype,device=mask.device)),1)
    o=body(inputs_embeds=emb,use_cache=True,past_key_values=past,attention_mask=mask,position_ids=npos+torch.arange(T,device=mask.device)[None])
    return [NUM.get(t,-999) for t in m.codi.get_base_model().lm_head(o.last_hidden_state[:,-1,:])[:,:50257].argmax(-1).tolist()]
@torch.no_grad()
def run(m,b,nl,dec,cap=None,patch=None,layer=None):
    """question pass; capture every block output, or overwrite masked positions at every block (or only at `layer`)"""
    body=body_of(m); hs=[]
    def hook(l):
        def f(mod,inp,out):
            h=out[0] if isinstance(out,tuple) else out
            if cap is not None: cap.append(h.detach())
            if patch is not None and (layer is None or l==layer):
                src,pm=patch; h=torch.where(pm[:,:,None],src[l],h)
                return ((h,)+tuple(out[1:])) if isinstance(out,tuple) else h
        return f
    if cap is not None or patch is not None: hs=[blk.register_forward_hook(hook(l)) for l,blk in enumerate(body.h)]
    mask=b['encoder_attention_mask'].long(); pos=(mask.cumsum(-1)-1).clamp(min=0)
    o=body(input_ids=b['encoder_input_ids'],use_cache=True,attention_mask=mask,position_ids=pos)
    for h in hs: h.remove()
    return finish(m,o,mask,nl,dec)
def rspans(tok,q):
    ids=tok(q.strip()+'\n').input_ids
    st=[i for i,t in enumerate(ids) if tok.decode([t]).strip()=='Rule']; wh=[i for i,t in enumerate(ids) if tok.decode([t])==' What'][-1]
    return st+[wh],len(ids)+1      # st[k-1]..st[k] is rule k
@torch.no_grad()
def means(m,tok,nl,dec):
    acc=None;n=0
    qs=[r['question'].strip() for r in base]
    for i in range(0,len(qs),32):
        b=batch_of(qs[i:i+32],tok,m); _,z=fast_answer(m,b,nl,dec_ids=dec); s=z.sum(0); acc=s if acc is None else acc+s; n+=z.size(0)
        if DEV=='mps': torch.mps.empty_cache()
    return acc/n
@torch.no_grad()
def ablated(m,tok,qs,nl,dec,mu):
    out=[None]*len(qs); idx=sorted(range(len(qs)),key=lambda i:len(qs[i]))
    for i in range(0,len(idx),32):
        ii=idx[i:i+32]; b=batch_of([qs[j] for j in ii],tok,m); lg,_=fast_answer(m,b,nl,mu=mu,dec_ids=dec)
        for j,t in zip(ii,lg.argmax(-1).tolist()): out[j]=NUM.get(t,-999)
        del lg,b
        if DEV=='mps': torch.mps.empty_cache()
    return out
def route(m,tok,nl,dec):
    res=[None]*len(FZ); idx=sorted(range(len(FZ)),key=lambda i:len(FZ[i]['q']))
    for i in range(0,len(idx),16):
        ii=idx[i:i+16]; bc=batch_of([FZ[j]['clean'] for j in ii],tok,m); be=batch_of([FZ[j]['q'] for j in ii],tok,m)
        T=bc['encoder_input_ids'].size(1); assert be['encoder_input_ids'].size(1)==T
        B=len(ii); mn=torch.zeros(B,T,dtype=torch.bool,device=DEV); ml=torch.zeros_like(mn); rel={}
        for r,j in enumerate(ii):
            q=FZ[j]; st,tot=rspans(tok,q['clean']); off=T-tot; jj=q['j']; n=q['level']
            mn[r,off+st[jj]:off+st[jj+1]]=True; ml[r,off+st[n-1]:off+st[n]]=True
            for k in range(1,q['dist']+1):
                rel.setdefault(k,torch.zeros_like(mn))[r,off+st[jj+k-1]:off+st[jj+k]]=True
        cc=[];ce=[]
        pc=run(m,bc,nl,dec,cap=cc); pe=run(m,be,nl,dec,cap=ce)
        Bn=run(m,be,nl,dec,patch=(cc,mn)); Bl=run(m,be,nl,dec,patch=(cc,ml)); An=run(m,bc,nl,dec,patch=(ce,mn)); Al=run(m,bc,nl,dec,patch=(ce,ml))
        grid={}
        for k,mk in rel.items():
            for l in range(12):
                grid[(k,l)]=run(m,bc,nl,dec,patch=(ce,mk),layer=l)
        for r,j in enumerate(ii):
            res[j]=dict(clean=pc[r],edit=pe[r],Bnext=Bn[r],Blast=Bl[r],Anext=An[r],Alast=Al[r],
                        grid={f'{k}|{l}':v[r] for (k,l),v in grid.items() if k<=FZ[j]['dist']})
        del cc,ce,bc,be
        if DEV=='mps': torch.mps.empty_cache()
    return res
STORE={}
MODELS=(('stab CODI-0',CK['stab0']%17820,0),('stab CODI',CK['stab']%17820,6),('lr3 CODI',CK['lr3'],6),('r1 CODI-0',CK['r10'],0),('r1 CODI',CK['r1'],6))
for name,ck,nl in MODELS:
    t0=time.time(); m,tok,ta=load(ck,nl,DEV); m.eval(); setup(tok)
    dec=torch.tensor([m.eot_id]+tok.encode('The answer is:'),device=DEV); s={}
    if nl:
        mu=means(m,tok,nl,dec); s['abl']=ablated(m,tok,[z['q'] for z in ZS],nl,dec,mu)
        s['intact_check']=ablated(m,tok,[z['q'] for z in ZS[:200]],nl,dec,None)
    s['route']=route(m,tok,nl,dec)
    STORE[name]=s; del m
    if DEV=='mps': torch.mps.empty_cache()
    print('##',name,f'{time.time()-t0:.0f}s',flush=True)
    json.dump(dict(FZ=FZ,store=STORE),open(OUT+('/b15_smoke.json' if SMOKE else '/b15_raw.json'),'w'))
print('DONE')
