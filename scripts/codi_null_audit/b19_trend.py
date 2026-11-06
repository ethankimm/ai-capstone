"""Do start-last CODI's latents chain? Readout and latent patches across epochs, on VALID only (TRAINING_PROTOCOL section 11.3).
Usage: b19_trend.py <this dir> <device> <output dir>. SMOKE=1 runs one checkpoint on a few items.
read  : ridge probes, fit on train items of one level, scored on valid items of that level. Which v_j does each site hold? Sites: h_0 (<bot>),
        h_1..h_6 (latent pass outputs). Chaining: v_j appears at later sites as j grows. Relay: one late value at every site from early on.
patch : pairs answered exactly right, edit before the last rule. Latents from the donor at one site (single), sites 0..s (prefix) or sites s..5
        (suffix), later latents recomputed. Chaining: a patch carries forward. Relay: the next latent re-reads the state and overwrites it.
tail  : latents mean-ablated from site s on, exact by level. Chaining: deeper levels need more latents.
Reference models: lr-3e-3 CODI (a known relay, calibrates the readout) and start-last CODI-0 (h_0 only: how far the tail gets with no latents)."""
import sys, json, time, random, os; sys.path.insert(0,sys.argv[1])
from harness import *
import numpy as np
from rc import *
DEV=sys.argv[2]; OUT=sys.argv[3]; SMOKE=int(os.environ.get('SMOKE','0'))
PER=150 if SMOKE else 2000; LEVELS=[2,3,4,5,6,7]
SL=BASE+'/codi_runs_l1to7w_startlast'; DL=BASE+'/synthetic_data_v2_l1to7w_startlast/depth2/'; DF=BASE+'/synthetic_data_v2_l1to7w/depth2/'
ck=lambda s:f'{SL}/codi/run/checkpoint-{s}'
ORDER=[('ep12',ck(17820),6,True),('lr3 relay',CK['lr3'],6,False),('ep11',ck(16335),6,True),('ep9',ck(13365),6,True),('ep7',ck(10395),6,True),
       ('ep10',ck(14850),6,True),('ep8',ck(11880),6,True),('ep6',ck(8910),6,True),('CODI-0 ep12',f'{SL}/codi0/run/checkpoint-17820',0,True)]
if SMOKE: ORDER=[o for o in ORDER if o[0]=='ep11']
def text(ru):
    k,t,a,b=ru
    if k=='P': return f"if it is even, halve it; otherwise add {a}"
    if k=='T': return f"if it is at least {t}, subtract {b}; otherwise add {a}"
    return f"if it is below {t}, double it; otherwise subtract {b}"
def render(x0,rules,last):
    body=' '.join(f"Rule {i+1}: {text(r)}." for i,r in enumerate(rules))
    return f"{body} Start with {x0}. What is the final number?" if last else f"Start with {x0}. {body} What is the final number?"
def inr(x): return 0<=x<=99
def legal(k,t,a,b):
    if k=='T': return 20<=t<=80 and 1<=a<=min(20,100-t) and 1<=b<=min(20,t)
    if k=='D': return 10<=t<=50 and 1<=b<=t
    return 1<=a<=25
def taken_const(ru,b):
    k=ru[0]
    if k=='P': return None if b else 'a'
    if k=='T': return 'b' if b else 'a'
    return None if b else 'b'
NUM={}
def setup(tok):
    if not NUM:
        for n in range(100): NUM[tok.encode(f"The answer is: {n}")[-1]]=n
def batch_of(qs,tok,m): return encode([dict(question=q,steps=['x'],answer='0') for q in qs],tok,m,DEV)
@torch.no_grad()
def run(m,b,nl,dec,ov=None):
    body=body_of(m); mask=b['encoder_attention_mask'].long(); pos=(mask.cumsum(-1)-1).clamp(min=0)
    o=body(input_ids=b['encoder_input_ids'],use_cache=True,attention_mask=mask,position_ids=pos)
    past=o.past_key_values; npos=mask.sum(-1,keepdim=True); h=o.last_hidden_state[:,-1:,:]; hs=[h[:,0]]; z=m.prj(h) if nl else None; zs=[]
    for s in range(nl):
        if ov is not None and s in ov: z=ov[s]
        zs.append(z[:,0])
        mask=torch.cat((mask,torch.ones(mask.size(0),1,dtype=mask.dtype,device=mask.device)),1)
        o=body(inputs_embeds=z,use_cache=True,past_key_values=past,attention_mask=mask,position_ids=npos)
        past=o.past_key_values; npos=npos+1; h=o.last_hidden_state[:,-1:,:]; hs.append(h[:,0]); z=m.prj(h)
    emb=m.get_embd(m.codi,m.model_name)(dec)[None].expand(mask.size(0),-1,-1); T=emb.size(1)
    mask=torch.cat((mask,torch.ones(mask.size(0),T,dtype=mask.dtype,device=mask.device)),1)
    o=body(inputs_embeds=emb,use_cache=True,past_key_values=past,attention_mask=mask,position_ids=npos+torch.arange(T,device=mask.device)[None])
    a=[NUM.get(t,-999) for t in m.codi.get_base_model().lm_head(o.last_hidden_state[:,-1,:])[:,:50257].argmax(-1).tolist()]
    return a,(torch.stack(zs,1) if nl else None),torch.stack(hs,1)
def forward_all(m,tok,qs,nl,dec,ovf=None):
    """Answers, latents and pass outputs for qs, in order. ovf(indices in this batch) gives the override dict."""
    A=[None]*len(qs); Z=[None]*len(qs); H=[None]*len(qs); order=sorted(range(len(qs)),key=lambda i:len(qs[i]))
    for i in range(0,len(order),32):
        ii=order[i:i+32]; b=batch_of([qs[j] for j in ii],tok,m); a,z,h=run(m,b,nl,dec,ovf(ii) if ovf else None); h=h.cpu()
        z=z.cpu() if nl else None
        for k,j in enumerate(ii):
            A[j]=a[k]; H[j]=h[k]
            if nl: Z[j]=z[k]
        del b
        if DEV=='mps': torch.mps.empty_cache()
    return A,Z,H
def ridge(Xa,Ya,Xb,vb):
    """Ridge to digit one-hots and the raw value, lambda picked on a held-out slice of the fit items. vb: [n, T] true values at the scored items."""
    mu,sd=Xa.mean(0),Xa.std(0)+1e-6; Xa=(Xa-mu)/sd; Xb=(Xb-mu)/sd; ym=Ya.mean(0); Yc=Ya-ym
    n=len(Xa); c=int(n*.85); T=vb.shape[1]
    def fit(X,Y):
        w,V=np.linalg.eigh(X.T@X); XY=V.T@(X.T@Y)
        return lambda l:V@(XY/(w+l)[:,None])
    def score(P,v):
        assert np.isfinite(P).all()
        ex=[];w2=[]
        for j in range(T):
            t=P[:,21*j:21*j+10].argmax(1); o=P[:,21*j+10:21*j+20].argmax(1)
            ex.append(float(np.mean(10*t+o==v[:,j]))); w2.append(float(np.mean(np.abs(np.rint(P[:,21*j+20]*50+50)-v[:,j])<=2)))
        return ex,w2
    f=fit(Xa[:c],Yc[:c]); va=Ya[c:]; vv=np.stack([10*va[:,21*j:21*j+10].argmax(1)+va[:,21*j+10:21*j+20].argmax(1) for j in range(T)],1)
    best=max((1e1,1e2,1e3,1e4,1e5),key=lambda l:float(np.mean(score(Xa[c:]@f(l)+ym,vv)[0])))
    return score(Xb@fit(Xa,Yc)(best)+ym,vb)
def targets(rows_):
    """[n, T] with v_0 (the start) to v_L, and the matching probe targets."""
    V=np.array([walk(*parse(r['question']))[0] for r in rows_]); assert V.min()>=0 and V.max()<=99
    n,T=V.shape; Y=np.zeros((n,21*T))
    for j in range(T):
        Y[np.arange(n),21*j+V[:,j]//10]=1; Y[np.arange(n),21*j+10+V[:,j]%10]=1; Y[:,21*j+20]=(V[:,j]-50)/50
    return V,Y
CACHE={}
def data(last):
    if last not in CACHE:
        D=DL if last else DF; va=json.load(open(D+'valid.json')); fit=rows(D+'train.json',per_level=PER,seed=21,levels=LEVELS)
        rng=random.Random(19); P=[]
        for r in va:
            x0,rules=parse(r['question']); n=len(rules); q=r['question'].strip()
            if n<2: continue
            assert render(x0,rules,last)==q
            v,br=walk(x0,rules); c=0; js=list(range(n-1)); rng.shuffle(js)   # edits before the last rule only
            for j in js:
                k,t,a,b=rules[j]; w=taken_const(rules[j],br[j])
                if not w: continue
                for d in rng.sample([3,-3,4,-4,5,-5,6,-6,7,-7,8,-8],12):
                    na,nb=(a+d,b) if w=='a' else (a,b+d)
                    if not legal(k,t,na,nb): continue
                    nr=list(rules); nr[j]=(k,t,na,nb); v2,_=walk(x0,nr)
                    if not all(inr(y) for y in v2) or abs(v2[-1]-v[-1])<6: continue
                    P.append(dict(level=n,left=n-1-j,rq=q,dq=render(x0,nr,last),gold=v[-1],dgold=v2[-1])); c+=1; break
                if c>=2: break
        CACHE[last]=(va[::20] if SMOKE else va,fit,P[::10] if SMOKE else P)
    return CACHE[last]
def measure(ckpt,nl,last):
    va,fit,P=data(last); m,tok,_=load(ckpt,nl,DEV); m.eval(); setup(tok)
    dec=torch.tensor([m.eot_id]+tok.encode('The answer is:'),device=DEV)
    qv=[r['question'].strip() for r in va]; gv=np.array([int(r['answer']) for r in va]); lv=np.array([int(r['level']) for r in va])
    Av,Zv,Hv=forward_all(m,tok,qv,nl,dec); _,_,Hf=forward_all(m,tok,[r['question'].strip() for r in fit],nl,dec)
    out={'exact':{int(L):float(np.mean(np.array(Av)[lv==L]==gv[lv==L])) for L in sorted(set(lv))},'read':{},'base':{}}
    lf=np.array([int(r['level']) for r in fit]); Hf=torch.stack(Hf).numpy().astype(np.float64); Hv=torch.stack(Hv).numpy().astype(np.float64)
    for L in LEVELS:
        a=np.where(lf==L)[0]; b=np.where(lv==L)[0]
        if len(b)<5: continue
        Vf,Yf=targets([fit[i] for i in a]); Vv,_=targets([va[i] for i in b])
        out['base'][L]=[float(np.mean(Vv[:,j]==np.bincount(Vf[:,j]).argmax())) for j in range(Vv.shape[1])]
        out['read'][L]=[ridge(Hf[a,s],Yf,Hv[b,s],Vv) for s in range(Hf.shape[1])]
    del Hf,Hv
    if nl:
        mu=torch.stack(Zv).mean(0).to(DEV); out['tail']={}
        for s in range(nl):
            At,_,_=forward_all(m,tok,qv,nl,dec,lambda ii,s=s:{t:mu[t][None,None,:].expand(len(ii),1,-1) for t in range(s,nl)})
            out['tail'][s]={int(L):float(np.mean(np.array(At)[lv==L]==gv[lv==L])) for L in sorted(set(lv))}
        qs=[p['rq'] for p in P]+[p['dq'] for p in P]; A,Z,_=forward_all(m,tok,qs,nl,dec); n=len(P)
        keep=[i for i,p in enumerate(P) if A[i]==p['gold'] and A[n+i]==p['dgold']]
        def cond(src,sites):
            """clean question, latents at `sites` from the donor (src=1) or the item itself (src=0)."""
            ov=lambda ii:{s:torch.stack([Z[src*n+keep[j]][s] for j in ii])[:,None,:].to(DEV) for s in sites}
            return forward_all(m,tok,[P[i]['rq'] for i in keep],nl,dec,ov)[0]
        C={'self':cond(0,list(range(nl))),'all':cond(1,list(range(nl)))}
        for s in range(nl):
            C[f'single_{s}']=cond(1,[s])
            if s<nl-1: C[f'prefix_{s}']=cond(1,list(range(s+1)))
            if s: C[f'suffix_{s}']=cond(1,list(range(s,nl)))
        out['patch']=dict(pairs=[dict(level=P[i]['level'],left=P[i]['left'],gold=P[i]['gold'],dgold=P[i]['dgold']) for i in keep],cond=C)
    del m
    if DEV=='mps': torch.mps.empty_cache()
    return out
PATH=OUT+('/b19_smoke.json' if SMOKE else '/b19_trend.json'); RES=json.load(open(PATH)) if os.path.exists(PATH) and not SMOKE else {}
while len(RES)<len(ORDER):
    todo=[o for o in ORDER if o[0] not in RES and os.path.exists(o[1]+'/vocab.json')]   # vocab.json syncs last
    if not todo: time.sleep(60); continue
    name,ckpt,nl,last=todo[0]; t0=time.time(); RES[name]=measure(ckpt,nl,last); json.dump(RES,open(PATH,'w'))
    print(time.strftime('%H:%M'),name,f'{time.time()-t0:.0f}s',{k:round(v,2) for k,v in RES[name]['exact'].items()},flush=True)
print('DONE')
