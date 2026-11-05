"""Donor patch, mediation and same-value donors on CODI latents (TRAINING_PROTOCOL section 11.3; paper draft RQ2).
Usage: b16_donor.py <this dir> <device> <output dir> [name=checkpoint:num_latent ...]. FORMAT=startlast uses the start-last format.
diff pair : donor = recipient with one taken constant changed, so the value after that rule differs and the gold answer moves by >= 6.
same pair : donor = different start and earlier rules, same value after rule i, same later rules, so the same gold answer.
Conditions, sites z_0..z_{K-1} as in ablate_latents_rulechain_codi.py, later latents recomputed unless overwritten:
  self       recipient question, its own latents written back: a control, must equal the clean answer
  all        recipient question, every latent from the donor (through latents)
  direct     donor question, every latent from the recipient (direct route)
  single_s   recipient question, z_s from the donor
  suffix_s   recipient question, z_t from the donor for t >= s"""
import sys, json, time, random, os; sys.path.insert(0,sys.argv[1])
from harness import *
import numpy as np
from rc import *
DEV=sys.argv[2]; OUT=sys.argv[3]; SMOKE=int(os.environ.get('SMOKE','0')); FMT=os.environ.get('FORMAT','')
TE=BASE+('/synthetic_data_v2_l1to7w_startlast' if FMT else '/synthetic_data_v2_l1to7w')+'/depth2/test.json'
def text(ru):
    k,t,a,b=ru
    if k=='P': return f"if it is even, halve it; otherwise add {a}"
    if k=='T': return f"if it is at least {t}, subtract {b}; otherwise add {a}"
    return f"if it is below {t}, double it; otherwise subtract {b}"
def render(x0,rules):
    body=' '.join(f"Rule {i+1}: {text(r)}." for i,r in enumerate(rules))
    return f"{body} Start with {x0}. What is the final number?" if FMT else f"Start with {x0}. {body} What is the final number?"
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
def rand_rule(rng):
    k=rng.choice('PTD')
    if k=='P': return ('P',0,rng.randint(1,25),0)
    if k=='T':
        t=rng.randint(20,80); return ('T',t,rng.randint(1,min(20,100-t)),rng.randint(1,min(20,t)))
    t=rng.randint(10,50); return ('D',t,0,rng.randint(1,t))
rng=random.Random(16)
base=rows(TE,per_level=(4 if SMOKE else 200),seed=61,levels=[2,3,4,5,6,7])
P=[]
for r in base:
    x0,rules=parse(r['question']); q=r['question'].strip(); assert render(x0,rules)==q
    n=len(rules); v,br=walk(x0,rules); gold=v[-1]; c=0
    js=list(range(n)); rng.shuffle(js)
    for j in js:
        k,t,a,b=rules[j]; w=taken_const(rules[j],br[j])
        if not w: continue
        for d in rng.sample([3,-3,4,-4,5,-5,6,-6,7,-7,8,-8],12):
            na,nb=(a+d,b) if w=='a' else (a,b+d)
            if not legal(k,t,na,nb): continue
            nr=list(rules); nr[j]=(k,t,na,nb); v2,b2=walk(x0,nr)
            if not all(inr(y) for y in v2) or abs(v2[-1]-gold)<6: continue
            P.append(dict(kind='diff',item=r['id'],level=n,rule=j+1,left=n-1-j,rq=q,dq=render(x0,nr),gold=gold,dgold=v2[-1],same_path=b2==br)); c+=1; break
        if c>=2: break
    if n>=2:
        i=rng.randint(1,n-1)
        for _ in range(4000):
            nx=rng.randint(1,99); pr=[rand_rule(rng) for _ in range(i)]; v2,b2=walk(nx,pr)
            if all(inr(y) for y in v2) and v2[-1]==v[i] and (nx,pr)!=(x0,rules[:i]):
                P.append(dict(kind='same',item=r['id'],level=n,rule=i,left=n-i,rq=q,dq=render(nx,pr+rules[i:]),gold=gold,dgold=gold,same_path=True)); break
print('pairs',{k:sum(p['kind']==k for p in P) for k in ('diff','same')},flush=True)
NUM={}
def setup(tok):
    if not NUM:
        for n in range(100): NUM[tok.encode(f"The answer is: {n}")[-1]]=n
def batch_of(qs,tok,m): return encode([dict(question=q,steps=['x'],answer='0') for q in qs],tok,m,DEV)
@torch.no_grad()
def run(m,b,nl,dec,ov=None):
    body=body_of(m); mask=b['encoder_attention_mask'].long(); pos=(mask.cumsum(-1)-1).clamp(min=0)
    o=body(input_ids=b['encoder_input_ids'],use_cache=True,attention_mask=mask,position_ids=pos)
    past=o.past_key_values; npos=mask.sum(-1,keepdim=True); z=m.prj(o.last_hidden_state[:,-1:,:]); zs=[]
    for s in range(nl):
        if ov is not None and s in ov: z=ov[s]
        zs.append(z[:,0])
        mask=torch.cat((mask,torch.ones(mask.size(0),1,dtype=mask.dtype,device=mask.device)),1)
        o=body(inputs_embeds=z,use_cache=True,past_key_values=past,attention_mask=mask,position_ids=npos)
        past=o.past_key_values; npos=npos+1; z=m.prj(o.last_hidden_state[:,-1:,:])
    emb=m.get_embd(m.codi,m.model_name)(dec)[None].expand(mask.size(0),-1,-1); T=emb.size(1)
    mask=torch.cat((mask,torch.ones(mask.size(0),T,dtype=mask.dtype,device=mask.device)),1)
    o=body(inputs_embeds=emb,use_cache=True,past_key_values=past,attention_mask=mask,position_ids=npos+torch.arange(T,device=mask.device)[None])
    a=[NUM.get(t,-999) for t in m.codi.get_base_model().lm_head(o.last_hidden_state[:,-1,:])[:,:50257].argmax(-1).tolist()]
    return a,torch.stack(zs,1)
def clean_pass(m,tok,qs,nl,dec):
    A={};Z={}; idx=sorted(qs,key=len)
    for i in range(0,len(idx),32):
        c=idx[i:i+32]; b=batch_of(c,tok,m); a,z=run(m,b,nl,dec); z=z.cpu()
        for q,x,y in zip(c,a,z): A[q]=x; Z[q]=y
        del b
        if DEV=='mps': torch.mps.empty_cache()
    return A,Z
def cond(m,tok,nl,dec,items,qof,ovof):
    """items: pair indices. qof(p) = question run; ovof(p) = (source question, sites)."""
    out={}; order=sorted(items,key=lambda i:len(qof(P[i])))
    for i in range(0,len(order),32):
        ii=order[i:i+32]; b=batch_of([qof(P[j]) for j in ii],tok,m)
        src=[ovof(P[j]) for j in ii]; sites=src[0][1]
        ov={s:torch.stack([Z[q][s] for q,_ in src])[:,None,:].to(DEV) for s in sites}
        a,_=run(m,b,nl,dec,ov)
        for j,x in zip(ii,a): out[j]=x
        del b,ov
        if DEV=='mps': torch.mps.empty_cache()
    return out
if len(sys.argv)>4: MODELS=[(s.split('=')[0],s.split('=',1)[1].rsplit(':',1)[0],int(s.rsplit(':',1)[1])) for s in sys.argv[4:]]
else: MODELS=[('stab CODI',CK['stab']%17820,6),('lr3 CODI',CK['lr3'],6),('r1 CODI',CK['r1'],6)]
qs=list({p['rq'] for p in P}|{p['dq'] for p in P}); STORE={}
for name,ck,nl in MODELS:
    t0=time.time(); m,tok,ta=load(ck,nl,DEV); m.eval(); setup(tok)
    dec=torch.tensor([m.eot_id]+tok.encode('The answer is:'),device=DEV)
    A,Z=clean_pass(m,tok,qs,nl,dec)
    keep=[i for i,p in enumerate(P) if A[p['rq']]==p['gold'] and A[p['dq']]==p['dgold']]
    diff=[i for i in keep if P[i]['kind']=='diff']; same=[i for i in keep if P[i]['kind']=='same']
    allS=list(range(nl)); R={}
    R['self']=cond(m,tok,nl,dec,diff,lambda p:p['rq'],lambda p:(p['rq'],allS))   # control: own latents must change nothing
    R['all']=cond(m,tok,nl,dec,diff+same,lambda p:p['rq'],lambda p:(p['dq'],allS))
    R['direct']=cond(m,tok,nl,dec,diff,lambda p:p['dq'],lambda p:(p['rq'],allS))
    for s in range(nl):
        R[f'single_{s}']=cond(m,tok,nl,dec,diff,lambda p:p['rq'],lambda p,s=s:(p['dq'],[s]))
        if s: R[f'suffix_{s}']=cond(m,tok,nl,dec,diff,lambda p:p['rq'],lambda p,s=s:(p['dq'],list(range(s,nl))))
    STORE[name]=dict(keep=keep,res={c:{str(k):v for k,v in d.items()} for c,d in R.items()},nl=nl); del m,Z
    if DEV=='mps': torch.mps.empty_cache()
    print('##',name,f'{time.time()-t0:.0f}s kept diff {len(diff)} same {len(same)}',flush=True)
    json.dump(dict(P=P,store=STORE),open(OUT+('/b16_smoke.json' if SMOKE else ('/b16_raw_startlast.json' if FMT else '/b16_raw.json')),'w'))
print('DONE')
