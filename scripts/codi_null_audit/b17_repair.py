"""Repair test: where do wrong answers come from? For an item the model gets wrong, patch in the latents of a donor that the model answers
correctly and that has the TRUE value before the last rule and the same last rule. If the answer becomes right, the latents and answer stage
are fine and the error was already in the value they received. Usage: b17_repair.py <this dir> <device> <output dir>
Control: a donor with a value 3 away, which should give the last rule applied to that wrong value.
Below, the b16 docstring lines:
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

rng=random.Random(17)
base=rows(TE,seed=71,per_level=500,levels=[3,4,5,6,7])
def donors(t,last,n,want):
    """chains of n rules: random rules ending at value t, then the recipient's last rule"""
    out=[]
    for _ in range(40000):
        if len(out)>=want: break
        nx=rng.randint(1,99); pr=[rand_rule(rng) for _ in range(n-1)]; v2,b2=walk(nx,pr)
        if all(inr(y) for y in v2) and v2[-1]==t: out.append(render(nx,pr+[last]))
    return out
def run_all(m,tok,nl,dec,recs,src,Z):
    res={}; order=sorted(range(len(recs)),key=lambda i:len(recs[i]))
    for i in range(0,len(order),32):
        ii=order[i:i+32]; b=batch_of([recs[j] for j in ii],tok,m)
        ov={s:torch.stack([Z[src[j]][s] for j in ii])[:,None,:].to(DEV) for s in range(nl)}
        a,_=run(m,b,nl,dec,ov)
        for j,x in zip(ii,a): res[j]=x
        del b,ov
        if DEV=='mps': torch.mps.empty_cache()
    return [res[i] for i in range(len(recs))]
STORE={}
for name,ck,nl in (('r1 CODI',CK['r1'],6),('lr3 CODI',CK['lr3'],6),('stab CODI',CK['stab']%17820,6)):
    t0=time.time(); m,tok,ta=load(ck,nl,DEV); m.eval(); setup(tok)
    dec=torch.tensor([m.eot_id]+tok.encode('The answer is:'),device=DEV)
    A,Z=clean_pass(m,tok,[r['question'].strip() for r in base],nl,dec)
    wrong=[]
    for r in base:
        x0,ru=parse(r['question']); v,br=walk(x0,ru); q=r['question'].strip()
        if A[q]!=v[-1]: wrong.append(dict(item=r['id'],level=len(ru),q=q,gold=v[-1],pred=A[q],t=v[-2],last=ru[-1]))
    rng.shuffle(wrong); wrong=wrong[:(30 if SMOKE else 600)]
    cand=[];own=[]
    for w in wrong:
        t2=w['t']+rng.choice((3,-3)); bad=apply(w['last'],t2) if inr(t2) else None
        w['bad']=bad if bad is not None and inr(bad) else None
        w['true_c']=donors(w['t'],w['last'],w['level'],6)
        w['short_c']=donors(w['t'],w['last'],2,3)
        w['bad_c']=donors(t2,w['last'],w['level'],6) if w['bad'] is not None else []
        cand+=w['true_c']+w['short_c']+w['bad_c']
    A2,Z2=clean_pass(m,tok,list(set(cand)),nl,dec); Z.update(Z2); A.update(A2)
    out=[]
    for kind,key,goal in (('same length donor','true_c','gold'),('short donor','short_c','gold'),('wrong value donor','bad_c','bad')):
        recs=[];src=[];idx=[]
        for i,w in enumerate(wrong):
            g=w[goal]
            if g is None: continue
            c=[q for q in w[key] if A[q]==g]
            if c: recs.append(w['q']); src.append(c[0]); idx.append(i)
        ans=run_all(m,tok,nl,dec,recs,src,Z) if recs else []
        for i,a in zip(idx,ans): out.append(dict(kind=kind,item=wrong[i]['item'],level=wrong[i]['level'],gold=wrong[i]['gold'],pred=wrong[i]['pred'],bad=wrong[i]['bad'],ans=a))
    STORE[name]=dict(n_wrong=len(wrong),rows=out,wrong=[dict(level=w['level'],gold=w['gold'],pred=w['pred'],t=w['t'],item=w['item'],q=w['q']) for w in wrong]); del m,Z
    if DEV=='mps': torch.mps.empty_cache()
    print('##',name,f'{time.time()-t0:.0f}s wrong {len(wrong)} tested {len(out)}',flush=True)
    for kind in ('same length donor','short donor','wrong value donor'):
        rr=[x for x in out if x['kind']==kind]
        if not rr: continue
        tgt=lambda x:x['bad'] if kind=='wrong value donor' else x['gold']
        print('  ',kind,'n',len(rr),'hits target',round(float(np.mean([x['ans']==tgt(x) for x in rr])),3),'within 2',round(float(np.mean([abs(x['ans']-tgt(x))<=2 for x in rr])),3),'unchanged',round(float(np.mean([x['ans']==x['pred'] for x in rr])),3),{L:(round(float(np.mean([x['ans']==tgt(x) for x in rr if x['level']==L])),2),sum(x['level']==L for x in rr)) for L in range(3,8) if any(x['level']==L for x in rr)},flush=True)
    json.dump(STORE,open(OUT+('/b17_smoke.json' if SMOKE else '/b17_raw.json'),'w'))
print('DONE')
