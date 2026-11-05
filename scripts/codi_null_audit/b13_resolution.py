"""How precise is the running value, and how does the answer get made? Usage: b13_resolution.py <this dir> <device> <output dir>
psy    : move one T/D threshold to margin m either side of the true running value -> psychometric curve per rule position.
bit    : tiny edit (x0 or a taken constant, +-1..4) whose first effect is a later branch flip, via parity or via a threshold.
fz     : freeze test. Edit rule j; patch hidden states of LATER rule tokens between clean and edited runs, to see whether
         the edit reaches the answer through the later rule tokens (serial) or directly from rule j (late combination).
sets   : zero-shot chains built to need, or not need, exact state; same grammar, level matched."""
import sys, json, time, random, os; sys.path.insert(0,sys.argv[1])
from harness import *
import numpy as np
from rc import *
from latentreasoning.tasks.synthetic_v2 import Rule, RuleChain, chain_accepted
DEV=sys.argv[2]; OUT=sys.argv[3]; SMOKE=int(os.environ.get('SMOKE','0'))
TE=BASE+'/synthetic_data_v2_l1to7w/depth2/test.json'
def text(ru):
    k,t,a,b=ru
    if k=='P': return f"if it is even, halve it; otherwise add {a}"
    if k=='T': return f"if it is at least {t}, subtract {b}; otherwise add {a}"
    return f"if it is below {t}, double it; otherwise subtract {b}"
def render(x0,rules):
    return f"Start with {x0}. "+' '.join(f"Rule {i+1}: {text(r)}." for i,r in enumerate(rules))+" What is the final number?"
def inr(x): return 0<=x<=99
def tail_from(s,rules):
    for r in rules:
        s=apply(r,s)
        if not inr(s): return None
    return s
def legal(k,t,a,b):
    if k=='T': return 20<=t<=80 and 1<=a<=min(20,100-t) and 1<=b<=min(20,t)
    if k=='D': return 10<=t<=50 and 1<=b<=t
    return 1<=a<=25
def margin(ru,s):
    k,t,a,b=ru
    return None if k=='P' else ((s-t+1) if s>=t else (t-s))
def taken_const(ru,b):
    """which constant the taken branch uses, if any"""
    k=ru[0]
    if k=='P': return None if b else 'a'
    if k=='T': return 'b' if b else 'a'
    return None if b else 'b'
rng=random.Random(13)
base=rows(TE,per_level=(4 if SMOKE else 200),seed=41,levels=[3,4,5,6,7])
Q=[]
for r in base:
    x0,rules=parse(r['question']); assert render(x0,rules)==r['question'].strip()
    n=len(rules); v,br=walk(x0,rules); gold=v[-1]; it=dict(item=r['id'],level=n,gold=gold)
    # psychometric: threshold placed by margin m on either side of the value entering rule j
    for j,ru in enumerate(rules):
        k,t,a,b=ru
        if k=='P': continue
        s=v[j]; hi=s-b; lo=(s+a) if k=='T' else 2*s
        if not(inr(hi) and inr(lo)): continue
        fh=tail_from(hi,rules[j+1:]); fl=tail_from(lo,rules[j+1:])
        if fh is None or fl is None or abs(fh-fl)<10: continue
        for m in (1,2,3,4,6):
            for side in ('hi','lo'):
                nt=s-(m-1) if side=='hi' else s+m
                if not legal(k,nt,a,b): continue
                nr=list(rules); nr[j]=(k,nt,a,b)
                Q.append(dict(it,kind='psy',j=j+1,m=m,side=side,rk=k,q=render(x0,nr),hi=fh,lo=fl))
    # small edits and their first effect
    cand=[('x0',0,d) for d in (1,-1,2,-2,3,-3,4,-4)]
    for j,ru in enumerate(rules):
        w=taken_const(ru,br[j])
        if w:
            for d in (1,-1,2,-2,3,-3,4,-4): cand.append((w,j,d))
    rng.shuffle(cand); seen={}
    for w,j,d in cand:
        if w=='x0':
            nx=x0+d; nr=list(rules); e=0
            if not 1<=nx<=99: continue
        else:
            k,t,a,b=rules[j]; na,nb=(a+d,b) if w=='a' else (a,b+d)
            if not legal(k,t,na,nb): continue
            nx=x0; nr=list(rules); nr[j]=(k,t,na,nb); e=j+1
        v2,b2=walk(nx,nr)
        if not all(inr(x) for x in v2): continue
        diff=[i for i in range(n) if b2[i]!=br[i]]
        if not diff: continue
        kk=diff[0]; s=nx; okp=True
        for i in range(n):
            s=apply(nr[i],s,br[i])
            if not inr(s): okp=False
        if not okp or abs(v2[-1]-s)<10: continue
        cls='P' if rules[kk][0]=='P' else 'TD'
        nh=sum(1 for i in range(e,kk) if rules[i][0]=='P' and br[i])
        nhall=sum(1 for i in range(0,kk) if rules[i][0]=='P' and br[i])
        key=(cls,min(nh,1))
        if seen.get(key,0)>=3: continue
        seen[key]=seen.get(key,0)+1
        Q.append(dict(it,kind='bit',cls=cls,nh=nh,nhall=nhall,e=e,k=kk+1,dist=kk+1-e,d=d,m=margin(nr[kk],v2[kk]),q=render(nx,nr),new=v2[-1],kept=s))
print('built psy/bit',{k:sum(q['kind']==k for q in Q) for k in ('psy','bit')},flush=True)
# freeze pairs
FZ=[]
fb=[r for r in base if int(r['level'])>=4][:(12 if SMOKE else 520)]
for r in fb:
    x0,rules=parse(r['question']); n=len(rules); v,br=walk(x0,rules); gold=v[-1]; it=dict(item=r['id'],level=n,gold=gold,clean=r['question'].strip())
    js=list(range(n-1)); rng.shuffle(js); c1=c2=0
    for j in js:
        k,t,a,b=rules[j]
        w=taken_const(rules[j],br[j])
        if w and c1<3:
            for d in rng.sample([3,-3],2):
                na,nb=(a+d,b) if w=='a' else (a,b+d)
                if not legal(k,t,na,nb): continue
                nr=list(rules); nr[j]=(k,t,na,nb); v2,b2=walk(x0,nr)
                if b2!=br or not all(inr(x) for x in v2) or v2[-1]==gold: continue
                FZ.append(dict(it,kind='const',j=j+1,dist=n-1-j,q=render(x0,nr),new=v2[-1],kept=v2[-1])); c1+=1; break
        if k!='P' and c2<3:
            s=v[j]
            for _ in range(60):
                nt=rng.randint(20,80) if k=='T' else rng.randint(10,50)
                cross=(s>=nt)!=(s>=t)
                if not cross or abs(s-nt)<4 or not legal(k,nt,a,b): continue
                nr=list(rules); nr[j]=(k,nt,a,b); v2,b2=walk(x0,nr)
                if not all(inr(x) for x in v2): continue
                sk=x0; okp=True
                for i in range(n):
                    sk=apply(nr[i],sk,b2[i] if i==j else br[i])
                    if not inr(sk): okp=False
                if okp and abs(v2[-1]-gold)>=8 and abs(sk-v2[-1])>=6 and abs(sk-gold)>=6:
                    FZ.append(dict(it,kind='flip',j=j+1,dist=n-1-j,q=render(x0,nr),new=v2[-1],kept=sk)); c2+=1; break
print('freeze',{k:sum(q['kind']==k for q in FZ) for k in ('const','flip')},flush=True)
# zero-shot sets
def build(tight,halve,L):
    while True:
        s=rng.randint(1,99); x0=s; rules=[]; halved=False; pafter=False; dead=False
        for j in range(L):
            for _ in range(40):
                kd=rng.choice('PTD')
                if kd=='P':
                    ru=('P',0,rng.randint(1,25),0)
                elif kd=='T':
                    t=rng.randint(20,80)
                    if tight:
                        mm=rng.choice((1,2)); t=s-(mm-1) if rng.random()<.5 else s+mm
                        if not 20<=t<=80: continue
                    ru=('T',t,rng.randint(1,min(20,100-t)),rng.randint(1,min(20,t)))
                else:
                    t=rng.randint(10,50)
                    if tight:
                        mm=rng.choice((1,2)); t=s-(mm-1) if rng.random()<.5 else s+mm
                        if not 10<=t<=50: continue
                    ru=('D',t,0,rng.randint(1,t))
                mg=margin(ru,s)
                if mg is not None and not tight and mg<5: continue
                ns=apply(ru,s)
                if not inr(ns): continue
                break
            else:
                dead=True; break
            if ru[0]=='P':
                if halved: pafter=True
                if branch(ru,s): halved=True
            rules.append(ru); s=ns
        if dead or pafter!=halve: continue
        if tight and sum(r[0]!='P' for r in rules)*2<L-1: continue
        c=RuleChain(x0,tuple(Rule(*r) for r in rules))
        if chain_accepted(c,rng): return x0,rules
ZS=[]
for L in (3,4,5,6,7):
    for tight in (False,True):
        for halve in (False,True):
            for _ in range(6 if SMOKE else 500):
                x0,ru=build(tight,halve,L); v,_b=walk(x0,ru)
                ZS.append(dict(level=L,tight=tight,halve=halve,q=render(x0,ru),gold=v[-1],np=sum(r[0]=='P' for r in ru)))
print('sets',len(ZS),flush=True)
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
def run(m,b,nl,dec,cap=None,patch=None):
    """question pass with optional capture of every block output, or overwrite at masked positions"""
    body=body_of(m); hs=[]
    def hook(l):
        def f(mod,inp,out):
            h=out[0] if isinstance(out,tuple) else out
            if cap is not None: cap.append(h.detach())
            if patch is not None:
                src,pm=patch; h=torch.where(pm[:,:,None],src[l],h)
                return ((h,)+tuple(out[1:])) if isinstance(out,tuple) else h
        return f
    if cap is not None or patch is not None: hs=[blk.register_forward_hook(hook(l)) for l,blk in enumerate(body.h)]
    mask=b['encoder_attention_mask'].long(); pos=(mask.cumsum(-1)-1).clamp(min=0)
    o=body(input_ids=b['encoder_input_ids'],use_cache=True,attention_mask=mask,position_ids=pos)
    for h in hs: h.remove()
    return finish(m,o,mask,nl,dec)
def answers(m,tok,qs,nl,dec):
    out=[None]*len(qs); idx=sorted(range(len(qs)),key=lambda i:len(qs[i]))
    for i in range(0,len(idx),32):
        ii=idx[i:i+32]; b=batch_of([qs[j] for j in ii],tok,m)
        for j,a in zip(ii,run(m,b,nl,dec)): out[j]=a
        del b
        if DEV=='mps': torch.mps.empty_cache()
    return out
def spans(tok,q,j):
    """token positions in the encoded question: first token of rule j+1 (j is the 1-based edited rule), the ' What', total incl. <bot>"""
    ids=tok(q.strip()+'\n').input_ids
    st=[i for i,t in enumerate(ids) if tok.decode([t]).strip()=='Rule']; wh=[i for i,t in enumerate(ids) if tok.decode([t])==' What'][-1]
    return st[j],wh,len(ids)+1
def freeze(m,tok,nl,dec):
    res=[None]*len(FZ); idx=sorted(range(len(FZ)),key=lambda i:len(FZ[i]['q']))
    for i in range(0,len(idx),16):
        ii=idx[i:i+16]; qc=[FZ[j]['clean'] for j in ii]; qe=[FZ[j]['q'] for j in ii]
        bc=batch_of(qc,tok,m); be=batch_of(qe,tok,m)
        assert bc['encoder_input_ids'].shape==be['encoder_input_ids'].shape
        T=bc['encoder_input_ids'].size(1); pm=torch.zeros(len(ii),T,dtype=torch.bool,device=DEV); good=[]
        for r,j in enumerate(ii):
            a,w,tot=spans(tok,FZ[j]['clean'],FZ[j]['j']); a2,w2,tot2=spans(tok,FZ[j]['q'],FZ[j]['j']); good.append((a,w,tot)==(a2,w2,tot2))
            off=T-tot; pm[r,off+a:off+w]=True
        cc=[];ce=[]
        pc=run(m,bc,nl,dec,cap=cc); pe=run(m,be,nl,dec,cap=ce)
        pA=run(m,bc,nl,dec,patch=(ce,pm)); pB=run(m,be,nl,dec,patch=(cc,pm))
        # controls: a run patched with its own states must not change; every position from the edited run must give the edited answer
        pI=run(m,bc,nl,dec,patch=(cc,pm)); pF=run(m,bc,nl,dec,patch=(ce,torch.ones_like(pm)))
        for r,j in enumerate(ii): res[j]=dict(clean=pc[r],edit=pe[r],A=pA[r],B=pB[r],I=pI[r],F=pF[r],aligned=good[r])
        del cc,ce,bc,be
        if DEV=='mps': torch.mps.empty_cache()
    return res
STORE={}
MODELS=(('stab CODI-0',CK['stab0']%17820,0),('stab CODI',CK['stab']%17820,6),('lr3 CODI',CK['lr3'],6),('r1 CODI-0',CK['r10'],0),('r1 CODI',CK['r1'],6))
for name,ck,nl in MODELS:
    t0=time.time(); m,tok,ta=load(ck,nl,DEV); m.eval(); setup(tok)
    dec=torch.tensor([m.eot_id]+tok.encode('The answer is:'),device=DEV)
    pb=answers(m,tok,[r['question'].strip() for r in base],nl,dec)
    pq=answers(m,tok,[q['q'] for q in Q],nl,dec)
    pz=answers(m,tok,[z['q'] for z in ZS],nl,dec)
    pf=freeze(m,tok,nl,dec)
    STORE[name]=dict(base=dict(zip([r['id'] for r in base],pb)),q=pq,z=pz,fz=pf); del m
    if DEV=='mps': torch.mps.empty_cache()
    print('##',name,f'{time.time()-t0:.0f}s',flush=True)
    json.dump(dict(Q=Q,FZ=FZ,ZS=ZS,store=STORE,base=[dict(id=r['id'],level=int(r['level']),question=r['question'],answer=r['answer']) for r in base]),open(OUT+('/b13_smoke.json' if SMOKE else '/b13_raw.json'),'w'))
print('DONE')
