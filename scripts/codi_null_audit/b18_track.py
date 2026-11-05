"""Latent use over training, on VALID only, on the Mac from checkpoints the watcher has synced (TRAINING_PROTOCOL section 11.2).
Usage: b18_track.py <this dir> <device> <output dir>. Runs until both start-last arms have reported their last checkpoint and the original
stabilised CODI has been measured as the reference. Per checkpoint: exact and within-2 accuracy by level; for CODI also all latents
mean-ablated, and the donor test (latents swapped, or question swapped) split by edits before and at the last rule."""
import sys, json, time, random, os, glob; sys.path.insert(0,sys.argv[1])
from harness import *
import numpy as np
from rc import *
DEV=sys.argv[2]; OUT=sys.argv[3]; SMOKE=int(os.environ.get('SMOKE','0'))
SL=BASE+'/codi_runs_l1to7w_startlast'; LAST=17820
ARMS=[('startlast CODI',SL+'/codi/run',6,True),('startlast CODI-0',SL+'/codi0/run',0,True),('original CODI',BASE+'/codi_runs_l1to7w_stab/codi/run',6,False)]
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
rng=random.Random(18)
VA=json.load(open(BASE+'/synthetic_data_v2_l1to7w/depth2/valid.json'))
sub=rows(BASE+'/synthetic_data_v2_l1to7w/depth2/valid.json',per_level=(6 if SMOKE else 60),seed=81,levels=[2,3,4,5,6,7])
PAIRS=[]
for r in sub:
    x0,rules=parse(r['question']); n=len(rules); v,br=walk(x0,rules); c=0
    js=list(range(n)); rng.shuffle(js)
    for j in js:
        k,t,a,b=rules[j]; w=taken_const(rules[j],br[j])
        if not w: continue
        for d in rng.sample([3,-3,4,-4,5,-5,6,-6,7,-7,8,-8],12):
            na,nb=(a+d,b) if w=='a' else (a,b+d)
            if not legal(k,t,na,nb): continue
            nr=list(rules); nr[j]=(k,t,na,nb); v2,b2=walk(x0,nr)
            if not all(inr(y) for y in v2) or abs(v2[-1]-v[-1])<6: continue
            PAIRS.append(dict(x0=x0,rules=rules,nr=nr,gold=v[-1],dgold=v2[-1],left=n-1-j,level=n)); c+=1; break
        if c>=2: break
NUM={}
def setup(tok):
    if not NUM:
        for n in range(100): NUM[tok.encode(f"The answer is: {n}")[-1]]=n
def batch_of(qs,tok,m): return encode([dict(question=q,steps=['x'],answer='0') for q in qs],tok,m,DEV)
@torch.no_grad()
def run(m,b,nl,dec,ov=None):
    body=body_of(m); mask=b['encoder_attention_mask'].long(); pos=(mask.cumsum(-1)-1).clamp(min=0)
    o=body(input_ids=b['encoder_input_ids'],use_cache=True,attention_mask=mask,position_ids=pos)
    past=o.past_key_values; npos=mask.sum(-1,keepdim=True); z=m.prj(o.last_hidden_state[:,-1:,:]) if nl else None; zs=[]
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
    return a,(torch.stack(zs,1) if nl else None)
def forward_all(m,tok,qs,nl,dec,ovf=None):
    """qs: list of questions. ovf(batch indices) -> override dict or None. Returns answers and latents in order."""
    A=[None]*len(qs); Z=[None]*len(qs); order=sorted(range(len(qs)),key=lambda i:len(qs[i]))
    for i in range(0,len(order),32):
        ii=order[i:i+32]; b=batch_of([qs[j] for j in ii],tok,m); a,z=run(m,b,nl,dec,ovf(ii) if ovf else None)
        for k,j in enumerate(ii):
            A[j]=a[k]
            if nl: Z[j]=z[k].cpu()
        del b
        if DEV=='mps': torch.mps.empty_cache()
    return A,Z
def score(golds,levels,A):
    e=np.abs(np.array(A)-np.array(golds)); lv=np.array(levels)
    return {int(L):[round(float(np.mean(e[lv==L]==0)),4),round(float(np.mean(e[lv==L]<=2)),4)] for L in sorted(set(levels))}
def measure(ck,nl,last):
    m,tok,ta=load(ck,nl,DEV); m.eval(); setup(tok)
    dec=torch.tensor([m.eot_id]+tok.encode('The answer is:'),device=DEV)
    items=VA[::33] if SMOKE else VA; pr=[parse(r['question']) for r in items]
    qs=[render(x,ru,last) for x,ru in pr]; golds=[int(r['answer']) for r in items]; lv=[int(r['level']) for r in items]
    A,Z=forward_all(m,tok,qs,nl,dec); out={'clean':score(golds,lv,A)}
    if nl:
        mu=torch.stack(Z).mean(0).to(DEV)
        Ab,_=forward_all(m,tok,qs,nl,dec,lambda ii:{s:mu[s][None,None,:].expand(len(ii),1,-1) for s in range(nl)})
        out['ablated']=score(golds,lv,Ab)
        rq=[render(p['x0'],p['rules'],last) for p in PAIRS]; dq=[render(p['x0'],p['nr'],last) for p in PAIRS]
        Ar,Zr=forward_all(m,tok,rq,nl,dec); Ad,Zd=forward_all(m,tok,dq,nl,dec)
        keep=[i for i,p in enumerate(PAIRS) if Ar[i]==p['gold'] and Ad[i]==p['dgold']]
        sw=lambda Zs,ks:(lambda ii:{s:torch.stack([Zs[ks[j]][s] for j in ii])[:,None,:].to(DEV) for s in range(nl)})
        T,_=forward_all(m,tok,[rq[i] for i in keep],nl,dec,sw(Zd,keep)) if keep else ([],None)
        Dr,_=forward_all(m,tok,[dq[i] for i in keep],nl,dec,sw(Zr,keep)) if keep else ([],None)
        def share(ans,sel):
            ii=[k for k,i in enumerate(keep) if sel(PAIRS[i])]
            return [round(float(np.mean([ans[k]==PAIRS[keep[k]]['dgold'] for k in ii])),3) if ii else None,len(ii)]
        out['route']={'through_latents_before_last':share(T,lambda p:p['left']>=1),'direct_before_last':share(Dr,lambda p:p['left']>=1),'through_latents_last':share(T,lambda p:p['left']==0),'direct_last':share(Dr,lambda p:p['left']==0)}
    del m
    if DEV=='mps': torch.mps.empty_cache()
    return out
PATH=OUT+('/b18_smoke.json' if SMOKE else '/b18_track.json')
RES=json.load(open(PATH)) if os.path.exists(PATH) else {}
def ready(d):
    f=d+'/pytorch_model.bin'
    return all(os.path.exists(d+'/'+x) for x in ('pytorch_model.bin','training_args.bin','trainer_state.json')) and time.time()-os.path.getmtime(f)>240
def report(k,o):
    c=o['clean']; g=lambda t:' '.join(f"L{L} {t[L][0]:.2f}" for L in sorted(t) if int(L)>=3)
    msg=f"{k}: {g(c)}"
    if 'ablated' in o: msg+=f" | ablated {g(o['ablated'])} | through latents {o['route']['through_latents_before_last']} direct {o['route']['direct_before_last']}"
    print(time.strftime('%H:%M'),msg,flush=True)
fails={}
if os.environ.get('ONLY'): ARMS=[a for a in ARMS if a[0]==os.environ['ONLY']]
while True:
    todo=[]
    for name,root,nl,last in ARMS:
        for d in sorted(glob.glob(root+'/checkpoint-*'),key=lambda p:int(p.rsplit('-',1)[1])):
            step=int(d.rsplit('-',1)[1]); k=f'{name}|{step}'
            if k not in RES and ready(d) and fails.get(k,0)<3: todo.append((k,d,nl,last))
    if todo:
        k,d,nl,last=todo[0]
        try:
            t0=time.time(); RES[k]=measure(d,nl,last); report(k,RES[k]); print('   ',f'{time.time()-t0:.0f}s',flush=True)
            json.dump(RES,open(PATH,'w'),indent=0)
        except Exception as e:
            fails[k]=fails.get(k,0)+1; print('FAILED',k,repr(e)[:200],flush=True)
        if SMOKE and len(RES)>=2: break
        continue
    done=all(f'{n}|{LAST}' in RES for n,_,_,_ in ARMS) or (SMOKE and RES)
    if done: break
    time.sleep(180)
print('DONE')
