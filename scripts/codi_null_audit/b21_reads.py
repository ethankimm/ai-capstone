"""Who reads what? Usage: b21_reads.py <this dir> <device> <output dir>. Valid only (TRAINING_PROTOCOL section 11.3). SMOKE=1: a few items.
Block one stage's attention to a set of positions, for that stage only, and see whether the answer changes. No probe involved.
Stages: latent pass 0..5, `lat` (all six passes), `ans` (the answer tokens).
Sets: rule_j, rules (all), start ("Start with N."), rest (the other question tokens, with <bot>), prev (earlier latents), prev1 (the latent just before).
Chain: blocking prev hurts, and pass s depends on one rule more than the others, moving with s. Relay: prev hardly matters and each pass re-reads the question.
Position 0, the attention sink, is never blocked. Controls: stage `none` must leave every answer unchanged; blocking the last rule at `ans` must change it (the answer stage applies that rule).
Also mean attention mass of each latent pass on each set, by layer."""
import sys, json, time, os; sys.path.insert(0,sys.argv[1])
from harness import *
import numpy as np
from rc import *
DEV=sys.argv[2]; OUT=sys.argv[3]; SMOKE=int(os.environ.get('SMOKE','0')); LEVELS=[2,3,4,5,6]; EXTRA=int(os.environ.get('EXTRA','0'))
SL=BASE+'/codi_runs_l1to7w_startlast'; DL=BASE+'/synthetic_data_v2_l1to7w_startlast/depth2/'; DF=BASE+'/synthetic_data_v2_l1to7w/depth2/'
MODELS=[('startlast CODI',f'{SL}/codi/run/checkpoint-17820',6,True),('lr3 relay',CK['lr3'],6,False)]
if os.environ.get('ONLY'): MODELS=[m for m in MODELS if m[0]==os.environ['ONLY']]
NUM={}
def setup(tok):
    if not NUM:
        for n in range(100): NUM[tok.encode(f"The answer is: {n}")[-1]]=n
def spans(tok,q):
    """Token indices in q+'\\n'+<bot>, unpadded: one range per rule, the start sentence, and everything else."""
    ti=lambda p:len(tok(q[:p]).input_ids); n=q.count('Rule ')
    ps=[q.index(f'Rule {k+1}:') for k in range(n)]; ps=[p-1 if p else 0 for p in ps]
    s=q.index('Start with'); s=s-1 if s else 0; e=q.index('.',s)+1; w=q.index(' What is')
    end=w if s<ps[0] else s   # rules end at the question sentence, or at the start sentence when it comes last
    rule=[(ti(ps[k]),ti(ps[k+1]) if k+1<n else ti(end)) for k in range(n)]
    total=len(tok(q+'\n').input_ids)+1
    start=(ti(s),ti(e)); used=set(range(start[0],start[1]))|{i for a,b in rule for i in range(a,b)}
    return rule,start,[i for i in range(total) if i not in used],total
def blockset(sp,W,nl,key,stage):
    """[positions] to mask, padded index. W = padded question width."""
    rule,start,rest,total=sp; off=W-total
    if key.startswith('rule_'): a,b=rule[int(key[5:])]; idx=range(a,b)
    elif key=='rules': idx=[i for a,b in rule for i in range(a,b)]
    elif key=='early': idx=[i for a,b in rule[:-1] for i in range(a,b)]    # every rule but the last
    elif key=='early2': idx=[i for a,b in rule[:-2] for i in range(a,b)]   # every rule but the last two
    elif key=='start': idx=range(*start)
    elif key=='rest': idx=rest
    else: return []
    return [off+i for i in idx if i>0]   # position 0 is the attention sink: blocking it damages the model whatever it holds, and it belongs to a different set in each format
@torch.no_grad()
def forward(m,past0,h0,mask0,npos0,dec,nl,stage,key,sps,attn=None):
    """Latent passes and answer tokens from a cached question, with `key` blocked at `stage`."""
    body=body_of(m); W=mask0.size(1); past=clone_cache(past0); z=m.prj(h0); mask=mask0; npos=npos0; B=mask.size(0)
    def blocked(cur,s):
        hit=stage==s or (stage=='lat' and s!='ans')
        if not hit or key is None: return cur
        c=cur.clone()
        if key in('prev','prev1'):
            if s=='ans': return cur
            lo=W if key=='prev' else W+s-1
            if s>0: c[:,lo:W+s]=0
            return c
        for r,sp in enumerate(sps): c[r,blockset(sp,W,nl,key,stage)]=0
        return c
    for s in range(nl):
        mask=torch.cat((mask,torch.ones(B,1,dtype=mask.dtype,device=mask.device)),1)
        o=body(inputs_embeds=z,use_cache=True,past_key_values=past,attention_mask=blocked(mask,s),position_ids=npos,output_attentions=attn is not None)
        if attn is not None: attn(s,o.attentions)
        past=o.past_key_values; npos=npos+1; z=m.prj(o.last_hidden_state[:,-1:,:])
    emb=m.get_embd(m.codi,m.model_name)(dec)[None].expand(B,-1,-1); T=emb.size(1)
    full=torch.cat((blocked(mask,'ans'),torch.ones(B,T,dtype=mask.dtype,device=mask.device)),1)
    o=body(inputs_embeds=emb,use_cache=True,past_key_values=past,attention_mask=full,position_ids=npos+torch.arange(T,device=mask.device)[None])
    return [NUM.get(t,-999) for t in m.codi.get_base_model().lm_head(o.last_hidden_state[:,-1,:])[:,:50257].argmax(-1).tolist()]
def conds(L,nl):
    if EXTRA: return [(None,None),('lat','early'),('lat','early2'),('ans','early')]
    keys=[f'rule_{j}' for j in range(L)]+['rules','start','rest']
    return [(None,None)]+[(s,k) for s in list(range(nl))+['lat','ans'] for k in keys]+[(s,k) for s in list(range(1,nl))+['lat'] for k in ('prev','prev1')]
def measure(ck,nl,last):
    D=DL if last else DF; m,tok,_=load(ck,nl,DEV); m.eval(); setup(tok)
    dec=torch.tensor([m.eot_id]+tok.encode('The answer is:'),device=DEV); va=json.load(open(D+'valid.json')); res={}
    for L in LEVELS:
        items=[r for r in va if int(r['level'])==L]; items=items[:8] if SMOKE else items
        items.sort(key=lambda r:len(r['question'])); C=conds(L,nl); ans={str(c):[] for c in C}; gold=[int(r['answer']) for r in items]
        A=np.zeros((nl,12,L+5)); An=0
        for i in range(0,len(items),32):
            rs=items[i:i+32]; qs=[r['question'].strip() for r in rs]; sps=[spans(tok,q) for q in qs]
            b=encode([dict(question=q,steps=['x'],answer='0') for q in qs],tok,m,DEV); W=b['encoder_input_ids'].size(1)
            if i==0:
                ids=[tok(q+'\n').input_ids for q in qs[:1]][0]; a0,b0=sps[0][0][0]; assert tok.decode(ids[a0:b0]).strip().startswith('Rule 1'),tok.decode(ids[a0:b0])
                a0,b0=sps[0][1]; assert tok.decode(ids[a0:b0]).strip().startswith('Start with') and tok.decode(ids[a0:b0]).strip().endswith('.'),tok.decode(ids[a0:b0])
            mask=b['encoder_attention_mask'].long(); pos=(mask.cumsum(-1)-1).clamp(min=0)
            with torch.no_grad(): o=body_of(m)(input_ids=b['encoder_input_ids'],use_cache=True,attention_mask=mask,position_ids=pos)
            past=o.past_key_values; h0=o.last_hidden_state[:,-1:,:]; npos=mask.sum(-1,keepdim=True)
            for c in C: ans[str(c)]+=forward(m,past,h0,mask,npos,dec,nl,c[0],c[1],sps)
            def mass(s,att):
                """att: per layer [B,H,1,K]; share of attention on each set (rules, start, rest, earlier latents, self, sink)."""
                for l,a in enumerate(att):
                    a=a[:,:,0,:].mean(1).float().cpu().numpy()
                    for r,sp in enumerate(sps):
                        for j in range(L): A[s,l,j]+=a[r,blockset(sp,W,nl,f'rule_{j}',0)].sum()
                        A[s,l,L]+=a[r,blockset(sp,W,nl,'start',0)].sum(); A[s,l,L+1]+=a[r,blockset(sp,W,nl,'rest',0)].sum()
                        A[s,l,L+2]+=a[r,W:W+s].sum(); A[s,l,L+3]+=a[r,W+s]; A[s,l,L+4]+=a[r,W-sp[3]]
            forward(m,past,h0,mask,npos,dec,nl,None,None,sps,attn=mass); An+=len(rs)
            del o,past,b
            if DEV=='mps': torch.mps.empty_cache()
        res[L]=dict(gold=gold,ans=ans,attn=(A/max(An,1)).tolist()); print(time.strftime('%H:%M'),f'L{L} n={len(items)} conds={len(C)}',flush=True)
    del m
    if DEV=='mps': torch.mps.empty_cache()
    return res
PATH=OUT+('/b21_smoke.json' if SMOKE else ('/b21_extra.json' if EXTRA else '/b21_reads.json')); RES=json.load(open(PATH)) if os.path.exists(PATH) and not SMOKE else {}
for name,ck,nl,last in MODELS:
    if name in RES: continue
    t0=time.time(); RES[name]=measure(ck,nl,last); json.dump(RES,open(PATH,'w')); print('##',name,f'{time.time()-t0:.0f}s',flush=True)
print('DONE')
