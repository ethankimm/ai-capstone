"""Does attention move through the problem, pass by pass? Usage: b24_pointer.py <codi_null_audit dir> <device> <output dir>. SMOKE=1: a few items.
Rows: <bot>, then latent passes 0-5. For every head: share of attention on each step's tokens, and the centre of mass over steps (0 = first step, 1 = last).
RuleChain: tokens of rule j. GSM8K: question numbers first used in step j. Position 0 (the sink) is never counted. Batch 1, so no padding."""
import sys, json, time, os, re; sys.path.insert(0,sys.argv[1])
from harness import *
import numpy as np
from rc import *
DEV=sys.argv[2]; OUT=sys.argv[3]; SMOKE=int(os.environ.get('SMOKE','0'))
SL=BASE+'/codi_runs_l1to7w_startlast'; DL=BASE+'/synthetic_data_v2_l1to7w_startlast/depth2/'; DF=BASE+'/synthetic_data_v2_l1to7w/depth2/'
REL=os.path.expanduser('~/.cache/huggingface/hub/models--zen-E--CODI-gpt2/snapshots/fd641b3d3edc59e4f534b55588e906588c9e36bb')
NUMRE=re.compile(r'\d[\d,]*(?:\.\d+)?')
def rule_sets(tok,q):
    ti=lambda p:len(tok(q[:p]).input_ids); n=q.count('Rule ')
    ps=[q.index(f'Rule {k+1}:') for k in range(n)]; ps=[p-1 if p else 0 for p in ps]
    s=q.index('Start with'); s=s-1 if s else 0; w=q.index(' What is'); end=w if s<ps[0] else s
    return [[i for i in range(ti(ps[k]),ti(ps[k+1]) if k+1<n else ti(end)) if i>0] for k in range(n)]
def num(s):
    try: return float(s.replace(',','').rstrip('.'))
    except: return None
def gsm_sets(tok,q,rationale):
    qb=q.encode(); ls=[len(bytes(tok.byte_decoder[c] for c in t)) for t in tok.convert_ids_to_tokens(tok.encode(q))]; assert sum(ls)==len(qb)
    off=[]; c=0
    for l in ls: off.append((c,c+l)); c+=l
    spans={}
    for m in re.finditer(NUMRE.pattern.encode(),qb): spans.setdefault(num(m.group().decode()),[]).append(m.span())
    used=set(); prior=set(); out=[]
    for ex,val in re.findall(r'<<([^<>=]*)=([^<>]*)>>',rationale):
        toks=[]
        for x in NUMRE.findall(ex):
            v=num(x)
            if v in prior or v in used or v not in spans: continue
            used.add(v)
            for a,b in spans[v]: toks+=[i for i,(c,d) in enumerate(off) if i>0 and c<b and d>a]
        out.append(sorted(set(toks))); prior.add(num(val.strip()) if num(val.strip()) is not None else None)
    return [t for t in out if t]
@torch.no_grad()
def one(m,ids,sets,nl):
    body=body_of(m); T=ids.size(1); J=len(sets); S=torch.zeros(T,J,device=DEV)
    for j,t in enumerate(sets): S[t,j]=1
    o=body(input_ids=ids,use_cache=True,output_attentions=True)
    rows_=[torch.cat([a[0,:,-1,:T] for a in o.attentions],0)]; past=o.past_key_values; z=m.prj(o.last_hidden_state[:,-1:,:])
    for s in range(nl):
        o=body(inputs_embeds=z,use_cache=True,past_key_values=past,output_attentions=True)
        rows_.append(torch.cat([a[0,:,0,:T] for a in o.attentions],0)); past=o.past_key_values; z=m.prj(o.last_hidden_state[:,-1:,:])
    A=torch.stack(rows_); M=A@S; tot=M.sum(-1); sh=M/(tot[...,None]+1e-9); com=sh@torch.linspace(0,1,J,device=DEV)
    return com.float().cpu().numpy(),sh.mean(1).float().cpu().numpy(),tot.mean(1).float().cpu().numpy()
def run_rule(name,ck,D,levels,res):
    m,tok,_=load(ck,6,DEV); m.eval(); va=json.load(open(D+'valid.json')); A=json.load(open(BASE+'/codi_audit_artifacts/b21_reads.json')).get(name,{})
    for L in levels:
        ev=[r for r in va if int(r['level'])==L]; ev.sort(key=lambda r:len(r['question'])); ok=None
        if str(L) in A: d=A[str(L)]; ok=[a==g for a,g in zip(d['ans'][str((None,None))],d['gold'])]
        n=8 if SMOKE else len(ev); C=[];Sh=[];To=[]
        for r in ev[:n]:
            q=r['question'].strip(); ids=torch.tensor([tok(q+'\n').input_ids+[m.bot_id]],device=DEV); sets=rule_sets(tok,q)
            if SMOKE and not C: print([tok.decode(ids[0,t].tolist()) for t in sets[-1]],flush=True)
            c,s,t=one(m,ids,sets,6); C.append(c);Sh.append(s);To.append(t)
            if DEV=='mps': torch.mps.empty_cache()
        res[f'{name}|L{L}']=dict(com=np.stack(C).astype(np.float16),share=np.stack(Sh).mean(0),tot=np.stack(To).mean(0),right=np.array(ok[:n] if ok else [True]*n)); print(time.strftime('%H:%M'),name,L,n,flush=True)
    del m
    if DEV=='mps': torch.mps.empty_cache()
def run_gsm(res):
    from latentreasoning.data.gsm8k_aug import load_gsm8k_aug
    m,tok,_=load(REL,6,DEV); m.eval(); ex=load_gsm8k_aug(split='test',n=None,seed=None)
    logged={int(json.loads(l)['idx']):json.loads(l)['greedy']['trunc_6']['correct'] for l in open(REPO+'/results/20260927-005051_codi_necessity-all-replacements/predictions.jsonl')}
    G={}
    for e in ex[:60] if SMOKE else ex:
        q=e.question.strip().replace('  ',' '); sets=gsm_sets(tok,q,e.rationale); n=len(e.intermediate_values)
        if len(sets)<2 or n<2: continue
        ids=torch.tensor([tok.encode(q)+[m.bot_id]],device=DEV); c,s,t=one(m,ids,sets,6); J=len(sets)
        G.setdefault(J,[]).append((c,s,t,logged[int(e.idx)],n))
        if DEV=='mps': torch.mps.empty_cache()
    for J,v in sorted(G.items()):
        res[f'GSM8K released|J{J}']=dict(com=np.stack([x[0] for x in v]).astype(np.float16),share=np.stack([x[1] for x in v]).mean(0),tot=np.stack([x[2] for x in v]).mean(0),right=np.array([x[3] for x in v]),steps=np.array([x[4] for x in v]))
        print(time.strftime('%H:%M'),'gsm J',J,len(v),flush=True)
    del m
    if DEV=='mps': torch.mps.empty_cache()
res={}
run_gsm(res)
run_rule('startlast CODI',f'{SL}/codi/run/checkpoint-17820',DL,[3] if SMOKE else [3,4,5,6],res)
if not SMOKE: run_rule('lr3 relay',CK['lr3'],DF,[4,5],res)
out={}
for k,v in res.items():
    for a,b in v.items(): out[f'{k}::{a}']=b
np.savez_compressed(OUT+('/b24_smoke.npz' if SMOKE else '/b24_pointer.npz'),**out); print('DONE')
