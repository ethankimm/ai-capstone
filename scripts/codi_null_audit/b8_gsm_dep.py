"""Latent-to-latent dependence on the released GSM8K CODI vs RuleChain CODI, measured identically.
Equal-length groups, so no padding exists and the released model's lack of pad_invariant cannot matter."""
import sys, json, glob, os, collections; sys.path.insert(0,sys.argv[1])
from harness import *
import numpy as np
torch.manual_seed(0); DEV=sys.argv[2]
def groups(qs,tok,m,suffix):
    by=collections.defaultdict(list)
    for q in qs:
        ids=tok(q+suffix).input_ids+[m.bot_id]
        by[len(ids)].append(ids)
    return [torch.tensor(v[:16],device=DEV) for k,v in sorted(by.items()) if len(v)>=4]
@torch.no_grad()
def measure(m,G):
    body=body_of(m); res=collections.defaultdict(list)
    for ids in G:
        B,T=ids.shape; mask=torch.ones(B,T,dtype=torch.long,device=DEV); pos0=torch.arange(T,device=DEV)[None].expand(B,-1)
        o=body(input_ids=ids,use_cache=True,attention_mask=mask,position_ids=pos0)
        past=o.past_key_values; z=m.prj(o.last_hidden_state[:,-1:,:]); npos=torch.full((B,1),T,device=DEV)
        perm=torch.roll(torch.arange(B,device=DEV),1)
        for s in range(6):
            _,zo,past2,mask2,npos2=latent_step(m,z,past,mask,npos)
            _,zx,_,_,_=latent_step(m,z[perm],past,mask,npos)
            res[s].append((zo[:,0].float().cpu(),zx[:,0].float().cpu()))
            z,past,mask,npos=zo,past2,mask2,npos2
    out=[]
    for s in range(6):
        zo=torch.cat([a for a,_ in res[s]]); zx=torch.cat([b for _,b in res[s]])
        spread=(zo-zo.mean(0,keepdim=True)).norm(dim=-1).mean()
        out.append(round(float((zo-zx).norm(dim=-1).mean()/(spread*2**.5)),3))
    return out,sum(len(a) for a,_ in res[0])
gq=json.load(open(glob.glob(os.path.expanduser('~/.cache/huggingface/hub/datasets--zen-E--GSM8k-Aug/snapshots/*/gsm8k_test.json'))[0]))['question']
rq=[r['question'].strip() for r in rows(BASE+'/synthetic_data_v2_l1to7w/depth2/test.json',per_level=60,seed=9,levels=[3,4,5,6,7])]
out={}
rel=glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--zen-E--CODI-gpt2/snapshots/*'))[0]
for name,ck,qs,suf in (('GSM8K released CODI',rel,gq,''),('RuleChain CODI lr3e-3',CK['lr3'],rq,'\n'),('RuleChain CODI round1',CK['r1'],rq,'\n'),('RuleChain CODI stab',CK['stab']%17820,rq,'\n')):
    m,tok,ta=load(ck,6,DEV); m.eval()
    d,n=measure(m,groups(qs,tok,m,suf)); out[name]=d
    print(name,'n',n,'dependence z_s->z_{s+1} for s=0..5:',d,flush=True)
    del m
    if DEV=='mps': torch.mps.empty_cache()
json.dump(out,open(sys.argv[1]+'/b8_dep.json','w'),indent=1)
print('DONE')
