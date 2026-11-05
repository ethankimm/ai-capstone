"""Released GSM8K CODI: are the latents forward-indexed (v_1 at h_1 for every n) or end-relative (v_{n-1})? Logit lens, no fitting."""
import sys, json, glob, os, re, collections; sys.path.insert(0,sys.argv[1])
from harness import *
import numpy as np
DEV='cpu'; torch.manual_seed(0)
d=json.load(open(glob.glob(os.path.expanduser('~/.cache/huggingface/hub/datasets--zen-E--GSM8k-Aug/snapshots/*/gsm8k_test.json'))[0]))
rel=glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--zen-E--CODI-gpt2/snapshots/*'))[0]
m,tok,ta=load(rel,6,DEV); m.eval(); body=body_of(m); head=m.codi.get_base_model().lm_head
items=[]
for q,c in zip(d['question'],d['cot']):
    v=re.findall(r'=\s*(-?[\d,]*\.?\d+)>>',c)
    v=[x.replace(',','') for x in v]
    if len(v)>=2: items.append((tok.encode(q.strip().replace('  ',' '))+[m.bot_id],v))
by=collections.defaultdict(list)
for ids,v in items: by[len(ids)].append((ids,v))
rec=[]
with torch.no_grad():
    for T,g in sorted(by.items()):
        for i in range(0,len(g),32):
            gg=g[i:i+32]; ids=torch.tensor([x[0] for x in gg]); B=ids.size(0)
            mask=torch.ones(B,T,dtype=torch.long); o=body(input_ids=ids,use_cache=True,attention_mask=mask,position_ids=torch.arange(T)[None].expand(B,-1))
            past=o.past_key_values; z=m.prj(o.last_hidden_state[:,-1:,:]); npos=torch.full((B,1),T); tops=[]
            for s in range(6):
                mask=torch.cat((mask,torch.ones(B,1,dtype=torch.long)),1)
                o=body(inputs_embeds=z,use_cache=True,past_key_values=past,attention_mask=mask,position_ids=npos)
                past=o.past_key_values; npos=npos+1; h=o.last_hidden_state[:,-1]; z=m.prj(o.last_hidden_state[:,-1:,:])
                tops.append(head(h)[:,:50257].argmax(-1).tolist())
            for b in range(B):
                rec.append(([tops[s][b] for s in range(6)],gg[b][1]))
print('items',len(rec))
print('top-1 token at latent pass output h_s equals ... (share of items), n = number of <<>> steps')
out={}
for n,lab in ((2,'n=2'),(3,'n=3'),(4,'n=4'),(5,'n>=5')):
    R=[r for r in rec if (len(r[1])==n if n<5 else len(r[1])>=5)]
    print(f' {lab} (items {len(R)})')
    def ft(x):
        x=x[:-2] if x.endswith('.0') else x
        return {tok.encode(s)[0] for s in (x,' '+x)}
    for s in (1,2,3,4,5,6):
        t=[r[0][s-1] for r in R]
        hit=lambda f:float(np.mean([a in ft(f(r[1])) for a,r in zip(t,R)]))
        row=dict(v1=hit(lambda v:v[0]),v2=hit(lambda v:v[1]),v_nm1=hit(lambda v:v[-2]),answer=hit(lambda v:v[-1]))
        out[f'{lab}_h{s}']={k:round(float(v),3) for k,v in row.items()}
        print(f'   h_{s}',out[f'{lab}_h{s}'])
json.dump(out,open(sys.argv[1]+'/b11_gsm_lens.json','w'),indent=1)
