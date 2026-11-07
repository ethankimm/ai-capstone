"""Accuracy by level from a checkpoint, valid only, through the eval path (TRAINING_PROTOCOL section 12.2).
Usage: b25_gated_track.py <this dir> <device> <checkpoint> <gate 0|1> [k...]. k = latent passes to run (default 6). Prints exact and within 2 by level."""
import sys, json, os; sys.path.insert(0,sys.argv[1])
from harness import *
from eval_codi import encode_and_think, tokenize_questions
DEV,CKPT,GATE=sys.argv[2],sys.argv[3],bool(int(sys.argv[4])); KS=[int(x) for x in sys.argv[5:]] or [6]
D=BASE+'/synthetic_data_v2_l1to7w_startlast/depth2/valid.json'
m,tok,ta=load(CKPT,6,DEV); m.eval(); ta.rule_gate=GATE; m.rule_gate=GATE; ta.batch_size=32
num={tok.encode(f"The answer is: {n}")[-1]:n for n in range(100)}; va=json.load(open(D)); va.sort(key=lambda r:(int(r['level']),len(r['question'])))
dec=[m.eot_id]+tok.encode('The answer is:'); emb0=m.get_embd(m.codi,m.model_name)(torch.tensor(dec,device=DEV))
for k in KS:
    ta.inf_latent_iterations=k; P=[]
    with torch.no_grad():
        for b in tokenize_questions(m,tok,[r['question'].strip()+'\n' for r in va],32,ta,DEV):
            past,pi,_=encode_and_think(m,b,ta); emb=emb0[None].expand(b['input_ids'].size(0),-1,-1)
            o=body_of(m)(inputs_embeds=emb,use_cache=True,past_key_values=past,**pi.next(len(dec)))
            P+=[num.get(t,-999) for t in m.codi.get_base_model().lm_head(o.last_hidden_state[:,-1,:])[:,:50257].argmax(-1).tolist()]
            del past,o
            if DEV=='mps': torch.mps.empty_cache()
    res={}
    for L in range(1,8):
        I=[i for i,r in enumerate(va) if int(r['level'])==L]; g=[int(va[i]['answer']) for i in I]; p=[P[i] for i in I]
        res[L]=(round(sum(a==b for a,b in zip(p,g))/len(I),3),round(sum(abs(a-b)<=2 for a,b in zip(p,g))/len(I),3))
    print('k',k,'exact/within2',' '.join(f'L{L} {e:.2f}/{w:.2f}' for L,(e,w) in res.items()),flush=True)
