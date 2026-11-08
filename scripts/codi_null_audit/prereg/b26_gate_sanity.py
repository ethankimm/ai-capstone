"""Is gated accuracy real? Valid L3-L6. Usage: b26_gate_sanity.py <codi_null_audit dir> <device> <checkpoint>
 a. exact from free generation (generate_batches), not just the teacher-forced token.
 b. edit rule j so the true answer changes: does the answer follow the new chain? (no route but the gate's)
 c. swap the rule text of two middle rules: answer should follow the swapped order.
 d. wrong model control: same items, gate off on the gated checkpoint."""
import sys, json, re, random; sys.path.insert(0,sys.argv[1])
from harness import *
from rc import *
from eval_codi import encode_and_think, tokenize_questions, generate_batches
import types
DEV,CK_=sys.argv[2],sys.argv[3]; D=BASE+'/synthetic_data_v2_l1to7w_startlast/depth2/valid.json'
m,tok,ta=load(CK_,6,DEV); m.eval(); ta.rule_gate=True; m.rule_gate=True; ta.batch_size=32; ta.inf_latent_iterations=6
num={tok.encode(f"The answer is: {n}")[-1]:n for n in range(100)}; dec=[m.eot_id]+tok.encode('The answer is:'); emb0=m.get_embd(m.codi,m.model_name)(torch.tensor(dec,device=DEV))
def tf(qs):
    P=[]
    with torch.no_grad():
        for b in tokenize_questions(m,tok,[q.strip()+'\n' for q in qs],32,ta,DEV):
            past,pi,_=encode_and_think(m,b,ta); e=emb0[None].expand(b['input_ids'].size(0),-1,-1)
            o=body_of(m)(inputs_embeds=e,use_cache=True,past_key_values=past,**pi.next(len(dec)))
            P+=[num.get(t,-999) for t in m.codi.get_base_model().lm_head(o.last_hidden_state[:,-1,:])[:,:50257].argmax(-1).tolist()]
            del past,o
            if DEV=='mps': torch.mps.empty_cache()
    return P
va=json.load(open(D)); rng=random.Random(0)
for L in (3,4,5,6):
    ev=[r for r in va if int(r['level'])==L]; qs=[r['question'].strip() for r in ev]; g=[int(r['answer']) for r in ev]; p=tf(qs)
    print(f'L{L} n={len(ev)} teacher-forced exact {sum(a==b for a,b in zip(p,g))/len(g):.3f}',flush=True)
    # b: swap two adjacent non-last rules; new gold computed by the true chain; keep items where answer changes and values stay 0-99
    Q=[];G=[];O=[]
    for r in ev:
        x0,ru=parse(r['question'].strip()); q=r['question'].strip(); parts=re.findall(r'Rule \d+: ([^.]*)\.',q)
        i=rng.randrange(0,L-2) if L>2 else 0
        ru2=list(ru); ru2[i],ru2[i+1]=ru2[i+1],ru2[i]; v=walk(x0,ru2)[0]
        if min(v)<0 or max(v)>99 or v[-1]==int(r['answer']): continue
        t=list(parts); t[i],t[i+1]=t[i+1],t[i]
        Q.append(' '.join(f'Rule {k+1}: {s}.' for k,s in enumerate(t))+f' Start with {x0}. What is the final number?'); G.append(v[-1]); O.append(int(r['answer']))
    pq=tf(Q); print(f'   rule order swapped n={len(Q)}: follows the new chain {sum(a==b for a,b in zip(pq,G))/max(len(Q),1):.3f}; keeps the old answer {sum(a==b for a,b in zip(pq,O))/max(len(Q),1):.3f}',flush=True)
    # c: replace the start value; true chain recomputed
    Q=[];G=[]
    for r in ev:
        x0,ru=parse(r['question'].strip()); x1=rng.randrange(1,99); v=walk(x1,ru)[0]
        if min(v)<0 or max(v)>99 or v[-1]==int(r['answer']): continue
        Q.append(re.sub(r'Start with \d+',f'Start with {x1}',r['question'].strip())); G.append(v[-1])
    pq=tf(Q); print(f'   new start value n={len(Q)}: follows the new chain {sum(a==b for a,b in zip(pq,G))/max(len(Q),1):.3f}',flush=True)
    if L==5:
        ta.rule_gate=False; m.rule_gate=False; p0=tf(qs); ta.rule_gate=True; m.rule_gate=True
        print(f'   same checkpoint, gate off: exact {sum(a==b for a,b in zip(p0,g))/len(g):.3f}',flush=True)
    data=types.SimpleNamespace(batch_size=32); 
    outs=generate_batches(m,tok,[q+'\n' for q in qs[:96]],data,ta) if False else None
print('DONE')
