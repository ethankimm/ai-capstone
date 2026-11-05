"""Independent reimplementation of CODI's train-time passes for auditing (fp32). Runs from the CODI checkout."""
import os, sys, json, random
from pathlib import Path
BASE=str(Path(__file__).resolve().parents[3])  # the folder holding ai-capstone/, codi/ and the data/run folders
CODI=BASE+'/codi'; REPO=BASE+'/ai-capstone'
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.chdir(CODI)
sys.path.insert(0,CODI); sys.path.insert(0,REPO+'/scripts'); sys.path.insert(0,REPO)
import torch, transformers
from torch.nn import functional as F
from src.model import DataArguments, ModelArguments, TrainingArguments
import diagnose_codi_latents  # noqa (patches torch.load for CPU)
from eval_codi import build_model
from transformers.cache_utils import DynamicCache

CK={
 'stab': BASE+'/codi_runs_l1to7w_stab/codi/run/checkpoint-%d',
 'stab0': BASE+'/codi_runs_l1to7w_stab/codi0/run/checkpoint-%d',
 'lr3': BASE+'/codi_runs_l1to7w_codi/checkpoint-17820',
 'r1': REPO+'/rulechain-checkpoints/gpt2_depth2_codi_bs64/checkpoint-31280',
 'r10': REPO+'/rulechain-checkpoints/gpt2_depth2_codi0/checkpoint-29716',
}
def load(ckpt, nl=6, device='cpu'):
    argv=f"--output_dir /tmp/u --model_name_or_path gpt2 --seed 11 --model_max_length 512 --lora_r 128 --lora_alpha 32 --lora_init --batch_size 64 --greedy True --num_latent {nl} --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 --inf_latent_iterations {nl} --inf_num_iterations 1 --remove_eos True --use_lora True --fix_attn_mask True --pad_invariant True --include_last_cot False --ckpt_dir {ckpt}".split()
    p=transformers.HfArgumentParser((ModelArguments,DataArguments,TrainingArguments))
    ma,da,ta=p.parse_args_into_dataclasses(args=argv)
    ta.print_ref_model_stats=False; ta.print_loss=False
    m,tok,lr=build_model(ma,ta,device=device,dtype=torch.float32)
    bad=[k for k in lr.missing_keys if not k.startswith('codi.base_model.model.lm_head')]
    assert not bad, bad[:5]
    m.training_args=ta; m.print_loss=False
    return m,tok,ta

def rows(path,per_level=None,seed=0,levels=None):
    d=json.load(open(path))
    if per_level is None and levels is None: return d
    r=random.Random(seed); out=[]
    by={}
    for x in d: by.setdefault(int(x['level']),[]).append(x)
    for L in sorted(by):
        if levels and L not in levels: continue
        v=by[L]; r.shuffle(v); out+=v[:per_level]
    return out

def encode(rows_, tok, model, device):
    """exactly train.py's preprocess + collator (remove_eos True, include_last_cot False)"""
    eos=tok.eos_token_id; bot=model.bot_id; eot=model.eot_id; pad=model.pad_token_id
    enc=[];dec=[];ref=[];lab=[];rlab=[];rpos=[];mpos=[]
    pr=tok.encode('The answer is:')
    for x in rows_:
        q=tok(x['question'].strip()+'\n').input_ids
        cot=tok('\n'.join(x['steps'][:-1])+'\n').input_ids
        a=tok(f"The answer is: {str(x['answer']).strip()}").input_ids+[eos]
        full=q+cot+a
        ref.append(torch.tensor(full)); rl=torch.tensor(full).clone(); rl[:len(q)]=-100; rlab.append(rl)
        enc.append(torch.tensor(q+[bot]))
        d=[eot]+a; dec.append(torch.tensor(d)); lab.append(torch.tensor(d))
        def find(t):
            for i in range(len(t)-len(pr)+1):
                if t[i:i+len(pr)]==pr: return i+len(pr)
            raise ValueError
        rpos.append(find(full)); mpos.append(find(d))
    ps=torch.nn.utils.rnn.pad_sequence
    e=ps([t.flip(0) for t in enc],batch_first=True,padding_value=pad).flip(1)
    r=ps(ref,batch_first=True,padding_value=pad)
    rl=ps(rlab,batch_first=True,padding_value=-100)
    dd=ps(dec,batch_first=True,padding_value=pad)
    l=ps(lab,batch_first=True,padding_value=-100)
    return {k:v.to(device) for k,v in dict(encoder_input_ids=e,decoder_input_ids=dd,ref_input_ids=r,labels=l,encoder_attention_mask=e.ne(pad),
            ref_answer_position=torch.tensor(rpos),model_answer_position=torch.tensor(mpos),ref_attention_mask=r.ne(pad),ref_labels=rl).items()}

def clone_cache(c):
    # this transformers version hands back legacy tuples, which the model never mutates; a DynamicCache it does
    if isinstance(c,tuple): return c
    n=DynamicCache()
    n.key_cache=[k for k in c.key_cache]; n.value_cache=[v for v in c.value_cache]
    n._seen_tokens=c._seen_tokens
    return n

def question(model,b):
    mask=b['encoder_attention_mask'].long()
    pos=(mask.cumsum(-1)-1).clamp(min=0)
    # transformer body, not the LM-head wrapper: same LoRA'd weights, no [B,T,V] logits (those exhausted memory)
    out=model.codi.get_base_model().transformer(input_ids=b['encoder_input_ids'],use_cache=True,attention_mask=mask,position_ids=pos)
    return out, mask, mask.sum(-1,keepdim=True)

def latent_step(model,z,past,mask,pos):
    """one latent pass; returns h (ln_f output), z_next, past, mask, pos+1"""
    mask=torch.cat((mask,torch.ones(mask.size(0),1,dtype=mask.dtype,device=mask.device)),1)
    out=model.codi.get_base_model().transformer(inputs_embeds=z,use_cache=True,past_key_values=past,attention_mask=mask,position_ids=pos)
    h=out.last_hidden_state[:,-1:,:]
    return h, model.prj(h), out.past_key_values, mask, pos+1

def decode(model,b,past,mask,pos):
    embd=model.get_embd(model.codi,model.model_name)(b['decoder_input_ids'])
    T=embd.size(1)
    m=torch.cat((mask,torch.ones(mask.size(0),T,dtype=mask.dtype,device=mask.device)),1)
    p=pos+torch.arange(T,device=mask.device)[None]
    return model.codi(inputs_embeds=embd,use_cache=True,output_hidden_states=True,past_key_values=past,attention_mask=m.bool(),position_ids=p)

def teacher(model,b):
    return model.codi.get_base_model().transformer(input_ids=b['ref_input_ids'],output_hidden_states=True,attention_mask=b['ref_attention_mask'])

def losses(model,b,out,tout):
    logits=out.logits
    ce=F.cross_entropy(logits[:,:-1].reshape(-1,logits.size(-1)),b['labels'][:,1:].reshape(-1),ignore_index=-100)
    mp=b['model_answer_position']-1; rp=b['ref_answer_position']-1
    dist=0
    per=[]
    for o,r in zip(out.hidden_states,tout.hidden_states):
        rs=r.gather(1,rp[:,None,None].expand(-1,-1,r.size(-1))); os_=o.gather(1,mp[:,None,None].expand(-1,-1,o.size(-1)))
        t=F.smooth_l1_loss(os_,rs.detach())/rs.std()
        dist=dist+t; per.append(t.detach())
    dist=dist/len(out.hidden_states)
    return ce,dist,per


def body_of(model): return model.codi.get_base_model().transformer

@torch.no_grad()
def fast_answer(model,b,nl,mu=None,want_z=False,dec_ids=None):
    """Forward-only student: transformer body for every pass (no [B,T,V] logits), lm_head only on the last position."""
    body=body_of(model)
    mask=b['encoder_attention_mask'].long(); pos=(mask.cumsum(-1)-1).clamp(min=0)
    o=body(input_ids=b['encoder_input_ids'],use_cache=True,attention_mask=mask,position_ids=pos)
    past=o.past_key_values; npos=mask.sum(-1,keepdim=True)
    z=model.prj(o.last_hidden_state[:,-1:,:]) if nl else None
    zs=[]
    for s in range(nl):
        if mu is not None: z=mu[s][None,None].expand(z.size(0),1,-1)
        zs.append(z[:,0])
        mask=torch.cat((mask,torch.ones(mask.size(0),1,dtype=mask.dtype,device=mask.device)),1)
        o=body(inputs_embeds=z,use_cache=True,past_key_values=past,attention_mask=mask,position_ids=npos)
        past=o.past_key_values; npos=npos+1
        z=model.prj(o.last_hidden_state[:,-1:,:])
    emb=model.get_embd(model.codi,model.model_name)(dec_ids)[None].expand(mask.size(0),-1,-1)
    T=emb.size(1)
    mask=torch.cat((mask,torch.ones(mask.size(0),T,dtype=mask.dtype,device=mask.device)),1)
    o=body(inputs_embeds=emb,use_cache=True,past_key_values=past,attention_mask=mask,position_ids=npos+torch.arange(T,device=mask.device)[None])
    h=o.last_hidden_state[:,-1,:]
    logits=model.codi.get_base_model().lm_head(h)[:,:50257]
    return logits,(torch.stack(zs,1) if nl else None)
