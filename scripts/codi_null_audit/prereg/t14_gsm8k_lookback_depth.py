"""Released GSM8K CODI, look-back cut by chain length on the full test set (per item). Usage as t12 (CODI checkout, CODI venv):
t14_gsm8k_lookback_depth.py <ai-capstone dir> <device> <out dir>. SMOKE=1: 30 items. Conditions per item: clean, nb_all (every latent iteration
masked from all earlier latent iterations), nb_even (earlier even sites only), nb_odd (earlier odd sites only); answer tokens see everything.
Split by the number of steps in the gold rationale (Example.intermediate_values). Copy of t12's model / run(); t12 docstring follows.
Released GSM8K CODI: which route carries a latent's value, and is t11's look-back cost about values? Usage (CODI checkout, CODI venv):
t12_gsm8k_route_split.py <ai-capstone dir> <device> <out dir>. SMOKE=1: 20 pairs, no full-test pass. Same model, pairs, sites and generation
path as t11_gsm8k_routes.py (zen-E/CODI-gpt2 fd641b3, fp32, batch 1, the 317 aligned pairs of 20260927-004340).
Site s = the latent iteration whose input is z_s; its keys/values sit at position Q + s (Q = question length incl. <bot>).
A. Look-back controls (t11's nb cut every earlier latent): nb_even = each iteration masked from earlier EVEN sites only (value sites);
   nb_odd = from earlier ODD sites only (punctuation sites). On qualified recipients and on all 1319 test items.
   If t11's drop is about values, nb_even costs far more than nb_odd.
B. Route split for s in 0, 2, 4, on qualified pairs whose two questions have the same token length (positions line up):
   both_s       twin's z_s fed into iteration s+1 (t11 single_s): both routes carry the twin's value
   input_s      twin's z_s fed in, then that iteration's keys/values overwritten with the recipient's own: only the recurrent route
                (the next input z_{s+1}, computed from the twin's z_s) carries it
   kv_s         recipient's z_s fed in (clean), that iteration's keys/values overwritten with the twin's: only attention carries it,
                to later iterations and to the answer tokens
   kv_noans_s   as kv_s, but the answer tokens cannot attend to position Q + s: the twin's value reaches the answer only if later latent
                iterations read it by attention and pass it on (a horizontal chain)
   Score: answer = twin answer. A vertical chain (gated / SIM on RuleChain) would show input_s ~ both_s and kv_s ~ 0; a horizontal chain,
   kv_noans_s well above 0."""
import sys, os, json, time
REPO = sys.argv[1]; DEV = sys.argv[2]; OUT = sys.argv[3]; SMOKE = int(os.environ.get('SMOKE', '0'))
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('HF_DATASETS_OFFLINE', '1'); os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, REPO + '/scripts'); sys.path.insert(0, REPO); sys.path.insert(0, os.getcwd())
import torch, transformers
from src.model import DataArguments, ModelArguments, TrainingArguments
import diagnose_codi_latents  # noqa
from eval_codi import build_model
from decode_patch_codi import encode_question
from latentreasoning.data.gsm8k_aug import load_gsm8k_aug
from latentreasoning.eval.metrics import extract_final_number, is_correct
CKPT = os.path.expanduser('~/.cache/huggingface/hub/models--zen-E--CODI-gpt2/snapshots/fd641b3d3edc59e4f534b55588e906588c9e36bb')
argv = (f"--output_dir /tmp/u --model_name_or_path gpt2 --seed 11 --model_max_length 512 --lora_r 128 --lora_alpha 32 --lora_init --greedy True "
        f"--num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 --inf_latent_iterations 6 --inf_num_iterations 1 "
        f"--remove_eos True --use_lora True --ckpt_dir {CKPT}").split()
ma, da, ta = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments)).parse_args_into_dataclasses(args=argv)
ta.print_ref_model_stats = False; ta.print_loss = False
model, tok, lr = build_model(ma, ta, device=DEV, dtype=torch.float32); model.eval(); NL = 6
emb = model.get_embd(model.codi, model.model_name)


def kv_at(pkv, pos):
    if isinstance(pkv, (tuple, list)): return [(k[:, :, pos].clone(), v[:, :, pos].clone()) for k, v in pkv]
    return [(pkv.key_cache[l][:, :, pos].clone(), pkv.value_cache[l][:, :, pos].clone()) for l in range(len(pkv.key_cache))]
def set_kv(pkv, pos, src):
    for l, (k, v) in enumerate(src):
        if isinstance(pkv, (tuple, list)): pkv[l][0][:, :, pos] = k; pkv[l][1][:, :, pos] = v
        else: pkv.key_cache[l][:, :, pos] = k; pkv.value_cache[l][:, :, pos] = v


@torch.no_grad()
def run(q, feed=None, kv=None, block=(), hide_ans=()):
    """feed {s: z} replaces the input of iteration s+1; kv {s: kv list} overwrites that iteration's keys/values after it runs;
    block = sites every later iteration cannot attend to; hide_ans = sites the answer tokens cannot attend to.
    Returns (answer, {s: z_s}, {s: kv list at site s})."""
    ids, attn = encode_question(model, tok, q.strip().replace('  ', ' '), DEV); Q = ids.size(1)
    o = model.codi(input_ids=ids, use_cache=True, output_hidden_states=True, past_key_values=None, attention_mask=attn)
    pkv = o.past_key_values; lat = model.prj(o.hidden_states[-1][:, -1, :].unsqueeze(1)); Z = {}; KV = {}
    for s in range(NL):
        x = feed[s] if feed and s in feed else lat; Z[s] = x
        m = torch.ones((1, Q + s + 1), dtype=torch.long, device=DEV)
        for b in block:
            if b < s: m[:, Q + b] = 0
        o = model.codi(inputs_embeds=x, use_cache=True, output_hidden_states=True, past_key_values=pkv, attention_mask=m)
        pkv = o.past_key_values; lat = model.prj(o.hidden_states[-1][:, -1, :].unsqueeze(1))
        if kv and s in kv: set_kv(pkv, Q + s, kv[s])
        KV[s] = kv_at(pkv, Q + s)
    out = emb(torch.tensor([model.eot_id], device=DEV)).unsqueeze(0); toks = []; L0 = Q + NL
    for t in range(64):  # decode_answer, with an optional mask over latent positions
        m = torch.ones((1, L0 + t + 1), dtype=torch.long, device=DEV)
        for h in hide_ans: m[:, Q + h] = 0
        o = model.codi(inputs_embeds=out, use_cache=True, past_key_values=pkv, attention_mask=m if hide_ans else None)
        pkv = o.past_key_values; nid = torch.argmax(o.logits[:, -1, :model.codi.config.vocab_size - 1], dim=-1)
        if nid.item() == tok.eos_token_id: break
        toks.append(nid.item()); out = emb(nid).unsqueeze(1)
    return extract_final_number(tok.decode(toks, skip_special_tokens=True)), Z, KV, Q



ex = load_gsm8k_aug(split='test', n=None, seed=None); ex = ex[:30] if SMOKE else ex
EVEN, ODD = [0, 2, 4], [1, 3, 5]; res = []; t0 = time.time()
for k, e in enumerate(ex):
    r = dict(idx=k, steps=len(e.intermediate_values), gold=e.answer)
    for c, blk in (('clean', ()), ('nb_all', EVEN + ODD), ('nb_even', EVEN), ('nb_odd', ODD)):
        a = run(e.question, block=blk)[0]; r[c] = a; r[c + '_ok'] = bool(is_correct(a, e.answer))
    res.append(r)
    if k % 200 == 0: print(time.strftime('%H:%M'), f'{k}/{len(ex)} {time.time() - t0:.0f}s', flush=True)
json.dump(res, open(OUT + ('/t14_smoke.json' if SMOKE else '/t14_gsm8k_lookback_depth.json'), 'w'))
import collections
by = collections.defaultdict(list)
for r in res: by[min(r['steps'], 6)].append(r)
print('| steps | n | clean acc | right kept: nb_all / nb_even / nb_odd |')
for n in sorted(by):
    R = by[n]; ok = [r for r in R if r['clean_ok']]
    kept = ' / '.join(f"{sum(r[c + '_ok'] for r in ok) / max(len(ok), 1):.2f}" for c in ('nb_all', 'nb_even', 'nb_odd'))
    print(f"| {n}{'+' if n == 6 else ''} | {len(R)} | {sum(r['clean_ok'] for r in R) / len(R):.3f} | {kept} (n right {len(ok)}) |")
print('DONE')
