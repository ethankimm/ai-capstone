"""t22 / t23: released SIM-CODI GPT-2 (internlm/SIM_COT-GPT2-CODI) vs released CODI GPT-2 (zen-E/CODI-gpt2) on GSM8K.
Pre-registered in scripts/codi_null_audit/prereg/t_predictions.md (t22, t23), commit 6a1e88b.

Usage (CODI checkout as cwd, CODI venv; via codi_audit_artifacts/run_guarded.sh):
  sim_gsm8k.py <ai-capstone dir> sweep <sim|codi> <device> <out dir>
  sim_gsm8k.py <ai-capstone dir> route <sim|codi> <device> <out dir>
SMOKE=1: 12 items / 6 pairs.

sweep: every GSM8K test item (1319). Answer after k = 0..6 latent passes (cache copied after each pass, answer decoded from it, the
  same generation path as t12 / CODI's decode_answer). At k = 6 also: SIM's step decoder applied to z_0..z_6 (greedy, '<<a op b=c>>'),
  and logit-lens top-1 token at the question's last position and at each latent pass, layers 4/6/8/10/12.
route: twin pairs = the 1471 candidates of 20260927-004340 (regenerated exactly: codi_audit_artifacts/t22/gsm8k_twin_candidates.json),
  qualified for THIS model (recipient and twin answered right, same question token length). Per site s = 0..5 (input of pass s):
  both_s (twin's z_s fed in), input_s (twin's z_s fed in, pass s's keys/values reset to the recipient's), kv_s (own z_s, twin's keys/values
  at pass s), kv_noans_s (kv_s with the answer masked from pass s); both_all, kv_all. After both_s: the step decoder on the downstream
  latents z_{s+1}..z_6 of the patched run. Score offline: answer = twin answer.
"""
import sys, os, json, time, copy, re
REPO, MODE, MODEL, DEV, OUT = sys.argv[1:6]; SMOKE = int(os.environ.get('SMOKE', '0'))
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1'); os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, REPO + '/scripts'); sys.path.insert(0, REPO); sys.path.insert(0, os.getcwd())
import torch, transformers
from safetensors.torch import load_file
from src.model import DataArguments, ModelArguments, TrainingArguments
import diagnose_codi_latents  # noqa
from eval_codi import build_model
from decode_patch_codi import encode_question
from latentreasoning.data.gsm8k_aug import load_gsm8k_aug
from latentreasoning.eval.metrics import extract_final_number, is_correct

BASE = os.path.dirname(REPO)
SIM_DIR = BASE + '/grid_checkpoints/simcot_gpt2_codi'
CODI_DIR = os.path.expanduser('~/.cache/huggingface/hub/models--zen-E--CODI-gpt2/snapshots/fd641b3d3edc59e4f534b55588e906588c9e36bb')
CK = SIM_DIR if MODEL == 'sim' else CODI_DIR
argv = (f"--output_dir /tmp/u --model_name_or_path gpt2 --seed 11 --model_max_length 512 --lora_r 128 --lora_alpha 32 --lora_init --greedy True "
        f"--num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 --inf_latent_iterations 6 --inf_num_iterations 1 "
        f"--remove_eos True --use_lora True --ckpt_dir {CK}").split()
ma, da, ta = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments)).parse_args_into_dataclasses(args=argv)
ta.print_ref_model_stats = False; ta.print_loss = False
model, tok, lr = build_model(ma, ta, device=DEV, dtype=torch.float32); model.eval(); NL = 6
bad = [k for k in lr.missing_keys if not k.startswith('codi.base_model.model.lm_head')]
assert not bad, bad[:5]
unexp = [k for k in lr.unexpected_keys if not k.startswith('decoder.')]
assert not unexp, unexp[:5]
emb = model.get_embd(model.codi, model.model_name)
lm = model.codi.get_base_model(); ln_f, head = lm.transformer.ln_f, lm.lm_head
VOC = model.codi.config.vocab_size - 1

# SIM's training-only step decoder (fresh GPT-2 + trained weights); applied to either model's latents
sd = {k[len('decoder.'):]: v for k, v in load_file(SIM_DIR + '/model.safetensors').items() if k.startswith('decoder.')}
decoder = transformers.GPT2LMHeadModel.from_pretrained('gpt2'); missing = decoder.load_state_dict(sd, strict=False)
assert not missing.unexpected_keys and all('attn.bias' in k or 'masked_bias' in k for k in missing.missing_keys), missing
decoder = decoder.to(DEV).float().eval()
LAYERS = (4, 6, 8, 10, 12)


def kv_at(pkv, pos):
    if isinstance(pkv, (tuple, list)): return [(k[:, :, pos].clone(), v[:, :, pos].clone()) for k, v in pkv]
    return [(pkv.key_cache[l][:, :, pos].clone(), pkv.value_cache[l][:, :, pos].clone()) for l in range(len(pkv.key_cache))]


def set_kv(pkv, pos, src):
    for l, (k, v) in enumerate(src):
        if isinstance(pkv, (tuple, list)): pkv[l][0][:, :, pos] = k; pkv[l][1][:, :, pos] = v
        else: pkv.key_cache[l][:, :, pos] = k; pkv.value_cache[l][:, :, pos] = v


@torch.no_grad()
def answer_from(pkv, Q, k, hide_ans=()):
    pkv = copy.deepcopy(pkv); out = emb(torch.tensor([model.eot_id], device=DEV)).unsqueeze(0); toks = []; L0 = Q + k
    for t in range(64):
        m = None
        if hide_ans:
            m = torch.ones((1, L0 + t + 1), dtype=torch.long, device=DEV)
            for h in hide_ans: m[:, Q + h] = 0
        o = model.codi(inputs_embeds=out, use_cache=True, past_key_values=pkv, attention_mask=m)
        pkv = o.past_key_values; nid = torch.argmax(o.logits[:, -1, :VOC], dim=-1)
        if nid.item() == tok.eos_token_id: break
        toks.append(nid.item()); out = emb(nid).unsqueeze(1)
    return extract_final_number(tok.decode(toks, skip_special_tokens=True))


def lens_top(hs):
    """hs: tuple of [1, T, d] hidden states (embeddings + 12 layers); top-1 token at the last position per layer in LAYERS."""
    return [int(head(hs[l][0, -1] if l == 12 else ln_f(hs[l][0, -1]))[:VOC].argmax()) for l in LAYERS]


@torch.no_grad()
def run(q, feed=None, kv=None, hide_ans=(), sweep=False, lens=False):
    """feed {s: z} replaces the input of pass s; kv {s: kv list} overwrites pass s's keys/values after it runs.
    Returns dict(ans, Z [z_0..z_6], KV {s}, Q, sweep [answer after k passes], lens)."""
    ids, attn = encode_question(model, tok, q.strip().replace('  ', ' '), DEV); Q = ids.size(1)
    o = model.codi(input_ids=ids, use_cache=True, output_hidden_states=True, past_key_values=None, attention_mask=attn)
    pkv = o.past_key_values; lat = model.prj(o.hidden_states[-1][:, -1, :].unsqueeze(1)); Z = []; KV = {}
    L = [lens_top(o.hidden_states)] if lens else None; sw = [answer_from(pkv, Q, 0)] if sweep else None
    for s in range(NL):
        x = feed[s] if feed and s in feed else lat; Z.append(x)
        m = torch.ones((1, Q + s + 1), dtype=torch.long, device=DEV)
        o = model.codi(inputs_embeds=x, use_cache=True, output_hidden_states=True, past_key_values=pkv, attention_mask=m)
        pkv = o.past_key_values; lat = model.prj(o.hidden_states[-1][:, -1, :].unsqueeze(1))
        if kv and s in kv: set_kv(pkv, Q + s, kv[s])
        KV[s] = kv_at(pkv, Q + s)
        if lens: L.append(lens_top(o.hidden_states))
        if sweep and s < NL - 1: sw.append(answer_from(pkv, Q, s + 1))
    Z.append(lat)
    a = answer_from(pkv, Q, NL, hide_ans)
    if sweep: sw.append(a)
    return dict(ans=a, Z=Z, KV=KV, Q=Q, sweep=sw, lens=L)


@torch.no_grad()
def dec_step(z):
    x = z; toks = []; past = None
    for t in range(40):
        o = decoder(inputs_embeds=x, past_key_values=past, use_cache=True)
        past = o.past_key_values; nid = int(o.logits[0, -1].argmax())
        if nid == tok.eos_token_id: break
        toks.append(nid); x = emb(torch.tensor([[nid]], device=DEV))
    return tok.decode(toks)


os.makedirs(OUT, exist_ok=True); t0 = time.time(); tag = '_smoke' if SMOKE else ''
if MODE == 'sweep':
    ex = load_gsm8k_aug(split='test', n=None, seed=None)
    if SMOKE: ex = ex[:12]
    res = []
    for i, e in enumerate(ex):
        r = run(e.question, sweep=True, lens=True)
        if SMOKE and i == 0:
            assert str(run(e.question)['ans']) == str(r['ans']), 'sweep path changed the full answer'
            assert str(run(e.question, kv={0: r['KV'][0]})['ans']) == str(r['ans']), 'own keys/values write-back changed the answer'
        res.append(dict(idx=e.idx, gold=e.answer, values=e.intermediate_values, ans=r['ans'], correct=is_correct(r['ans'], e.answer),
                        sweep=r['sweep'], sweep_correct=[is_correct(a, e.answer) for a in r['sweep']], lens=r['lens'],
                        decoded=[dec_step(z) for z in r['Z']]))
        if i % 100 == 0: print(time.strftime('%H:%M'), f'{MODEL} item {i}/{len(ex)} {time.time() - t0:.0f}s acc so far {sum(x["correct"] for x in res) / len(res):.3f}', flush=True)
    json.dump(dict(model=MODEL, ckpt=CK, layers=LAYERS, items=res), open(f'{OUT}/t22_sweep_{MODEL}{tag}.json', 'w'))
    print(f'{MODEL} accuracy {sum(x["correct"] for x in res) / len(res):.3f}; by k ' +
          ' '.join(f'{sum(x["sweep_correct"][k] for x in res) / len(res):.3f}' for k in range(NL + 1)))
elif MODE == 'zdump':
    # t25 post hoc: raw latent vectors z_0..z_6 and the question's last hidden state (the no-latent comparison), fp16
    import numpy as np
    ex = load_gsm8k_aug(split='test', n=None, seed=None)
    if SMOKE: ex = ex[:12]
    Zs, Hs, ids = [], [], []
    for i, e in enumerate(ex):
        with torch.no_grad():
            qids, attn = encode_question(model, tok, e.question.strip().replace('  ', ' '), DEV)
            o = model.codi(input_ids=qids, use_cache=True, output_hidden_states=True, attention_mask=attn)
            Hs.append(o.hidden_states[-1][0, -1].float().cpu().numpy())
        r = run(e.question)
        Zs.append(torch.cat([z[0] for z in r['Z']], 0).float().cpu().numpy()); ids.append(e.idx)
        if i % 300 == 0: print(time.strftime('%H:%M'), f'{MODEL} zdump {i}/{len(ex)}', flush=True)
    np.savez_compressed(f'{OUT}/t25_z_{MODEL}{tag}.npz', Z=np.stack(Zs).astype(np.float16), H=np.stack(Hs).astype(np.float16), idx=np.array(ids))
    print('saved', np.stack(Zs).shape)
elif MODE == 'conf':
    # t25 baseline: answer confidence at k = 6 (same greedy path; log-probs of the generated tokens)
    ex = load_gsm8k_aug(split='test', n=None, seed=None)
    if SMOKE: ex = ex[:12]
    res = []
    for i, e in enumerate(ex):
        with torch.no_grad():
            ids, attn = encode_question(model, tok, e.question.strip().replace('  ', ' '), DEV); Q = ids.size(1)
            o = model.codi(input_ids=ids, use_cache=True, output_hidden_states=True, attention_mask=attn)
            pkv = o.past_key_values; lat = model.prj(o.hidden_states[-1][:, -1, :].unsqueeze(1))
            for s in range(NL):
                o = model.codi(inputs_embeds=lat, use_cache=True, output_hidden_states=True, past_key_values=pkv,
                               attention_mask=torch.ones((1, Q + s + 1), dtype=torch.long, device=DEV))
                pkv = o.past_key_values; lat = model.prj(o.hidden_states[-1][:, -1, :].unsqueeze(1))
            out = emb(torch.tensor([model.eot_id], device=DEV)).unsqueeze(0); toks = []; lps = []
            for t in range(64):
                o = model.codi(inputs_embeds=out, use_cache=True, past_key_values=pkv)
                pkv = o.past_key_values; lp = torch.log_softmax(o.logits[0, -1, :VOC].float(), -1); nid = int(lp.argmax())
                lps.append(float(lp[nid]))
                if nid == tok.eos_token_id: break
                toks.append(nid); out = emb(torch.tensor([nid], device=DEV)).unsqueeze(1)
        text = tok.decode(toks, skip_special_tokens=True); a = extract_final_number(text)
        # log-probs of the number tokens only (the answer value), plus all tokens
        num_lp = [lp for tkn, lp in zip(toks, lps) if any(ch.isdigit() for ch in tok.decode([tkn]))]
        res.append(dict(idx=e.idx, ans=a, correct=is_correct(a, e.answer), text=text, lp=lps, num_lp=num_lp))
        if i % 200 == 0: print(time.strftime('%H:%M'), f'{MODEL} conf item {i}/{len(ex)} {time.time() - t0:.0f}s', flush=True)
    json.dump(dict(model=MODEL, items=res), open(f'{OUT}/t25_conf_{MODEL}{tag}.json', 'w'))
    print(f'{MODEL} accuracy {sum(x["correct"] for x in res) / len(res):.3f}')
elif MODE == 'route':
    pairs = json.load(open(BASE + '/codi_audit_artifacts/t22/gsm8k_twin_candidates.json'))
    if SMOKE: pairs = pairs[:40]
    res = []; nq = 0
    for k, p in enumerate(pairs):
        R = run(p['recipient_question'])
        if not is_correct(R['ans'], p['recipient_gold']): continue
        T = run(p['twin_question'])
        if not is_correct(T['ans'], p['twin_answer']) or R['Q'] != T['Q']: continue
        nq += 1; rq = p['recipient_question']
        r = dict(recipient_idx=p['recipient_idx'], step=p['step'], chain_len=p['chain_len'], gold=p['recipient_gold'], twin_answer=p['twin_answer'],
                 recipient_values=p['recipient_values'], twin_values=p['twin_values'], base=R['ans'], twin_pred=T['ans'],
                 base_decoded=[dec_step(z) for z in R['Z']], twin_decoded=[dec_step(z) for z in T['Z']])
        for s in range(NL):
            B = run(rq, feed={s: T['Z'][s]})
            r[f'both_{s}'] = B['ans']; r[f'both_{s}_decoded'] = [dec_step(z) for z in B['Z'][s + 1:]]
            r[f'input_{s}'] = run(rq, feed={s: T['Z'][s]}, kv={s: R['KV'][s]})['ans']
            r[f'kv_{s}'] = run(rq, kv={s: T['KV'][s]})['ans']
            r[f'kv_noans_{s}'] = run(rq, kv={s: T['KV'][s]}, hide_ans=(s,))['ans']
        r['both_all'] = run(rq, feed={s: T['Z'][s] for s in range(NL)})['ans']
        r['kv_all'] = run(rq, kv={s: T['KV'][s] for s in range(NL)})['ans']
        if SMOKE and nq == 1:
            assert str(run(rq, kv={2: R['KV'][2]})['ans']) == str(R['ans']), 'own keys/values write-back changed the answer'
            assert str(run(rq, feed={3: R['Z'][3]})['ans']) == str(R['ans']), 'own input write-back changed the answer'
        res.append(r)
        if nq % 25 == 1: print(time.strftime('%H:%M'), f'{MODEL} candidate {k}/{len(pairs)} qualified {nq} {time.time() - t0:.0f}s', flush=True)
        if SMOKE and nq >= 6: break
    json.dump(dict(model=MODEL, ckpt=CK, pairs=res), open(f'{OUT}/t22_route_{MODEL}{tag}.json', 'w'))
    fol = lambda c: sum(str(r[c]) == str(r['twin_answer']) for r in res) / max(len(res), 1)
    print(f'{MODEL} qualified {len(res)}; all sites both {fol("both_all"):.3f} kv {fol("kv_all"):.3f}')
    for s in range(NL):
        print(f'  site {s}: ' + ' | '.join(f'{c} {fol(f"{c}_{s}"):.3f}' for c in ('both', 'input', 'kv', 'kv_noans')))
print('DONE', f'{time.time() - t0:.0f}s')
