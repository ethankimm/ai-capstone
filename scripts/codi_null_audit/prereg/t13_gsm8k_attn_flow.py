"""Attention flow in released GSM8K CODI, the t9 measures for the row t9 lacked. Usage (CODI checkout, CODI venv):
t13_gsm8k_attn_flow.py <ai-capstone dir> <device> <out dir>. SMOKE=1: 20 items. Full: all 1319 GSM8K-Aug test items.
Model and generation path as t11 / t12 (zen-E/CODI-gpt2 fd641b3, fp32, batch 1, no padding). For the query of latent iteration s (input z_s),
every layer: raw attention (mean over heads) and the norm share ||sum_h a v W_O|| (W_O from c_proj, LoRA included), over key sets:
question tokens (attention sink = first token excluded), earlier EVEN sites (value sites), earlier ODD sites (punctuation sites), self, sink.
Also the share by distance d (site s reading site s-d). Split by whether the model answers the item right.
Compare with t9 (RuleChain): start-last .09-.27 on earlier latents, chains .01-.09 while computing, relay .02-.18. Descriptive only."""
import sys, os, json, time
REPO = sys.argv[1]; DEV = sys.argv[2]; OUT = sys.argv[3]; SMOKE = int(os.environ.get('SMOKE', '0'))
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('HF_DATASETS_OFFLINE', '1'); os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, REPO + '/scripts'); sys.path.insert(0, REPO); sys.path.insert(0, os.getcwd())
import numpy as np, torch, transformers
from src.model import DataArguments, ModelArguments, TrainingArguments
import diagnose_codi_latents  # noqa
from eval_codi import build_model
from decode_patch_codi import encode_question, decode_answer
from latentreasoning.data.gsm8k_aug import load_gsm8k_aug
from latentreasoning.eval.metrics import extract_final_number, is_correct
CKPT = os.path.expanduser('~/.cache/huggingface/hub/models--zen-E--CODI-gpt2/snapshots/fd641b3d3edc59e4f534b55588e906588c9e36bb')
argv = (f"--output_dir /tmp/u --model_name_or_path gpt2 --seed 11 --model_max_length 512 --lora_r 128 --lora_alpha 32 --lora_init --greedy True "
        f"--num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 --inf_latent_iterations 6 --inf_num_iterations 1 "
        f"--remove_eos True --use_lora True --ckpt_dir {CKPT}").split()
ma, da, ta = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments)).parse_args_into_dataclasses(args=argv)
ta.print_ref_model_stats = False; ta.print_loss = False
model, tok, lr = build_model(ma, ta, device=DEV, dtype=torch.float32); model.eval(); NL = 6
body = model.codi.get_base_model().transformer
WO = []
with torch.no_grad():
    for blk in body.h:
        I = torch.eye(768, device=DEV); WO.append((blk.attn.c_proj(I) - blk.attn.c_proj(torch.zeros(1, 768, device=DEV))).reshape(12, 64, 768))
SETS = ['question', 'even_prev', 'odd_prev', 'self', 'sink']
ex = load_gsm8k_aug(split='test', n=None, seed=None); ex = ex[:20] if SMOKE else ex
acc = {g: {k: np.zeros((NL, 12, len(SETS))) for k in ('raw', 'norm')} for g in ('right', 'wrong')}
dist = {g: np.zeros((NL, 12, NL)) for g in ('right', 'wrong')}; cnt = {'right': 0, 'wrong': 0}; t0 = time.time()
with torch.no_grad():
    for n, e in enumerate(ex):
        ids, attn = encode_question(model, tok, e.question.strip().replace('  ', ' '), DEV); Q = ids.size(1)
        o = model.codi(input_ids=ids, use_cache=True, output_hidden_states=True, past_key_values=None, attention_mask=attn)
        pkv = o.past_key_values; lat = model.prj(o.hidden_states[-1][:, -1, :].unsqueeze(1)); A = {k: np.zeros((NL, 12, len(SETS))) for k in ('raw', 'norm')}; Dd = np.zeros((NL, 12, NL))
        for s in range(NL):
            o = model.codi(inputs_embeds=lat, use_cache=True, output_hidden_states=True, output_attentions=True, past_key_values=pkv)
            pkv = o.past_key_values; lat = model.prj(o.hidden_states[-1][:, -1, :].unsqueeze(1)); K = Q + s + 1
            sel = np.zeros((len(SETS), K), bool); sel[0, 1:Q] = True; sel[4, 0] = True; sel[3, Q + s] = True
            for b in range(s): sel[1 if b % 2 == 0 else 2, Q + b] = True
            for l in range(12):
                a = o.attentions[l][0, :, 0, :K]
                v = pkv[l][1][0, :, :K, :] if isinstance(pkv[l], (tuple, list)) else pkv.value_cache[l][0, :, :K, :]
                c = torch.einsum('hk,hkd,hde->ke', a, v, WO[l]).norm(dim=-1)
                for key, val in (('raw', a.mean(0)), ('norm', c)):
                    val = (val / val.sum()).cpu().numpy(); A[key][s, l] = sel @ val
                    if key == 'norm':
                        for d in range(1, s + 1): Dd[s, l, d - 1] = val[Q + s - d]
        ans = extract_final_number(decode_answer(model, tok, pkv, lat, DEV, 64)); g = 'right' if is_correct(ans, e.answer) else 'wrong'
        for k in A: acc[g][k] += A[k]
        dist[g] += Dd; cnt[g] += 1
        if n % 200 == 0: print(time.strftime('%H:%M'), f'{n}/{len(ex)} {time.time() - t0:.0f}s', flush=True)
out = {g: dict(n=cnt[g], sets=SETS, **{f'{k}_share': (acc[g][k] / max(cnt[g], 1)).tolist() for k in acc[g]}, norm_by_distance=(dist[g] / max(cnt[g], 1)).tolist()) for g in cnt}
json.dump(out, open(OUT + ('/t13_smoke.json' if SMOKE else '/t13_gsm8k_attn_flow.json'), 'w'))
for g in cnt:
    if not cnt[g]: continue
    S = np.array(out[g]['norm_share']).mean(1); ns = S[:, :4] / S[:, :4].sum(1, keepdims=True)
    print(f'{g} n={cnt[g]} norm, non-sink, by site 0-5: earlier latents ' + ' '.join(f'{x:.2f}' for x in ns[:, 1] + ns[:, 2])
          + ' | of which even ' + ' '.join(f'{x:.2f}' for x in ns[:, 1]) + ' | question ' + ' '.join(f'{x:.2f}' for x in ns[:, 0]))
print('DONE')
