"""Does released GSM8K CODI chain or pool? Route and site tests on the 317 aligned minimal pairs of 20260927-004340.
Usage (from the CODI checkout, CODI venv): t11_gsm8k_routes.py <ai-capstone dir> <device> <out dir>. SMOKE=1: 20 pairs, no full-test pass.
Model: zen-E/CODI-gpt2 @ fd641b3 (local HF cache), loaded with that record's flags, fp32 on MPS (the record used bf16 on CUDA, so expect
small differences). Batch 1 and the record's generation path (scripts/decode_patch_codi.py: encode_question, run_thoughts, decode_answer).
Pairs: recipient = test item recipient_idx (load_gsm8k_aug test, unshuffled), twin = the recipient with one number perturbed (twin_question,
twin_answer). Sites as the record: z_s feeds iteration s+1 (z_0 = projection of the <bot> state); released GSM8K CODI carries values on
even sites (z_0, z_2, z_4) and CoT punctuation on odd sites.
Conditions on pairs (recipient question, latents from the twin, later latents recomputed unless stated; scored: answer = twin answer):
  all            z_0..z_5 from the twin (record: .839)
  single_s       z_s from the twin, s in 0, 2, 4 (record: .199 / .202 / .360)
  even_pairs     two of z_0, z_2, z_4 from the twin; even_all: all three
  prefix_2       z_0..z_2 from the twin (record .539): does a recomputed z_3.. carry the twin's value forward?
  nb_*           the same with every latent iteration masked from attending to earlier latent iterations (answer still sees all)
On all 1319 test items: clean accuracy and nb accuracy (look-back cut), the route test of t10.
Reading against RuleChain (20261108-042557): a chain keeps nb accuracy and its swaps propagate under nb; start-last CODI loses accuracy
under nb and its single swaps never propagate (votes)."""
import sys, os, json, time, itertools
REPO = sys.argv[1]; DEV = sys.argv[2]; OUT = sys.argv[3]; SMOKE = int(os.environ.get('SMOKE', '0'))
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('HF_DATASETS_OFFLINE', '1'); os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, REPO + '/scripts'); sys.path.insert(0, REPO); sys.path.insert(0, os.getcwd())
import torch, transformers
from src.model import DataArguments, ModelArguments, TrainingArguments
import diagnose_codi_latents  # noqa (patches torch.load for CPU)
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


@torch.no_grad()
def think(q, override=None, noback=False):
    """run_thoughts (decode_patch_codi.py) with an optional look-back cut; returns (pkv, latent, {s: z_s})."""
    ids, attn = encode_question(model, tok, q.strip().replace('  ', ' '), DEV); Q = ids.size(1)
    o = model.codi(input_ids=ids, use_cache=True, output_hidden_states=True, past_key_values=None, attention_mask=attn)
    pkv = o.past_key_values; lat = o.hidden_states[-1][:, -1, :].unsqueeze(1); lat = model.prj(lat) if model.use_prj else lat; Z = {}
    for it in range(1, NL + 1):
        feed = override[it - 1] if override is not None and (it - 1) in override else lat
        Z[it - 1] = feed
        m = None
        if noback and it > 1:
            m = torch.ones((1, Q + it), dtype=torch.long, device=DEV); m[:, Q:Q + it - 1] = 0
        o = model.codi(inputs_embeds=feed, use_cache=True, output_hidden_states=True, past_key_values=pkv, attention_mask=m)
        pkv = o.past_key_values; h = o.hidden_states[-1][:, -1, :].unsqueeze(1); lat = model.prj(h) if model.use_prj else h
    return pkv, lat, Z


def answer(q, override=None, noback=False):
    pkv, lat, Z = think(q, override, noback)
    return extract_final_number(decode_answer(model, tok, pkv, lat, DEV, 64)), Z


ex = load_gsm8k_aug(split='test', n=None, seed=None)
pairs = [json.loads(l) for l in open(REPO + '/results/20260927-004340_codi_minimal-pair-patch-aligned/predictions.jsonl')]
if SMOKE: pairs = pairs[:20]
EVEN = [0, 2, 4]
COND = {'all': list(range(NL)), 'single_0': [0], 'single_2': [2], 'single_4': [4], 'even_02': [0, 2], 'even_04': [0, 4], 'even_24': [2, 4],
        'even_all': EVEN, 'prefix_2': [0, 1, 2]}
res = []; t0 = time.time()
for k, p in enumerate(pairs):
    rq = ex[p['recipient_idx']].question; ra, Zr = answer(rq); ta_, Zt = answer(p['twin_question'])
    r = dict(recipient_idx=p['recipient_idx'], step=p['step'], gold=p['recipient_gold'], twin_answer=p['twin_answer'], base=ra, twin_pred=ta_,
             rec_all=p['all_slot']['matches_twin'], rec_single={s: p['single_slot'][str(s)]['matches_twin'] for s in EVEN}, rec_prefix2=p['prefix']['2']['matches_twin'])
    for nb in (False, True):
        for c, sites in COND.items():
            # prefix: twin sites given, later sites recomputed from them; others: given sites patched, rest recomputed as well
            a, _ = answer(rq, {s: Zt[s] for s in sites}, noback=nb); r[('nb_' if nb else '') + c] = a
        if nb: r['nb_clean'], _ = answer(rq, None, noback=True)
    res.append(r)
    if k % 50 == 0: print(time.strftime('%H:%M'), f'pair {k}/{len(pairs)} {time.time() - t0:.0f}s', flush=True)
acc = {}
if not SMOKE:
    for nb in (False, True):
        ok = [is_correct(answer(e.question, None, noback=nb)[0], e.answer) for e in ex]; acc['nb' if nb else 'clean'] = sum(ok) / len(ok)
        print(time.strftime('%H:%M'), f"full test {'nb' if nb else 'clean'} acc {acc['nb' if nb else 'clean']:.3f}", flush=True)
json.dump(dict(pairs=res, full_test=acc), open(OUT + ('/t11_smoke.json' if SMOKE else '/t11_gsm8k_routes.json'), 'w'))
q = [r for r in res if str(r['base']) == str(r['gold']) and str(r['twin_pred']) == str(r['twin_answer'])]
f = lambda c: sum(str(r[c]) == str(r['twin_answer']) for r in q) / max(len(q), 1)
print(f'qualified here {len(q)}/{len(res)} | record all {sum(r["rec_all"] for r in res) / len(res):.3f}')
for c in ['all', 'single_0', 'single_2', 'single_4', 'even_02', 'even_04', 'even_24', 'even_all', 'prefix_2']:
    print(f'  {c:9s} follows twin {f(c):.3f} | nb {f("nb_" + c):.3f}')
print(f'  recipient right under nb: {sum(str(r["nb_clean"]) == str(r["gold"]) for r in q) / max(len(q), 1):.3f}')
print('DONE')
