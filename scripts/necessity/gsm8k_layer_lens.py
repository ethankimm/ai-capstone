"""t51: per-layer logit lens inside each pass of the released CODI-family checkpoints on GSM8K. Pre-registered in
prereg/t_predictions.md (t51).

  ../codi/.venv/bin/python scripts/necessity/gsm8k_layer_lens.py <gsm8k_codi | gsm8k_sim | gsm8k_coconut | gsm8k_simcoconut | gsm8k_llama | gsm8k_simllama1b | gsm8k_simllama3b> <device> <out json>

Natural (unpatched) runs on the 1,319 GSM8K-Aug test items; items answered right are scored. At each serial position (pass 0 = the
question's last position, passes 1..6 = the latent passes) and each layer l = 0..L, the last-position hidden state is read with the
logit lens: the final norm (GPT-2 ln_f, Llama model.norm; HF's hidden_states[L] already has it) and the output head, top-10 tokens.
A rationale step's value counts as read if one of its single-token forms (with or without a leading space) is in that top 10.
Kept steps: integer values that are single tokens, not among the question's numbers, and not equal to another step's value.

Per record (one kept step of one right item): s* = first pass whose output (layer L) reads the value; onset l* = the lowest layer from
which the value stays read through layer L in pass s*. Chance: the next record at the same depth (another item, a different value not
in this question) scored on this record's tokens. Same-pass parent/child pairs: a step and the deepest step it uses, both with s*
in the same pass; their onsets are compared.
"""
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adapters  # noqa: E402

MODEL, DEV, OUT = sys.argv[1:4]; SMOKE = int(os.environ.get('SMOKE', '0')); TOPK = 10
BASE = adapters.BASE
if DEV == 'cpu': torch.set_num_threads(int(os.environ.get('THREADS', '4')))
if MODEL in adapters.LLAMA_GSM8K:
    ad = adapters.LLAMA_GSM8K[MODEL][1](DEV)
elif MODEL in ('gsm8k_codi', 'gsm8k_sim'):
    ad = adapters.GSM8KCODI(DEV, ckpt=str(BASE / 'grid_checkpoints/simcot_gpt2_codi') if MODEL == 'gsm8k_sim' else None)
elif MODEL in ('gsm8k_coconut', 'gsm8k_simcoconut'):  # t51b: GPT-2 Coconut / SIM-Coconut
    ad = adapters.GridCoconut('gsm8k', DEV, 'coconut',
                              path=str(BASE / 'grid_checkpoints/simcot_gpt2_coconut/checkpoint_28') if MODEL == 'gsm8k_simcoconut' else None)
    if MODEL == 'gsm8k_simcoconut': ad.n_latents = 10  # as hops.py (t36) and t39: SIM-CoT gsm_simcot.yaml, c_thought 2 x max_latent_stage 5
else:
    raise SystemExit(f'unknown model {MODEL}')
COC = isinstance(ad, adapters.GridCoconut)
base = ad.lm if COC else ad.m.codi.get_base_model(); head = base.lm_head
norm = base.transformer.ln_f if hasattr(base, 'transformer') else base.model.norm
L = base.config.num_hidden_layers; K = ad.n_latents; tok = ad.tok
print(f'{MODEL}: {L} layers, {K} latents, dtype {next(base.parameters()).dtype}', flush=True)
from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.eval.counterfactual import parse_steps  # noqa: E402
from latentreasoning.eval.metrics import is_correct  # noqa: E402
NUM = re.compile(r'\d+(?:\.\d+)?')


def tok_ids(v):
    try:
        f = float(str(v).replace(',', ''))
    except ValueError:
        return None
    if not f.is_integer(): return None
    s = str(int(f)); ids = set()
    for form in (s, ' ' + s):
        t = tok.encode(form, add_special_tokens=False)
        if len(t) == 1: ids.add(t[0])
    return ids or None


def canon(v):
    try:
        f = float(str(v).replace(',', ''))
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return None


@torch.no_grad()
def lens_tops(hs):
    """hs: HF hidden_states tuple (L + 1 tensors [1, T, d]); returns [L + 1, TOPK] top token ids at the last position."""
    x = torch.stack([h[0, -1] for h in hs])  # [L + 1, d]
    x = torch.cat([norm(x[:L]), x[L:]])  # hidden_states[L] is already normed
    return head(x).float().topk(TOPK, dim=-1).indices.cpu()


@torch.no_grad()
def run(question):
    """natural run; returns (answer, tops [K + 1, L + 1, TOPK])."""
    if COC:  # Coconut: question + <|start-latent|>, then K passes fed the post-ln_f last hidden state (GridCoconut.begin / latent)
        ids = torch.tensor([ad.qids(question) + [ad.sl]], device=ad.dev)
        o = ad.body(input_ids=ids, use_cache=True, output_hidden_states=True)
        tops = [lens_tops(o.hidden_states)]; ctx = dict(Q=ids.size(1)); past = o.past_key_values; z = o.last_hidden_state[:, -1, :]
        for s in range(K):
            o = ad.body(inputs_embeds=z[:, None], use_cache=True, past_key_values=past, output_hidden_states=True)
            tops.append(lens_tops(o.hidden_states)); past = o.past_key_values; z = o.last_hidden_state[:, -1, :]
        return ad.finish(ctx, past)[0], torch.stack(tops)
    ids, attn = ad.enc(ad.m, ad.tok, question.strip().replace('  ', ' '), ad.dev)
    o = ad.m.codi(input_ids=ids, use_cache=True, output_hidden_states=True, past_key_values=None, attention_mask=attn)
    tops = [lens_tops(o.hidden_states)]; past = o.past_key_values; ctx = dict(Q=ids.size(1))
    z = ad.m.prj(o.hidden_states[-1][:, -1, :].unsqueeze(1))[:, 0]
    for s in range(K):
        m = torch.ones((1, ctx['Q'] + s + 1), dtype=torch.long, device=ad.dev)
        o = ad.m.codi(inputs_embeds=z[:, None], use_cache=True, output_hidden_states=True, past_key_values=o.past_key_values if s else past,
                      attention_mask=m)
        tops.append(lens_tops(o.hidden_states)); z = ad.m.prj(o.hidden_states[-1][:, -1, :].unsqueeze(1))[:, 0]
    return ad.finish(ctx, o.past_key_values)[0], torch.stack(tops)


def steps_of(e):
    st = parse_steps(e.rationale); qn = {canon(x) for x in NUM.findall(e.question.replace(',', ''))}
    vals = [canon(s['val']) for s in st]; depth, parent, out = {}, {}, []
    for k, s in enumerate(st):
        ops = {canon(x) for x in NUM.findall(s['expr'].replace(',', ''))}
        used = [j for j in range(k) if vals[j] in ops]
        depth[k] = 1 + max(depth[j] for j in used) if used else 1
        parent[k] = max(used, key=lambda j: (depth[j], j)) if used else None
        v = vals[k]
        if v is None or v in qn or vals.count(v) > 1: continue
        ids = tok_ids(v)
        if ids is None: continue
        out.append(dict(k=k, val=v, ids=ids, depth=depth[k], parent=parent[k], final=k == len(st) - 1))
    return out


te = load_gsm8k_aug(split='test', n=None, seed=None)
if SMOKE: te = te[:SMOKE]
recs, tops_by_item, n_right, t0 = [], {}, 0, time.time()
for i, e in enumerate(te):
    ans, T = run(e.question)
    ok = bool(ans is not None and ans != '' and is_correct(str(ans), str(e.answer)))
    if ok:
        n_right += 1; tops_by_item[i] = T; qn = {canon(x) for x in NUM.findall(e.question.replace(',', ''))}
        for r in steps_of(e):
            hit = torch.zeros(K + 1, L + 1, dtype=torch.bool)
            for t in r['ids']: hit |= (T == t).any(-1)
            recs.append(dict(item=i, k=r['k'], val=r['val'], depth=r['depth'], parent=r['parent'], final=r['final'],
                             qnums=sorted(x for x in qn if x), R=hit.tolist()))
    if i % 50 == 0 or i == len(te) - 1:
        print(time.strftime('%H:%M'), f'{i + 1}/{len(te)} right {n_right} records {len(recs)} {(time.time() - t0) / (i + 1):.2f}s/item', flush=True)
    if DEV == 'mps' and i % 20 == 0: torch.mps.empty_cache()


def first_onset(R):
    for s in range(K + 1):
        if R[s][L]:
            l = L
            while l > 0 and R[s][l - 1]: l -= 1
            return s, l
    return None, None


for r in recs: r['s_star'], r['onset'] = first_onset(r['R'])
D = lambda r: min(r['depth'], 4)
grp = defaultdict(list)
for r in recs: grp[D(r)].append(r)
for g in grp.values():  # chance surface: next record at the same depth from another item, value not in this question
    for x in g:
        x['C'] = None
    for j, x in enumerate(g):
        for y in g[j + 1:] + g[:j]:
            if y['item'] != x['item'] and y['val'] != x['val'] and y['val'] not in x['qnums']:
                T = tops_by_item[x['item']]; hit = torch.zeros(K + 1, L + 1, dtype=torch.bool)
                for t in tok_ids(y['val']): hit |= (T == t).any(-1)
                x['C'] = hit.tolist(); break
rng = np.random.default_rng(0)
ci = lambda v: [float(x) for x in np.percentile([rng.choice(v, len(v)).mean() for _ in range(2000)], [2.5, 97.5])] if len(v) > 1 else None

summ = {}
for d in sorted(grp):
    g = grp[d]; R = np.array([r['R'] for r in g], dtype=float); C = np.array([r['C'] for r in g if r['C'] is not None], dtype=float)
    read = [r for r in g if r['s_star'] is not None]
    sdist = {s: sum(r['s_star'] == s for r in read) / max(len(read), 1) for s in range(K + 1)}
    on = defaultdict(list)
    for r in read: on[r['s_star']].append(r['onset'])
    summ[d] = dict(n=len(g), n_read=len(read), read_rate=len(read) / len(g), rate=R.mean(0).round(4).tolist(),
                   chance=C.mean(0).round(4).tolist() if len(C) else None, s_star_dist=sdist,
                   share_before_d_minus_1=sum(r['s_star'] < d - 1 for r in read) / max(len(read), 1),
                   onset_by_s={s: dict(n=len(v), median=float(np.median(v)), mean=float(np.mean(v)), frac_median=float(np.median(v)) / L)
                               for s, v in sorted(on.items())})
    s = summ[d]
    print(f'depth {d}{"+" if d == 4 else ""}: n={s["n"]} read at some pass output {s["read_rate"]:.2f} | s* dist '
          + ' '.join(f'{x:.2f}' for x in sdist.values()) + f' | before pass d-1 {s["share_before_d_minus_1"]:.2f} | onset by s* '
          + ' '.join(f's{k}:{v["median"]:.0f}/{L}(n{v["n"]})' for k, v in s['onset_by_s'].items()), flush=True)
    rL = R[:, :, L].mean(0); cL = C[:, :, L].mean(0) if len(C) else np.zeros(K + 1)
    print('   read at layer L by pass:', ' '.join(f'{x:.2f}' for x in rL), '| chance', ' '.join(f'{x:.2f}' for x in cL), flush=True)

by_key = {(r['item'], r['k']): r for r in recs}; pp = []
for r in recs:
    p = by_key.get((r['item'], r['parent'])) if r['parent'] is not None else None
    if p and r['s_star'] is not None and p['s_star'] is not None and r['s_star'] == p['s_star']:
        pp.append(dict(item=r['item'], s=r['s_star'], child_depth=r['depth'], child_onset=r['onset'], parent_onset=p['onset']))
gt = sum(x['child_onset'] > x['parent_onset'] for x in pp); lt = sum(x['child_onset'] < x['parent_onset'] for x in pp)
from math import comb
sign_p = (sum(comb(gt + lt, j) for j in range(gt, gt + lt + 1)) / 2 ** (gt + lt)) if gt + lt else None
same_pass = dict(n=len(pp), child_later=gt, child_earlier=lt, tie=len(pp) - gt - lt, sign_p_one_sided=sign_p,
                 mean_gap=float(np.mean([x['child_onset'] - x['parent_onset'] for x in pp])) if pp else None,
                 gap_ci=ci([x['child_onset'] - x['parent_onset'] for x in pp]) if len(pp) > 1 else None)
print(f'same-pass parent/child pairs: n={len(pp)} child later {gt} earlier {lt} tie {len(pp) - gt - lt} | sign p {sign_p} | mean gap '
      f'{same_pass["mean_gap"]}', flush=True)
for r in recs: r.pop('qnums')
json.dump(dict(model=MODEL, layers=L, latents=K, n_items=len(te), n_right=n_right, summary=summ, same_pass=same_pass, pairs=pp, records=recs),
          open(OUT, 'w'))
print('DONE', f'{time.time() - t0:.0f}s')
