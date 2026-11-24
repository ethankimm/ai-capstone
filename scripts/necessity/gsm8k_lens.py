"""t30: GSM8K readability by loop crossings for released CODI / Coconut / SIM-CODI, read by vocabulary projection of each latent
(top-10 tokens, Dilgren & Wiegreffe's readout). Pre-registered in prereg/t_predictions.md (t30).

  ../codi/.venv/bin/python scripts/necessity/gsm8k_lens.py <gsm8k_codi | gsm8k_coconut | gsm8k_sim | gsm8k_llama (t48)> <device> <out json>

Same pairs, step depths, refreshes and controls as t28b (sim_parallel.py), with the readout replaced: a value of B is read if its number
token (with or without a leading space, single-token values only) is among the top-10 tokens of ANY latent input z_0..z_5, projected with
the model's output head (GPT-2 ties input and output embeddings, so CODI's projected latents are scored the same way).

t48: env READ=z (default; t30's readout of the fed-back latent inputs) or READ=h (the logit lens on the output hidden state of each pass
before CODI's projection, i.e. HF hidden_states[-1], after the final norm; z_s = prj(h_{s-1}), so the same positions and refreshes are
read). The Llama release needs READ=h: its projected latents project to unrelated tokens. CODI models only.
"""
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adapters  # noqa: E402

MODEL, DEV, OUT = sys.argv[1:4]; SMOKE = int(os.environ.get('SMOKE', '0')); TOPK = 10
BASE = adapters.BASE; T26 = BASE / 'codi_audit_artifacts/t26'
if MODEL == 'gsm8k_coconut':
    ad = adapters.GridCoconut('gsm8k', DEV, 'coconut'); head = ad.head; tok = ad.tok
elif MODEL in adapters.LLAMA_GSM8K:  # t48 / t49 / t50: released Llama-3.2 CODI 1B / SIM-CODI 1B / SIM-CODI 3B (READ=h, see the docstring)
    ad = adapters.LLAMA_GSM8K[MODEL][1](DEV)
    head = ad.m.codi.get_base_model().lm_head; tok = ad.tok
else:
    ad = adapters.GSM8KCODI(DEV, ckpt=str(BASE / 'grid_checkpoints/simcot_gpt2_codi') if MODEL == 'gsm8k_sim' else None)
    head = ad.m.codi.get_base_model().lm_head; tok = ad.tok
K = ad.n_latents
READ = os.environ.get('READ', 'h' if MODEL in adapters.LLAMA_GSM8K else 'z')
assert READ in ('z', 'h') and (READ == 'z' or isinstance(ad, adapters.GSM8KCODI)), READ
print(f'{MODEL}: READ={READ}', flush=True)
from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.data.minimal_pairs import _reexecute  # noqa: E402
from latentreasoning.eval.counterfactual import parse_steps  # noqa: E402
from latentreasoning.eval.metrics import is_correct  # noqa: E402
EX = {e.question: e for e in load_gsm8k_aug(split='test', n=None, seed=None)}
NUM = re.compile(r'\d+(?:\.\d+)?')


def tok_ids(v):
    """single-token ids for the number v (with and without a leading space); None if neither form is one token."""
    try:
        f = float(str(v).replace(',', ''))
    except ValueError:
        return None
    if not f.is_integer(): return None
    s = str(int(f)); ids = set()
    for form in (s, ' ' + s):
        t = tok.encode(form, add_special_tokens=False)  # Llama prepends BOS otherwise; GPT-2 adds none, so unchanged there
        if len(t) == 1: ids.add(t[0])
    return ids or None


def twin_steps(qa, qb):
    e = EX[qa]; st = parse_steps(e.rationale)
    na, nb = NUM.findall(qa), NUM.findall(qb); diff = [(a, b) for a, b in zip(na, nb) if a != b]
    if len(na) != len(nb) or len(diff) != 1: return None
    nr = _reexecute(st, {diff[0][0]: diff[0][1]})
    if nr is None: return None
    out = []; depth = {}
    for k, (s, v) in enumerate(zip(st, nr)):
        if is_correct(str(v), s['val']): continue
        ops = NUM.findall(s['expr'].replace(',', '')); used = [depth[j] for j in depth if st[j]['val'] in ops]
        depth[k] = 1 + max(used) if used else 1
        out.append(dict(k=k, a=s['val'], b=str(v), depth=depth[k]))
    return out


@torch.no_grad()
def _begin_h(rows):
    """GSM8KCODI.begin, also returning the pre-projection hidden state h (same computation, same z)."""
    (r,) = rows
    ids, attn = ad.enc(ad.m, ad.tok, r['question'].strip().replace('  ', ' '), ad.dev)
    o = ad.m.codi(input_ids=ids, use_cache=True, output_hidden_states=True, past_key_values=None, attention_mask=attn)
    h = o.hidden_states[-1][:, -1, :]
    return dict(Q=ids.size(1)), o.past_key_values, ad.m.prj(h.unsqueeze(1))[:, 0], h


@torch.no_grad()
def _latent_h(ctx, past, z, s):
    """GSM8KCODI.latent, also returning h."""
    m = torch.ones((1, ctx['Q'] + s + 1), dtype=torch.long, device=ad.dev)
    o = ad.m.codi(inputs_embeds=z[:, None], use_cache=True, output_hidden_states=True, past_key_values=past, attention_mask=m)
    h = o.hidden_states[-1][:, -1, :]
    return o.past_key_values, ad.m.prj(h.unsqueeze(1))[:, 0], h


@torch.no_grad()
def nat_run(rows, feed=None):
    """Returns (answers, nat, rd): nat = latent inputs z_0..z_K (fed forward); rd = what is read, z itself (READ=z) or the h each z was
    projected from (READ=h)."""
    if READ == 'z':
        ctx, past, z = ad.begin(rows); nat = [z]
        for s in range(K):
            zin = feed[s].to(device=z.device, dtype=z.dtype) if feed is not None else z
            past, z = ad.latent(ctx, past, zin, s); nat.append(z)
        return ad.finish(ctx, past), nat, nat
    ctx, past, z, h = _begin_h(rows); nat, rd = [z], [h]
    for s in range(K):
        zin = feed[s].to(device=z.device, dtype=z.dtype) if feed is not None else z
        past, z, h = _latent_h(ctx, past, zin, s); nat.append(z); rd.append(h)
    return ad.finish(ctx, past), nat, rd


@torch.no_grad()
def top_tokens(zs):
    """set of top-k token ids over all latents in zs (list of [1, d])."""
    out = set()
    for z in zs:
        out |= set(head(z.reshape(1, -1)).topk(TOPK, dim=-1).indices[0].tolist())
    return out


if MODEL in adapters.LLAMA_GSM8K:  # t47 / t49 / t50 pairs are QUAL=own; keep those right on both, as the t26 (QUAL=gold) pairs are
    T47 = BASE / 'codi_audit_artifacts' / adapters.LLAMA_GSM8K[MODEL][0]
    pairs = [p for f in (MODEL, 'long_' + MODEL) for p in json.load(open(T47 / f'{f}.json'))['pairs']
             if p['qualified'] and p.get('R_right') and p.get('D_right')]
else:
    pairs = [p for f in (MODEL, 'long_' + MODEL) for p in json.load(open(T26 / f'{f}.json'))['pairs'] if p['qualified']]
if SMOKE: pairs = pairs[:6]
items = []; t0 = time.time(); n_skip = 0
for i, p in enumerate(pairs):
    tw = twin_steps(p['R']['question'], p['D']['question'])
    if not tw: continue
    _, natA, rdA = nat_run([p['R']]); feed = natA[:K]; tops = [top_tokens(rdA[:K])]
    for n in range(1, K + 1):
        _, nat, rd = nat_run([p['D']], feed); feed = nat[:K]; tops.append(top_tokens(rd[:K]))
    for t in tw:
        ib, ia = tok_ids(t['b']), tok_ids(t['a'])
        if ib is None: n_skip += 1; continue
        items.append(dict(pair=i, depth=t['depth'], b=t['b'], a=t['a'], B=[bool(ib & T) for T in tops],
                          A=[bool(ia & T) for T in tops] if ia else [False] * (K + 1), tops=[sorted(T) for T in tops]))
    if i % 25 == 0: print(time.strftime('%H:%M'), f'pair {i}/{len(pairs)} {time.time() - t0:.0f}s', flush=True)
    if DEV == 'mps': torch.mps.empty_cache()
grp = defaultdict(list)
for it in items: grp[it['depth']].append(it)
for g in grp.values():  # chance: the next item's B value at the same depth, scored on this item's tokens
    for x, y in zip(g, g[1:] + g[:1]):
        ids = tok_ids(y['b']); x['C'] = [bool(ids & set(T)) and y['b'] != x['b'] for T in x['tops']]
summ = {}
for d in sorted(grp):
    xs = grp[d]; own = [it for it in xs if it['B'][K]]
    if len(xs) < 5: continue
    summ[d] = dict(n=len(xs), own_rate=len(own) / len(xs), n_own=len(own),
                   B=[sum(it['B'][n] for it in own) / max(len(own), 1) for n in range(K + 1)],
                   A=[sum(it['A'][n] for it in own) / max(len(own), 1) for n in range(K + 1)],
                   chance=[sum(it['C'][n] for it in xs) / len(xs) for n in range(K + 1)])
    s = summ[d]; f = lambda v: ' '.join(f'{x:.2f}' for x in v)
    print(f'depth {d}: n={s["n"]} read in own run {s["own_rate"]:.2f} | given own-read, B by refresh {f(s["B"])} | stale A {f(s["A"])} '
          f'| chance {f(s["chance"])}', flush=True)
print(f'skipped {n_skip} multi-token values')
for it in items: it.pop('tops')
json.dump(dict(model=MODEL, read=READ, summary=summ, items=items, skipped=n_skip), open(OUT, 'w'))
print('DONE', f'{time.time() - t0:.0f}s')
