"""t28 / t28b: does released SIM-CODI compute GSM8K steps in parallel from the question, or chained through the loop?
Pre-registered in scripts/codi_null_audit/prereg/t_predictions.md (t28) before running.

  ../codi/.venv/bin/python scripts/necessity/sim_parallel.py <device> <out json>     (SMOKE=1: 6 pairs)

Pairs: SIM's qualified t26 / t27a GSM8K pairs (A = recipient, B = donor). B's gold step values are recomputed from A's solution with
the one changed number. Changed steps of B are split into
  direct    : uses the edited number and no earlier changed step (computable from B's question alone, one pass)
  dependent : uses an earlier changed step's result (a chain needs that result to cross the loop first)
Latents decoded with SIM's own step decoder (training-only GPT-2 decoder, greedy '<<a op b=c>>'), at every latent input z_0..z_5,
no assumption about which latent holds which step: a step counts as read if ANY latent decodes to B's value for it.
Conditions: refresh 0 (A's latents copied: floor), refresh 1 (<= 1 loop crossing), refresh 2 (<= 2), B's own run (ceiling).
Chain: dependent steps read at refresh 1 well below B's own run, rising at refresh 2. Parallel: refresh 1 already close to B's own run.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adapters  # noqa: E402

DEV, OUT = sys.argv[1:3]; SMOKE = int(os.environ.get('SMOKE', '0'))
BASE = adapters.BASE; T26 = BASE / 'codi_audit_artifacts/t26'
SIM_DIR = BASE / 'grid_checkpoints/simcot_gpt2_codi'
LATENT = os.environ.get('LATENT', 'sim')  # t28c: LATENT=codi decodes released CODI's latents with SIM's decoder (negative control)
ad = adapters.GSM8KCODI(DEV, ckpt=str(SIM_DIR) if LATENT == 'sim' else None); K = ad.n_latents; tok = ad.tok

import transformers  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
sd = {k[len('decoder.'):]: v for k, v in load_file(str(SIM_DIR / 'model.safetensors')).items() if k.startswith('decoder.')}
decoder = transformers.GPT2LMHeadModel.from_pretrained('gpt2'); missing = decoder.load_state_dict(sd, strict=False)
assert not missing.unexpected_keys and all('attn.bias' in k or 'masked_bias' in k for k in missing.missing_keys), missing
decoder = decoder.to(DEV).float().eval()

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.data.minimal_pairs import _reexecute  # noqa: E402
from latentreasoning.eval.counterfactual import parse_steps  # noqa: E402
from latentreasoning.eval.metrics import is_correct  # noqa: E402
EX = {e.question: e for e in load_gsm8k_aug(split='test', n=None, seed=None)}
NUM = re.compile(r'\d+(?:\.\d+)?'); STEP = re.compile(r'<<([^=<>]*)=([^<>]*)>>')


@torch.no_grad()
def dec_step(z):
    """sim_gsm8k.py's dec_step: greedy decode of one latent [1, d] into step text."""
    x = z.reshape(1, 1, -1); toks = []; past = None
    for _ in range(40):
        o = decoder(inputs_embeds=x, past_key_values=past, use_cache=True)
        past = o.past_key_values; nid = int(o.logits[0, -1].argmax())
        if nid == tok.eos_token_id: break
        toks.append(nid); x = ad.emb(torch.tensor([[nid]], device=DEV))
    return tok.decode(toks)


def result(s):
    m = STEP.search(s or '')
    return m.group(2).strip() if m else None


@torch.no_grad()
def nat_run(rows, feed=None):
    ctx, past, z = ad.begin(rows); nat = [z]
    for s in range(K):
        zin = feed[s].to(device=z.device, dtype=z.dtype) if feed is not None else z
        past, z = ad.latent(ctx, past, zin, s); nat.append(z)
    return ad.finish(ctx, past), nat


def twin_steps(qa, qb):
    """A's gold steps, B's recomputed step values, and per changed step whether it is direct or dependent."""
    e = EX[qa]; st = parse_steps(e.rationale)
    na, nb = NUM.findall(qa), NUM.findall(qb)
    diff = [(a, b) for a, b in zip(na, nb) if a != b]
    if len(na) != len(nb) or len(diff) != 1: return None
    old, new = diff[0]; nr = _reexecute(st, {old: new})
    if nr is None: return None
    out = []; depth = {}
    for k, (s, v) in enumerate(zip(st, nr)):
        if is_correct(str(v), s['val']): continue
        ops = NUM.findall(s['expr'].replace(',', ''))
        used = [depth[j] for j in depth if st[j]['val'] in ops]
        depth[k] = 1 + max(used) if used else 1  # loop crossings a chain needs before this value can sit in a latent
        out.append(dict(k=k, a=s['val'], b=str(v), kind='dependent' if used else 'direct', depth=depth[k]))
    return out


pairs = [p for f in (f'gsm8k_{LATENT}', f'long_gsm8k_{LATENT}') for p in json.load(open(T26 / f'{f}.json'))['pairs'] if p['qualified']]
if SMOKE: pairs = pairs[:6]
res = []; t0 = time.time()
for i, p in enumerate(pairs):
    tw = twin_steps(p['R']['question'], p['D']['question'])
    if not tw: continue
    A, B = [p['R']], [p['D']]
    _, natA = nat_run(A); aB, natB = nat_run(B)
    feed = natA[:K]; lat = {'refresh0': feed}
    for n in range(1, K + 1):
        _, nat = nat_run(B, feed); feed = nat[:K]; lat[f'refresh{n}'] = feed
    lat['own'] = natB[:K]
    dec = {c: [dec_step(z) for z in zs] for c, zs in lat.items()}
    rec = dict(id=p['id'], s=p['s'], hops=p['hops'], steps=tw, decoded=dec,
               refreshK_is_own=all(torch.allclose(a, b, atol=1e-5) for a, b in zip(lat[f'refresh{K}'], lat['own'])))
    for c in lat:
        R = [result(x) for x in dec[c]]
        for t in tw: t[f'read_{c}'] = any(r is not None and is_correct(r, t['b']) for r in R)
    res.append(rec)
    if i % 25 == 0: print(time.strftime('%H:%M'), f'pair {i}/{len(pairs)} {time.time() - t0:.0f}s', flush=True)
    if DEV == 'mps': torch.mps.empty_cache()

C = [f'refresh{n}' for n in range(K + 1)] + ['own']
summ = {}; print('refresh K = own run on', sum(r['refreshK_is_own'] for r in res), 'of', len(res), flush=True)
for d in sorted({t['depth'] for r in res for t in r['steps']}):
    xs = [t for r in res for t in r['steps'] if t['depth'] == d]; ro = [t for t in xs if t['read_own']]
    summ[d] = dict(n=len(xs), n_own=len(ro), own_rate=len(ro) / len(xs),
                   all={c: sum(t[f'read_{c}'] for t in xs) / len(xs) for c in C},
                   given_own={c: sum(t[f'read_{c}'] for t in ro) / max(len(ro), 1) for c in C})
    print(f'depth {d}: n={len(xs)} readable in own run {len(ro) / len(xs):.2f} | given own-readable: '
          + ' '.join(f"{n} {summ[d]['given_own'][f'refresh{n}']:.2f}" for n in range(K + 1)), flush=True)
json.dump(dict(pairs=res, summary=summ), open(OUT, 'w'))
print('DONE', f'{time.time() - t0:.0f}s')
