"""t56 / t57: inference-only validity tests of RuleChain and the hop count. Pre-registered in prereg/t_predictions.md (t56, t57).

  ../codi/.venv/bin/python scripts/necessity/validity.py <model> <device> <mode> <out json>
  model: gated | sim | startlast | simsf | relay | stab | startlast0 | stab0   (adapters.RuleChainCODI keys)
  mode   sets    (t56) accuracy by level and hop counts on the sets of validity_sets.py: orig, rw, hc, sv (L2-L7, K = 6) and l8
                 (K = 6, 7, 8 latent passes at inference); CODI-0 models: accuracy only
         wide    (t56 addendum) the wide set (values 0..115, readout extended to 0..115), K = 6
         robust  (t57) on this model's t31 pairs (QUAL=own, qualified): reproduction of the t31 count, the swapped pair (B as
                 recipient, A as donor), an unrelated donor (another item of the same level supplies M_0's latents), and on new
                 valid-split pairs: small (|delta| <= 2) vs large (|delta| >= 10) edits of the same item and position.
  env SMOKE=1: a handful of items / pairs. CAP (default 40) pairs per (level, edit position).
Every refresh run also stores the logits of the 100 number tokens, so margin-based counts are scored offline (validity_score.py).
Hop runs as in hops.py: M_0 = B's question with the donor's natural latents, M_n = B's question with the latents M_{n-1} produced.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adapters  # noqa: E402
import validity_sets as VS  # noqa: E402

sys.path.insert(0, str(adapters.REPO / 'scripts/codi_null_audit'))
import rc as RCmod  # noqa: E402

MODEL, DEV, MODE, OUT = sys.argv[1:5]
SMOKE = int(os.environ.get('SMOKE', '0')); CAP = int(os.environ.get('CAP', '40'))
ORDER = 'last' if MODEL in ('gated', 'sim', 'startlast', 'startlast0') else 'first'
NOLOOP = MODEL.endswith('0')
A = adapters.BASE / 'codi_audit_artifacts'
SETS = A / 't56/sets'

ad = adapters.RuleChainCODI(MODEL, DEV)
K0 = ad.n_latents
NUMIDS = torch.tensor([t for t, n in sorted(ad.num.items(), key=lambda kv: kv[1])], device=DEV)  # token id of number n at index n


# ---------------------------------------------------------------- runs
@torch.no_grad()
def finish_logits(ctx, past):
    """adapters.RuleChainCODI.finish, returning (answers, logits of the 100 number tokens)."""
    pi = ctx['pi']
    if ad.gate: pi.mask[:, :ctx['W']] = ctx['g'].answer()
    emb = ad.m.get_embd(ad.m.codi, ad.m.model_name)(ad.dec)[None].expand(ctx['B'], -1, -1)
    o = ad.body(inputs_embeds=emb, use_cache=True, past_key_values=past, **pi.next(emb.size(1)))
    lg = ad.m.codi.get_base_model().lm_head(o.last_hidden_state[:, -1, :])[:, :50257].float()
    ans = [ad.num.get(x, -999) for x in lg.argmax(-1).tolist()]
    return ans, lg[:, NUMIDS].cpu()


@torch.no_grad()
def run(rows, feed=None):
    """One forward with K = ad.n_latents passes; feed = list of K input vectors (None: natural). Returns (answers, logits, nat)."""
    ctx, past, z = ad.begin(rows); nat = [z]
    for s in range(ad.n_latents):
        zin = feed[s].to(device=z.device, dtype=z.dtype) if feed is not None else z
        past, z = ad.latent(ctx, past, zin, s); nat.append(z)
    a, lg = finish_logits(ctx, past)
    return a, lg, [x.detach() for x in nat]


@torch.no_grad()
def refresh(D, feed0):
    """M_0 .. M_K on B's rows D, starting from latents feed0. Returns answers [K+1][B] and number logits [K+1][B][100]."""
    K = ad.n_latents; feed = feed0[:K]; As, Ls = [], []
    for n in range(K + 1):
        a, lg, nat = run(D, feed); As.append(a); Ls.append(lg); feed = nat[:K]
    return As, Ls


def hop(As, b, tgt):
    fol = [As[n][b] == tgt for n in range(len(As))]
    return next((n for n in range(len(fol)) if all(fol[n:])), None), fol


def q(r): return dict(question=r['question'])


def accuracy(items):
    right = defaultdict(int); tot = defaultdict(int); ans = {}
    for i in range(0, len(items), ad.batch_size):
        rr = items[i:i + ad.batch_size]; a, _, _ = run([q(r) for r in rr])
        for r, x in zip(rr, a):
            ans[r['id']] = x; tot[r['level']] += 1; right[r['level']] += x == int(r['answer'])
    return {str(L): right[L] / tot[L] for L in sorted(tot)}, ans


# ---------------------------------------------------------------- structural edits
CONST = dict(P=('a',), T=('t', 'a', 'b'), D=('t', 'b'))


def edit(item, pos, rng, fam, x0r, size=None, vmax=99):
    """Change the start value (pos 0) or one constant of rule pos; the walk stays valid, values before pos are unchanged, the value
    after rule pos and the final answer change. fam None: each rule's own family (training or held-out ranges). size: None, 'small' (|delta| <= 2) or 'large' (|delta| >= 10)."""
    x0 = item['x0']; rules = [VS.rule_of(d) for d in item['rules']]; old = VS.walk(x0, rules, x0r, vmax)
    for _ in range(200):
        if pos == 0: lo, hi = x0r; cur = x0
        else:
            r = rules[pos - 1]; c = rng.choice(CONST[r.kind]); cur = getattr(r, c)
            f = fam if fam is not None else (VS.TRAIN if VS.rule_ok(r.kind, r.t, r.a, r.b, VS.TRAIN) else VS.HELD)
            lo, hi = f[r.kind][c]
        if size == 'small': v = cur + rng.choice((-2, -1, 1, 2))
        elif size == 'large': v = cur + rng.choice((-1, 1)) * rng.randint(10, 40)
        else: v = rng.randint(lo, hi)
        if v == cur or not lo <= v <= hi: continue
        if pos == 0: nx, nr = v, rules
        else:
            d = dict(t=r.t, a=r.a, b=r.b); d[c] = v
            if not VS.rule_ok(r.kind, d['t'], d['a'], d['b'], f): continue
            nx, nr = x0, rules[:pos - 1] + [VS.S.Rule(r.kind, **d)] + rules[pos:]
        new = VS.walk(nx, nr, x0r, vmax)
        if new is None or new[-1] == old[-1] or new[pos] == old[pos] or new[:pos] != old[:pos]: continue
        return dict(item, id=item['id'] + f'-e{pos}', x0=nx, rules=[VS.rule_json(x) for x in nr], values=new, answer=str(new[-1]),
                    question=VS.render(nx, nr, ORDER, item.get('reworded', False)), delta=v - cur, pos=pos)
    return None


def make_pairs(items, ans, fam, x0r, seed, size=None, cap=None, vmax=99):
    """QUAL=own: every recipient, one edit per (recipient, position); pairs whose own answers differ (both numbers) are run."""
    cap = cap or (4 if SMOKE else CAP); by = defaultdict(list)
    for r in items: by[r['level']].append(r)
    P = []
    for L in sorted(by):
        for pos in range(L + 1):
            rng = random.Random(f'{seed}-{L}-{pos}'); idx = by[L][:]; rng.shuffle(idx); n = 0
            for r in idx:
                e = edit(r, pos, rng, fam, x0r, size, vmax)
                if e is None: continue
                P.append(dict(level=L, pos=pos, edit='start' if pos == 0 else f'rule{pos}', s=L if pos == 0 else L - pos + 1,
                              delta=e['delta'], R=r, D=e))
                n += 1
                if n >= cap: break
    return P


def hop_pairs(P, tag, donor=None):
    """Hop count on pairs P. donor(p) -> rows whose natural latents seed M_0 (default: the recipient R)."""
    out = []; t0 = time.time(); B = ad.batch_size
    for i in range(0, len(P), B):
        pp = P[i:i + B]
        aR, _, natR = run([q(p['R']) for p in pp]); aD, lgD, _ = run([q(p['D']) for p in pp])
        if donor is not None:
            aX, _, natR = run([q(donor(p)) for p in pp])
        As, Ls = refresh([q(p['D']) for p in pp], natR)
        for b, p in enumerate(pp):
            rec = dict(level=p['level'], pos=p['pos'], s=p['s'], edit=p['edit'], delta=p.get('delta'), idR=p['R']['id'], idD=p['D']['id'],
                       aR=aR[b], aD=aD[b], goldR=int(p['R']['answer']), goldD=int(p['D']['answer']), A=[As[n][b] for n in range(len(As))])
            if donor is not None: rec['aX'] = aX[b]
            ref = rec['aX'] if donor is not None else aR[b]
            rec['qualified'] = aR[b] != -999 and aD[b] != -999 and aD[b] != ref and (donor is None or aR[b] != aD[b])
            rec['mk_matches_donor'] = As[-1][b] == aD[b]
            if rec['qualified']:
                rec['hops'], rec['follows'] = hop(As, b, aD[b])
                rec['margin'] = [float(Ls[n][b, aD[b]] - Ls[n][b, ref]) for n in range(len(As))]
            out.append(rec)
        if (i // B) % 10 == 0: print(time.strftime('%H:%M'), f'{tag} pair {i}/{len(P)} {time.time() - t0:.0f}s', flush=True)
        if DEV == 'mps': torch.mps.empty_cache()
    Q = [r for r in out if r['qualified']]
    print(f'{tag}: qualified {len(Q)}/{len(out)}; M_K = donor on {sum(r["mk_matches_donor"] for r in out)}/{len(out)}; '
          f'mean hops {np.mean([r["hops"] for r in Q]) if Q else float("nan"):.2f}', flush=True)
    return out


def load_set(sid):
    items = json.load(open(SETS / f'{sid}_{ORDER}.json'))
    if SMOKE: items = [r for r in items if r['level'] == 3][:24] + [r for r in items if r['level'] == 4][:24] + [r for r in items if r['level'] == 8][:16]
    return items


# ---------------------------------------------------------------- modes
R = dict(model=MODEL, order=ORDER, mode=MODE, cap=CAP)
if MODE == 'sets':
    fams = dict(orig=(VS.TRAIN, (1, 99)), rw=(VS.TRAIN, (1, 99)), hc=(VS.HELD, (1, 99)), hc1=(None, (1, 99)), sv=(VS.TRAIN, VS.SV_RANGE), l8=(VS.TRAIN, (1, 99)))
    for sid, (fam, x0r) in fams.items():
        items = load_set(sid); Ks = (6, 7, 8) if sid == 'l8' and not NOLOOP else (K0,)
        for K in Ks:
            ad.n_latents = K; tag = f'{sid}_K{K}'
            acc, ans = accuracy(items); print(time.strftime('%H:%M'), tag, 'accuracy', {k: round(v, 3) for k, v in acc.items()}, flush=True)
            R[tag] = dict(acc=acc)
            if NOLOOP: continue
            P = make_pairs(items, ans, fam, x0r, seed=f't56-{sid}')
            R[tag]['pairs'] = hop_pairs(P, tag)
        ad.n_latents = K0
elif MODE == 'robust':
    assert not NOLOOP
    t31 = [p for p in json.load(open(A / f't31/{MODEL}.json'))['pairs'] if p.get('qualified')]
    if SMOKE: t31 = t31[:24]
    cvt = lambda d, i: dict(id=i, question=d['question'], answer=str(d['answer']))
    P = [dict(level=p['level'], pos=0 if p['edit'] == 'start' else int(p['edit'][4:]), s=p['s'], edit=p['edit'], t31_hops=p['hops'],
              R=cvt(p['R'], f"{p['id']}-R-{p['edit']}"), D=cvt(p['D'], f"{p['id']}-D-{p['edit']}")) for p in t31]
    R['repro'] = hop_pairs(P, 'repro')
    for r, p in zip(R['repro'], P): r['t31_hops'] = p['t31_hops']
    R['swap'] = hop_pairs([dict(p, R=p['D'], D=p['R']) for p in P], 'swap')
    rng = random.Random('t57-donor'); byL = defaultdict(list)
    for p in P: byL[p['level']].append(p['R'])
    pick = {}
    for j, p in enumerate(P):
        cand = [x for x in byL[p['level']] if x['question'] != p['R']['question'] and x['question'] != p['D']['question']]
        pick[j] = rng.choice(cand)
    for j, p in enumerate(P): p['X'] = pick[j]
    R['unrelated'] = hop_pairs(P, 'unrelated', donor=lambda p: p['X'])
    # edit size on valid items (structural copies of the valid split, this model's order)
    data = 'synthetic_data_v2_l1to7w_startlast' if ORDER == 'last' else 'synthetic_data_v2_l1to7w'
    rows = [r for r in json.load(open(adapters.BASE / data / 'depth2/valid.json')) if int(r['level']) in (2, 3, 4, 5, 6)]
    if SMOKE: rows = [r for r in rows if int(r['level']) in (3, 4)][:40]
    items = []
    for r in rows:
        x0, rules = RCmod.parse(r['question'])
        rl = [VS.S.Rule(k, t=t, a=a, b=b) for k, t, a, b in rules]
        items.append(dict(id=r['id'], level=int(r['level']), x0=x0, rules=[VS.rule_json(x) for x in rl], answer=r['answer'],
                          question=VS.render(x0, rl, ORDER)))
    assert all(it['question'] == r['question'].strip() for it, r in zip(items, rows)), 'valid re-render differs'
    for size in ('small', 'large'):
        Ps = make_pairs(items, None, VS.TRAIN, (1, 99), seed='t57-size', size=size, cap=4 if SMOKE else 25)
        R[f'size_{size}'] = hop_pairs(Ps, f'size_{size}')
elif MODE == 'wide':
    # t56 addendum: every value in 0..115; the answer readout is extended to the 116 number tokens 0..115 (single GPT-2 tokens)
    W = VS.WIDE_MAX
    ad.num = {ad.tok.encode(f"The answer is: {n}")[-1]: n for n in range(W + 1)}; assert len(ad.num) == W + 1
    assert all(ad.tok.decode([t]).strip() == str(n) for t, n in ad.num.items())
    NUMIDS = torch.tensor([t for t, n in sorted(ad.num.items(), key=lambda kv: kv[1])], device=DEV)
    items = load_set('wide'); tag = f'wide_K{K0}'
    acc, ans = accuracy(items); print(time.strftime('%H:%M'), tag, 'accuracy', {k: round(v, 3) for k, v in acc.items()}, flush=True)
    R[tag] = dict(acc=acc, answers={r['id']: ans[r['id']] for r in items})
    if not NOLOOP:
        P = make_pairs(items, ans, VS.TRAIN, (1, W), seed='t56-wide', vmax=W)
        R[tag]['pairs'] = hop_pairs(P, tag)
else:
    raise SystemExit(f'unknown mode {MODE}')
json.dump(R, open(OUT, 'w'), default=str)
print('DONE', time.strftime('%H:%M'))
