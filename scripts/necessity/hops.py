"""t26 hop count: how many recurrent edges does the information that changes the answer cross? Pre-registered in
scripts/codi_null_audit/prereg/t_predictions.md (t26).

  ../codi/.venv/bin/python scripts/necessity/hops.py <model> <device> <out json>
  model: gated | sim | simsf | startlast | relay | stab   (RuleChain valid L2-L6)
         gsm8k_codi | gsm8k_sim | gsm8k_coconut          (GSM8K test, the 1471 twin candidates of t22)
  env SMOKE=1: a few pairs plus the write-back checks. CAP (default 80) recipients per (level, edit) cell on RuleChain.

Recurrent edges: question pass -> z_0, pass s -> z_{s+1}. Attention reads other positions at the same layer and adds no layers, so the
only way past one pass of depth is across these edges. Hop-limited recomputation on a minimal pair (recipient R, donor D):
  M_0: D's question, every latent input z_s set to R's natural z_s        -> donor info reaches the answer with 0 hops
  M_n: D's question, z_s set to the natural z_s produced inside M_{n-1}   -> <= n hops
  M_K (K = 6) is D's own run (checked exactly). Only z vectors move between runs, never keys/values, so token lengths need not match.
F_n = share of qualified pairs (R and D answered right) whose M_n answer is D's. hops* = smallest n with D's answer from n on.
"""
from __future__ import annotations

import gc
import json
import os
import random
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

MODEL, DEV, OUT = sys.argv[1:4]
SMOKE = int(os.environ.get('SMOKE', '0')); CAP = int(os.environ.get('CAP', '80'))
RC = MODEL in ('gated', 'sim', 'simsf', 'startlast', 'relay', 'stab')
BASE = adapters.BASE


# ---------------------------------------------------------------- hop-limited recomputation
@torch.no_grad()
def nat_run(ad, rows, feed=None):
    """Latent loop; feed = list of K input vectors [B, d] (None: natural). Returns (answers, nat) with nat[s] = the input pass s would get
    naturally (nat[0] = question pass output; nat[s+1] = pass s output)."""
    ctx, past, z = ad.begin(rows)
    nat = [z]
    for s in range(ad.n_latents):
        zin = feed[s].to(device=z.device, dtype=z.dtype) if feed is not None else z
        past, z = ad.latent(ctx, past, zin, s)
        nat.append(z)
    return ad.finish(ctx, past), [x.detach() for x in nat]


@torch.no_grad()
def hop_runs(ad, R, D):
    aR, natR = nat_run(ad, R)
    aD, natD = nat_run(ad, D)
    K = ad.n_latents; feed = natR[:K]; A = []
    for n in range(K + 1):
        a, nat = nat_run(ad, D, feed)
        A.append(list(a)); feed = nat[:K]
    return list(aR), list(aD), A, (natR, natD)


# ---------------------------------------------------------------- RuleChain pairs
sys.path.insert(0, str(adapters.REPO / 'scripts/codi_null_audit'))
NUM = re.compile(r'\d+')


def rc_edit(q, pos, rng, parse, walk):
    """Edit the start value (pos 0) or one number of rule pos (1..L), same digit count; v_pos must change, all values in 0..99 and the
    answer must change. Returns (new question, new final) or None."""
    x0, rules = parse(q); old = walk(x0, rules)[0]
    if pos == 0:
        m = re.search(r'Start with (\d+)\.', q); spans = [m.span(1)]
    else:
        m = re.search(rf'Rule {pos}: (.*?)(?= Rule {pos + 1}: | Start with | What is)', q)
        if m is None: return None
        spans = [(m.start(1) + x.start(), m.start(1) + x.end()) for x in NUM.finditer(m.group(1))]
    for _ in range(60):
        a, b = rng.choice(spans); s = q[a:b]; nd = len(s)
        v = rng.randint(10 ** (nd - 1) if nd > 1 else 1, 10 ** nd - 1)
        if v == int(s): continue
        nq = q[:a] + str(v) + q[b:]
        try:
            nx, nr = parse(nq)
        except Exception:
            continue
        new = walk(nx, nr)[0]
        if min(new) < 0 or max(new) > 99 or new[-1] == old[-1] or new[pos] == old[pos] or new[:pos] != old[:pos]: continue
        return nq, new[-1]
    return None


def rc_pairs(ad, rows, parse, walk):
    """Recipients answered right, then one edit per (recipient, position); CAP recipients per (level, position)."""
    ans = [None] * len(rows); order = sorted(range(len(rows)), key=lambda i: len(rows[i]['question']))
    for i in range(0, len(order), ad.batch_size):
        ii = order[i:i + ad.batch_size]; a, _ = nat_run(ad, [rows[k] for k in ii])
        for k, x in zip(ii, a): ans[k] = x
    by = defaultdict(list)
    for i, r in enumerate(rows):
        if ans[i] == int(r['answer']): by[int(r['level'])].append(i)
    acc = {L: len(by[L]) / sum(int(r['level']) == L for r in rows) for L in sorted(by)}
    print('clean acc by level', {L: round(a, 3) for L, a in acc.items()}, flush=True)
    P = []
    for L in sorted(by):
        for pos in range(L + 1):
            rng = random.Random(1000 * L + pos); idx = by[L][:]; rng.shuffle(idx); n = 0
            for i in idx:
                e = rc_edit(rows[i]['question'], pos, rng, parse, walk)
                if e is None: continue
                P.append(dict(level=L, edit='start' if pos == 0 else f'rule{pos}', s=L if pos == 0 else L - pos + 1, id=rows[i]['id'],
                              R=dict(question=rows[i]['question'], answer=int(rows[i]['answer'])), D=dict(question=e[0], answer=e[1])))
                n += 1
                if n >= (4 if SMOKE else CAP): break
    return P, acc


# ---------------------------------------------------------------- models
if RC:
    import rc as RCmod  # noqa: E402  (parse / walk)
    ad = adapters.RuleChainCODI(MODEL, DEV)
    data = 'synthetic_data_v2_l1to7w_startlast' if MODEL in ('gated', 'sim', 'startlast') else 'synthetic_data_v2_l1to7w'
    rows = [r for r in json.load(open(BASE / data / 'depth2/valid.json')) if 2 <= int(r['level']) <= 6]
    if SMOKE: rows = [r for r in rows if int(r['level']) in (3, 4)][:120]
    same = lambda a, b: a == b
    P, clean_acc = rc_pairs(ad, rows, RCmod.parse, RCmod.walk)
else:
    from latentreasoning.eval.metrics import is_correct  # noqa: E402
    if MODEL == 'gsm8k_coconut':
        ad = adapters.GridCoconut('gsm8k', DEV, 'coconut')
    else:
        ad = adapters.GSM8KCODI(DEV, ckpt=str(BASE / 'grid_checkpoints/simcot_gpt2_codi') if MODEL == 'gsm8k_sim' else None)
    cand = json.load(open(os.environ.get('CANDS', BASE / 'codi_audit_artifacts/t22/gsm8k_twin_candidates.json')))  # t27a: CANDS = long twins
    if SMOKE: cand = cand[:30]
    same = lambda a, b: a is not None and a != '' and is_correct(str(a), str(b))
    P = [dict(level=c['chain_len'], edit=f"step{c['step']}", s=c['chain_len'] - c['step'], id=c['recipient_idx'],
              R=dict(question=c['recipient_question'], answer=c['recipient_gold']), D=dict(question=c['twin_question'], answer=c['twin_answer']))
         for c in cand]
    clean_acc = None
print(time.strftime('%H:%M'), f'{MODEL}: {len(P)} candidate pairs', flush=True)

# ---------------------------------------------------------------- smoke checks
if SMOKE:
    r0 = [P[0]['R']]; a0, nat0 = nat_run(ad, r0); a1, _ = nat_run(ad, r0, nat0[:ad.n_latents])
    assert list(a0) == list(a1), 'own natural z write-back changed the answer'
    print('  own write-back check passes', flush=True)

# ---------------------------------------------------------------- main loop
res = []; t0 = time.time(); B = ad.batch_size; memo = {}
K = ad.n_latents
for i in range(0, len(P), B):
    pp = P[i:i + B]
    if B == 1:
        p = pp[0]; key = p['R']['question']
        if key not in memo: memo[key] = list(nat_run(ad, [p['R']])[0])
        if not same(memo[key][0], p['R']['answer']):
            res.append(dict(p, qualified=False, aR=memo[key][0])); continue
        dk = p['D']['question']
        if dk not in memo: memo[dk] = list(nat_run(ad, [p['D']])[0])
        if not same(memo[dk][0], p['D']['answer']):
            res.append(dict(p, qualified=False, aR=memo[key][0], aD=memo[dk][0])); continue
    aR, aD, A, _ = hop_runs(ad, [p['R'] for p in pp], [p['D'] for p in pp])
    for b, p in enumerate(pp):
        rec = dict(p, aR=aR[b], aD=aD[b], A=[A[n][b] for n in range(K + 1)])
        rec['qualified'] = bool(same(aR[b], p['R']['answer']) and same(aD[b], p['D']['answer']))
        rec['mk_matches_donor'] = A[K][b] == aD[b]
        fol = [bool(same(A[n][b], p['D']['answer'])) for n in range(K + 1)]
        rec['follows'] = fol; rec['keeps'] = [bool(same(A[n][b], p['R']['answer'])) for n in range(K + 1)]
        rec['hops'] = next((n for n in range(K + 1) if all(fol[n:])), None)
        res.append(rec)
    if (i // B) % (25 if B == 1 else 5) == 0:
        nq = sum(r['qualified'] for r in res)
        print(time.strftime('%H:%M'), f'pair {i}/{len(P)} qualified {nq} {time.time() - t0:.0f}s', flush=True)
    if len(memo) > 600: memo.clear()
    gc.collect()
    if DEV == 'mps': torch.mps.empty_cache()
    if SMOKE and sum(r['qualified'] for r in res) >= 8: break

# ---------------------------------------------------------------- summary
Q = [r for r in res if r['qualified']]
bad = [r for r in res if 'mk_matches_donor' in r and not r['mk_matches_donor']]
ran = sum('mk_matches_donor' in r for r in res)
print(f'{MODEL}: qualified {len(Q)} / {len(res)}; M_K = donor run on {ran - len(bad)} of {ran} pairs run', flush=True)


def curve(xs):
    if not xs: return None
    return dict(n=len(xs), F=[float(np.mean([x['follows'][n] for x in xs])) for n in range(K + 1)],
                Rk=[float(np.mean([x['keeps'][n] for x in xs])) for n in range(K + 1)],
                hops_mean=float(np.mean([x['hops'] for x in xs])), hops_hist=[sum(x['hops'] == n for x in xs) for n in range(K + 1)])


summ = dict(all=curve(Q), by_s={s: curve([x for x in Q if x['s'] == s]) for s in sorted({x['s'] for x in Q})},
            by_level={L: curve([x for x in Q if x['level'] == L]) for L in sorted({x['level'] for x in Q})},
            by_level_edit={f"L{L}_{e}": curve([x for x in Q if x['level'] == L and x['edit'] == e])
                           for L, e in sorted({(x['level'], x['edit']) for x in Q})})
f = lambda c: '-' if c is None else f"n={c['n']:4d} F " + ' '.join(f'{v:.2f}' for v in c['F']) + f" | hops {c['hops_mean']:.2f}"
print('all  ', f(summ['all']))
for s, c in summ['by_s'].items(): print(f's={s:<3}', f(c))
for L, c in summ['by_level'].items(): print(f'L{L:<3} ', f(c))
json.dump(dict(model=MODEL, clean_acc=clean_acc, n_candidates=len(P), pairs=res, summary=summ,
               mk_mismatch=len(bad)), open(OUT, 'w'), default=str)
print('DONE', f'{time.time() - t0:.0f}s')
