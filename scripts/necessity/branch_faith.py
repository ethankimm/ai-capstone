"""t45: branch faithfulness on wrong answers (RuleChain). Pre-registered in scripts/codi_null_audit/prereg/t_predictions.md (t45).

  ../codi/.venv/bin/python scripts/necessity/branch_faith.py <gated|sim|simsf|startlast|relay|stab|simcoconut_rc> <device> <out json>

Branch of rule k read from latent z_j by a binary ridge probe (natural latents, train L2-L7 pooled). On wrong items, the chain is replayed
from the question's start value with every branch forced to the one read; faithful = the replay gives the model's own answer.
Branch-explainable errors = some branch pattern gives the model's answer. IN / OUT: each rule's latent chosen among z_j <= C / z_j > C.
"""
import itertools
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
sys.path.insert(0, str(adapters.REPO / 'scripts/codi_null_audit'))
import rc as RCmod  # noqa: E402

MODEL, DEV, OUT = sys.argv[1:4]; SMOKE = int(os.environ.get('SMOKE', '0'))
BASE = adapters.BASE; ART = BASE / 'codi_audit_artifacts'; LV = [2, 3, 4, 5, 6, 7]
if MODEL == 'simcoconut_rc':
    ad = adapters.GridCoconut('rulechain', DEV, 'coconut', max_new=16, path=str(adapters.REPO / 'rulechain-checkpoints/pilot_k8_simcot_c1m6/checkpoint_26'))
    DD = BASE / 'synthetic_data_v2/depth2'; NFIT = 300; HOPF = [ART / 't37/hops.json']
else:
    ad = adapters.RuleChainCODI(MODEL, DEV); NFIT = 800
    DD = BASE / ('synthetic_data_v2_l1to7w_startlast' if MODEL in ('gated', 'sim', 'startlast') else 'synthetic_data_v2_l1to7w') / 'depth2'
    HOPF = [ART / f't31/{MODEL}.json', ART / f't41/{MODEL}.json']
if SMOKE: NFIT = 40


@torch.no_grad()
def nat_run(rows):
    ctx, past, z = ad.begin(rows); nat = [z]
    for s in range(ad.n_latents):
        past, z = ad.latent(ctx, past, z, s); nat.append(z)
    return list(ad.finish(ctx, past)), torch.stack(nat, 1).float().cpu().numpy()


def run_all(rows, tag):
    ans, Z = [None] * len(rows), [None] * len(rows); B = ad.batch_size; t0 = time.time()
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]['question']))
    for i in range(0, len(order), B):
        ii = order[i:i + B]; a, z = nat_run([rows[k] for k in ii])
        for n, k in enumerate(ii): ans[k] = a[n]; Z[k] = z[n]
        if (i // B) % (500 if B == 1 else 25) == 0: print(time.strftime('%H:%M'), f'{MODEL} {tag} {i}/{len(rows)} ({time.time() - t0:.0f}s)', flush=True)
        if DEV == 'mps' and B == 1 and i % 50 == 0: torch.mps.empty_cache()
    return ans, np.stack(Z).astype(np.float64)


def bacc(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    if len(y) < 10 or y.all() or (~y).all(): return float('nan')
    return float(((p & y).sum() / y.sum() + (~p & ~y).sum() / (~y).sum()) / 2)


def fit(X, y):
    """binary ridge (+-1), penalty on a held-out 15%."""
    mu, sd = X.mean(0), X.std(0) + 1e-6; Xs = (X - mu) / sd; c = int(.85 * len(X)); Y = np.where(y, 1.0, -1.0)
    def sol(A, t):
        w, V = np.linalg.eigh(A.T @ A); m = t.mean(); XY = V.T @ (A.T @ (t - m))
        return lambda lam: (V @ (XY / (np.maximum(w, 0) + lam)), m)
    f = sol(Xs[:c], Y[:c]); lams = (1e1, 1e2, 1e3, 1e4)
    sc = [bacc(y[c:], (Xs[c:] @ f(l)[0] + f(l)[1]) > 0) for l in lams]
    W, b = sol(Xs, Y)(lams[int(np.nanargmax(sc))])
    return lambda Xn: ((Xn - mu) / sd) @ W + b > 0


walk = lambda q: RCmod.walk(*RCmod.parse(q))
def forced(x0, rules, brs):
    s = x0
    for r, b in zip(rules, brs): s = RCmod.apply(r, s, bool(b))
    return s

tr_all = json.load(open(DD / 'train.json')); r_ = random.Random(45); tr = []
for L in LV: tr += r_.sample([r for r in tr_all if int(r['level']) == L], NFIT)
ev = [r for s in ('valid.json', 'test.json') for r in json.load(open(DD / s)) if int(r['level']) in LV]
if SMOKE: ev = random.Random(0).sample(ev, 200)
print(time.strftime('%H:%M'), f'{MODEL}: fit {len(tr)}, eval {len(ev)}', flush=True)
_, Zt = run_all(tr, 'fit'); ans, Ze = run_all(ev, 'eval'); K1 = Zt.shape[1]
Bt = [walk(r['question'])[1] for r in tr]; Be = [walk(r['question'])[1] for r in ev]
Lt = np.array([int(r['level']) for r in tr]); Le = np.array([int(r['level']) for r in ev])
READ = np.full((len(ev), K1, 8), -1, int)  # branch of rule k (1-based) read at z_(j+1)
for j in range(K1):
    for k in range(1, 8):
        m = Lt >= k
        if m.sum() < 50: continue
        pr = fit(Zt[m, j], np.array([Bt[i][k - 1] for i in np.where(m)[0]], bool)); me = Le >= k
        READ[me, j, k] = pr(Ze[me, j]).astype(int)
gold = np.array([int(r['answer']) for r in ev]); A = np.array([a if isinstance(a, (int, np.integer)) else -999 for a in ans])
right = A == gold; wrong = (~right) & (A >= 0) & (A < 100)
# best latent per (level, rule) on correct items; read accuracy table
acc_tab, best = {}, {}
for L in LV:
    ii = np.where(right & (Le == L))[0]
    for k in range(1, L + 1):
        sc = [bacc([Be[i][k - 1] for i in ii], READ[ii, j, k]) for j in range(K1)]
        acc_tab[f'L{L}_r{k}'] = sc; best[(L, k)] = sc
pool = {}  # fallback: rule k's latent chosen on correct items of every level >= k (levels with too few correct items)
for k in range(1, 8):
    ii = np.where(right & (Le >= k))[0]
    pool[k] = [bacc([Be[i][k - 1] for i in ii], READ[ii, j, k]) for j in range(K1)]
lev = defaultdict(list)
for f in HOPF:
    for p in json.load(open(f))['pairs']:
        if p.get('qualified') and p.get('hops') is not None and p['edit'] == 'start': lev[int(p['level'])].append(p['hops'])
CL = {L: float(np.mean(v)) for L, v in lev.items()}
Cof = lambda L: int(round(CL.get(L, CL[max(CL)])))
def pick(L, k, allowed):
    for tab in (best[(L, k)], pool[k]):
        sc = [(s if j in allowed and np.isfinite(s) else -1) for j, s in enumerate(tab)]
        if max(sc) >= 0: return int(np.argmax(sc))
    return None
items = []
for i in np.where(wrong)[0]:
    L = int(Le[i]); x0, rules = RCmod.parse(ev[i]['question']); C = Cof(L)
    pats = list(itertools.product([0, 1], repeat=L)); hits = [forced(x0, rules, p) == A[i] for p in pats]
    rec = dict(i=int(i), L=L, C=C, model_ans=int(A[i]), gold=int(gold[i]), explainable=bool(any(hits)), chance=float(np.mean(hits)),
               gold_check=forced(x0, rules, Be[i]) == gold[i])
    for nm, allowed in (('ALL', set(range(K1))), ('IN', set(range(min(C, K1)))), ('OUT', set(range(min(C, K1), K1)))):
        js = [pick(L, k, allowed) for k in range(1, L + 1)]
        if any(j is None for j in js): rec[nm] = None; continue
        brs = [READ[i, j, k] for k, j in zip(range(1, L + 1), js)]
        rec[nm] = bool(forced(x0, rules, brs) == A[i]); rec[nm + '_flips'] = int(sum(b != g for b, g in zip(brs, Be[i])))
    items.append(rec)


def boot(x):
    x = np.asarray(x, float)
    if len(x) < 5: return dict(rate=float(x.mean()) if len(x) else None, n=int(len(x)), ci=None)
    r = np.random.default_rng(0); bs = [x[r.integers(0, len(x), len(x))].mean() for _ in range(1000)]
    return dict(rate=float(x.mean()), n=int(len(x)), ci=[float(v) for v in np.percentile(bs, [2.5, 97.5])])


E = [x for x in items if x['explainable']]
S = dict(n_wrong=len(items), n_explainable=len(E), gold_check=float(np.mean([x['gold_check'] for x in items])) if items else None,
         chance=float(np.mean([x['chance'] for x in E])) if E else None)
for nm in ('ALL', 'IN', 'OUT'):
    xs = [x for x in E if x.get(nm) is not None]
    S[nm] = boot([x[nm] for x in xs]); S[nm + '_chance'] = float(np.mean([x['chance'] for x in xs])) if xs else None
    S[nm + '_flips_mean'] = float(np.mean([x[nm + '_flips'] for x in xs])) if xs else None
S['by_level'] = {L: dict(n=sum(x['L'] == L for x in E), ALL=boot([x['ALL'] for x in E if x['L'] == L and x.get('ALL') is not None])['rate'])
                 for L in LV}
S['read_on_right'] = {k: max([v for v in s if np.isfinite(v)] or [np.nan]) for k, s in acc_tab.items()}
json.dump(dict(model=MODEL, acc={L: float(right[Le == L].mean()) for L in LV}, CL=CL, summary=S, items=items, read_table=acc_tab),
          open(OUT, 'w'), default=float)
print(f"{MODEL}: wrong {len(items)}, branch-explainable {len(E)}, gold-branch check {S['gold_check']}")
for nm in ('ALL', 'IN', 'OUT'):
    print(f"  {nm}: reproduces {S[nm]} chance {S[nm + '_chance']} mean flips {S[nm + '_flips_mean']}")
print('  by level', S['by_level'])
print('DONE')
