"""t46 hop count on Liang & Pan's released polynomial-iteration CODI models (arXiv 2602.00449v2). Pre-registered in
scripts/codi_null_audit/prereg/t_predictions.md (t46). Sibling of hops.py (which runs at import time, so nat_run / hop_runs are copied
here unchanged, minus the FIRST option); the hop definition, M_0..M_K and QUAL=own are hops.py's.

  ../codi/.venv/bin/python scripts/necessity/hops_poly.py <poly_N_m> <device> <out json>
  model key poly_<N>_<m>: N in {3, 4, 8, 10} trained lengths, m in {50, 41}. Their repo: adapters.LP_REPO (../latent-cot-thinking).
  env GATE=1: gate 2 only (adapter natural run == their CODI.forward on 200 test items per length), then exit.
  env SMOKE=1: 2 items per length at k = 3, 4 (<= 16 pairs), write-back checks. CAP (default 80) items per (k, edit position).
  env WEIGHTS (default pytorch_model.bin).

Task: x_1..x_k uniform in 1..m-1, s_1 = x_1, s_t = s_{t-1} * x_t + 1 mod m, answer s_k. Items: their shipped test split, k = 2..N.
Input: questions are left-padded (PAD, masked) to the length-N width, as in their training and test.py collation (see adapters.PolyCODI;
the handoff's 'no padding' would put short questions at positions never seen in training). One k per batch.
Pairs (handoff 2026-11-15): B = a test item, A = B with one input x_i replaced. Every replacement v in 1..m-1, v != x_i, is run; the
share that changes the model's answer (and the gold answer) is recorded per (item, i); A is drawn uniformly from the replacements that
change the model's answer (none: the (item, i) has no pair). In hops.py's names R = A (latent source), D = B (the question run):
M_0 = B's question with A's natural latent inputs; M_n = B's question with the inputs M_{n-1} produced; hops = first n from which the
answer is the model's own answer to B (QUAL=own; right or wrong). Serial distance: edit to x_1 -> s = k - 1; x_i, i >= 2 -> s = k - i + 1.
"""
from __future__ import annotations

import ast
import json
import os
import random
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adapters  # noqa: E402

MODEL, DEV, OUT = sys.argv[1:4]
SMOKE = int(os.environ.get('SMOKE', '0')); CAP = int(os.environ.get('CAP', '80')); GATE = int(os.environ.get('GATE', '0'))
_, N, MOD = MODEL.split('_'); N, MOD = int(N), int(MOD)
torch.manual_seed(0)
ad = adapters.PolyCODI(N, MOD, DEV, weights=os.environ.get('WEIGHTS', 'pytorch_model.bin'))
K = ad.n_latents
DATA = adapters.LP_REPO / f'data/polynomial_data/seq_{N}/num_5000/mod_{MOD}/codi'


def load(k, split='test'):
    a = np.load(DATA / f'{split}_{k}.npy')
    return [dict(x=[int(v) for v in r[1:k + 1]], answer=int(r[-1])) for r in a]


def gold(x):
    s = x[0]
    for v in x[1:]: s = (s * v + 1) % MOD
    return s


def serial(k, i):
    return k - 1 if i == 1 else k - i + 1


# ---------------------------------------------------------------- hop-limited recomputation (hops.py, unchanged but FIRST)
@torch.no_grad()
def nat_run(ad, rows, feed=None):
    ctx, past, z = ad.begin(rows)
    nat = [z]
    for s in range(ad.n_latents):
        zin = feed[s].to(device=z.device, dtype=z.dtype) if feed is not None else z
        past, z = ad.latent(ctx, past, zin, s)
        nat.append(z)
    return ad.finish(ctx, past), [x.detach() for x in nat], None


@torch.no_grad()
def hop_runs(ad, R, D):
    aR, natR, _ = nat_run(ad, R)
    aD, natD, _ = nat_run(ad, D)
    feed = natR[:K]; A = []
    for n in range(K + 1):
        a, nat, _ = nat_run(ad, D, feed)
        A.append(list(a)); feed = nat[:K]
    return list(aR), list(aD), A


def answers(rows):
    out = []
    for j in range(0, len(rows), ad.batch_size):
        out += nat_run(ad, rows[j:j + ad.batch_size])[0]
    return out


# ---------------------------------------------------------------- gate 2: adapter == their CODI.forward (test.py inputs)
def their_forward_batches():
    """Their test.py path: Polynomial_CODI.load_data over lengths 1..N (test split), test.py's own preprocess_polynomial (taken from its
    source with ast; test.py runs at import), src/test_helpers' collator over the WHOLE set at once (so every question is left-padded to
    the length-N width, as in test.py), then sliced per length like test_model(). Returns {k: (batch slice of the first 200, gold)}."""
    for name in ('matplotlib', 'matplotlib.pyplot'):  # test_helpers imports pyplot and never uses it
        sys.modules.setdefault(name, types.ModuleType(name))
    from typing import Dict
    from src.test_helpers import DataCollatorForSupervisedDataset
    from src.data.data_processing import Polynomial_CODI
    tree = ast.parse((adapters.LP_REPO / 'test.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'preprocess_polynomial')
    ns = dict(np=np, torch=torch, Dict=Dict, POLYNOMIAL_TOKEN_DICT=ad.T)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'test.py', 'exec'), ns)
    raw = Polynomial_CODI(mod=MOD, save_dir=DATA, cot=True).load_data([i + 1 for i in range(N)], 'test')
    dd = ns['preprocess_polynomial'](raw)
    batch = DataCollatorForSupervisedDataset(tokenizer=None, data_name='polynomial')([{k: v[i] for k, v in dd.items()} for i in range(len(raw))])
    step = len(raw) // N
    return {k: ({kk: v[(k - 1) * step:(k - 1) * step + 200].to(DEV) for kk, v in batch.items()}, raw[(k - 1) * step:(k - 1) * step + 200])
            for k in range(1, N + 1)}


@torch.no_grad()
def their_forward_preds(batch):
    import contextlib, io
    ad.m.training_args.print_ref_model_stats = True
    with contextlib.redirect_stdout(io.StringIO()):
        out = ad.m.forward(**batch)
    pos = batch['model_answer_position'] - 1
    return out['logits'][torch.arange(pos.size(0)), pos].argmax(-1).tolist(), float(out['student_accuracy'])


def _states(x):
    s = [x[0]]
    for v in x[1:]: s.append((s[-1] * v + 1) % MOD)
    return s


if GATE:
    res = {}; ok_all = True
    TB = their_forward_batches()
    for k in range(1, N + 1):
        rows = load(k)[:200]
        assert all(gold(r['x']) == r['answer'] for r in rows), 'generator mismatch'
        batch, raw = TB[k]
        assert [list(map(int, r[1:k + 1])) for r in raw] == [r['x'] for r in rows], 'their loader and ours give different items'
        theirs, sacc = their_forward_preds(batch)
        mine = answers(rows); full = []
        for j in range(0, len(rows), ad.batch_size):
            nat_run(ad, rows[j:j + ad.batch_size]); full += ad.last_full
        eq = sum(a == b for a, b in zip(mine, theirs)); eqf = sum(a == b for a, b in zip(full, theirs))
        acc = float(np.mean([a == r['answer'] for a, r in zip(mine, rows)]))
        res[k] = dict(n=len(rows), adapter_eq_theirs=eq, fullvocab_eq_theirs=eqf, adapter_acc=acc, their_student_acc=sacc)
        ok_all &= eq == len(rows) and eqf == len(rows)
        print(f'k={k:2d} adapter==theirs {eq}/{len(rows)}  full-vocab {eqf}/{len(rows)}  acc adapter {acc:.3f} theirs {sacc:.3f}', flush=True)
    print('GATE 2', 'PASS' if ok_all else 'FAIL')
    json.dump(dict(model=MODEL, weights=ad.weights, gate2=res, passed=bool(ok_all)), open(OUT, 'w'), indent=1)
    sys.exit(0 if ok_all else 1)


# ---------------------------------------------------------------- accuracy by k (all shipped test items)
acc = {}
for k in range(1, N + 1):
    rows = load(k); a = answers(rows); acc[k] = float(np.mean([x == r['answer'] for x, r in zip(a, rows)]))
print('test acc by k', {k: round(v, 4) for k, v in acc.items()}, flush=True)

# ---------------------------------------------------------------- pairs: exhaustive single-input edits
LEN = (3, 4) if SMOKE else tuple(range(2, N + 1)); cap = 2 if SMOKE else CAP
P = []; edits = []; t0 = time.time()
for k in LEN:
    rows = load(k); order = list(range(len(rows))); random.Random(4600 + k).shuffle(order); items = order[:cap]
    base = answers([rows[j] for j in items])
    for i in range(1, k + 1):
        cand = []
        for jj, j in enumerate(items):
            for v in range(1, MOD):
                if v != rows[j]['x'][i - 1]:
                    x = rows[j]['x'][:]; x[i - 1] = v; cand.append((jj, j, v, x))
        a = answers([dict(x=c[3]) for c in cand])
        by = defaultdict(list)
        for c, ans in zip(cand, a): by[c[0]].append((c, ans))
        for jj, j in enumerate(items):
            r = rows[j]; lst = by[jj]
            ch = [(c, ans) for c, ans in lst if ans != base[jj]]
            gch = [c for c, _ in lst if gold(c[3]) != r['answer']]
            both = [c for c, ans in lst if ans != base[jj] and gold(c[3]) != r['answer']]
            edits.append(dict(k=k, i=i, s=serial(k, i), item=j, n_values=len(lst), n_model_change=len(ch), n_gold_change=len(gch),
                              n_both_change=len(both), B_right=base[jj] == r['answer']))
            if not ch: continue
            (c, aA) = random.Random(hash((k, i, j)) % 2 ** 31).choice(ch)
            P.append(dict(k=k, i=i, s=serial(k, i), item=j, v_old=r['x'][i - 1], v_new=c[2],
                          A=dict(x=c[3], answer=gold(c[3])), B=dict(x=r['x'], answer=r['answer'])))
    print(time.strftime('%H:%M'), f'k={k}: {sum(p["k"] == k for p in P)} pairs, {time.time() - t0:.0f}s', flush=True)

# ---------------------------------------------------------------- smoke checks
r0 = [P[0]['B']]; a0, nat0, _ = nat_run(ad, r0); a1, _, _ = nat_run(ad, r0, nat0[:K])
assert list(a0) == list(a1), 'own natural z write-back changed the answer'
print('own write-back check passes', flush=True)
if SMOKE: P = P[:16]

# ---------------------------------------------------------------- main loop (one k per batch: no padding)
res = []
for k in sorted({p['k'] for p in P}):
    Pk = [p for p in P if p['k'] == k]
    for j in range(0, len(Pk), ad.batch_size):
        pp = Pk[j:j + ad.batch_size]
        aA, aB, A = hop_runs(ad, [p['A'] for p in pp], [p['B'] for p in pp])  # R = A (latents), D = B (question)
        for b, p in enumerate(pp):
            rec = dict(p, aA=aA[b], aB=aB[b], M=[A[n][b] for n in range(K + 1)])
            rec['qualified'] = aA[b] != aB[b]
            rec['A_right'] = aA[b] == p['A']['answer']; rec['B_right'] = aB[b] == p['B']['answer']
            rec['gold_differs'] = p['A']['answer'] != p['B']['answer']
            rec['mk_matches_B'] = A[K][b] == aB[b]
            fol = [A[n][b] == aB[b] for n in range(K + 1)]
            rec['follows'] = fol; rec['keeps'] = [A[n][b] == aA[b] for n in range(K + 1)]
            rec['neither'] = [not (f or kk) for f, kk in zip(fol, rec['keeps'])]
            rec['hops'] = next((n for n in range(K + 1) if all(fol[n:])), None)
            res.append(rec)
    print(time.strftime('%H:%M'), f'k={k} done, {len(res)} pairs', flush=True)

Q = [r for r in res if r['qualified']]
bad = [r for r in res if not r['mk_matches_B']]
print(f'{MODEL}: qualified {len(Q)} / {len(res)}; M_K = B run on {len(res) - len(bad)} of {len(res)}', flush=True)
assert not bad, 'M_K differs from B own run'


# ---------------------------------------------------------------- summary
def boot_mean(v, n=2000, seed=0):
    v = np.asarray(v, float)
    if len(v) == 0: return None
    rng = np.random.default_rng(seed); bs = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def fe_slope(xs):
    """Within-item OLS slope of hops on s (item fixed effects: demean both within (k, item))."""
    g = defaultdict(list)
    for x in xs: g[(x['k'], x['item'])].append((x['s'], x['hops']))
    num = den = 0.0
    for v in g.values():
        if len(v) < 2: continue
        s = np.array([a for a, _ in v], float); h = np.array([b for _, b in v], float)
        num += ((s - s.mean()) * (h - h.mean())).sum(); den += ((s - s.mean()) ** 2).sum()
    return num / den if den else None


def boot_slope(xs, n=2000, seed=0):
    g = defaultdict(list)
    for x in xs: g[(x['k'], x['item'])].append(x)
    keys = list(g); rng = np.random.default_rng(seed); est = fe_slope(xs)
    if est is None: return None
    bs = [fe_slope([y for kk in rng.choice(len(keys), len(keys)) for y in g[keys[kk]]]) for _ in range(n)]
    bs = [b for b in bs if b is not None]
    return [float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def curve(xs):
    if not xs: return None
    return dict(n=len(xs), hops=boot_mean([x['hops'] for x in xs]), hist=[sum(x['hops'] == n for x in xs) for n in range(K + 1)],
                F=[float(np.mean([x['follows'][n] for x in xs])) for n in range(K + 1)],
                keepA=[float(np.mean([x['keeps'][n] for x in xs])) for n in range(K + 1)],
                neither=[float(np.mean([x['neither'][n] for x in xs])) for n in range(K + 1)])


def split(xs):
    return dict(all=curve(xs), slope=boot_slope(xs),
                by_s={s: curve([x for x in xs if x['s'] == s]) for s in sorted({x['s'] for x in xs})},
                by_k={k: curve([x for x in xs if x['k'] == k]) for k in sorted({x['k'] for x in xs})},
                by_k_i={f'k{k}_i{i}': curve([x for x in xs if x['k'] == k and x['i'] == i])
                        for k, i in sorted({(x['k'], x['i']) for x in xs})})


def edit_share(es):
    if not es: return None
    return dict(n_item_pos=len(es), share_values_change_model=float(sum(e['n_model_change'] for e in es) / sum(e['n_values'] for e in es)),
                share_values_change_gold=float(sum(e['n_gold_change'] for e in es) / sum(e['n_values'] for e in es)),
                share_item_pos_never_change=float(np.mean([e['n_model_change'] == 0 for e in es])))


summ = dict(main=split(Q), gold_differs=split([x for x in Q if x['gold_differs']]),
            both_right=split([x for x in Q if x['A_right'] and x['B_right']]),
            gold_differs_by_s_hops={s: boot_mean([x['hops'] for x in Q if x['gold_differs'] and x['s'] == s]) for s in sorted({x['s'] for x in Q})},
            edits=dict(by_k_i={f'k{k}_i{i}': edit_share([e for e in edits if e['k'] == k and e['i'] == i])
                               for k, i in sorted({(e['k'], e['i']) for e in edits})},
                       by_s={s: edit_share([e for e in edits if e['s'] == s]) for s in sorted({e['s'] for e in edits})},
                       by_k={k: edit_share([e for e in edits if e['k'] == k]) for k in sorted({e['k'] for e in edits})},
                       early_k4plus=edit_share([e for e in edits if e['k'] >= 4 and e['i'] <= e['k'] - 2])))
f = lambda c: '-' if c is None else (f"n={c['n']:4d} hops {c['hops'][0]:.2f} [{c['hops'][1]:.2f},{c['hops'][2]:.2f}] F "
                                     + ' '.join(f'{v:.2f}' for v in c['F']) + f" | neither M1 {c['neither'][1]:.2f}")
print('all  ', f(summ['main']['all']), '| within-item slope', summ['main']['slope'])
for s, c in summ['main']['by_s'].items(): print(f's={s:<3}', f(c))
for k, c in summ['main']['by_k'].items(): print(f'k={k:<3}', f(c), f"| acc {acc[k]:.3f}")
for s, e in summ['edits']['by_s'].items(): print(f'edits s={s}', e)
print('early edits k>=4', summ['edits']['early_k4plus'])
json.dump(dict(model=MODEL, weights=ad.weights, qual='own', cap=cap, test_acc=acc, n_pairs=len(res), pairs=res, edits=edits,
               summary=summ, mk_mismatch=len(bad)), open(OUT, 'w'), default=str)
print('DONE', f'{time.time() - t0:.0f}s')
