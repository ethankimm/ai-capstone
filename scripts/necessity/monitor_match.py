"""t44b (matched fit; head shared with t44 monitor_select.py): does a crossing-aware monitor (read only the latents an item's answer crosses) beat reading all latents?
Pre-registered in scripts/codi_null_audit/prereg/t_predictions.md (t44).

  ../codi/.venv/bin/python scripts/necessity/monitor_select.py <model> <device> <out json>
  model: gated | sim | simsf | startlast | relay | stab | simcoconut_rc            (RuleChain, pooled over L2-L7)
         gsm8k_codi | gsm8k_sim | gsm8k_coconut | gsm8k_simcoconut                (GSM8K)
  env SMOKE=1: tiny sets (code check).

Latents z_1 .. z_(K+1) (z_1 = question-pass output, z_(j+1) = output of pass j). Dual ridge probes on standardized features, penalty per
target on a held-out 15% of the fit set, fit on train items for every prefix z_1..z_c (c = 1..K+1) and suffix z_(c+1)..z_(K+1).
Monitors per test item (C = item crossings): ALL = full prefix; CROSS = prefix C; SHUF = prefix of a random other test item's C; OUT =
suffix after C; DEPTH (GSM8K) = prefix from the item's serial depth via the model's mean hops at that depth (t42). C = 0 reads z_1.
T1 gold steps (+ answer features), T2 own answer without answer features (wrong items), T3 answer wrong (+ answer features).
"""
import json
import math
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

MODEL, DEV, OUT = sys.argv[1:4]; SMOKE = int(os.environ.get('SMOKE', '0'))
BASE = adapters.BASE; ART = BASE / 'codi_audit_artifacts'
RC = not MODEL.startswith('gsm8k')
LV = [2, 3, 4, 5, 6, 7]
rng = np.random.default_rng(44)


@torch.no_grad()
def nat_run(ad, rows):
    ctx, past, z = ad.begin(rows); nat = [z]
    for s in range(ad.n_latents):
        past, z = ad.latent(ctx, past, z, s); nat.append(z)
    return list(ad.finish(ctx, past)), torch.stack(nat, 1).float().cpu().numpy()


def run_all(ad, rows, tag):
    ans, Z = [None] * len(rows), [None] * len(rows); B = ad.batch_size; t0 = time.time()
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]['question']))
    for i in range(0, len(order), B):
        ii = order[i:i + B]; a, z = nat_run(ad, [rows[k] for k in ii])
        for n, k in enumerate(ii): ans[k] = a[n]; Z[k] = z[n]
        if (i // B) % (250 if B == 1 else 25) == 0: print(time.strftime('%H:%M'), f'{MODEL} {tag} {i}/{len(rows)} ({time.time() - t0:.0f}s)', flush=True)
        if DEV == 'mps' and B == 1 and i % 50 == 0: torch.mps.empty_cache()
    return ans, np.stack(Z).astype(np.float64)


def dual_ridge(Xa, Ya, kinds, Xb):
    """Ya [n, t] (binary: +-1 / 0 missing) or class ids (kind 'cls', -1 missing). Returns predictions on Xb per target."""
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6; A = (Xa - mu) / sd / math.sqrt(Xa.shape[1]); B = (Xb - mu) / sd / math.sqrt(Xa.shape[1])
    lams = (1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2); out = []
    for t, kind in enumerate(kinds):
        y = Ya[:, t]; ok = (y != 0) if kind == 'bin' else (y >= 0); X = A[ok]; yy = y[ok]
        if kind == 'cls':
            cl = np.unique(yy); Y = (yy[:, None] == cl[None]).astype(float)
        else:
            Y = yy[:, None].astype(float)
        c = int(.85 * len(X)); best, bl = -1, lams[0]
        w, V = np.linalg.eigh(X[:c] @ X[:c].T); ym = Y[:c].mean(0); VY = V.T @ (Y[:c] - ym); KX = X[c:] @ X[:c].T
        for lam in lams:
            P = KX @ (V @ (VY / (np.maximum(w, 0) + lam)[:, None])) + ym
            sc = bacc(yy[c:], P[:, 0]) if kind == 'bin' else float(np.mean(cl[P.argmax(1)] == yy[c:]))
            if sc > best: best, bl = sc, lam
        ym = Y.mean(0); al = np.linalg.solve(X @ X.T + bl * np.eye(len(X)), Y - ym); P = B @ (X.T @ al) + ym
        out.append(P[:, 0] if kind == 'bin' else cl[P.argmax(1)])
    return out


def bacc(y, p):
    y, p = np.asarray(y) > 0, np.asarray(p) > 0
    if len(y) < 10 or y.all() or (~y).all(): return float('nan')
    return float(((p & y).sum() / y.sum() + (~p & ~y).sum() / (~y).sum()) / 2)


# ---------------------------------------------------------------- data, labels, crossings
if RC:
    sys.path.insert(0, str(adapters.REPO / 'scripts/codi_null_audit'))
    import rc as RCmod  # noqa: E402
    if MODEL == 'simcoconut_rc':
        ad = adapters.GridCoconut('rulechain', DEV, 'coconut', max_new=16,
                                  path=str(adapters.REPO / 'rulechain-checkpoints/pilot_k8_simcot_c1m6/checkpoint_26'))
        DD = BASE / 'synthetic_data_v2/depth2'; NFIT = 250; HOPF = [ART / 't37/hops.json']; EV = 'test.json'
    else:
        ad = adapters.RuleChainCODI(MODEL, DEV); NFIT = 500; EV = 'valid.json'
        DD = BASE / ('synthetic_data_v2_l1to7w_startlast' if MODEL in ('gated', 'sim', 'startlast') else 'synthetic_data_v2_l1to7w') / 'depth2'
        HOPF = [ART / f't31/{MODEL}.json', ART / f't41/{MODEL}.json']
    if SMOKE: NFIT = 40
    hs, hmax = {}, defaultdict(int)
    for f in HOPF:
        for p in json.load(open(f))['pairs']:
            if p.get('qualified') and p.get('hops') is not None:
                if p['edit'] == 'start': hs[p['id']] = int(p['hops'])
                hmax[p['id']] = max(hmax[p['id']], int(p['hops']))
    CI = {i: hs.get(i, hmax[i]) for i in hmax}
    tr_all = json.load(open(DD / 'train.json')); r_ = random.Random(44); tr = []
    for L in LV: tr += r_.sample([r for r in tr_all if int(r['level']) == L], NFIT)
    te = [r for s in ('valid.json', 'test.json') for r in json.load(open(DD / s)) if int(r['level']) in LV]  # t44b: all eval items
    if SMOKE: te = random.Random(0).sample(te, 150)
    TG = [f'branch_r{k}' for k in range(2, 7)]
    def labels(rows):
        Y = np.zeros((len(rows), len(TG)))
        for i, r in enumerate(rows):
            brs = RCmod.walk(*RCmod.parse(r['question']))[1]
            for j, k in enumerate(range(2, 7)):
                if k <= len(brs): Y[i, j] = 1 if brs[k - 1] else -1
        return Y
    def afeat(a):
        x = np.zeros(100); x[a] = 1 if isinstance(a, (int, np.integer)) and 0 <= a < 100 else 0; return x
    own_cls = lambda a: int(a) if isinstance(a, (int, np.integer)) and 0 <= a < 100 else -1
    is_right = lambda a, r: a == int(r['answer'])
    depth_c = None
else:
    from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
    from latentreasoning.eval.metrics import is_correct  # noqa: E402
    # data first: the adapters switch the HF hub to offline mode
    tr = [dict(question=e.question, e=e) for e in load_gsm8k_aug(split='train', n=150 if SMOKE else 2000, seed=39)]
    allte = load_gsm8k_aug(split='test', n=None, seed=None)
    if MODEL in ('gsm8k_coconut', 'gsm8k_simcoconut'):
        ad = adapters.GridCoconut('gsm8k', DEV, 'coconut', **({} if MODEL == 'gsm8k_coconut' else
                                  dict(path=str(BASE / 'grid_checkpoints/simcot_gpt2_coconut/checkpoint_28'))))
        if MODEL == 'gsm8k_simcoconut': ad.n_latents = 10
    else:
        ad = adapters.GSM8KCODI(DEV, ckpt=str(BASE / 'grid_checkpoints/simcot_gpt2_codi') if MODEL == 'gsm8k_sim' else None)
    m = MODEL.split('_', 1)[1]
    HOPF = [ART / 't36/gsm8k_simcoconut.json', ART / 't36/long_gsm8k_simcoconut.json'] if m == 'simcoconut' else \
        [ART / f't31/{MODEL}.json', ART / f't31/long_{MODEL}.json']
    hop = defaultdict(list)
    for f in HOPF:
        for p in json.load(open(f))['pairs']:
            if p.get('qualified') and p.get('hops') is not None: hop[' '.join(p['R']['question'].split())].append(p['hops'])
    qk = lambda q: ' '.join(q.split())
    te = [dict(question=e.question, e=e, id=e.idx) for e in allte]  # t44b: all test items
    if SMOKE: te = random.Random(0).sample(te, 80)
    CI = {r['id']: max(1, int(round(float(np.mean(hop[qk(r['question'])]))))) for r in te if hop.get(qk(r['question']))}
    sys.argv = ['x', '/dev/null']; ns = {}
    exec(open(HERE / 'gsm8k_depth.py').read().split("qtext = lambda")[0], ns)  # depth_of (t42, strict)
    D42 = json.load(open(ART / 't42/gsm8k_depth_strict.json'))['models'][m]['hops_by_depth']
    dbin = lambda d: None if d is None else ('1' if d <= 1 else '2' if d == 2 else '3' if d == 3 else '4+')
    def depth_c(r):
        d = ns['depth_of'](r['e'].question, r['e'].rationale, r['e'].answer)['strict']['depth']; b = dbin(d)
        return max(1, int(round(D42[b]['mean']))) if b and D42[b]['mean'] is not None else None
    import re
    STEP = re.compile(r'<<([^<>=]*)=([^<>]*)>>')
    def num(a):
        try: return float(str(a).replace(',', ''))
        except (TypeError, ValueError): return None
    def steps(e): return [(bool(re.search(r'[*/]', ex)), num(res)) for ex, res in STEP.findall(e.rationale)]
    MED = [float(np.median([s[k][1] for r in tr if len(s := steps(r['e'])) > k and s[k][1] is not None])) for k in range(3)]
    TG = ['op_1', 'op_2', 'op_3', 'big_1', 'big_2', 'big_3']
    def labels(rows):
        Y = np.zeros((len(rows), 6))
        for i, r in enumerate(rows):
            st = steps(r['e'])
            for k in range(3):
                if k < len(st):
                    Y[i, k] = 1 if st[k][0] else -1
                    if st[k][1] is not None: Y[i, 3 + k] = 1 if st[k][1] >= MED[k] else -1
        return Y
    def afeat(a):
        v = num(a); x = np.zeros(16)
        if v is None or not math.isfinite(v): x[15] = 1; return x
        iv = int(round(abs(v))) if abs(v) < 1e12 else 0
        x[0] = math.log10(1 + abs(v)); x[1] = float(v < 0); x[2] = float(v == round(v)); x[3] = iv % 2; x[4 + iv % 10] = 1
        x[14] = len(str(iv)); return x
    QB = None
    def own_cls(a):
        v = num(a)
        return -1 if v is None or QB is None else int(np.searchsorted(QB, math.log10(1 + abs(v))))
    is_right = lambda a, r: a not in (None, '') and is_correct(str(a), r['e'].answer)

print(time.strftime('%H:%M'), f'{MODEL}: fit {len(tr)}, test with crossings {len(te)}', flush=True)
ansA, ZA = run_all(ad, tr, 'fit'); ansB, ZB = run_all(ad, te, 'test')
K1 = ZA.shape[1]
if not RC:
    QB = np.quantile([math.log10(1 + abs(num(a))) for a in ansA if num(a) is not None], [.2, .4, .6, .8])
Ya, Yb = labels(tr), labels(te)
AA, AB = np.stack([afeat(a) for a in ansA]), np.stack([afeat(a) for a in ansB])
wa = np.array([-1 if is_right(a, r) else 1 for a, r in zip(ansA, tr)]); wb = np.array([-1 if is_right(a, r) else 1 for a, r in zip(ansB, te)])
oa = np.array([own_cls(a) for a in ansA]); ob = np.array([own_cls(a) for a in ansB])
ga = np.array([own_cls(int(r['answer']) if RC else r['e'].answer) for r in tr]); gb = np.array([own_cls(int(r['answer']) if RC else r['e'].answer) for r in te])

# ---------------------------------------------------------------- t44b: group crossings, matched fits
if RC:
    lev = defaultdict(list)
    for f in HOPF:
        for p in json.load(open(f))['pairs']:
            if p.get('qualified') and p.get('hops') is not None and p['edit'] == 'start': lev[int(p['level'])].append(p['hops'])
    CL = {L: float(np.mean(v)) for L, v in lev.items()}
    grp = lambda r: int(np.clip(round(CL.get(int(r['level']), CL[max(CL)])), 1, K1))
else:
    grp = lambda r: int(np.clip(depth_c(r) or 1, 1, K1))
ctr = np.array([grp(r) for r in tr]); cte = np.array([grp(r) for r in te])
cme = np.array([int(np.clip(CI[r['id']], 1, K1)) if r['id'] in CI else 0 for r in te])
NT = len(TG); kinds = ['bin'] * (NT + 1)
def fitpred(idx, rows_tr, rows_te):
    Xa = ZA[rows_tr][:, idx].reshape(len(rows_tr), -1); Xb = ZB[rows_te][:, idx].reshape(len(rows_te), -1)
    p1 = dual_ridge(np.concatenate([Xa, AA[rows_tr]], 1), np.concatenate([Ya[rows_tr], wa[rows_tr, None]], 1), kinds,
                    np.concatenate([Xb, AB[rows_te]], 1))
    p2 = dual_ridge(Xa, oa[rows_tr, None], ['cls'], Xb)
    return np.stack(p1[:-1], 1), p1[-1], p2[0]
P = {m: dict(T1=np.zeros((len(te), NT)), T3=np.zeros(len(te)), T2=np.full(len(te), -2)) for m in ('CROSS_m', 'ALL_m', 'ALL_p', 'CROSS_meas')}
def put(m, rows, out):
    P[m]['T1'][rows], P[m]['T3'][rows], P[m]['T2'][rows] = out
t0 = time.time(); allz = list(range(K1)); every = np.arange(len(tr))
put('ALL_p', np.arange(len(te)), fitpred(allz, every, np.arange(len(te))))
groups = sorted(set(ctr))
near = lambda c: min(groups, key=lambda g: (abs(g - c), -g))
for g in sorted(set(cte)):
    S = np.where(ctr == near(g))[0]; T = np.where(cte == g)[0]
    put('CROSS_m', T, fitpred(list(range(g)), S, T)); put('ALL_m', T, fitpred(allz, S, T))
    print(time.strftime('%H:%M'), f'group c={g}: fit {len(S)} test {len(T)} ({time.time() - t0:.0f}s)', flush=True)
for c in sorted(set(cme) - {0}):
    T = np.where(cme == c)[0]; S = np.where(ctr == near(c))[0]
    put('CROSS_meas', T, fitpred(list(range(c)), S, T))
def score(task, m, items):
    if task == 'T1':
        Pm = P[m]['T1']; return float(np.nanmean([bacc(Yb[items, t][Yb[items, t] != 0], Pm[items, t][Yb[items, t] != 0]) for t in range(NT)]))
    if task == 'T3': return bacc(wb[items], P[m]['T3'][items])
    w = items[(wb[items] > 0) & (ob[items] >= 0)]
    return float(np.mean(P[m]['T2'][w] == ob[w])) if len(w) >= 10 else float('nan')
allidx = np.arange(len(te)); meas = np.where(cme > 0)[0]
RES = dict(model=MODEL, n_fit=len(tr), n_test=len(te), K1=K1, acc_test=float(np.mean(wb < 0)), n_wrong=int((wb > 0).sum()),
           groups_train={int(g): int((ctr == g).sum()) for g in groups}, groups_test={int(g): int((cte == g).sum()) for g in set(cte)},
           n_measured=int(len(meas)), scores={}, diffs={}, by_group={})
for task in ('T1', 'T2', 'T3'):
    RES['scores'][task] = {m: score(task, m, allidx) for m in ('CROSS_m', 'ALL_m', 'ALL_p')}
    RES['scores'][task]['on_measured'] = {m: score(task, m, meas) for m in ('CROSS_meas', 'CROSS_m', 'ALL_m')} if len(meas) else None
    for a, b, items in (('CROSS_m', 'ALL_m', allidx), ('ALL_m', 'ALL_p', allidx), ('CROSS_m', 'ALL_p', allidx), ('CROSS_meas', 'ALL_m', meas)):
        if not len(items): continue
        bs = []
        for _ in range(500):
            ii = items[rng.integers(0, len(items), len(items))]; bs.append(score(task, a, ii) - score(task, b, ii))
        RES['diffs'][f'{task}:{a}-{b}'] = dict(diff=score(task, a, items) - score(task, b, items), ci=[float(v) for v in np.nanpercentile(bs, [2.5, 97.5])])
    RES['by_group'][task] = {int(g): {m: score(task, m, np.where(cte == g)[0]) for m in ('CROSS_m', 'ALL_m', 'ALL_p')} for g in sorted(set(cte))}
print(f"{MODEL}: test {len(te)} (acc {RES['acc_test']:.3f}, wrong {RES['n_wrong']}), groups train {RES['groups_train']} test {RES['groups_test']}")
for task in ('T1', 'T2', 'T3'):
    s = RES['scores'][task]; print(f'  {task}: ' + ' '.join(f'{m} {v:.3f}' for m, v in s.items() if m != 'on_measured')
                                  + (f" | measured-C items: " + ' '.join(f'{m} {v:.3f}' for m, v in s['on_measured'].items()) if s['on_measured'] else ''))
    print('     by group', {g: {m: round(v, 3) for m, v in d.items()} for g, d in RES['by_group'][task].items()})
for k, v in RES['diffs'].items(): print(f"  {k}: {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
json.dump(RES, open(OUT, 'w'), default=float)
print('DONE')
