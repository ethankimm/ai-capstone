"""t29: RuleChain readability by loop crossings (probe version of t28b). Pre-registered in prereg/t_predictions.md (t29).

  ../codi/.venv/bin/python scripts/necessity/rc_readability.py <model> <device> <out json>
  model: gated | sim | simsf | startlast | relay | stab.  SMOKE=1: tiny.

Probes: for each level L (2-6), latent input site j (0-5) and intermediate value v_k (1 <= k <= L-1), a multinomial logistic regression
on the natural latent inputs of 800 train items per level, validated on 200 more. A site is legible for (L, k) if its probe validates
at >= .5. No assumption about which site holds which step: a value is read if ANY legible site's probe predicts it.
Pairs: the qualified t26 pairs (A = recipient, B = donor). For every intermediate value of B that differs from A's, d = loop crossings a
chain needs before it can sit in a latent (edit at rule i, value after rule k: d = k - max(i, 1) + 1). Refresh n latents as in hops.py
(refresh 0 = A's latents, refresh 6 = B's own). Scored: B's value read, A's (stale) value read, chance (another pair's B value at the
same (L, k)).
"""
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adapters  # noqa: E402

MODEL, DEV, OUT = sys.argv[1:4]; SMOKE = int(os.environ.get('SMOKE', '0'))
BASE = adapters.BASE; T26 = BASE / 'codi_audit_artifacts/t26'
sys.path.insert(0, str(adapters.REPO / 'scripts/codi_null_audit'))
import rc as RC  # noqa: E402
ad = adapters.RuleChainCODI(MODEL, DEV); K = ad.n_latents
data = 'synthetic_data_v2_l1to7w_startlast' if MODEL in ('gated', 'sim', 'startlast') else 'synthetic_data_v2_l1to7w'
LEVELS = (3, 4) if SMOKE else (2, 3, 4, 5, 6)
NTR, NVA = (120, 60) if SMOKE else (800, 200)


@torch.no_grad()
def nat_run(rows, feed=None):
    ctx, past, z = ad.begin(rows); nat = [z]
    for s in range(K):
        zin = feed[s].to(device=z.device, dtype=z.dtype) if feed is not None else z
        past, z = ad.latent(ctx, past, zin, s); nat.append(z)
    return ad.finish(ctx, past), nat


def latents(rows):
    """natural latent inputs [N, K, d] (float32 numpy)."""
    out = []
    for i in range(0, len(rows), ad.batch_size):
        _, nat = nat_run(rows[i:i + ad.batch_size]); out.append(torch.stack(nat[:K], 1).float().cpu().numpy())
        if DEV == 'mps': torch.mps.empty_cache()
    return np.concatenate(out)


vals = lambda q: RC.walk(*RC.parse(q))[0]  # v_0 (start) .. v_L

# ---------------------------------------------------------------- probes
t0 = time.time()
tr = json.load(open(BASE / data / 'depth2/train.json')); rng = random.Random(0); by = defaultdict(list)
for r in tr:
    if int(r['level']) in LEVELS: by[int(r['level'])].append(r)
probe, legible = {}, {}
for L in LEVELS:
    rows = rng.sample(by[L], NTR + NVA); Z = latents(rows); V = np.array([vals(r['question']) for r in rows])
    for k in range(1, L):
        for j in range(K):
            clf = LogisticRegression(max_iter=400, C=0.5)
            clf.fit(Z[:NTR, j], V[:NTR, k]); acc = float((clf.predict(Z[NTR:, j]) == V[NTR:, k]).mean())
            legible[(L, k, j)] = acc
            if acc >= .5: probe[(L, k, j)] = clf
    print(time.strftime('%H:%M'), f'L{L} probes: ' + ' '.join(
        f"v{k}:" + ''.join('#' if legible[(L, k, j)] >= .5 else '.' for j in range(K)) for k in range(1, L)), f'{time.time() - t0:.0f}s', flush=True)

# ---------------------------------------------------------------- pairs and refreshes
P = [p for p in json.load(open(T26 / f'{MODEL}.json'))['pairs'] if p['qualified'] and p['level'] in LEVELS]
cnt = defaultdict(int); keep = []
for p in P:
    if cnt[(p['level'], p['edit'])] < (3 if SMOKE else 60): cnt[(p['level'], p['edit'])] += 1; keep.append(p)
P = keep; print(f'{len(P)} pairs', flush=True)
REF = []  # per pair: [K+1, K, d] latents at refresh 0..K
for i in range(0, len(P), ad.batch_size):
    pp = P[i:i + ad.batch_size]; A = [p['R'] for p in pp]; B = [p['D'] for p in pp]
    _, natA = nat_run(A); _, natB = nat_run(B); feed = natA[:K]; R = [torch.stack(feed, 1)]
    for n in range(1, K + 1):
        _, nat = nat_run(B, feed); feed = nat[:K]; R.append(torch.stack(feed, 1))
    assert torch.allclose(R[K], torch.stack(natB[:K], 1), atol=1e-4), 'refresh K differs from the donor run'
    R = torch.stack(R, 1).float().cpu().numpy()
    REF += list(R)
    if DEV == 'mps': torch.mps.empty_cache()
print(time.strftime('%H:%M'), 'refreshes done (refresh K = donor run on every pair)', flush=True)

# ---------------------------------------------------------------- scoring
items = []
for p, R in zip(P, REF):
    L = p['level']; va, vb = vals(p['R']['question']), vals(p['D']['question'])
    i = 0 if p['edit'] == 'start' else int(p['edit'][4:])
    for k in range(1, L):
        if vb[k] == va[k]: continue
        sites = [j for j in range(K) if (L, k, j) in probe]
        pred = np.array([[probe[(L, k, j)].predict(R[n, j][None])[0] for j in sites] for n in range(K + 1)]) if sites else None
        items.append(dict(L=L, k=k, d=k - max(i, 1) + 1, a=int(va[k]), b=int(vb[k]), legible=bool(sites),
                          pred=pred.tolist() if pred is not None else None))
grp = defaultdict(list)
for it in items: grp[(it['L'], it['k'])].append(it)
for g in grp.values():  # chance target: the next item's B value in the same (L, k) group, if different
    for x, y in zip(g, g[1:] + g[:1]): x['chance'] = y['b'] if y['b'] != x['b'] else None
read = lambda it, n, v: it['pred'] is not None and v is not None and v in it['pred'][n]
summ = {}
for d in sorted({it['d'] for it in items}):
    xs = [it for it in items if it['d'] == d]; own = [it for it in xs if read(it, K, it['b'])]
    if len(xs) < 5: continue
    summ[d] = dict(n=len(xs), legible=sum(it['legible'] for it in xs) / len(xs), own_rate=len(own) / len(xs), n_own=len(own),
                   B=[sum(read(it, n, it['b']) for it in own) / max(len(own), 1) for n in range(K + 1)],
                   A=[sum(read(it, n, it['a']) for it in own) / max(len(own), 1) for n in range(K + 1)],
                   chance=[sum(read(it, n, it['chance']) for it in xs) / len(xs) for n in range(K + 1)])
    s = summ[d]; f = lambda v: ' '.join(f'{x:.2f}' for x in v)
    print(f'd={d}: n={s["n"]} legible {s["legible"]:.2f} read in own run {s["own_rate"]:.2f} | given own-read, B by refresh {f(s["B"])} '
          f'| stale A {f(s["A"])} | chance {f(s["chance"])}', flush=True)
json.dump(dict(model=MODEL, legible={f'{a}_{b}_{c}': v for (a, b, c), v in legible.items()}, summary=summ, items=items), open(OUT, 'w'))
print('DONE', f'{time.time() - t0:.0f}s')
