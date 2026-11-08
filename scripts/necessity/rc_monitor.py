"""Monitor steps 2-3 on RuleChain CODI: decode at the causal location, then test whether the decoded value is faithful.

  ../codi/.venv/bin/python scripts/necessity/rc_monitor.py <model gated|sim|startlast|relay> <device> <out json> [n_train_per_level=600]

Decoders (linear, 100-way, standardised features), one per site j and location:
  input   z_j, the latent's input vector (the recurrent route)
  entries site j's stored keys and values at layers 3, 6, 9, 12 (the attention route), 6144 dims
Trained on clean runs of train-split questions (levels 2-6, n per level), target v_{ks(j)} (ks as t1/t15: gated min(j, L-1),
sim min(j+1, L-1), others L-1; v_k = value after k rules). Monitor answer from a decoded value u at site j = u pushed through the
recipient's remaining rules (rules ks(j)+1 .. L).
Tested on valid L3-L5 (t15's items and donors):
  clean_right / clean_wrong  decode accuracy; P(monitor answer = model answer) (on wrong items: does the decoder predict the error?)
  swap_j -> decode at j' >= j (t15's both_j donor): P(decoded = donor-implied v_{ks(j')}), P(monitor answer = actual answer)
On-route flag per (L, j) from t15: input-only follow >= .5. Claim under test: monitors at on-route latents track own errors and
swap outcomes; accurate decoders at off-route latents (clean accuracy high) do not.
"""
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adapters  # noqa: E402
import core  # noqa: E402

MODEL, dev, out = sys.argv[1:4]; NPL = int(sys.argv[4]) if len(sys.argv) > 4 else 600
ad = adapters.RuleChainCODI(MODEL, dev)
from rc import parse, walk  # noqa: E402  (path set by harness)
DD = adapters.BASE / ('synthetic_data_v2_l1to7w_startlast' if MODEL in ('gated', 'sim', 'startlast') else 'synthetic_data_v2_l1to7w') / 'depth2'
T15 = json.load(open(adapters.BASE / 'codi_audit_artifacts' / f't15_{MODEL}.json'))
NL = 6; LAY = (2, 5, 8, 11)


def ks(j, L):
    return min(j, L - 1) if MODEL == 'gated' else (min(j + 1, L - 1) if MODEL == 'sim' else L - 1)


def feats(rows, feed=None):
    """-> answers [N], input feats [N, 6, 768], entry feats [N, 6, 6144] (float16 numpy)."""
    A, Xi, Xe = [], [], []
    tl = np.array([len(ad.tok(r['question'].strip() + '\n').input_ids) for r in rows]); order = np.argsort(tl, kind='stable')
    res = [None] * len(rows)
    for ii in core._batches(len(rows), ad.batch_size, order):
        gc.collect(); f = feed(ii) if feed else None
        a, Z, KV = core.run(ad, [rows[k] for k in ii], feed=f)
        zi = torch.stack([Z[s] for s in range(NL)], 1).float().cpu().numpy().astype(np.float16)
        ze = torch.stack([torch.cat([torch.cat([KV[s][l][0].flatten(1), KV[s][l][1].flatten(1)], 1) for l in LAY], 1) for s in range(NL)], 1)
        ze = ze.float().cpu().numpy().astype(np.float16)
        for n, k in enumerate(ii): res[k] = (a[n], zi[n], ze[n])
        if dev == 'mps': torch.mps.empty_cache()
    return np.array([r[0] for r in res]), np.stack([r[1] for r in res]), np.stack([r[2] for r in res])


def fit(X, y, epochs=200, wd=1e-3):
    torch.manual_seed(0)
    X = torch.tensor(X, dtype=torch.float32); mu, sd = X.mean(0), X.std(0) + 1e-4; X = (X - mu) / sd
    W = torch.nn.Linear(X.size(1), 100); opt = torch.optim.Adam(W.parameters(), lr=3e-3, weight_decay=wd); Y = torch.tensor(y)
    for _ in range(epochs):
        opt.zero_grad(); torch.nn.functional.cross_entropy(W(X), Y).backward(); opt.step()
    return lambda Z: W((torch.tensor(Z, dtype=torch.float32) - mu) / sd).argmax(-1).numpy()


t0 = time.time()
tr = [r for r in json.load(open(DD / 'train.json')) if 2 <= int(r['level']) <= 6]
by = {}
for r in tr: by.setdefault(int(r['level']), []).append(r)
tr = []
for L in sorted(by): random.Random(L).shuffle(by[L]); tr += by[L][:NPL]
Ltr = np.array([int(r['level']) for r in tr]); Vtr = [walk(*parse(r['question']))[0] for r in tr]
_, Itr, Etr = feats(tr); print(time.strftime('%H:%M'), f'{MODEL} train features {len(tr)} {time.time() - t0:.0f}s', flush=True)
DEC = {}
for j in range(NL):
    ok = np.array([0 <= Vtr[n][ks(j, Ltr[n])] <= 99 for n in range(len(tr))]); y = np.array([Vtr[n][ks(j, Ltr[n])] for n in range(len(tr))])[ok]
    DEC[('input', j)] = fit(Itr[ok, j].astype(np.float32), y); DEC[('entries', j)] = fit(Etr[ok, j].astype(np.float32), y)
print(time.strftime('%H:%M'), f'{MODEL} decoders trained {time.time() - t0:.0f}s', flush=True)
del Itr, Etr

va = json.load(open(DD / 'valid.json')); RES = {}
for L in (3, 4, 5):
    ev = [r for r in va if int(r['level']) == L]; ref = T15[str(L)]; assert [r['id'] for r in ev] == ref['ids']
    P = [parse(r['question']) for r in ev]; V = np.array([walk(*p)[0] for p in P]); gold = V[:, -1]
    mon = lambda u, i, j: (walk(int(u), P[i][1][ks(j, L):])[0][-1] if 0 <= u <= 99 else -1)
    A0, I0, E0 = feats(ev); right = A0 == gold
    assert (A0 == np.array(ref['clean_ans'])).all(), 'clean answers differ from t15'
    onroute = {j: (ref['res'][f'input_{j}']['follows'] or 0) >= .5 for j in range(NL)}
    o = dict(clean=float(right.mean()), onroute=onroute, sites={})
    for loc, X in (('input', I0), ('entries', E0)):
        for j in range(NL):
            u = DEC[(loc, j)](X[:, j].astype(np.float32)); tv = V[:, ks(j, L)]; m = np.array([mon(u[i], i, j) for i in range(len(ev))])
            o['sites'][f'{loc}_{j}'] = dict(dec_right=float((u == tv)[right].mean()), dec_wrong=float((u == tv)[~right].mean()) if (~right).any() else None,
                                            mon_right=float((m == A0)[right].mean()), mon_wrong=float((m == A0)[~right].mean()) if (~right).any() else None,
                                            n_wrong=int((~right).sum()))
    # swaps: t15's both_j donors, decode downstream
    for j in range(NL):
        dn = np.array(ref['res'][f'both_{j}']['donor']); tg = np.array(ref['res'][f'both_{j}']['target'])
        ok = right & (tg >= 0)
        Ad, Id, Ed = feats([ev[d] for d in dn])  # donor runs, aligned with recipients
        Zd = torch.tensor(Id[:, j].astype(np.float32))
        As, Is, Es = feats(ev, feed=lambda ii: {j: Zd[ii].to(dev)})
        for jp in range(j, NL):
            # donor-implied value at site jp: donor's v_{ks(j)} pushed through the recipient's rules up to ks(jp)
            imp = np.array([walk(int(V[d, ks(j, L)]), P[i][1][ks(j, L):ks(jp, L)])[0][-1] for i, d in enumerate(dn)])
            for loc, X in (('input', Is), ('entries', Es)):
                u = DEC[(loc, jp)](X[:, jp].astype(np.float32)); m = np.array([mon(u[i], i, jp) for i in range(len(ev))])
                o['sites'][f'swap{j}_{loc}_{jp}'] = dict(n=int(ok.sum()), dec_donor=float((u == imp)[ok].mean()),
                                                         mon_actual=float((m == As)[ok].mean()), answer_follows=float((As == tg)[ok].mean()))
    RES[L] = o
    s = o['sites']
    print(time.strftime('%H:%M'), f'{MODEL} L{L} clean {o["clean"]:.3f} onroute {[j for j in range(NL) if onroute[j]]}', flush=True)
    for loc in ('input', 'entries'):
        print(f'  {loc:7s} dec_right ' + ' '.join(f"{s[f'{loc}_{j}']['dec_right']:.2f}" for j in range(NL))
              + ' | mon_wrong ' + ' '.join(f"{(s[f'{loc}_{j}']['mon_wrong'] or 0):.2f}" for j in range(NL))
              + ' | swap_j->j mon_actual ' + ' '.join(f"{s[f'swap{j}_{loc}_{j}']['mon_actual']:.2f}" for j in range(NL)), flush=True)
    json.dump(dict(model=MODEL, n_train=len(tr), res=RES), open(out, 'w'))
print('DONE')
