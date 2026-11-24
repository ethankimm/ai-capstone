"""t38a / t38b: does the hop count tell a monitor WHEN (per item) and WHERE (which latent) to read? Offline on the t29 readability
records (t26/rcread_<model>.json) joined back to their t26 pairs (t26/<model>.json). Pre-registered in t_predictions.md (t38).

  python3 scripts/necessity/hop_guided.py <out json>          (no model needed; rebuilds rc_readability.py's pair selection exactly)

Per changed value v_k of a pair (edit at rule i, level L, hops h of the pair, structural depth d = k - max(i, 1) + 1):
  first read n_r = first refresh from which B's value is read at every later refresh (t29: read = any legible site's probe says B).
  Predictors of n_r: structural d (t29's alignment) and the pair's own hop count, aligned to the value, h_k = h - (L - 1 - k).
  Where: the site(s) whose probe first reads B at refresh n_r, vs z_{n_r - 1} (the latent input that refresh n_r newly makes B's own).
"""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path('/Users/henninglindig/Documents/Penn/CIS5980'); T26 = BASE / 'codi_audit_artifacts/t26'
sys.path.insert(0, str(BASE / 'ai-capstone/scripts/codi_null_audit'))
import rc as RC  # noqa: E402

vals = lambda q: RC.walk(*RC.parse(q))[0]
OUT = sys.argv[1]; K = 6; RES = {}
for m in ['gated', 'sim', 'simsf', 'relay', 'startlast', 'stab']:
    R = json.load(open(T26 / f'rcread_{m}.json')); LEG = R['legible']; levels = sorted({it['L'] for it in R['items']})
    P = [p for p in json.load(open(T26 / f'{m}.json'))['pairs'] if p['qualified'] and p['level'] in (2, 3, 4, 5, 6)]
    cnt = defaultdict(int); keep = []
    for p in P:
        if cnt[(p['level'], p['edit'])] < 60: cnt[(p['level'], p['edit'])] += 1; keep.append(p)
    rows = []
    for p in keep:  # same loop as rc_readability.py's scoring, so items line up one to one
        L = p['level']; va, vb = vals(p['R']['question']), vals(p['D']['question']); i = 0 if p['edit'] == 'start' else int(p['edit'][4:])
        for k in range(1, L):
            if vb[k] != va[k]: rows.append((p, L, i, k, int(va[k]), int(vb[k])))
    assert len(rows) == len(R['items']), (m, len(rows), len(R['items']))
    for (p, L, i, k, a, b), it in zip(rows, R['items']):
        assert (it['L'], it['k'], it['a'], it['b']) == (L, k, a, b), m
    out = []
    for (p, L, i, k, a, b), it in zip(rows, R['items']):
        if it['pred'] is None or b not in it['pred'][K]: continue  # value not read in B's own run
        sites = [j for j in range(K) if LEG[f'{L}_{k}_{j}'] >= .5]
        rd = [b in it['pred'][n] for n in range(K + 1)]
        nr = next(n for n in range(K + 1) if all(rd[n:]))
        first_sites = [sites[c] for c, v in enumerate(it['pred'][nr]) if v == b] if nr > 0 else []
        h = p['hops']; hk = h - (L - 1 - k)
        out.append(dict(L=L, edit=p['edit'], k=k, d=it['d'], h=h, hk=hk, nr=nr, sites=sites, first_sites=first_sites,
                        new_site_reads=(nr - 1) in first_sites if nr > 0 else None, n_sites=len(sites)))
    # cells where the hop count varies across pairs of the same structure: does h track n_r beyond d?
    cell = defaultdict(list)
    for o in out: cell[(o['L'], o['edit'], o['k'])].append(o)
    within = [(o['hk'] - np.mean([x['hk'] for x in g]), o['nr'] - np.mean([x['nr'] for x in g])) for g in cell.values() if len(g) >= 5
              and len({x['h'] for x in g}) > 1 for o in g]
    wr = float(np.corrcoef(*zip(*within))[0, 1]) if len(within) >= 20 else None
    off = [o for o in out if o['hk'] != o['d']]
    pos = [o for o in out if o['nr'] > 0]
    RES[m] = dict(n_read=len(out), n_items=len(R['items']),
                  match_d=float(np.mean([o['nr'] == o['d'] for o in out])) if out else None,
                  match_h=float(np.mean([o['nr'] == o['hk'] for o in out])) if out else None,
                  mae_d=float(np.mean([abs(o['nr'] - o['d']) for o in out])) if out else None,
                  mae_h=float(np.mean([abs(o['nr'] - o['hk']) for o in out])) if out else None,
                  n_disagree=len(off), disagree_match_d=float(np.mean([o['nr'] == o['d'] for o in off])) if off else None,
                  disagree_match_h=float(np.mean([o['nr'] == o['hk'] for o in off])) if off else None,
                  within_cell_r=wr, n_within=len(within),
                  new_site=float(np.mean([o['new_site_reads'] for o in pos])) if pos else None,
                  new_site_given_legible=float(np.mean([o['new_site_reads'] for o in pos if (o['nr'] - 1) in o['sites']]))
                  if [o for o in pos if (o['nr'] - 1) in o['sites']] else None,
                  mean_legible_sites=float(np.mean([o['n_sites'] for o in out])) if out else None, items=out)
    r = RES[m]; f = lambda x: '—' if x is None else f'{x:.2f}'
    print(f"{m}: read {r['n_read']}/{r['n_items']} | n_r = d {f(r['match_d'])} vs = h_k {f(r['match_h'])} (MAE {f(r['mae_d'])} / {f(r['mae_h'])}) "
          f"| where d != h_k (n {r['n_disagree']}): d {f(r['disagree_match_d'])} h_k {f(r['disagree_match_h'])} | within-cell r {f(r['within_cell_r'])} "
          f"(n {r['n_within']}) | first site = z_(n_r-1) {f(r['new_site'])} (given legible {f(r['new_site_given_legible'])}; "
          f"legible sites per value {f(r['mean_legible_sites'])})", flush=True)
json.dump(RES, open(OUT, 'w'))
