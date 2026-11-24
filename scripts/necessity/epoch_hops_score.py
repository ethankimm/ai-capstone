"""t55 scoring (pre-registered in prereg/t_predictions.md, t55). Offline: ../codi_audit_artifacts/t55/*.json and t31/startlast.json.

  python3 scripts/necessity/epoch_hops_score.py <out json>

Pairs answered right on both sides. A = mean hops at s = 3-4 pooled; B = across-pair OLS slope of hops on s over s = 1-5.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

A = Path('/Users/henninglindig/Documents/Penn/CIS5980/codi_audit_artifacts'); rng = np.random.default_rng(55)


def load(f):
    return [p for p in json.load(open(f))['pairs'] if p.get('qualified') and p.get('hops') is not None and p['R_right'] and p['D_right']]


def metrics(P):
    a = [p['hops'] for p in P if p['s'] in (3, 4)]
    xs = np.array([p['s'] for p in P if 1 <= p['s'] <= 5], float); ys = np.array([p['hops'] for p in P if 1 <= p['s'] <= 5], float)
    B = float(np.polyfit(xs, ys, 1)[0]) if len(set(xs)) > 1 else None
    idx = rng.integers(0, len(xs), (2000, len(xs)))
    Bci = [float(v) for v in np.percentile([np.polyfit(xs[i], ys[i], 1)[0] for i in idx[:500]], [2.5, 97.5])] if B is not None else None
    by = defaultdict(list)
    for p in P: by[p['s']].append(p['hops'])
    return dict(n=len(P), A=float(np.mean(a)) if a else None, nA=len(a), B=B, B_ci=Bci,
                by_s={str(s): (len(v), round(float(np.mean(v)), 2)) for s, v in sorted(by.items())},
                start_by_L={str(L): (len(v), round(float(np.mean(v)), 2)) for L, v in sorted(
                    {L: [p['hops'] for p in P if p['edit'] == 'start' and p['level'] == L] for L in sorted({p['level'] for p in P})}.items()) if v})


R = {}
for ep, st in ((6, 8910), (8, 11880), (10, 14850), (12, 17820)):
    f = A / f't55/startlast_{st}.json'
    if f.exists(): R[f'startlast_e{ep}'] = metrics(load(f))
if (A / 't55/startlast_17820.json').exists():
    new = {(p['id'], p['edit'], p['D']['question']): p['hops'] for p in json.load(open(A / 't55/startlast_17820.json'))['pairs'] if p.get('hops') is not None}
    old = {(p['id'], p['edit'], p['D']['question']): p['hops'] for p in json.load(open(A / 't31/startlast.json'))['pairs'] if p.get('hops') is not None}
    common = set(new) & set(old)
    R['P0'] = dict(n=len(common), match=float(np.mean([new[k] == old[k] for k in common])) if common else None)
if (A / 't55/r1.json').exists(): R['r1_e40'] = metrics(load(A / 't55/r1.json'))
for k, v in R.items(): print(k, v)
e6, e12 = R.get('startlast_e6'), R.get('startlast_e12')
if e6 and e12:
    print('P1 A(12) - A(6) =', round(e12['A'] - e6['A'], 2), '(n', e6['nA'], e12['nA'], ') held' if e12['A'] - e6['A'] >= .30 and min(e6['nA'], e12['nA']) >= 15 else ') missed')
    print('P2 B(12) - B(6) =', round(e12['B'] - e6['B'], 2), 'held' if e12['B'] - e6['B'] >= .10 else 'missed')
if 'r1_e40' in R:
    r = R['r1_e40']; print('P3 r1 A', r['A'], 'B', r['B'], 'held' if (r['A'] or 0) <= .50 and (r['B'] or 0) <= .20 else 'missed')
json.dump(R, open(sys.argv[1], 'w'), indent=1)
