"""t56 / t57 scoring (pre-registered in prereg/t_predictions.md, t56 and t57). Offline; reads ../codi_audit_artifacts/t56/<model>_{sets,robust}.json.

  python3 scripts/necessity/validity_score.py <out json>      (from ai-capstone/; scores whatever files exist)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
import os
A = REPO.parent / 'codi_audit_artifacts'; T = Path(os.environ.get('T56DIR', A / 't56'))
LOOP = ['gated', 'sim', 'simsf', 'startlast', 'relay', 'stab']; CHAIN = ['gated', 'sim', 'simsf']; STD = ['startlast', 'relay', 'stab']
REFS = ['startlast0', 'stab0']
TEST_RECORD = dict(gated='20261107-204814_codi_v2-l1to7w-startlast-gated-depth2-codi', sim='20261107-225603_codi_v2-l1to7w-startlast-sim-depth2-codi',
                   simsf='20261108-130905_codi_v2-l1to7w-sim-depth2-codi', startlast='20261106-083843_codi_v2-l1to7w-startlast-depth2-codi',
                   relay='20261103-193425_codi_v2-l1to7w-depth2-codi', stab='20261105-044444_codi_v2-l1to7w-stab-depth2-codi',
                   startlast0='20261106-052644_codi_v2-l1to7w-startlast-depth2-codi0', stab0='20261105-024111_codi_v2-l1to7w-stab-depth2-codi0')
test_acc = lambda m: json.load(open(REPO / 'results' / TEST_RECORD[m] / 'manifest.json'))['metrics']['extra']['per_level_accuracy']
SHIFT = ['rw', 'hc', 'hc1', 'sv']
mean = lambda xs: float(np.mean(xs)) if len(xs) else None
Q = lambda ps: [p for p in ps if p.get('qualified') and p.get('hops') is not None]


def margin_counts(p):
    m = p['margin']; K = len(m) - 1; d = m[K] - m[0]
    mh = next((n for n in range(K + 1) if all(x > 0 for x in m[n:])), None)
    if d <= 0: return mh, None, None
    u = [(x - m[0]) / d for x in m]
    onset = next((n for n in range(K + 1) if u[n] >= .1), None)
    half = next((n for n in range(K + 1) if all(x >= .5 for x in u[n:])), None)
    return mh, onset, half


def by(ps, key):
    out = defaultdict(list)
    for p in ps: out[key(p)].append(p)
    return out


def cell(ps): return dict(n=len(ps), hops=mean([p['hops'] for p in ps]))


R = {}; V = {}
# ---------------------------------------------------------------- t56
for m in LOOP + REFS:
    f = T / f'{m}_sets.json'
    if not f.exists(): continue
    d = json.load(open(f)); out = {}; ta = test_acc(m)
    for tag, v in d.items():
        if not isinstance(v, dict) or 'acc' not in v: continue
        o = dict(acc=v['acc'])
        if 'pairs' in v:
            ps = v['pairs']; q = Q(ps)
            o['mk'] = f"{sum(p['mk_matches_donor'] for p in ps)}/{len(ps)}"
            st = by([p for p in q if p['pos'] == 0], lambda p: p['level'])
            o['start_by_L'] = {str(L): cell(st[L]) for L in sorted(st)}
            o['by_s'] = {str(s): cell(x) for s, x in sorted(by(q, lambda p: p['s']).items())}
            o['by_s_both_right'] = {str(s): cell(x) for s, x in sorted(by([p for p in q if p['aR'] == p['goldR'] and p['aD'] == p['goldD']],
                                                                           lambda p: p['s']).items())}
            rule = [p for p in q if p['pos'] > 0]
            o['rule_exact'] = mean([p['hops'] == p['level'] - p['pos'] for p in rule]); o['n_rule'] = len(rule)
        out[tag] = o
    ok_ = next((t for t in out if t.startswith('orig_K')), None)  # orig_K6, or orig_K0 for CODI-0
    out['orig_vs_test'] = {L: round(out[ok_]['acc'][L] - ta[L], 3) for L in out[ok_]['acc']} if ok_ else None
    R[m] = out

ok = lambda c, f: c is not None and c['n'] >= 15 and f(c['hops'])
V['t56'] = {}
if R:
    V['t56']['P0'] = {m: dict(mk=[R[m][t]['mk'] for t in R[m] if isinstance(R[m][t], dict) and 'mk' in R[m][t]],
                              orig_within_05=all(abs(x) <= .05 for x in R[m]['orig_vs_test'].values())) for m in R}
    V['t56']['P1'] = {m: {s: {L: (R[m][f'{s}_K6']['start_by_L'].get(str(L)), ok(R[m][f'{s}_K6']['start_by_L'].get(str(L)), lambda h: h >= L - 1))
                              for L in range(2, 7)} for s in SHIFT if f'{s}_K6' in R[m]} for m in CHAIN if m in R}
    V['t56']['P2'] = {s: (R['gated'][f'{s}_K6']['rule_exact'], R['gated'][f'{s}_K6']['n_rule']) for s in SHIFT if 'gated' in R and f'{s}_K6' in R['gated']}
    V['t56']['P3'] = {m: {s: {L: (R[m][f'{s}_K6']['start_by_L'].get(str(L)), ok(R[m][f'{s}_K6']['start_by_L'].get(str(L)), lambda h: h <= 2.5))
                              for L in range(2, 7)} for s in ['orig'] + SHIFT if f'{s}_K6' in R[m]} for m in STD if m in R}
    pool = lambda a: float(np.mean([a[str(L)] for L in (3, 4, 5)]))
    V['t56']['P4'] = {m: dict(hc1=pool(R[m]['hc1_K6']['acc']), hc=pool(R[m]['hc_K6']['acc']), held=pool(R[m]['hc1_K6']['acc']) > pool(R[m]['hc_K6']['acc']))
                      for m in LOOP if m in R and 'hc1_K6' in R[m]}
    V['t56']['P5'] = {m: {K: R[m][f'l8_K{K}']['acc']['8'] for K in (6, 7, 8) if f'l8_K{K}' in R[m]} for m in LOOP if m in R}
    V['t56']['acc_table'] = {m: {t: R[m][t]['acc'] for t in R[m] if isinstance(R[m][t], dict) and 'acc' in R[m][t]} for m in R}

# ---------------------------------------------------------------- t56 addendum: wide set
V['wide'] = {}
for m in LOOP + REFS:
    f = T / f'{m}_wide.json'
    if not f.exists(): continue
    d = json.load(open(f)); d = d[next(k for k in d if k.startswith('wide_K'))]; items = {r['id']: r for r in json.load(open(T / 'sets/wide_last.json'))}
    rows = [(items[i], a) for i, a in d['answers'].items()]
    hi = [a == int(r['answer']) for r, a in rows if int(r['answer']) >= 100]
    lo = defaultdict(list)
    for r, a in rows:
        if int(r['answer']) <= 99: lo[r['level']].append(a == int(r['answer']))
    first100 = defaultdict(list)
    for r, a in rows:
        k = next(j for j, v in enumerate(r['values']) if v >= 100); first100['start' if k == 0 else 'later'].append(a == int(r['answer']))
    o = dict(acc=d['acc'], W1_acc_answer_ge100=(mean(hi), len(hi)), acc_answer_le99_by_L={str(L): (mean(v), len(v)) for L, v in sorted(lo.items())},
             acc_by_first_ge100={k: (mean(v), len(v)) for k, v in first100.items()})
    ok_ = next((t for t in R.get(m, {}) if t.startswith('orig_K')), None)
    if ok_:
        orig = float(np.mean([R[m][ok_]['acc'][str(L)] for L in (2, 3, 4, 5)]))
        le = float(np.mean([x for L in (2, 3, 4, 5) for x in lo[L]]))
        o['W2'] = dict(orig_L2_5=orig, wide_le99_L2_5=le, drop=orig - le, held=orig - le >= .30)
    if 'pairs' in d:
        ps = d['pairs']; q = Q(ps); st = by([p for p in q if p['pos'] == 0], lambda p: p['level'])
        o['W0_mk'] = f"{sum(p['mk_matches_donor'] for p in ps)}/{len(ps)}"
        o['W3_start_by_L'] = {str(L): (cell(st[L]), ok(cell(st[L]), lambda h, L=L: h >= L - 1)) for L in range(2, 7) if L in st}
        o['by_s'] = {str(s): cell(x) for s, x in sorted(by(q, lambda p: p['s']).items())}
    V['wide'][m] = o

# ---------------------------------------------------------------- t57
t31 = lambda m: json.load(open(A / f't31/{m}.json'))
V['t57'] = {}
for m in LOOP:
    f = T / f'{m}_robust.json'
    if not f.exists(): continue
    d = json.load(open(f)); o = {}
    rp = Q(d['repro']); o['P0_repro_equal'] = mean([p['hops'] == p['t31_hops'] for p in rp]); o['n'] = len(rp)
    key = lambda p: (p['idR'].rsplit('-R-', 1)[0], p['edit'])
    sw = {key(dict(p, idR=p['idD'].replace('-D-', '-R-'))): p for p in Q(d['swap'])}
    diffs = [abs(sw[key(p)]['hops'] - p['hops']) for p in rp if key(p) in sw]
    o['P1'] = dict(n=len(diffs), within1=mean([x <= 1 for x in diffs]), equal=mean([x == 0 for x in diffs]), swap_mean=mean([sw[key(p)]['hops'] for p in rp if key(p) in sw]),
                   repro_mean=mean([p['hops'] for p in rp if key(p) in sw]))
    mc = [(p, *margin_counts(p)) for p in rp]
    o['P2'] = dict(margin_equal=mean([mh == p['hops'] for p, mh, _, _ in mc]),
                   onset_equal=mean([on == p['hops'] for p, _, on, _ in mc if on is not None]),
                   onset_mean=mean([on for _, _, on, _ in mc if on is not None]),
                   half_by_s={str(s): dict(n=len(x), half=mean([h for _, _, _, h in x if h is not None]), hops=mean([p['hops'] for p, *_ in x]))
                              for s, x in sorted(by(mc, lambda t: t[0]['s']).items())})
    sm, lg = Q(d['size_small']), Q(d['size_large'])
    bs, bl = by(sm, lambda p: p['s']), by(lg, lambda p: p['s'])
    o['P3'] = dict(cells={str(s): dict(small=cell(bs[s]), large=cell(bl[s])) for s in sorted(set(bs) | set(bl))},
                   pooled_diff=(mean([p['hops'] for p in lg]) or 0) - (mean([p['hops'] for p in sm]) or 0))
    o['P3']['cells_ok'] = all(abs(c['large']['hops'] - c['small']['hops']) <= .3 for c in o['P3']['cells'].values()
                              if c['small']['n'] >= 15 and c['large']['n'] >= 15)
    un = by(Q(d['unrelated']), lambda p: p['level'])
    st31 = by([p for p in t31(m)['pairs'] if p.get('qualified') and p.get('hops') is not None and p['edit'] == 'start'], lambda p: p['level'])
    o['P4'] = {str(L): dict(unrelated=cell(un[L]), t31_start=cell(st31[L])) for L in sorted(un)}
    V['t57'][m] = o

json.dump(dict(per_model=R, verdicts=V), open(sys.argv[1], 'w'), indent=1, default=str)
print(json.dumps(V, indent=1, default=str)[:20000])
