"""t53 scoring (pre-registered in prereg/t_predictions.md, t53 + P2 correction). Offline, reads ../codi_audit_artifacts/t53/<model>.json.

  ../codi/.venv/bin/python scripts/necessity/prefix_sub_score.py <out json>

e = smallest c with S_c = h_0 - h_c > 0 (K + 1 if none); the first latent carrying the edit is z_{e-1}.
Position: RuleChain i (start = 0, rule i); GSM8K k = gold step index of the edited step, s and item depth by gsm8k_depth.py (strict).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT_PATH = sys.argv[1]
sys.argv = [sys.argv[0], '/dev/null']
exec(open(Path(__file__).with_name('gsm8k_depth.py')).read().split('qtext = lambda')[0])  # depth_of, down, A, load_gsm8k_aug
import os
T = Path(os.environ.get('T53DIR', A / 't53'))
rng = np.random.default_rng(53); NB = 2000
qtext = lambda q: ' '.join(q.split())
te = load_gsm8k_aug(split='test', n=None, seed=None)
INFO = [depth_of(e.question, e.rationale, e.answer) for e in te]
BYQ = {qtext(e.question): i for i, e in enumerate(te)}


def entry(h):
    return next((c for c in range(1, len(h)) if h[0] - h[c] > 0), len(h))


def demeaned_slope(rows, g):
    """OLS slope of e on x after demeaning both within group g (level or item); group bootstrap."""
    by = defaultdict(list)
    for r in rows: by[r[g]].append(r)
    grp = [v for v in by.values() if len({x['x'] for x in v}) > 1]
    def sl(sel):
        xs, ys = [], []
        for v in sel:
            mx, my = np.mean([x['x'] for x in v]), np.mean([x['e'] for x in v])
            xs += [x['x'] - mx for x in v]; ys += [x['e'] - my for x in v]
        xs, ys = np.array(xs, float), np.array(ys, float)
        return float((xs * ys).sum() / (xs * xs).sum()) if (xs * xs).sum() else float('nan')
    if len(grp) < 5: return dict(n_groups=len(grp), slope=None, ci=None)
    bs = [sl([grp[j] for j in rng.integers(0, len(grp), len(grp))]) for _ in range(NB)]
    return dict(n_groups=len(grp), n=sum(len(v) for v in grp), slope=sl(grp), ci=[float(x) for x in np.nanpercentile(bs, [2.5, 97.5])])


def ratio(r, j):
    return r['dist'][j] / r['ref_dist'][j] if r.get('ref_dist') and r['ref_dist'][j] > 0 else np.nan


R = {}
for m in ('gated', 'startlast', 'gsm8k_codi', 'gsm8k_sim'):
    f = T / f'{m}.json'
    if not f.exists(): print(m, 'missing'); continue
    d = json.load(open(f)); K = d['K']; P = d['pairs']; rows = []
    for p in P:
        e = entry(p['h'])
        if m in ('gated', 'startlast'):
            x = 0 if p['edit'] == 'start' else int(p['edit'][4:])
            rows.append(dict(p, e=e, x=x, grp=p['level'], depth=p['level']))
        else:
            i = BYQ.get(qtext(p['R']['question']))
            if i is None: continue
            k = int(p['edit'][4:]); sd = down(INFO[i]['strict'], k)
            a, b = NUM.findall(p['R']['question']), NUM.findall(p['D']['question'])
            v = [x for x, y in zip(a, b) if x != y] if len(a) == len(b) else []
            ex = [x for x, _ in STEP.findall(te[i].rationale)]
            uses = [j for j, x in enumerate(ex) if len(v) == 1 and any(same(fnum(t), fnum(v[0])) for t in NUM.findall(x))]
            du = [dd for dd in (down(INFO[i]['strict'], j) for j in uses) if dd is not None]
            rows.append(dict(p, e=e, x=k, grp=i, sd=sd, au=max(du) if du else None, depth=INFO[i]['strict']['depth']))
    M = R[m] = dict(n=len(rows), K=K)
    M['checks'] = dict(h0_t31=float(np.mean([r['h'][0] == r['t31_hops'] for r in rows])), own=float(np.mean([r['own_ok'] for r in rows])),
                       cK=float(np.mean([r['ck_last'] for r in rows])))
    M['mean_h_c'] = [float(v) for v in np.mean([r['h'] for r in rows], axis=0)]
    M['median_ratio_j'] = [float(np.nanmedian([ratio(r, j) for r in rows])) for j in range(K + 1)]
    sub = [r for r in rows if r['x'] > 0] if m in ('gated', 'startlast') else rows
    M['slope_e_on_pos'] = demeaned_slope(sub, 'grp')
    M['slope_h0_on_pos'] = demeaned_slope([dict(r, e=r['h'][0]) for r in sub], 'grp')
    by = defaultdict(list)
    for r in rows: by[r['x']].append(r)
    M['by_pos'] = {str(x): dict(n=len(v), e_mean=float(np.mean([r['e'] for r in v])), h0=float(np.mean([r['h'][0] for r in v])),
                                S=[float(np.mean([r['h'][0] - r['h'][c] for r in v])) for c in range(K + 1)],
                                ratio=[float(np.nanmedian([ratio(r, j) for r in v])) for j in range(K + 1)])
                   for x, v in sorted(by.items())}
    if m == 'gated':
        rr = [r for r in rows if r['x'] > 0]; off = [r['e'] - r['x'] for r in rr]; vals, cnt = np.unique(off, return_counts=True)
        M['P1'] = dict(mode_offset=int(vals[cnt.argmax()]), mode_share=float(cnt.max() / len(off)), slope=M['slope_e_on_pos']['slope'])
        M['P1']['held'] = bool(M['P1']['mode_share'] >= .90 and .8 <= (M['P1']['slope'] or 0) <= 1.2)
        before = defaultdict(list); at = []
        for r in rr:
            for j in range(0, r['e'] - 1): before[j - (r['e'] - 1)].append(ratio(r, j))
            if 1 <= r['e'] <= K + 1 and r['e'] - 1 <= K: at.append(ratio(r, r['e'] - 1))
        M['P2'] = dict(before={str(k): float(np.nanmedian(v)) for k, v in sorted(before.items())}, at=float(np.nanmedian(at)))
        M['P2']['held'] = bool(all(v <= .05 for v in M['P2']['before'].values()) and M['P2']['at'] >= .20)
    if m == 'startlast':
        M['P3'] = dict(slope=M['slope_e_on_pos']['slope'], held=bool((M['slope_e_on_pos']['slope'] or 9) < .30))
    if m.startswith('gsm8k'):
        late = [r for r in rows if r['x'] >= 2]
        M['P4'] = dict(slope=M['slope_e_on_pos']['slope'], held=bool((M['slope_e_on_pos']['slope'] or 9) < .30))
        M['P5'] = dict(n=len(late), S1=float(np.mean([r['h'][0] - r['h'][1] for r in late])))
        M['P5']['held'] = bool(M['P5']['S1'] >= .30)
        M['P6'] = dict(n=len(late), ratio0=float(np.nanmedian([ratio(r, 0) for r in late])))
        M['P6']['held'] = bool(M['P6']['ratio0'] >= .25)
        bys = defaultdict(list)
        for r in rows:
            if r['sd'] is not None: bys[r['sd']].append(r)
        M['by_s'] = {str(s): dict(n=len(v), e_mean=float(np.mean([r['e'] for r in v])), h0=float(np.mean([r['h'][0] for r in v])))
                     for s, v in sorted(bys.items())}
        M['slope_e_on_s'] = demeaned_slope([dict(r, x=r['sd']) for r in rows if r['sd'] is not None], 'grp')
        # exploratory (post hoc, after t54): pairs with h_0 >= 1 only (e is undefined otherwise), and the any-use gold distance
        h1 = [r for r in rows if r['h'][0] >= 1]
        M['explore'] = dict(n_h0ge1=len(h1), slope_e_on_k_h0ge1=demeaned_slope(h1, 'grp'),
                            slope_e_on_anyuse_h0ge1=demeaned_slope([dict(r, x=r['au']) for r in h1 if r['au'] is not None], 'grp'),
                            slope_h0_on_anyuse=demeaned_slope([dict(r, x=r['au'], e=r['h'][0]) for r in rows if r['au'] is not None], 'grp'),
                            slope_h0_on_label=demeaned_slope([dict(r, x=r['sd'], e=r['h'][0]) for r in rows if r['sd'] is not None], 'grp'),
                            S1_late_h0ge1=float(np.mean([r['h'][0] - r['h'][1] for r in h1 if r['x'] >= 2])) if h1 else None)
        print('  exploratory', json.dumps(M['explore']))
    print(f"\n== {m}: n {M['n']} checks {M['checks']}")
    print('  mean h_c:', ' '.join(f'{v:.2f}' for v in M['mean_h_c']), '| median d_j/ref_j:', ' '.join(f'{v:.2f}' for v in M['median_ratio_j']))
    print('  slope e on position:', M['slope_e_on_pos'], '| slope h0 on position:', M['slope_h0_on_pos']['slope'])
    for x, v in M['by_pos'].items():
        print(f"  pos {x:>2} n={v['n']:4d} e {v['e_mean']:.2f} h0 {v['h0']:.2f} S_c", ' '.join(f'{s:.2f}' for s in v['S']),
              '| d/ref', ' '.join(f'{s:.2f}' for s in v['ratio']))
    for k in ('P1', 'P2', 'P3', 'P4', 'P5', 'P6'):
        if k in M: print(' ', k, M[k])
json.dump(R, open(OUT_PATH, 'w'), indent=1)
