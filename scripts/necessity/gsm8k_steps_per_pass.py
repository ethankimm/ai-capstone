"""Post hoc reanalysis (2026-11-16, no pre-registered prediction): steps per pass on GSM8K.

A pair at serial distance s that flips after n refreshes carries its s steps through n + 1 passes, so s / (n + 1) is the
number of steps per pass. Reported for all qualified on-path pairs and for pairs the model answers correctly on both
questions (model answer equals the recomputed gold answer for the recipient R and the donor D), by serial distance.
Inputs: the t31 / t36 hop pairs of the four released checkpoints. Serial distance as in gsm8k_depth.py (strict rule);
its definitions are reused by executing that file up to its first module-level statement.

Run from ai-capstone/:  ../codi/.venv/bin/python scripts/necessity/gsm8k_steps_per_pass.py <out.json>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT_PATH = sys.argv[1]
sys.argv = [sys.argv[0], '/dev/null']
exec(open(Path(__file__).with_name('gsm8k_depth.py')).read().split('qtext = lambda')[0])  # depth_of, down, fnum, same, A, load_gsm8k_aug

qtext = lambda q: ' '.join(q.split())
te = load_gsm8k_aug(split='test', n=None, seed=None)
INFO = [depth_of(e.question, e.rationale, e.answer) for e in te]
BYQ = {qtext(e.question): i for i, e in enumerate(te)}
HOPF = dict(codi=['t31/gsm8k_codi.json', 't31/long_gsm8k_codi.json'], sim=['t31/gsm8k_sim.json', 't31/long_gsm8k_sim.json'],
            coconut=['t31/gsm8k_coconut.json', 't31/long_gsm8k_coconut.json'],
            simcoconut=['t36/gsm8k_simcoconut.json', 't36/long_gsm8k_simcoconut.json'])
if (A / 't47/long_gsm8k_llama.json').exists():  # t47: released Llama-3.2-1B CODI, same pairs
    HOPF['llama'] = ['t47/gsm8k_llama.json', 't47/long_gsm8k_llama.json']
for key, t in (('simllama1b', 't49'), ('simllama3b', 't50')):  # released SIM-CODI Llama-3.2 1B (fp32) / 3B (bf16), same pairs
    if (A / f'{t}/long_gsm8k_{key}.json').exists():
        HOPF[key] = [f'{t}/gsm8k_{key}.json', f'{t}/long_gsm8k_{key}.json']
rng = np.random.default_rng(0)


def ci(v):
    v = np.asarray(v, float)
    return [float(x) for x in np.percentile([rng.choice(v, len(v)).mean() for _ in range(2000)], [2.5, 97.5])]


def within_item_slope(rows):
    """Item-demeaned regression of hops on serial distance, item bootstrap (as gsm8k_depth_within.py)."""
    by = defaultdict(list)
    for r in rows: by[r['i']].append(r)
    items = [v for v in by.values() if len({x['s'] for x in v}) > 1]
    def slope(sel):
        xs, ys = [], []
        for v in sel:
            ms, mh = np.mean([x['s'] for x in v]), np.mean([x['h'] for x in v])
            xs += [x['s'] - ms for x in v]; ys += [x['h'] - mh for x in v]
        xs, ys = np.array(xs), np.array(ys)
        return float((xs * ys).sum() / (xs * xs).sum()) if (xs * xs).sum() > 0 else float('nan')
    if len(items) < 10: return dict(n_items=len(items), slope=None, ci=None)
    bs = [slope([items[j] for j in rng.integers(0, len(items), len(items))]) for _ in range(2000)]
    return dict(n_items=len(items), slope=slope(items), ci=[float(x) for x in np.nanpercentile(bs, [2.5, 97.5])])


R = {}
for m, files in HOPF.items():
    rows, long_n, long_ok = [], 0, 0
    for f in files:
        for p in json.load(open(A / f))['pairs']:
            if not p.get('qualified') or p.get('hops') is None: continue
            ok = same(fnum(p['aR']), fnum(p['R'].get('answer'))) and same(fnum(p['aD']), fnum(p['D'].get('answer')))
            if 'long_' in f: long_n += 1; long_ok += ok
            i = BYQ.get(qtext(p['R']['question']))
            if i is None: continue
            sd = down(INFO[i]['strict'], int(p['edit'][4:]))
            if sd is None: continue
            rows.append(dict(i=i, s=sd, h=int(p['hops']), ok=bool(ok), d=INFO[i]['strict']['depth'], long='long_' in f))
    M = R[m] = dict(n_pairs=len(rows), n_correct=sum(r['ok'] for r in rows), long_pairs=long_n, long_correct=int(long_ok), by_s={})
    print(f"\n{m}: on-path pairs {len(rows)}, both correct {M['n_correct']} | long pairs {long_n}, long correct {int(long_ok)}")
    for s in (2, 3, 4, 5):
        row = {}
        for lab, sel in (('all', lambda r: True), ('correct', lambda r: r['ok']), ('wrong', lambda r: not r['ok'])):
            h = np.array([r['h'] for r in rows if r['s'] == s and sel(r)])
            if len(h) < 15: row[lab] = dict(n=int(len(h))); continue
            spp = s / (h + 1)
            row[lab] = dict(n=int(len(h)), hops=float(h.mean()), hops_ci=ci(h), steps_per_pass=float(spp.mean()), spp_ci=ci(spp),
                            share_hops0=float(np.mean(h == 0)), share_le=[float(np.mean(h <= k)) for k in range(5)],
                            share_ge_s_minus_1=float(np.mean(h >= s - 1)))
        M['by_s'][str(s)] = row
        a, c, w = row['all'], row['correct'], row['wrong']
        fmt = lambda d: 'n<15' if 'hops' not in d else f"{d['hops']:.2f} [{d['hops_ci'][0]:.2f},{d['hops_ci'][1]:.2f}] spp {d['steps_per_pass']:.2f} h0 {d['share_hops0']:.2f} >=s-1 {d['share_ge_s_minus_1']:.2f}"
        print(f"  s={s} all n={a['n']:4d} {fmt(a)} | correct n={c['n']:3d} ({c['n'] / a['n']:.2f}) {fmt(c)} | wrong n={w['n']:4d} {fmt(w)}")
    # Table cells: correct pairs whose edit is at the start of the item's longest dependency path (s == item depth D),
    # and correct pairs by item depth whatever the edit position.
    M['start_of_path'] = {}; M['by_item_depth'] = {}
    for D in (2, 3, 4):
        h = np.array([r['h'] for r in rows if r['ok'] and r['d'] == D and r['s'] == D])
        if len(h) >= 15:
            spp = D / (h + 1)
            M['start_of_path'][str(D)] = dict(n=int(len(h)), hops=float(h.mean()), hops_ci=ci(h), steps_per_pass=float(spp.mean()), spp_ci=ci(spp))
        g = np.array([r['h'] for r in rows if r['ok'] and r['d'] == D])
        if len(g) >= 15: M['by_item_depth'][str(D)] = dict(n=int(len(g)), hops=float(g.mean()), hops_ci=ci(g))
        e = np.array([r['h'] for r in rows if r['ok'] and r['d'] == D and r['s'] == 2])
        if len(e) >= 10: M['by_item_depth'].setdefault(str(D), {})['edit_two_from_answer'] = dict(n=int(len(e)), hops=float(e.mean()))
    lg = [r for r in rows if r['long']]
    M['long_pooled'] = dict(all=float(np.mean([r['h'] for r in lg])), correct=float(np.mean([r['h'] for r in lg if r['ok']])), n=len(lg), n_correct=sum(r['ok'] for r in lg))
    print('  start of path (s = D), correct:', {D: (v['n'], round(v['hops'], 2), [round(x, 2) for x in v['hops_ci']], round(v['steps_per_pass'], 2)) for D, v in M['start_of_path'].items()})
    print('  by item depth, correct:', {D: (v.get('n'), round(v['hops'], 2) if 'hops' in v else None, v.get('edit_two_from_answer')) for D, v in M['by_item_depth'].items()})
    print('  long pairs pooled (on path): all', round(M['long_pooled']['all'], 3), 'correct', round(M['long_pooled']['correct'], 3), M['long_pooled']['n'], M['long_pooled']['n_correct'])
    M['within_item'] = {lab: within_item_slope([r for r in rows if sel(r)]) for lab, sel in (('all', lambda r: True), ('correct', lambda r: r['ok']))}
    ok = [r for r in rows if r['ok'] and r['s'] <= 4]
    M['across_pair_slope_correct_s2to4'] = float(np.polyfit([r['s'] for r in ok], [r['h'] for r in ok], 1)[0])
    print('  within item:', {k: (v['n_items'], None if v['slope'] is None else round(v['slope'], 3), None if v['ci'] is None else [round(x, 3) for x in v['ci']]) for k, v in M['within_item'].items()},
          '| across-pair slope, correct, s 2-4:', round(M['across_pair_slope_correct_s2to4'], 3))
json.dump(R, open(OUT_PATH, 'w'), indent=1)
print('\nwrote', OUT_PATH)
