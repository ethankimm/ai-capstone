"""t54 scoring (pre-registered in prereg/t_predictions.md, t54). Offline; reads ../codi_audit_artifacts/t54/<m>.json and the t31 pairs.

  ../codi/.venv/bin/python scripts/necessity/cot_route_score.py <out json>

Model route: the written steps (<<expr=res>>) before "The answer is:" in the model's own CoT, parsed with gsm8k_depth.depth_of against
the model's own answer (the unwritten final step becomes the implicit final step). Edit distance on the model route: the edited number
is the question number that differs between R and D; distance = max down() over R's written steps whose expression contains it,
1 if none does.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT_PATH = sys.argv[1]
sys.argv = [sys.argv[0], '/dev/null']
exec(open(Path(__file__).with_name('gsm8k_depth.py')).read().split('qtext = lambda')[0])  # depth_of, down, fnum, same, NUM, STEP, A
rng = np.random.default_rng(54); NB = 2000
qtext = lambda q: ' '.join(q.split())
te = load_gsm8k_aug(split='test', n=None, seed=None)
GOLD = [depth_of(e.question, e.rationale, e.answer) for e in te]
BYQ = {qtext(e.question): i for i, e in enumerate(te)}


def route(q, row):
    t = row['text']; w = t.split('The answer is:')[0]
    info = depth_of(q, w, row['answer'] if row['answer'] is not None else 'nan')
    return info, [e for e, _ in STEP.findall(w)]


def edited_number(qr, qd):
    a, b = NUM.findall(qr), NUM.findall(qd)
    if len(a) != len(b): return None
    d = [x for x, y in zip(a, b) if x != y]
    return d[0] if len(d) == 1 else None


def slope(rows, x='x', y='h'):
    by = defaultdict(list)
    for r in rows: by[r['i']].append(r)
    grp = [v for v in by.values() if len({r[x] for r in v}) > 1]
    def sl(sel):
        xs, ys = [], []
        for v in sel:
            mx, my = np.mean([r[x] for r in v]), np.mean([r[y] for r in v])
            xs += [r[x] - mx for r in v]; ys += [r[y] - my for r in v]
        xs, ys = np.array(xs, float), np.array(ys, float)
        return float((xs * ys).sum() / (xs * xs).sum()) if (xs * xs).sum() else float('nan')
    if len(grp) < 10: return dict(n_items=len(grp), slope=None, ci=None)
    bs = [sl([grp[j] for j in rng.integers(0, len(grp), len(grp))]) for _ in range(NB)]
    return dict(n_items=len(grp), n_pairs=sum(len(v) for v in grp), slope=sl(grp), ci=[float(v) for v in np.nanpercentile(bs, [2.5, 97.5])])


def ci(v):
    v = np.asarray(v, float)
    return [float(x) for x in np.percentile([rng.choice(v, len(v)).mean() for _ in range(NB)], [2.5, 97.5])]


R = {}
for m in ('codi', 'sim'):
    f = A / f't54/{m}.json'
    if not f.exists(): print(m, 'missing'); continue
    G = {r['question']: r for r in json.load(open(f))['rows']}
    M = R[m] = {}
    tst = [(i, e) for i, e in enumerate(te) if e.question in G]
    M['P0'] = dict(parse=float(np.mean([G[e.question]['answer'] is not None for _, e in tst])),
                   acc=float(np.mean([same(fnum(G[e.question]['answer']), fnum(e.answer)) for _, e in tst])), n=len(tst))
    M['P0']['held'] = bool(M['P0']['parse'] >= .95 and .35 <= M['P0']['acc'] <= .50)
    md = [route(e.question, G[e.question])[0]['strict']['depth'] for _, e in tst]; gd = [GOLD[i]['strict']['depth'] for i, _ in tst]
    ok = [(a, b) for a, b in zip(md, gd) if a is not None and b is not None]
    M['P1'] = dict(exact=float(np.mean([a == b for a, b in ok])), within1=float(np.mean([abs(a - b) <= 1 for a, b in ok])),
                   model_depth_dist={str(k): int(v) for k, v in sorted(zip(*np.unique([min(a, 6) for a, _ in ok], return_counts=True)))})
    M['P1']['held'] = bool(.40 <= M['P1']['exact'] <= .80)
    rows = []; skipped = defaultdict(int)
    for fn in (f't31/gsm8k_{m}.json', f't31/long_gsm8k_{m}.json'):
        for p in json.load(open(A / fn))['pairs']:
            if not (p.get('qualified') and p.get('R_right') and p.get('D_right') and p.get('hops') is not None): continue
            qr, qd = p['R']['question'], p['D']['question']
            if qr not in G: skipped['no_cot'] += 1; continue
            i = BYQ.get(qtext(qr)); v = edited_number(qr, qd)
            if v is None: skipped['edit_not_found'] += 1; continue
            info, exprs = route(qr, G[qr]); si = info['strict']
            if si['depth'] is None: skipped['no_route'] += 1; continue
            use = [k for k, ex in enumerate(exprs) if any(same(fnum(x), fnum(v)) for x in NUM.findall(ex))]
            ds = [down(si, k) for k in use]; ds = [d for d in ds if d is not None]
            dist = max(ds) if ds else 1
            gk = int(p['edit'][4:]); gdist = down(GOLD[i]['strict'], gk) if i is not None else None
            rows.append(dict(i=i if i is not None else qr, h=int(p['hops']), x=dist, mdepth=si['depth'], gdist=gdist,
                             gdepth=GOLD[i]['strict']['depth'] if i is not None else None, in_written=bool(ds)))
    M['n_pairs'] = len(rows); M['skipped'] = dict(skipped)
    M['P2'] = slope(rows); M['P2']['held'] = bool(M['P2']['slope'] is not None and M['P2']['slope'] < .30)
    M['gold_slope_same_pairs'] = slope([r for r in rows if r['gdist'] is not None], x='gdist')
    sop = defaultdict(list)
    for r in rows:
        if r['x'] == r['mdepth']: sop[r['mdepth']].append(r['h'])
    M['start_of_path'] = {str(d): dict(n=len(v), hops=float(np.mean(v)), ci=ci(v)) for d, v in sorted(sop.items()) if len(v) >= 5}
    a, b = sop.get(2, []), sop.get(3, [])
    M['P3'] = dict(d2=float(np.mean(a)) if a else None, d3=float(np.mean(b)) if b else None, n=[len(a), len(b)])
    M['P3']['held'] = bool(len(a) >= 15 and len(b) >= 15 and np.mean(b) - np.mean(a) >= .50)
    d1 = [r['h'] for r in rows if not r['in_written']]
    M['P4'] = dict(n=len(d1), hops=float(np.mean(d1)) if d1 else None, ci=ci(d1) if len(d1) > 1 else None)
    M['P4']['held'] = bool(d1 and np.mean(d1) >= 1.0)
    bd = defaultdict(list)
    for r in rows: bd[r['x']].append(r['h'])
    M['hops_by_model_distance'] = {str(k): dict(n=len(v), hops=float(np.mean(v))) for k, v in sorted(bd.items())}
    agree = [(r['x'], r['gdist']) for r in rows if r['gdist'] is not None]
    M['P5'] = dict(edit_distance_exact=float(np.mean([a == b for a, b in agree])),
                   crosstab_model_vs_gold={f'{a}|{b}': c for (a, b), c in sorted(Counter(agree).items())})
    print(f"\n== {m}: P0 {M['P0']}\n  P1 {M['P1']}\n  pairs {len(rows)} skipped {dict(skipped)}")
    print('  P2 within-item slope on model-route distance', M['P2'], '| gold-distance slope, same pairs', M['gold_slope_same_pairs'])
    print('  hops by model-route distance', M['hops_by_model_distance'])
    print('  start of path by model-route depth', M['start_of_path'], '| P3', M['P3'])
    print('  P4 (edit not in any written step)', M['P4'], '\n  P5 exact agreement of edit distance', M['P5']['edit_distance_exact'])
json.dump(R, open(OUT_PATH, 'w'), indent=1, default=str)
