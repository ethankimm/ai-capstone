"""t42: GSM8K serial chain depth (offline). Pre-registered in scripts/codi_null_audit/prereg/t_predictions.md (t42).

  ../codi/.venv/bin/python scripts/necessity/gsm8k_depth.py <out json>

Depth: parse the calculator rationale (<<expr=result>> per step). Step j depends on step i < j when a number in j's expression
equals i's result (latest such i). strict: only when that number does not also appear in the question (as a digit string or a
small number word); loose: regardless. Step depth = 1 + max parent depth. Item depth = depth of the last step whose result equals
the gold answer; if no step does, one implicit final step over the sink steps is added (flag 'implicit').
Copy steps (<<50=50>>) inherit the depth of the step they copy. env MODE=strict|loose picks the rule for the downstream analyses.
Uses: Kevin's matched GSM8K CODI / CODI-0 predictions (accuracy, teacher), t39 per-item outputs (released models' accuracy and
readability), t31 / t36 hop pairs (crossings by downstream serial distance from the edited step).
"""
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug

ROOT = Path('/Users/henninglindig/Documents/Penn/CIS5980'); A = ROOT / 'codi_audit_artifacts'; RES = ROOT / 'ai-capstone/results'
OUT = sys.argv[1]; rng = np.random.default_rng(0); NB = 2000
STEP = re.compile(r'<<([^<>=]*)=([^<>]*)>>'); NUM = re.compile(r'\d[\d,]*(?:\.\d+)?')
WORDS = dict(one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10, eleven=11, twelve=12, fifteen=15,
             twenty=20, thirty=30, forty=40, fifty=50, hundred=100, thousand=1000, twice=2, double=2, triple=3, dozen=12, half=2,
             third=3, quarter=4)


def fnum(s):
    try: return float(str(s).replace(',', '').replace('$', '').replace('%', '').strip())
    except ValueError: return None


def same(a, b): return a is not None and b is not None and abs(a - b) <= 1e-6 * max(1.0, abs(a))


def depth_of(question, rationale, answer):
    qn = [fnum(x) for x in NUM.findall(question)] + [float(v) for w, v in WORDS.items() if re.search(rf'\b{w}', question.lower())]
    st = [(expr, fnum(res)) for expr, res in STEP.findall(rationale)]
    out = {}
    for mode in ('strict', 'loose'):
        d, par = [], []
        for j, (expr, res) in enumerate(st):
            if fnum(expr) is not None:  # copy step (<<50=50>>): inherits the depth of the step it copies, no new computation
                i = next((i for i in range(j - 1, -1, -1) if same(st[i][1], res)), None)
                par.append({i} if i is not None else set()); d.append(d[i] if i is not None else 1); continue
            ps = set()
            for x in NUM.findall(expr):
                v = fnum(x)
                if mode == 'strict' and any(same(v, q) for q in qn): continue
                i = next((i for i in range(j - 1, -1, -1) if same(st[i][1], v)), None)
                if i is not None: ps.add(i)
            par.append(ps); d.append(1 + max((d[i] for i in ps), default=0))
        ga = fnum(answer); fin = next((j for j in range(len(st) - 1, -1, -1) if same(st[j][1], ga)), None)
        if not st: out[mode] = dict(depth=None, step_depth=[], implicit=False, parents=[], final=None); continue
        if fin is None:
            used = set().union(*par); sinks = [j for j in range(len(st)) if j not in used]
            item = 1 + max(d[j] for j in sinks); implicit = True
        else: item = d[fin]; implicit = False
        out[mode] = dict(depth=item, step_depth=d, implicit=implicit, parents=[sorted(p) for p in par], final=fin)
    out['n_steps'] = len(st)
    return out


def down(info, k):
    """Longest serial path, in steps, from step k to the answer (k and the answer step inclusive; implicit final counts as one)."""
    par, fin, n = info['parents'], info['final'], len(info['parents'])
    ch = defaultdict(list)
    for j, ps in enumerate(par):
        for i in ps: ch[i].append(j)
    memo = {}
    def L(i):  # longest path from i to the target
        if i in memo: return memo[i]
        if fin is not None and i == fin: memo[i] = 1; return 1
        if not ch[i]: memo[i] = (2 if fin is None else None); return memo[i]  # sink: reaches only the implicit final step
        sub = [v for v in (L(j) for j in ch[i]) if v is not None]
        memo[i] = 1 + max(sub) if sub else None; return memo[i]
    return L(k) if k < n else None


def boot_mean(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 5: return dict(mean=float(np.mean(x)) if len(x) else None, n=int(len(x)), ci=None)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(NB)]
    return dict(mean=float(x.mean()), n=int(len(x)), ci=[float(v) for v in np.percentile(bs, [2.5, 97.5])])


def boot_gain(c0, c1):
    c0, c1 = np.asarray(c0, float), np.asarray(c1, float); n = len(c0)
    if n < 5: return dict(gain=None, n=n)
    idx = [rng.integers(0, n, n) for _ in range(NB)]; bs = [(c1[i] - c0[i]).mean() for i in idx]
    return dict(n=n, acc0=float(c0.mean()), acc1=float(c1.mean()), gain=float((c1 - c0).mean()),
                ci=[float(v) for v in np.percentile(bs, [2.5, 97.5])], only1=int(((c1 == 1) & (c0 == 0)).sum()), only0=int(((c0 == 1) & (c1 == 0)).sum()))


qtext = lambda q: ' '.join(q.split())
MODE = os.environ.get('MODE', 'strict')  # which depth rule drives sections B-E
te = load_gsm8k_aug(split='test', n=None, seed=None)
INFO = [depth_of(e.question, e.rationale, e.answer) for e in te]
BYQ = {qtext(e.question): i for i, e in enumerate(te)}
dep = np.array([np.nan if x[MODE]['depth'] is None else x[MODE]['depth'] for x in INFO])
depL = np.array([np.nan if x['loose']['depth'] is None else x['loose']['depth'] for x in INFO])
nst = np.array([x['n_steps'] for x in INFO]); imp = np.array([x['strict']['implicit'] for x in INFO])
dbin = lambda d: None if not np.isfinite(d) else ('1' if d <= 1 else '2' if d == 2 else '3' if d == 3 else '4+')
sbin = lambda n: str(n) if n <= 4 else '5+'
R = dict(mode=MODE, n=len(te), implicit=int(imp.sum()))

# ---- A. depth vs step count
tab = defaultdict(lambda: defaultdict(int)); tabL = defaultdict(lambda: defaultdict(int))
for n, d, dl in zip(nst, dep, depL):
    tab[sbin(n)][dbin(d)] += 1; tabL[sbin(n)][dbin(dl)] += 1
R['depth_by_steps'] = {s: dict(v) for s, v in sorted(tab.items())}; R['depth_by_steps_loose'] = {s: dict(v) for s, v in sorted(tabL.items())}
R['median_depth_by_steps'] = {s: float(np.nanmedian(dep[[sbin(n) == s for n in nst]])) for s in sorted(tab)}
R['agree_strict_loose'] = float(np.mean(dep[np.isfinite(dep)] == depL[np.isfinite(dep)]))
print('items', len(te), 'implicit final', int(imp.sum()), '| strict = loose on', round(R['agree_strict_loose'], 3))
for s in sorted(tab): print(f'  steps {s:3s} n {sum(tab[s].values()):4d}  depth', dict(sorted(tab[s].items(), key=str)), ' median', R['median_depth_by_steps'][s])

# ---- B. accuracy: Kevin's matched pair + teacher
def load_preds(prefix): return {r['question']: r for r in map(json.loads, open(next(RES.glob(prefix + '*')) / 'predictions.jsonl'))}
c0, c1 = load_preds('20261103-225715'), load_preds('20261104-012327')
rows = []
for q, r in c1.items():
    i = BYQ.get(qtext(q))
    if i is None: continue
    rows.append(dict(i=i, d=dep[i], n=nst[i], c0=float(c0[q]['correct']), c1=float(r['correct']), t=r['teacher_correct']))
print(f'Kevin pair joined {len(rows)} / {len(c1)}')
R['acc_pair'] = {}; R['acc_pair_teacher_right'] = {}; R['teacher_by_depth'] = {}; R['acc_pair_by_steps_depth'] = {}
for b in ('1', '2', '3', '4+'):
    S = [x for x in rows if dbin(x['d']) == b]
    R['acc_pair'][b] = boot_gain([x['c0'] for x in S], [x['c1'] for x in S])
    T = [x for x in S if x['t']]; R['acc_pair_teacher_right'][b] = boot_gain([x['c0'] for x in T], [x['c1'] for x in T])
    tt = [x['t'] for x in S if x['t'] is not None]; R['teacher_by_depth'][b] = dict(n=len(tt), acc=float(np.mean(tt)) if tt else None)
for s in ('3', '4', '5+'):
    for b in ('1', '2', '3', '4+'):
        S = [x for x in rows if sbin(x['n']) == s and dbin(x['d']) == b]
        if len(S) >= 20: R['acc_pair_by_steps_depth'][f'{s}|{b}'] = boot_gain([x['c0'] for x in S], [x['c1'] for x in S])
print('CODI vs CODI-0 by depth:')
for b, g in R['acc_pair'].items():
    if g.get('gain') is None: continue
    t = R['acc_pair_teacher_right'][b]
    print(f"  d {b:3s} n {g['n']:4d}  CODI-0 {g['acc0']:.3f} CODI {g['acc1']:.3f} gain {g['gain']:+.3f} [{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}] "
          f"({g['only1']}/{g['only0']}) | teacher {R['teacher_by_depth'][b]['acc']:.3f} | teacher-right gain {t['gain']:+.3f} (n {t['n']})")
print('  by steps|depth:', {k: (v['n'], round(v['gain'], 3)) for k, v in R['acc_pair_by_steps_depth'].items()})

# ---- C/D/E. released models: accuracy, readability, position (t39); crossings (t31 / t36)
HOPF = dict(codi=['t31/gsm8k_codi.json', 't31/long_gsm8k_codi.json'], sim=['t31/gsm8k_sim.json', 't31/long_gsm8k_sim.json'],
            coconut=['t31/gsm8k_coconut.json', 't31/long_gsm8k_coconut.json'], simcoconut=['t36/gsm8k_simcoconut.json', 't36/long_gsm8k_simcoconut.json'])


def bacc(y, p):
    y, p = np.asarray(y) > 0, np.asarray(p) > 0
    if y.all() or (~y).all() or len(y) < 10: return float('nan')
    return float(((p & y).sum() / y.sum() + (~p & ~y).sum() / (~y).sum()) / 2)


R['models'] = {}
for m in ('codi', 'sim', 'coconut', 'simcoconut'):
    M = R['models'][m] = {}
    T = json.load(open(A / f't39/gsm8k_{m}.json')); it = T['items']; y = np.array(it['y']); P = {k: np.array(v) for k, v in it['pred'].items()}
    assert len(y) == len(te) and it['idx'] == [e.idx for e in te]
    right = (y[:, 6] == -1).astype(float)
    M['acc_by_depth'] = {b: boot_mean(right[[dbin(d) == b for d in dep]]) for b in ('1', '2', '3', '4+')}
    M['acc_by_steps_depth'] = {f'{s}|{b}': boot_mean(right[[(sbin(n) == s and dbin(d) == b) for n, d in zip(nst, dep)]])
                               for s in ('3', '4', '5+') for b in ('1', '2', '3', '4+')}
    # crossings, per pair, by serial distance from the edited step to the answer
    pr = []
    for f in HOPF[m]:
        for p in json.load(open(A / f))['pairs']:
            if not p.get('qualified') or p.get('hops') is None: continue
            i = BYQ.get(qtext(p['R']['question']))
            if i is None: continue
            k = int(p['edit'][4:]); sd = down(INFO[i][MODE], k)
            pr.append(dict(h=float(p['hops']), s=int(p['s']), sd=sd, d=dep[i], i=i))
    M['n_pairs'] = len(pr); M['pairs_off_path'] = sum(x['sd'] is None for x in pr)
    M['hops_by_s'] = {str(s): boot_mean([x['h'] for x in pr if x['s'] == s]) for s in sorted({x['s'] for x in pr})}
    M['hops_by_serial_s'] = {str(s): boot_mean([x['h'] for x in pr if x['sd'] == s]) for s in sorted({x['sd'] for x in pr if x['sd'] is not None})}
    M['hops_by_depth'] = {b: boot_mean([x['h'] for x in pr if dbin(x['d']) == b]) for b in ('1', '2', '3', '4+')}
    ok = [x for x in pr if x['sd'] is not None]
    M['hops_slope_on_serial_s'] = float(np.polyfit([x['sd'] for x in ok], [x['h'] for x in ok], 1)[0]) if len(ok) > 10 else None
    M['hops_slope_on_s'] = float(np.polyfit([x['s'] for x in pr], [x['h'] for x in pr], 1)[0])
    # readability: per-item gain of the best latent set over QA, target op_k / big_k by the depth of step k
    cell = T['cells']; G = {}
    for t in ('op_2', 'op_3', 'big_2', 'big_3'):
        c = T['targets'].index(t); k = int(t[-1]) - 1; best = max(('QAZ', 'QAH'), key=lambda s: cell[f'{s}|{t}'])
        g = defaultdict(list)
        for j in range(len(te)):
            if y[j, c] == 0 or len(INFO[j][MODE]['step_depth']) <= k: continue
            dk = INFO[j][MODE]['step_depth'][k]
            g[str(min(dk, 3))].append(float(P[best][j, c] == y[j, c]) - float(P['QA'][j, c] == y[j, c]))
        G[t] = {b: boot_mean(v) for b, v in sorted(g.items())}
        # position: best single pass by whether step k is serial (depth >= 2) or not
        pos = {}
        for lab, sel in (('indep', lambda dk: dk == 1), ('serial', lambda dk: dk >= 2)):
            J = [j for j in range(len(te)) if y[j, c] != 0 and len(INFO[j][MODE]['step_depth']) > k and sel(INFO[j][MODE]['step_depth'][k])]
            sc = {h: bacc(y[J, c], P[h][J, c]) for h in P if h[0] == 'H' and h[1:].isdigit()}
            pos[lab] = dict(n=len(J), best=max(sc, key=lambda h: -1 if np.isnan(sc[h]) else sc[h]), scores=sc)
        G[t]['position'] = pos
    M['readability_by_step_depth'] = G
    a = M['acc_by_depth']; hs = M['hops_by_serial_s']
    print(f"\n{m}: acc by depth " + ' '.join(f"{b}:{v['mean']:.3f}(n{v['n']})" for b, v in a.items() if v['mean'] is not None)
          + f" | pairs {len(pr)} (off path {M['pairs_off_path']})")
    print('  acc steps|depth', {k: (v['n'], round(v['mean'], 3)) for k, v in M['acc_by_steps_depth'].items() if v['n'] >= 20})
    print('  hops by index s      ', {s: (v['n'], round(v['mean'], 2)) for s, v in M['hops_by_s'].items()}, 'slope', round(M['hops_slope_on_s'], 3))
    print('  hops by serial dist  ', {s: (v['n'], round(v['mean'], 2)) for s, v in hs.items()}, 'slope', None if M['hops_slope_on_serial_s'] is None else round(M['hops_slope_on_serial_s'], 3))
    print('  hops by item depth   ', {b: (v['n'], None if v['mean'] is None else round(v['mean'], 2)) for b, v in M['hops_by_depth'].items()})
    for t, v in G.items():
        print(f'  {t} gain by depth of step: ' + ' '.join(f"{b}:{w['mean']:+.3f}(n{w['n']})" for b, w in v.items() if b != 'position')
              + f" | best H indep {v['position']['indep']['best']} (n{v['position']['indep']['n']}) serial {v['position']['serial']['best']} (n{v['position']['serial']['n']})")
json.dump(R, open(OUT, 'w'), indent=1, default=float)
