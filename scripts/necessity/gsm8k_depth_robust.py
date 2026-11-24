"""t52: robustness of the GSM8K depth results to the depth rule (offline). Pre-registered in prereg/t_predictions.md (t52).

  ../codi/.venv/bin/python scripts/necessity/gsm8k_depth_robust.py <out json>

Depth rules, all on the gold calculator rationale (<<expr=result>> per step; copy steps inherit, as gsm8k_depth.py):
  strict   t42: a number in step j's expression links j to the latest earlier step with that result, unless the number is also in the
           question (digits or a small number word). This is the rule used in t42-t50.
  loose    links regardless of the question.
  resolved loose, but a number that is also in the question links to the earlier step only when that step's result is otherwise
           unused under strict (a dangling intermediate); otherwise it is read as the question's number.
  clean    strict on the items where no number is ambiguous (in the question and an earlier result), the answer is some step's result,
           and every operand is explained (question number, its percent form, an earlier result, or a common constant); the rules
           coincide there.
  ops      exploratory: arithmetic operations instead of steps. A step's expression is parsed; a leaf's distance is the number of
           binary operations above it; on the resolved links, a path's length is the sum of the entering leaves' distances.
Measures redone per rule: accuracy by item depth (5 CODI-family checkpoints), correct start-of-path crossings and steps per pass
(edit at the start of the item's longest path, s = D), the within-item slope, READ=h readability on clean items, and the t51 layer
lens on clean items. Hop pairs, readability and layer-lens files are the ones of t31 / t47 / t48 / t49 / t50 / t51.
"""
import ast
import json
import re
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

OUT_PATH = sys.argv[1]
sys.argv = [sys.argv[0], '/dev/null']
exec(open(Path(__file__).with_name('gsm8k_depth.py')).read().split('qtext = lambda')[0])  # depth_of, down, fnum, same, STEP, NUM, WORDS, A
qtext = lambda q: ' '.join(q.split())
te = load_gsm8k_aug(split='test', n=None, seed=None)
BYQ = {qtext(e.question): i for i, e in enumerate(te)}
CONST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 24, 30, 52, 60, 100, 1000, 365, 0.5, 0.25, 0.1, 0.01]
rng = np.random.default_rng(0)


def qnums(q):
    return [fnum(x) for x in NUM.findall(q)] + [float(v) for w, v in WORDS.items() if re.search(rf'\b{w}', q.lower())]


def leaves(expr):
    """[(value, number of binary operations above the leaf)] or None if the expression does not parse."""
    s = re.sub(r'(?<![\d.])0+(\d)', r'\1', expr.replace(',', '').replace('$', '').replace('%', '').replace('x', '*').replace('×', '*'))
    try: tree = ast.parse(s.strip(), mode='eval').body
    except SyntaxError: return None
    out = []
    def walk(n, d):
        if isinstance(n, ast.BinOp): walk(n.left, d + 1); walk(n.right, d + 1)
        elif isinstance(n, ast.UnaryOp): walk(n.operand, d)
        elif isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): out.append((float(n.value), d))
        else: raise ValueError
    try: walk(tree, 0)
    except ValueError: return None
    return out


def analyse(question, rationale, answer):
    qn = qnums(question); st = [(e, fnum(r)) for e, r in STEP.findall(rationale)]
    if not st: return None
    ga = fnum(answer); fin = next((j for j in range(len(st) - 1, -1, -1) if same(st[j][1], ga)), None)
    inq = lambda v: any(same(v, q) for q in qn)
    explained = lambda v: inq(v) or any(same(v * 100, q) or same(v / 100, q) for q in qn if q) or any(same(v, c) for c in CONST)
    ops, copies, amb, unexp = [], [], False, False  # ops[j] = [(value, parent or None, in question)]
    for j, (expr, res) in enumerate(st):
        latest = lambda v: next((i for i in range(j - 1, -1, -1) if same(st[i][1], v)), None)
        if fnum(expr) is not None: copies.append(latest(res)); ops.append([]); continue
        copies.append('no')
        row = []
        for x in NUM.findall(expr):
            v = fnum(x); i = latest(v); q = inq(v)
            if i is not None and q: amb = True
            if i is None and not explained(v): unexp = True
            row.append((v, i, q))
        ops.append(row)
    strict_used = {i for row in ops for v, i, q in row if i is not None and not q} | {c for c in copies if c not in ('no', None)}
    info = {}
    for rule in ('strict', 'loose', 'resolved'):
        par, d = [], []
        for j in range(len(st)):
            if copies[j] != 'no':
                i = copies[j]; par.append({i} if i is not None else set()); d.append(d[i] if i is not None else 1); continue
            ps = set()
            for v, i, q in ops[j]:
                if i is None: continue
                if rule == 'loose' or not q or (rule == 'resolved' and i not in strict_used and i != fin): ps.add(i)
            par.append(ps); d.append(1 + max((d[i] for i in ps), default=0))
        if fin is None:
            used = set().union(*par); item = 1 + max(d[j] for j in range(len(st)) if j not in used)
        else: item = d[fin]
        info[rule] = dict(depth=item, step_depth=d, implicit=fin is None, parents=[sorted(p) for p in par], final=fin)
    info['clean'] = not amb and not unexp and fin is not None
    info['ambiguous'], info['unexplained'] = amb, unexp
    # ops on the resolved links: o[j] = longest operation count from any question leaf to step j's result
    L = [leaves(e) if copies[j] == 'no' else [] for j, (e, _) in enumerate(st)]; par = info['resolved']['parents']
    if all(l is not None for l in L) and fin is not None:
        o = []
        for j in range(len(st)):
            if copies[j] != 'no': o.append(o[copies[j]] if copies[j] is not None else 0); continue
            best = 0
            for v, dist in L[j]:
                p = next((i for i in par[j] if same(st[i][1], v)), None)
                best = max(best, dist + (o[p] if p is not None else 0))
            o.append(best)
        info['ops'] = dict(step_ops=o, depth=o[fin])
    else: info['ops'] = None
    info['_st'], info['_L'] = st, L
    return info


def ops_from_edit(info, vR):
    """operations from the edited question number (value vR) to the answer, on the resolved links; None if off path."""
    if info is None or info['ops'] is None: return None
    st, L, par, fin = info['_st'], info['_L'], info['resolved']['parents'], info['resolved']['final']
    E = []
    for j in range(len(st)):
        if not L[j]:  # copy step
            p = par[j][0] if par[j] else None; E.append(E[p] if p is not None else None); continue
        best = None
        for v, dist in L[j]:
            p = next((i for i in par[j] if same(st[i][1], v)), None)
            if p is not None and E[p] is not None: best = max(best or 0, E[p] + dist)
            elif p is None and same(v, vR): best = max(best or 0, dist)
        E.append(best)
    return E[fin]


INFO = [analyse(e.question, e.rationale, e.answer) for e in te]
OK = [x is not None for x in INFO]
R = dict(n_items=len(te), n_parsed=sum(OK), n_clean=sum(1 for x in INFO if x and x['clean']),
         n_ambiguous=sum(1 for x in INFO if x and x['ambiguous']), n_unexplained=sum(1 for x in INFO if x and x['unexplained']),
         n_implicit=sum(1 for x in INFO if x and x['strict']['implicit']), n_ops=sum(1 for x in INFO if x and x['ops']))
dbin = lambda d: None if d is None else ('1' if d <= 1 else '4+' if d >= 4 else str(d))
RULES = ('strict', 'loose', 'resolved', 'clean')
depth = lambda x, r: None if x is None or (r == 'clean' and not x['clean']) else x['strict' if r == 'clean' else r]['depth']
R['depth_dist'] = {r: {b: sum(1 for x in INFO if dbin(depth(x, r)) == b) for b in ('1', '2', '3', '4+')} for r in RULES}
R['agree_with_strict'] = {r: float(np.mean([depth(x, r) == x['strict']['depth'] for x in INFO if x])) for r in ('loose', 'resolved')}
print('items', R['n_items'], 'parsed', R['n_parsed'], '| clean', R['n_clean'], 'ambiguous', R['n_ambiguous'], 'unexplained operand',
      R['n_unexplained'], 'implicit', R['n_implicit'], '| item depth = strict:', {k: round(v, 3) for k, v in R['agree_with_strict'].items()})
for r in RULES: print(f'  depth distribution {r:8s}', R['depth_dist'][r])
ops_by_d = defaultdict(list)
for x in INFO:
    if x and x['ops'] and x['clean']: ops_by_d[dbin(x['strict']['depth'])].append(x['ops']['depth'])
R['ops_per_item_depth_clean'] = {b: dict(n=len(v), mean=float(np.mean(v))) for b, v in sorted(ops_by_d.items())}
print('  clean items, operations on the longest path by step depth:', {b: (v['n'], round(v['mean'], 2)) for b, v in R['ops_per_item_depth_clean'].items()})


def boot(v):
    v = np.asarray(v, float)
    if len(v) < 5: return dict(n=int(len(v)), mean=float(v.mean()) if len(v) else None, ci=None)
    return dict(n=int(len(v)), mean=float(v.mean()), ci=[float(x) for x in np.percentile([rng.choice(v, len(v)).mean() for _ in range(2000)], [2.5, 97.5])])


# ---- accuracy by item depth
right = {}
for m in ('codi', 'sim', 'coconut', 'simcoconut'):
    it = json.load(open(A / f't39/gsm8k_{m}.json'))['items']; assert it['idx'] == [e.idx for e in te]
    right[m] = (np.array(it['y'])[:, 6] == -1).astype(float)
POS = {e.idx: i for i, e in enumerate(te)}
for m, f in (('llama', 't47/gsm8k_llama_acc.json'), ('simllama1b', 't49/gsm8k_simllama1b_acc.json'), ('simllama3b', 't50/gsm8k_simllama3b_acc.json')):
    v = np.full(len(te), np.nan)
    for x in json.load(open(A / f))['items']: v[POS[x['idx']]] = float(x['right'])
    right[m] = v
MODELS = ('codi', 'sim', 'llama', 'simllama1b', 'simllama3b', 'coconut', 'simcoconut')  # t52b added the two GPT-2 Coconut checkpoints
NAME = dict(codi='GPT-2 CODI', sim='GPT-2 SIM-CODI', llama='Llama CODI 1B', simllama1b='SIM-CODI 1B', simllama3b='SIM-CODI 3B',
            coconut='GPT-2 Coconut', simcoconut='GPT-2 SIM-Coconut')
R['acc_by_depth'] = {}
print('\naccuracy by item depth (1 / 2 / 3 / 4+):')
for m in MODELS:
    R['acc_by_depth'][m] = {}
    for r in RULES:
        R['acc_by_depth'][m][r] = {b: boot([right[m][i] for i, x in enumerate(INFO) if dbin(depth(x, r)) == b and np.isfinite(right[m][i])])
                                   for b in ('1', '2', '3', '4+')}
    print(f'  {NAME[m]:15s}', ' | '.join(f"{r} " + ' '.join(f"{c['mean']:.2f}" for c in R['acc_by_depth'][m][r].values()) for r in RULES))

# ---- hop count: correct start-of-path crossings, steps per pass, within-item slope
HOPF = dict(codi=['t31/gsm8k_codi.json', 't31/long_gsm8k_codi.json'], sim=['t31/gsm8k_sim.json', 't31/long_gsm8k_sim.json'],
            llama=['t47/gsm8k_llama.json', 't47/long_gsm8k_llama.json'], simllama1b=['t49/gsm8k_simllama1b.json', 't49/long_gsm8k_simllama1b.json'],
            simllama3b=['t50/gsm8k_simllama3b.json', 't50/long_gsm8k_simllama3b.json'],
            coconut=['t31/gsm8k_coconut.json', 't31/long_gsm8k_coconut.json'], simcoconut=['t36/gsm8k_simcoconut.json', 't36/long_gsm8k_simcoconut.json'])


def within_item_slope(rows):
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
    bs = [slope([items[j] for j in rng.integers(0, len(items), len(items))]) for _ in range(1000)]
    return dict(n_items=len(items), slope=slope(items), ci=[float(x) for x in np.nanpercentile(bs, [2.5, 97.5])])


R['hops'] = {}
print('\ncorrect start-of-path crossings at depth 2 / 3 / 4 (steps per pass in brackets):')
for m in MODELS:
    rows = []
    for f in HOPF[m]:
        for p in json.load(open(A / f))['pairs']:
            if not p.get('qualified') or p.get('hops') is None: continue
            i = BYQ.get(qtext(p['R']['question']))
            if i is None or INFO[i] is None: continue
            ok = same(fnum(p['aR']), fnum(p['R'].get('answer'))) and same(fnum(p['aD']), fnum(p['D'].get('answer')))
            k = int(p['edit'][4:]); x = INFO[i]
            na, nb = NUM.findall(p['R']['question']), NUM.findall(p['D']['question']); diff = [a for a, b in zip(na, nb) if a != b]
            row = dict(i=i, h=int(p['hops']), ok=bool(ok), clean=x['clean'],
                       ops=ops_from_edit(x, fnum(diff[0])) if len(na) == len(nb) and len(diff) == 1 else None)
            for r in ('strict', 'loose', 'resolved'):
                row['s_' + r] = down(x[r], k) if k < len(x[r]['parents']) else None; row['D_' + r] = x[r]['depth']
            rows.append(row)
    M = R['hops'][m] = dict(n_pairs=len(rows), start_of_path={}, within_item={})
    for r in RULES:
        rr = 'strict' if r == 'clean' else r; sel = [w for w in rows if (w['clean'] or r != 'clean') and w['s_' + rr] is not None]
        M['start_of_path'][r] = {}
        for D in (2, 3, 4):
            g = [w for w in sel if w['ok'] and w['D_' + rr] == D and w['s_' + rr] == D]
            if len(g) < 15: M['start_of_path'][r][str(D)] = dict(n=len(g)); continue
            h = np.array([w['h'] for w in g]); c = boot(h); spp = boot(D / (h + 1))
            cell = dict(n=len(g), hops=c['mean'], hops_ci=c['ci'], steps_per_pass=spp['mean'], spp_ci=spp['ci'])
            o = [(w['ops'], w['h']) for w in g if w['ops']]
            if r in ('resolved', 'clean') and len(o) >= 15:
                cell.update(n_ops=len(o), ops_on_path=float(np.mean([a for a, _ in o])), ops_per_pass=float(np.mean([a / (b + 1) for a, b in o])))
            M['start_of_path'][r][str(D)] = cell
        M['within_item'][r] = within_item_slope([dict(i=w['i'], s=w['s_' + rr], h=w['h']) for w in sel])
    fmt = lambda c: f"{c['hops']:.2f} ({c['steps_per_pass']:.2f}) n{c['n']}" if 'hops' in c else f"n{c['n']}<15"
    for r in RULES:
        sp = M['start_of_path'][r]; wi = M['within_item'][r]
        print(f'  {NAME[m]:15s} {r:8s}', ' / '.join(fmt(sp[D]) for D in ('2', '3', '4')),
              f"| slope {wi['slope']:+.3f} [{wi['ci'][0]:+.3f}, {wi['ci'][1]:+.3f}]" if wi.get('slope') is not None else '')
    for r in ('resolved', 'clean'):
        sp = M['start_of_path'][r]
        print(f'  {"":15s} ops {r:8s}', ' / '.join(f"{sp[D]['ops_on_path']:.2f} ops, {sp[D]['ops_per_pass']:.2f}/pass n{sp[D]['n_ops']}" if 'ops_per_pass' in sp[D] else '-'
                                                for D in ('2', '3', '4')))

# ---- READ=h readability on clean items
LENS = dict(codi=('t48/gsm8k_codi_lens_h.json', [A / 't26/gsm8k_codi.json', A / 't26/long_gsm8k_codi.json'], False),
            llama=('t48/gsm8k_llama_lens_h.json', [A / 't47/gsm8k_llama.json', A / 't47/long_gsm8k_llama.json'], True),
            simllama1b=('t49/gsm8k_simllama1b_lens_h.json', [A / 't49/gsm8k_simllama1b.json', A / 't49/long_gsm8k_simllama1b.json'], True),
            simllama3b=('t50/gsm8k_simllama3b_lens_h.json', [A / 't50/gsm8k_simllama3b.json', A / 't50/long_gsm8k_simllama3b.json'], True))
R['readability'] = {}
print('\nREAD=h, own-readable values readable at refresh d - 1 (d = 2 / 3), all items vs clean items:')
for m, (lf, pf, own) in LENS.items():
    pairs = [p for f in pf for p in json.load(open(f))['pairs'] if p['qualified'] and (not own or (p.get('R_right') and p.get('D_right')))]
    L = json.load(open(A / lf)); its = L['items']; K = len(its[0]['B']) - 1
    out = {}
    for lab, keep in (('all', lambda it: True), ('clean', lambda it: (lambda i: i is not None and INFO[i] is not None and INFO[i]['clean'])(BYQ.get(qtext(pairs[it['pair']]['R']['question']))))):
        out[lab] = {}
        for d in (2, 3, 4):
            g = [it for it in its if it['depth'] == d and keep(it)]; own_r = [it for it in g if it['B'][K]]
            out[lab][str(d)] = dict(n=len(g), n_own=len(own_r), B=[float(np.mean([it['B'][n] for it in own_r])) if own_r else None for n in range(K + 1)])
    R['readability'][m] = out
    chk = {d: L['summary'][d]['B'][int(d) - 1] for d in ('2', '3') if d in L['summary']}
    print(f'  {NAME[m]:15s}', ' | '.join(f"{lab}: " + ' / '.join(f"{out[lab][d]['B'][int(d) - 1]:.2f} (n{out[lab][d]['n_own']})" for d in ('2', '3')) for lab in out),
          '| file summary', {d: round(v, 2) for d, v in chk.items()})

# ---- t51 layer lens on clean items
R['layer_lens'] = {}; pool = [0, 0]
print('\nt51 layer lens, all items vs clean items: depth-2 share with s* = pass 0 | depth-1 s* = 0 median onset | same-pass child later / earlier')
for m in [m for m in MODELS if (A / f't51/layer_lens_gsm8k_{m}.json').exists()]:
    d = json.load(open(A / f't51/layer_lens_gsm8k_{m}.json')); Ld = d['layers']; recs = d['records']; out = {}
    for lab, keep in (('all', lambda r: True), ('clean', lambda r: INFO[r['item']] is not None and INFO[r['item']]['clean'])):
        rs = [r for r in recs if keep(r)]; by = {(r['item'], r['k']): r for r in rs}
        d2 = [r for r in rs if r['depth'] == 2 and r['s_star'] is not None]
        on = [r['onset'] for r in rs if r['depth'] == 1 and r['s_star'] == 0]
        gt = lt = tie = 0
        for r in rs:
            p = by.get((r['item'], r['parent'])) if r['parent'] is not None else None
            if p and r['s_star'] is not None and r['s_star'] == p['s_star']:
                gt += r['onset'] > p['onset']; lt += r['onset'] < p['onset']; tie += r['onset'] == p['onset']
        out[lab] = dict(n_records=len(rs), d2_share_s0=float(np.mean([r['s_star'] == 0 for r in d2])) if d2 else None, n_d2=len(d2),
                        d1_s0_median_onset=float(np.median(on)) if on else None, layers_left=Ld - float(np.median(on)) if on else None,
                        same_pass=dict(later=gt, earlier=lt, tie=tie))
        if lab == 'clean': pool[0] += gt; pool[1] += lt
    R['layer_lens'][m] = out
    print(f'  {NAME[m]:15s} L={Ld}', ' | '.join(f"{lab}: {o['d2_share_s0']:.2f} (n{o['n_d2']}), onset {o['d1_s0_median_onset']:.0f}, {o['same_pass']['later']}/{o['same_pass']['earlier']}" for lab, o in out.items()))
g, l = pool
R['layer_lens_clean_pooled'] = dict(later=g, earlier=l, sign_p_one_sided=sum(comb(g + l, j) for j in range(g, g + l + 1)) / 2 ** (g + l) if g + l else None)
print('  clean pooled same-pass:', R['layer_lens_clean_pooled'])
json.dump(R, open(OUT_PATH, "w"), indent=1)
print("\nwrote", OUT_PATH)
