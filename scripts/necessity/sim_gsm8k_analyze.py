"""Analysis for t22 / t23 (sim_gsm8k.py outputs). Usage: uv run python scripts/necessity/sim_gsm8k_analyze.py <t22 dir> [sweep|route|all]
Reads t22_sweep_{sim,codi}.json and t22_route_{sim,codi}.json; prints the tables the pre-registration asks for and writes
t22_summary.json next to them."""
import json, re, sys, collections
from pathlib import Path
from transformers import AutoTokenizer

D = Path(sys.argv[1]); WHAT = sys.argv[2] if len(sys.argv) > 2 else 'all'
TOK = AutoTokenizer.from_pretrained('gpt2')
STEP = re.compile(r'<<([^=<>]*)=([^<>]*)>>')
LAYERS = (4, 6, 8, 10, 12)


def num(s):
    try: return float(str(s).replace(',', '').strip())
    except Exception: return None


def same(a, b):
    x, y = num(a), num(b)
    return x is not None and y is not None and abs(x - y) < 1e-6


def result(dec):
    m = STEP.search(dec or '')
    return m.group(2).strip() if m else None


def nbin(n): return min(n, 5)


def load(kind, model):
    p = D / f't22_{kind}_{model}.json'
    return json.load(open(p)) if p.exists() else None


out = {}
if WHAT in ('sweep', 'all'):
    for model in ('sim', 'codi'):
        d = load('sweep', model)
        if d is None: print(f'[no sweep for {model}]'); continue
        it = d['items']; o = out.setdefault(model, {})
        acc = sum(x['correct'] for x in it) / len(it); o['acc'] = acc
        print(f'\n== {model}: test accuracy {acc:.3f} (n={len(it)})')
        print('accuracy after k passes, all items: ' + ' '.join(f'k{k} {sum(x["sweep_correct"][k] for x in it) / len(it):.3f}' for k in range(7)))
        by = collections.defaultdict(list)
        for x in it: by[nbin(len(x['values']))].append(x)
        o['sweep_by_n'] = {}
        for n in sorted(by):
            xs = by[n]; right = [x for x in xs if x['correct']]
            row = [sum(x['sweep_correct'][k] for x in xs) / len(xs) for k in range(7)]
            # on items right at k=6: share also right at k
            rr = [sum(x['sweep_correct'][k] for x in right) / max(len(right), 1) for k in range(7)]
            settle = [min(k for k in range(7) if all(x['sweep_correct'][j] for j in range(k, 7))) for x in right]
            o['sweep_by_n'][n] = dict(n=len(xs), acc_by_k=row, right_kept_by_k=rr, mean_settle=sum(settle) / max(len(settle), 1))
            print(f'  n={n}{"+" if n == 5 else ""} items {len(xs):4d} acc by k ' + ' '.join(f'{v:.2f}' for v in row)
                  + f' | right items kept by k ' + ' '.join(f'{v:.2f}' for v in rr) + f' | mean settling pass {o["sweep_by_n"][n]["mean_settle"]:.2f}')
        # decoder reads: z_{k-1} -> step k result, for k <= n-1 (shown steps)
        print('  step decoder: z_{k-1} decodes the gold result of step k (right items | wrong items), steps with a target (k <= n-1):')
        o['decoder'] = {}
        for k in range(1, 7):
            for tag, cond in (('right', True), ('wrong', False)):
                xs = [x for x in it if x['correct'] == cond and len(x['values']) - 1 >= k]
                if not xs: continue
                hit = sum(same(result(x['decoded'][k - 1]), x['values'][k - 1]) for x in xs) / len(xs)
                o['decoder'][f'{tag}_k{k}'] = (hit, len(xs))
            r, w = o['decoder'].get(f'right_k{k}'), o['decoder'].get(f'wrong_k{k}')
            if r: print(f'    k={k}: right {r[0]:.2f} (n={r[1]})' + (f' | wrong {w[0]:.2f} (n={w[1]})' if w else ''))
        # error visible in the trace: any targeted step decoded wrong
        def trace_off(x):
            ks = range(1, min(len(x['values']), 7))
            return any(not same(result(x['decoded'][k - 1]), x['values'][k - 1]) for k in ks) if len(ks) else None
        for tag, cond in (('right', True), ('wrong', False)):
            xs = [x for x in it if x['correct'] == cond and len(x['values']) >= 2]
            v = sum(bool(trace_off(x)) for x in xs) / max(len(xs), 1); o[f'trace_off_{tag}'] = (v, len(xs))
            print(f'  trace has a wrong intermediate ({tag} items, n>=2): {v:.2f} (n={len(xs)})')
        # own-error: does the model's wrong answer equal some decoded result? vs another item's decoded results
        wr = [x for x in it if not x['correct'] and x['ans'] is not None]
        own = sum(any(same(result(dz), x['ans']) for dz in x['decoded']) for x in wr) / max(len(wr), 1)
        shift = sum(any(same(result(dz), wr[(i + 1) % len(wr)]['ans']) for dz in x['decoded']) for i, x in enumerate(wr)) / max(len(wr), 1)
        o['own_error_any_decoded'] = (own, shift, len(wr))
        print(f'  wrong answer equals a decoded step result: {own:.2f} vs another item\'s answer {shift:.2f} (n={len(wr)})')
        # lens: top-1 token equals a gold intermediate; per pass, best layer; and t23 cell (n = 2, pass 5, v_1)
        def lens_hit(x, p, v):
            return any(same(TOK.decode([t]).strip(), v) for t in x['lens'][p])
        two = [x for x in it if len(x['values']) == 2 and x['correct']]
        if two:
            row = [sum(lens_hit(x, p, x['values'][0]) for x in two) / len(two) for p in range(7)]
            o['lens_n2_v1'] = row
            print('  lens reads v_1 on right n=2 items, question tail then passes 0-5 (any layer): ' + ' '.join(f'{v:.2f}' for v in row))
if WHAT in ('route', 'all'):
    for model in ('sim', 'codi'):
        d = load('route', model)
        if d is None: print(f'[no route for {model}]'); continue
        P = d['pairs']; o = out.setdefault(model, {})
        fol = lambda ps, c: sum(str(r[c]) == str(r['twin_answer']) for r in ps) / max(len(ps), 1)
        print(f'\n== {model} route split: {len(P)} qualified pairs; all sites both {fol(P, "both_all"):.3f} kv {fol(P, "kv_all"):.3f}')
        o['route'] = {}
        for grp, ps in (('all', P), ('n=2', [r for r in P if r['chain_len'] == 2]), ('n=3', [r for r in P if r['chain_len'] == 3]), ('n>=4', [r for r in P if r['chain_len'] >= 4])):
            if not ps: continue
            print(f'  {grp} (pairs {len(ps)}): all sites both {fol(ps, "both_all"):.2f} kv {fol(ps, "kv_all"):.2f}')
            o['route'][grp] = dict(n=len(ps), both_all=fol(ps, 'both_all'), kv_all=fol(ps, 'kv_all'))
            for s in range(6):
                v = {c: fol(ps, f'{c}_{s}') for c in ('both', 'input', 'kv', 'kv_noans')}; o['route'][grp][s] = v
                tag = ''
                if v['both'] >= .2 and v['input'] >= .5 * v['both']: tag = 'VERTICAL'
                elif v['both'] > 0 and v['kv'] >= .5 * v['both'] and v['input'] <= .2 * v['both']: tag = 'slots'
                print(f'    site {s}: ' + ' | '.join(f'{c} {v[c]:.2f}' for c in v) + f'  {tag}')
        # swap faithfulness: after both_s moves the answer to the twin's, does a downstream decoded step give the twin's intermediate?
        print('  after a both_s swap that moved the answer to the twin answer: a downstream decoded result equals a twin intermediate that differs from the recipient\'s')
        o['swap_faith'] = {}
        for s in range(6):
            moved = [r for r in P if str(r[f'both_{s}']) == str(r['twin_answer'])]
            def ok(r):
                diff = [tv for tv, rv in zip(r['twin_values'], r['recipient_values']) if not same(tv, rv)]
                return any(any(same(result(dz), tv) for tv in diff) for dz in r[f'both_{s}_decoded'])
            if moved:
                v = sum(ok(r) for r in moved) / len(moved); o['swap_faith'][s] = (v, len(moved))
                base = [r for r in P if str(r[f'both_{s}']) != str(r['twin_answer'])]
                vb = sum(ok(r) for r in base) / max(len(base), 1)
                print(f'    site {s}: moved {len(moved)}: {v:.2f}  (not moved {len(base)}: {vb:.2f})')
json.dump(out, open(D / 't22_summary.json', 'w'), indent=1, default=str)
