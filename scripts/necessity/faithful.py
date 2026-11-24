"""t43: per-latent faithfulness aligned to the hop count. Pre-registered in scripts/codi_null_audit/prereg/t_predictions.md (t43).

  ../codi/.venv/bin/python scripts/necessity/faithful.py <model> <device> <out json>
  model: gated | sim | simsf | startlast | relay | stab | simcoconut_rc   (RuleChain)   gsm8k_codi | gsm8k_sim   (GSM8K)
  env SMOKE=1: tiny fit / eval sets (code check).

Latents are indexed as in the paper: z_1 = question-pass output (the first loop crossing), z_j = output of pass j-1, z_(K+1) = last pass
output (never fed back). A latent is FAITHFUL on an item the model answers wrong when continuing the chain from the value read at z_j
reproduces the model's own wrong answer. Natural runs only (in hop-test mixed runs a latent does not feed the next pass of the same run).
RuleChain: 100-class ridge probe per (z_j, step k) on natural latents, fit on train L2-L7 pooled over levels; continuation re-applies the
item's rules k+1..L to the value read. k*(j, L) = the step z_j reads best on correctly answered eval items of level L.
GSM8K: numbers among the top-10 vocabulary-projection tokens of z_j (t30's readout); a read number put in place of a solution step and
re-executed through the remaining steps (strict dependency parse, t42) must give the model's answer. Control: the next wrong item's tokens.
Item crossings C: RuleChain = hops of the item's start-edit pair (t31 / t41 / t37); GSM8K = mean hops over the item's t31 pairs.
"""
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adapters  # noqa: E402

MODEL, DEV, OUT = sys.argv[1:4]; SMOKE = int(os.environ.get('SMOKE', '0'))
BASE = adapters.BASE; ART = BASE / 'codi_audit_artifacts'
RC = MODEL in ('gated', 'sim', 'simsf', 'startlast', 'relay', 'stab', 'simcoconut_rc')
LV = [2, 3, 4, 5, 6, 7]


@torch.no_grad()
def nat_run(ad, rows):
    ctx, past, z = ad.begin(rows); nat = [z]
    for s in range(ad.n_latents):
        past, z = ad.latent(ctx, past, z, s); nat.append(z)
    return list(ad.finish(ctx, past)), torch.stack(nat, 1).float().cpu().numpy()  # [B, K+1, d]


def run_all(ad, rows, tag):
    ans, Z = [None] * len(rows), [None] * len(rows); B = ad.batch_size; t0 = time.time()
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]['question']))
    for i in range(0, len(order), B):
        ii = order[i:i + B]; a, z = nat_run(ad, [rows[k] for k in ii])
        for n, k in enumerate(ii): ans[k] = a[n]; Z[k] = z[n]
        if (i // B) % (200 if B == 1 else 20) == 0: print(time.strftime('%H:%M'), f'{MODEL} {tag} {i}/{len(rows)} ({time.time() - t0:.0f}s)', flush=True)
        if DEV == 'mps' and B == 1 and i % 50 == 0: torch.mps.empty_cache()
    return ans, np.stack(Z)


def boot(x):
    x = np.asarray(x, float)
    if len(x) < 5: return dict(rate=float(x.mean()) if len(x) else None, n=int(len(x)), ci=None)
    r = np.random.default_rng(0); bs = [x[r.integers(0, len(x), len(x))].mean() for _ in range(1000)]
    return dict(rate=float(x.mean()), n=int(len(x)), ci=[float(v) for v in np.percentile(bs, [2.5, 97.5])])


RES = dict(model=MODEL)
# ================================================================ RuleChain
if RC:
    sys.path.insert(0, str(adapters.REPO / 'scripts/codi_null_audit'))
    import rc as RCmod  # noqa: E402
    if MODEL == 'simcoconut_rc':
        ad = adapters.GridCoconut('rulechain', DEV, 'coconut', max_new=16,
                                  path=str(adapters.REPO / 'rulechain-checkpoints/pilot_k8_simcot_c1m6/checkpoint_26'))
        DD = BASE / 'synthetic_data_v2/depth2'; NFIT = 300; HOPF = [ART / 't37/hops.json']
    else:
        ad = adapters.RuleChainCODI(MODEL, DEV); NFIT = 800
        DD = BASE / ('synthetic_data_v2_l1to7w_startlast' if MODEL in ('gated', 'sim', 'startlast') else 'synthetic_data_v2_l1to7w') / 'depth2'
        HOPF = [ART / f't31/{MODEL}.json', ART / f't41/{MODEL}.json']
    if SMOKE: NFIT = 40
    K1 = ad.n_latents + 1
    tr_all = json.load(open(DD / 'train.json')); rng = random.Random(43)
    tr = []
    for L in LV:
        xs = [r for r in tr_all if int(r['level']) == L]; tr += rng.sample(xs, NFIT)
    ev = [r for s in ('valid.json', 'test.json') for r in json.load(open(DD / s)) if int(r['level']) in LV]
    if SMOKE: ev = [r for r in ev if int(r['level']) in (3, 5)][:120]
    walk = lambda q: RCmod.walk(*RCmod.parse(q))[0]
    print(time.strftime('%H:%M'), f'{MODEL}: fit {len(tr)}, eval {len(ev)}', flush=True)
    _, Zt = run_all(ad, tr, 'fit'); Vt = [walk(r['question']) for r in tr]
    ans, Ze = run_all(ad, ev, 'eval'); Ve = [walk(r['question']) for r in ev]
    # probes: per (j, k), 100-class ridge, penalty on a held-out 15%
    def fit(X, y):
        X = X.astype(np.float64); mu, sd = X.mean(0), X.std(0) + 1e-6; Xs = (X - mu) / sd; Y = np.eye(100)[y]; c = int(.85 * len(X))
        def sol(A, Yt):
            w, V = np.linalg.eigh(A.T @ A); ym = Yt.mean(0); XY = V.T @ (A.T @ (Yt - ym))
            return lambda lam: (V @ (XY / (np.maximum(w, 0) + lam)[:, None]), ym)
        f = sol(Xs[:c], Y[:c]); lams = (1e1, 1e2, 1e3, 1e4)
        acc = [((Xs[c:] @ f(l)[0] + f(l)[1]).argmax(1) == y[c:]).mean() for l in lams]
        W, b = sol(Xs, Y)(lams[int(np.argmax(acc))])
        return lambda Xn: (((Xn.astype(np.float64) - mu) / sd) @ W + b).argmax(1)
    Lt = np.array([int(r['level']) for r in tr]); Le = np.array([int(r['level']) for r in ev])
    READ = np.full((len(ev), K1, 8), -1, int)  # value read at z_(j+1) for step k
    for j in range(K1):
        for k in range(8):
            m = Lt >= k
            if m.sum() < 50: continue
            pr = fit(Zt[m, j], np.array([Vt[i][k] for i in np.where(m)[0]]))
            me = Le >= k; READ[me, j, k] = pr(Ze[me, j])
    gold = np.array([int(r['answer']) for r in ev]); A = np.array([a if isinstance(a, (int, np.integer)) else -999 for a in ans])
    right = A == gold; wrong = (~right) & (A >= 0) & (A < 100)
    # readability on correct items, and each latent's step k*(j, L)
    readab = {}; kstar = {}
    for L in LV:
        ii = np.where(right & (Le == L))[0]
        for j in range(K1):
            sc = {k: float(np.mean([READ[i, j, k] == Ve[i][k] for i in ii])) if len(ii) else float('nan') for k in range(L + 1)}
            kb = max(sc, key=lambda k: -1 if np.isnan(sc[k]) else sc[k]); kstar[(L, j)] = kb
            readab[f'L{L}_z{j + 1}'] = dict(best_step=kb, read=sc[kb], by_step=sc)
    # item crossings
    hop = {}
    for f in HOPF:
        if not f.exists(): continue
        for p in json.load(open(f))['pairs']:
            if p.get('qualified') and p.get('hops') is not None and p['edit'] == 'start': hop[p['id']] = int(p['hops'])
    rules = [RCmod.parse(r['question'])[1] for r in ev]
    def cont(i, k, v): return int(RCmod.walk(int(v), rules[i][k:])[0][-1]) if k < len(rules[i]) else int(v)
    items = []
    for i in np.where(wrong)[0]:
        L = int(Le[i]); C = hop.get(ev[i]['id'])
        for j in range(K1):
            k = kstar[(L, j)]; v = int(READ[i, j, k])
            if v < 0: continue
            try: hit = cont(i, k, v) == A[i]
            except Exception: hit = False
            ch = np.mean([cont(i, k, u) == A[i] for u in range(100)])
            items.append(dict(i=int(i), id=ev[i]['id'], L=L, z=j + 1, k=k, C=C, read=v, gold_v=int(Ve[i][k]), hit=bool(hit),
                              gold_track=bool(v == Ve[i][k]), chance=float(ch), readable=readab[f'L{L}_z{j + 1}']['read']))
    acc = {L: float(right[Le == L].mean()) for L in LV}
    RES.update(acc=acc, n_wrong=int(wrong.sum()), readability=readab, items=items, n_hops=len(hop))
    by = defaultdict(list)
    for x in items: by[('z', x['z'])].append(x); by[('Lz', x['L'], x['z'])].append(x)
    S = {}
    for key, xs in by.items():
        S['_'.join(map(str, key))] = dict(faithful=boot([x['hit'] for x in xs]), chance=float(np.mean([x['chance'] for x in xs])),
                                          gold_track=float(np.mean([x['gold_track'] for x in xs])), readable=float(np.mean([x['readable'] for x in xs])))
    xc = [x for x in items if x['C'] is not None]
    S['inside_C'] = dict(faithful=boot([x['hit'] for x in xc if x['z'] <= x['C']]), chance=float(np.mean([x['chance'] for x in xc if x['z'] <= x['C']] or [np.nan])))
    S['outside_C'] = dict(faithful=boot([x['hit'] for x in xc if x['z'] > x['C']]), chance=float(np.mean([x['chance'] for x in xc if x['z'] > x['C']] or [np.nan])))
    S['z3_C_ge3'] = boot([x['hit'] for x in xc if x['z'] == 3 and x['C'] >= 3]); S['z3_C_lt3'] = boot([x['hit'] for x in xc if x['z'] == 3 and x['C'] < 3])
    RES['summary'] = S
    print(f'{MODEL}: acc', {L: round(a, 3) for L, a in acc.items()}, f'| wrong {int(wrong.sum())} | start-edit hops for {len(hop)} items')
    for j in range(K1):
        s = S.get(f'z_{j + 1}')
        if s: print(f"  z{j + 1}: faithful {s['faithful']['rate']:.3f} (n {s['faithful']['n']}) chance {s['chance']:.3f} gold-track {s['gold_track']:.3f} readable {s['readable']:.2f}")
    for nm in ('inside_C', 'outside_C'):
        s = S[nm]; print(f"  {nm}: faithful {s['faithful']['rate']} (n {s['faithful']['n']}) chance {s['chance']}")
    print('  z3 by C>=3 / C<3:', S['z3_C_ge3'], S['z3_C_lt3'])
    if int(os.environ.get('LOCAL', '0')):  # t43b: local consistency between successive latents on wrong items
        lev = defaultdict(list)
        for f in HOPF:
            if not f.exists(): continue
            for p in json.load(open(f))['pairs']:
                if p.get('qualified') and p.get('hops') is not None and p['edit'] == 'start': lev[int(p['level'])].append(p['hops'])
        CL = {L: float(np.mean(v)) for L, v in lev.items()}
        def step(i, ka, kb, v):
            return int(RCmod.walk(int(v), rules[i][ka:kb])[0][-1]) if kb > ka else int(v)
        links = []
        POOL = np.where(right)[0] if int(os.environ.get('CEIL', '0')) else np.where(wrong)[0]  # t43c: same test on correct items = readout ceiling
        for i in POOL:
            L = int(Le[i]); c = CL.get(L, CL[max(CL)] if L > max(CL) else None)
            for j in range(K1 - 1):
                ka, kb = kstar[(L, j)], kstar[(L, j + 1)]; va, vb = int(READ[i, j, ka]), int(READ[i, j + 1, kb])
                if va < 0 or vb < 0: continue
                try: ok = step(i, ka, kb, va) == vb
                except Exception: ok = False
                ch = float(np.mean([step(i, ka, kb, u) == vb for u in range(100)]))
                links.append(dict(i=int(i), L=L, j=j + 1, ka=ka, kb=kb, C=c, ok=bool(ok), chance=ch, kind='compute' if kb > ka else 'copy',
                                  nontrivial=bool(va != Ve[i][ka] or vb != Ve[i][kb])))
        NT = links if int(os.environ.get('CEIL', '0')) else [x for x in links if x['nontrivial']]
        LS = dict(n_links=len(links), n_nontrivial=len(NT), CL=CL,
                  by_link={f'z{j}->z{j + 1}': dict(cons=boot([x['ok'] for x in NT if x['j'] == j]),
                                                   chance=float(np.mean([x['chance'] for x in NT if x['j'] == j] or [np.nan]))) for j in range(1, K1)})
        for nm, sel in (('inside_C', lambda x: x['C'] is not None and x['j'] + 1 <= round(x['C'])),
                        ('outside_C', lambda x: x['C'] is not None and x['j'] + 1 > round(x['C']))):
            xs = [x for x in NT if sel(x)]
            LS[nm] = dict(cons=boot([x['ok'] for x in xs]), chance=float(np.mean([x['chance'] for x in xs] or [np.nan])))
        for kind in ('compute', 'copy'):  # added after t43b registration: computing links vs copies
            for nm, sel in (('inside_C', lambda x: x['C'] is not None and x['j'] + 1 <= round(x['C'])),
                            ('outside_C', lambda x: x['C'] is not None and x['j'] + 1 > round(x['C']))):
                xs = [x for x in NT if x['kind'] == kind and sel(x)]
                LS[f'{kind}_{nm}'] = dict(cons=boot([x['ok'] for x in xs]), chance=float(np.mean([x['chance'] for x in xs] or [np.nan])))
        RES['local'] = LS; RES['local_links'] = links
        print(f"  t43b local consistency: {len(NT)} non-trivial links of {len(links)}")
        for k, v in LS['by_link'].items(): print(f"    {k}: {v['cons']['rate']} (n {v['cons']['n']}) chance {v['chance']:.3f}")
        for nm in ('inside_C', 'outside_C', 'compute_inside_C', 'compute_outside_C', 'copy_inside_C', 'copy_outside_C'):
            print(f"    {nm}: {LS[nm]['cons']} chance {LS[nm]['chance']:.3f}")

# ================================================================ GSM8K
else:
    from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
    from latentreasoning.eval.metrics import is_correct  # noqa: E402
    te = load_gsm8k_aug(split='test', n=None, seed=None)
    if SMOKE: te = te[:60]
    ad = adapters.GSM8KCODI(DEV, ckpt=str(BASE / 'grid_checkpoints/simcot_gpt2_codi') if MODEL == 'gsm8k_sim' else None)
    head = ad.m.codi.get_base_model().lm_head; tok = ad.tok; K1 = ad.n_latents + 1
    STEP = re.compile(r'<<([^<>=]*)=([^<>]*)>>'); NUMR = re.compile(r'\d[\d,]*(?:\.\d+)?')
    fnum = lambda s: (lambda t: float(t) if re.fullmatch(r'-?\d+(\.\d+)?', t) else None)(str(s).replace(',', '').replace('$', '').strip())
    same = lambda a, b: a is not None and b is not None and abs(a - b) <= 1e-6 * max(1.0, abs(a))

    def parse(e):
        """steps as (expr, result, refs) with refs = [(span, step i)] for numbers taken from an earlier result (strict rule, t42)."""
        qn = [fnum(x) for x in NUMR.findall(e.question)]
        st = []
        for expr, res in STEP.findall(e.rationale):
            r = fnum(res); refs = []
            for m in NUMR.finditer(expr):
                v = fnum(m.group())
                if v is None or any(same(v, q) for q in qn): continue
                i = next((i for i in range(len(st) - 1, -1, -1) if same(st[i][1], v)), None)
                if i is not None: refs.append((m.span(), i))
            st.append((expr, r, refs))
        ga = fnum(e.answer); fin = next((j for j in range(len(st) - 1, -1, -1) if same(st[j][1], ga)), None)
        return st, fin

    def reexec(st, fin, k, v):
        vals = [s[1] for s in st]; vals[k] = v
        for j in range(k + 1, fin + 1):
            expr, _, refs = st[j]
            if not refs: continue
            out, last = '', 0
            for (a, b), i in refs: out += expr[last:a] + repr(vals[i]); last = b
            out += expr[last:]
            if not re.fullmatch(r'[\d.+\-*/() ]+', out.replace(',', '')): return None
            try: vals[j] = float(eval(out.replace(',', '')))
            except Exception: return None
        return vals[fin]

    @torch.no_grad()
    def tops(z):
        T = head(torch.tensor(z, device=DEV)).topk(10, dim=-1).indices.tolist()  # [K1, 10]
        out = []
        for row in T:
            vs = set()
            for t in row:
                v = fnum(tok.decode([t]).strip())
                if v is not None: vs.add(v)
            out.append(vs)
        return out

    ans, Ze = run_all(ad, [dict(question=e.question) for e in te], 'eval')
    hop = defaultdict(list)
    for f in (ART / f't31/{MODEL}.json', ART / f't31/long_{MODEL}.json'):
        for p in json.load(open(f))['pairs']:
            if p.get('qualified') and p.get('hops') is not None: hop[' '.join(p['R']['question'].split())].append(p['hops'])
    rows = []
    for i, e in enumerate(te):
        a = fnum(ans[i]) if ans[i] not in (None, '') else None
        st, fin = parse(e)
        rows.append(dict(i=i, ans=a, right=a is not None and is_correct(str(ans[i]), e.answer), st=st, fin=fin, T=tops(Ze[i]),
                         C=float(np.mean(hop[' '.join(e.question.split())])) if hop.get(' '.join(e.question.split())) else None))
    W = [r for r in rows if not r['right'] and r['ans'] is not None and r['fin'] is not None]

    def hit(r, vs):
        for k in range(r['fin'] + 1):
            for v in vs:
                if same(v, r['st'][k][1]): continue
                o = reexec(r['st'], r['fin'], k, v)
                if o is not None and same(o, r['ans']): return True
        return False

    items = []
    for n, r in enumerate(W):
        sh = W[(n + 1) % len(W)]
        for j in range(K1):
            items.append(dict(i=r['i'], z=j + 1, C=r['C'], hit=hit(r, r['T'][j]), shuf=hit(r, sh['T'][j]),
                              gold_track=any(same(v, s[1]) for v in r['T'][j] for s in r['st'][:r['fin'] + 1])))
    R_ = [r for r in rows if r['right'] and r['fin'] is not None]
    readable = {f'z{j + 1}': float(np.mean([any(same(v, s[1]) for v in r['T'][j] for s in r['st'][:r['fin'] + 1]) for r in R_])) for j in range(K1)}
    S = {}
    for j in range(K1):
        xs = [x for x in items if x['z'] == j + 1]
        S[f'z_{j + 1}'] = dict(faithful=boot([x['hit'] for x in xs]), shuffled=boot([x['shuf'] for x in xs]),
                               gold_track=float(np.mean([x['gold_track'] for x in xs])), readable_right=readable[f'z{j + 1}'])
    xc = [x for x in items if x['C'] is not None]
    for nm, sel in (('inside_C', lambda x: x['z'] <= round(x['C'])), ('outside_C', lambda x: x['z'] > round(x['C']))):
        xs = [x for x in xc if sel(x)]
        S[nm] = dict(faithful=boot([x['hit'] for x in xs]), shuffled=boot([x['shuf'] for x in xs]))
    RES.update(acc=float(np.mean([r['right'] for r in rows])), n_wrong=len(W), summary=S, items=items)
    print(f"{MODEL}: acc {RES['acc']:.3f} | wrong items scored {len(W)}")
    for j in range(K1):
        s = S[f'z_{j + 1}']
        print(f"  z{j + 1}: faithful {s['faithful']['rate']:.3f} shuffled {s['shuffled']['rate']:.3f} gold-track(wrong) {s['gold_track']:.3f} readable(right) {s['readable_right']:.3f}")
    for nm in ('inside_C', 'outside_C'):
        print(f"  {nm}: faithful {S[nm]['faithful']} shuffled {S[nm]['shuffled']['rate']}")
json.dump(RES, open(OUT, 'w'), default=str)
print('DONE')
