"""Regression check: the general harness reproduces t15 (RuleChain) and t12 (GSM8K) per-item answers.

  ../codi/.venv/bin/python scripts/necessity/regress.py rulechain <device> [model=gated] [level=3]
  ../codi/.venv/bin/python scripts/necessity/regress.py gsm8k <device> [n_pairs=40]

Compares every condition's per-item answer with the stored t15_<model>.json / t12_gsm8k_route_split.json in codi_audit_artifacts;
prints mismatches per condition and exits 1 on any.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adapters  # noqa: E402
import core  # noqa: E402

ART = adapters.BASE / 'codi_audit_artifacts'
what, dev = sys.argv[1], sys.argv[2]
bad = 0

if what == 'rulechain':
    model = sys.argv[3] if len(sys.argv) > 3 else 'gated'; L = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    ad = adapters.RuleChainCODI(model, dev)
    from rc import parse, walk
    data = adapters.BASE / ('synthetic_data_v2_l1to7w_startlast' if model in ('gated', 'sim', 'startlast') else 'synthetic_data_v2_l1to7w') / 'depth2/valid.json'
    ev = [r for r in json.load(open(data)) if int(r['level']) == L]
    P = [parse(r['question']) for r in ev]; V = np.array([walk(*p)[0] for p in P]); gold = V[:, -1]
    tl = np.array([len(ad.tok(r['question'].strip() + '\n').input_ids) for r in ev]); order = np.argsort(tl, kind='stable')
    ks_of = lambda j: min(j, L - 1) if model == 'gated' else (min(j + 1, L - 1) if model == 'sim' else L - 1)
    donors, targets = {}, {}
    for j in list(range(6)) + ['all']:  # t15's donor rule, verbatim
        k = ks_of(L - 1) if j == 'all' else ks_of(j); rng = random.Random(1000 * L + (7 if j == 'all' else j)); dn, tg = [], []
        for i in range(len(ev)):
            cand = [d for d in range(len(ev)) if d != i and tl[d] == tl[i]]; rng.shuffle(cand); pick = None
            for d in cand:
                if V[d, k] == V[i, k]: continue
                vals = walk(int(V[d, k]), P[i][1][k:])[0]
                if min(vals) < 0 or max(vals) > 99 or vals[-1] == gold[i]: continue
                pick = (d, vals[-1]); break
            dn.append(ev[pick[0]] if pick else ev[i]); tg.append(int(pick[1]) if pick else None)
        donors[j], targets[j] = dn, tg
    clean = core.answers(ad, ev, order=order)
    out = core.route_split(ad, ev, donors, targets, order=order)
    ref = json.load(open(ART / f't15_{model}.json'))[str(L)]
    m0 = sum(a != b for a, b in zip(clean, ref['clean_ans'])); print(f'clean: {m0} mismatches / {len(ev)}'); bad += m0
    for c, ans in out.items():
        r = ref['res'][c]['ans']; mm = sum((a if a is not None else -999) != b for a, b in zip(ans, r)); bad += mm
        if mm: print(f'  {c}: {mm} mismatches')
    S = core.summarise(out, list(gold), clean, targets); print(core.fmt(S, 6))

elif what == 'gsm8k':
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    ad = adapters.GSM8KCODI(dev)
    from latentreasoning.data.gsm8k_aug import load_gsm8k_aug
    ex = load_gsm8k_aug(split='test', n=None, seed=None)
    ref = json.load(open(ART / 't12_gsm8k_route_split.json'))['pairs'][:n]
    pairs = [json.loads(l) for l in open(adapters.REPO / 'results/20260927-004340_codi_minimal-pair-patch-aligned/predictions.jsonl')][:n]
    for p, rr in zip(pairs, ref):
        R = dict(question=ex[p['recipient_idx']].question); T = dict(question=p['twin_question'])
        a0, Zr, KVr = core.run(ad, [R]); tw, Zt, KVt = core.run(ad, [T])
        got = dict(base=a0[0], twin_pred=tw[0])
        if rr['same_len']:
            for s in (0, 2, 4):
                got[f'both_{s}'] = core.run(ad, [R], feed={s: Zt[s]})[0][0]
                got[f'input_{s}'] = core.run(ad, [R], feed={s: Zt[s]}, kv={s: KVr[s]})[0][0]
                got[f'kv_{s}'] = core.run(ad, [R], kv={s: KVt[s]})[0][0]
                got[f'kv_noans_{s}'] = core.run(ad, [R], kv={s: KVt[s]}, hide=(s,))[0][0]
        for c, v in got.items():
            if str(v) != str(rr[c]): bad += 1; print(f'  pair {p["recipient_idx"]} {c}: got {v} ref {rr[c]}')
    print(f'gsm8k: {n} pairs compared')

print('REGRESSION', 'PASS' if bad == 0 else f'FAIL ({bad} mismatches)')
sys.exit(1 if bad else 0)
