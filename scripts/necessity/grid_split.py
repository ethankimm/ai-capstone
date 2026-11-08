"""Route split (necessity check) on the Dilgren & Wiegreffe grid and other batch-1 adapters.

  ../codi/.venv/bin/python scripts/necessity/grid_split.py <task prosqa|prontoqa|gsm8k> <method codi|coconut> <device> <out json> [cap]

Pairs: ProsQA / ProntoQA from pairs.py (one path relation rewired so the answer flips, equal token length); sampled with seed 0, ProsQA
up to `cap` (default 300) pairs per path length (3-6), ProntoQA `cap` pairs spread evenly over the 5 edited hops. GSM8K Coconut: the
231 twin pairs of 20260927-003331 (one number in the question changed), kept where both questions have the same token length.
Qualified pair: recipient answered right, donor answered its implied answer. Per qualified pair: for each site j = 0..5 the four
conditions (core.SITE_CONDS) and both_all / kv_all; score = answer equals the donor's answer. SMOKE=1: first 6 qualified pairs plus
the own write-back checks.
"""
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adapters  # noqa: E402
import core  # noqa: E402
import pairs as PR  # noqa: E402
import multi  # noqa: E402

task, method, dev, out = sys.argv[1:5]; cap = int(sys.argv[5]) if len(sys.argv) > 5 else 300; SMOKE = int(os.environ.get('SMOKE', '0'))
ad = multi.GridCODIMulti(task, dev) if method == 'codi' else multi.GridCoconutMulti(task, dev, 'coconut'); VERIFY = 10
D = adapters.GRID_REPO / 'data'; rng = random.Random(0)
if method == 'codi':
    tlen = lambda q: ad.qids(q).size(1)
else:
    tlen = lambda q: len(ad.qids(q))
if task == 'gsm8k':
    sys.path.insert(0, str(adapters.REPO / 'scripts')); from coconut_common import num_match  # noqa: E402
    from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
    ex = load_gsm8k_aug(split='test', n=None, seed=None)
    src = [json.loads(l) for l in open(adapters.REPO / 'results/20260927-003331_coconut_minimal-pair-patch-full/predictions.jsonl')]
    rows = [dict(question=ex[p['recipient_idx']].question, answer=str(p['recipient_gold'])) for p in src]
    P = [dict(i=k, hop=p['step'], n_hops=len(ex[p['recipient_idx']].intermediate_values), donor=dict(question=p['twin_question'], answer=p['twin_answer']),
              target=str(p['twin_answer']), edit=(p['perturbed_number'], p['delta'])) for k, p in enumerate(src)]
    P = [p for p in P if tlen(rows[p['i']]['question']) == tlen(p['donor']['question'])]
    same = lambda a, g: bool(a) and num_match([str(a)], str(g))
else:
    rows = json.load(open(D / f'{task}_test.json'))
    allp = [p for p in (PR.prosqa_pairs if task == 'prosqa' else PR.prontoqa_pairs)(rows, tlen) if p['donor']]
    rng.shuffle(allp); key = 'n_hops' if task == 'prosqa' else 'hop'; per = cap if task == 'prosqa' else cap // 5
    cnt = defaultdict(int); P = []
    for p in allp:
        if cnt[p[key]] < per: cnt[p[key]] += 1; P.append(p)
    P.sort(key=lambda p: (p['i'], p['hop']))
    same = lambda a, g: adapters.norm_text(str(a)) == adapters.norm_text(str(g))
print(time.strftime('%H:%M'), f'{task} {method}: {len(P)} candidate pairs', flush=True)

memo = {}
def clean(row):
    q = row['question']
    if q not in memo: memo[q] = core.run(ad, [row])
    return memo[q]


res = []; t0 = time.time(); nq = 0; checked = False
with torch.no_grad():
    for k, p in enumerate(P):
        r = rows[p['i']]; d = p['donor']
        a0, Zr, KVr = clean(r); ad_, Zd, KVd = clean(d)
        rec = dict(i=p['i'], hop=p['hop'], n_hops=p['n_hops'], edit=p['edit'], gold=r['answer'], target=p['target'], clean=a0[0], donor_clean=ad_[0])
        rec['qualified'] = same(a0[0], r['answer']) and same(ad_[0], p['target'])
        if rec['qualified']:
            nq += 1
            if SMOKE and not checked:
                assert core.run(ad, [r], kv={0: KVr[0]})[0] == a0 and core.run(ad, [r], feed={0: Zr[0]})[0] == a0, 'own write-back changed answer'
                print('  own write-back checks pass', flush=True); checked = True
            S = range(ad.n_latents); names, edits = ['own'], [{}]
            for j in S:
                names += [f'both_{j}', f'input_{j}', f'kv_{j}', f'kv_noans_{j}']
                edits += [dict(feed={j: Zd[j]}), dict(feed={j: Zd[j]}, kv={j: KVr[j]}), dict(kv={j: KVd[j]}), dict(kv={j: KVd[j]}, hide=(j,))]
            names += ['both_all', 'kv_all', 'input_all']
            edits += [dict(feed={s: Zd[s] for s in S}), dict(kv={s: KVd[s] for s in S}), dict(feed={s: Zd[s] for s in S}, kv={s: KVr[s] for s in S})]
            A = multi.run_multi(ad, r, edits)
            for c, a in zip(names, A): rec[c] = a
            rec['own_matches_clean'] = A[0] == a0[0]
            if nq <= VERIFY:  # batched vs one-condition-at-a-time
                single = [core.run(ad, [r], feed=e.get('feed'), kv=e.get('kv'), hide=e.get('hide', ()))[0][0] for e in edits]
                rec['verify_agree'] = sum(x == y for x, y in zip(single, A)) / len(A)
        res.append(rec)
        if len(memo) > 400: memo.clear()
        if k % 25 == 0: print(time.strftime('%H:%M'), f'pair {k}/{len(P)} qualified {nq} {time.time() - t0:.0f}s', flush=True)
        if SMOKE and nq >= 6: break
        if dev == 'mps' and k % 25 == 0: torch.mps.empty_cache()

Q = [x for x in res if x['qualified']]
CONDS = [f'{c}_{j}' for j in range(ad.n_latents) for c in core.SITE_CONDS] + ['both_all', 'kv_all', 'input_all']
def rates(xs):
    return {c: (sum(same(x[c], x['target']) for x in xs) / len(xs) if xs else None) for c in CONDS} | {'n': len(xs)}
summary = dict(all=rates(Q), by_hop={h: rates([x for x in Q if x['hop'] == h]) for h in sorted({x['hop'] for x in Q})},
               by_len={h: rates([x for x in Q if x['n_hops'] == h]) for h in sorted({x['n_hops'] for x in Q})},
               keeps_all={c: sum(same(x[c], x['gold']) for x in Q) / max(len(Q), 1) for c in CONDS})
A = summary['all']; f = lambda v: '  - ' if v is None else f'{v:.2f}'
print(f'{task} {method}: qualified {len(Q)} / {len(res)}; own row = clean {sum(x["own_matches_clean"] for x in Q)}/{len(Q)}; '
      f'batched = single on verified pairs {np.mean([x["verify_agree"] for x in Q if "verify_agree" in x]):.4f}')
print(' | '.join(f"j{j}: b {f(A[f'both_{j}'])} i {f(A[f'input_{j}'])} k {f(A[f'kv_{j}'])} na {f(A[f'kv_noans_{j}'])}" for j in range(ad.n_latents))
      + f" | all: b {f(A['both_all'])} kv {f(A['kv_all'])} in {f(A['input_all'])}")
json.dump(dict(task=task, method=method, pairs=res, summary=summary), open(out, 'w'))
print('DONE')
