"""Need test on the Dilgren & Wiegreffe grid: accuracy by gold path length, per method, with latent-count truncation.

  ../codi/.venv/bin/python scripts/necessity/grid_accuracy.py <task prosqa|prontoqa|gsm8k> <method codi|coconut|no_cot|cot> <device> <out json> [n]

codi: 6 latents and 0 latents (question + <bot> + <eot>). coconut: 0..6 latents (question, <start-latent>, k passes, <end-latent>).
no_cot / cot: the grid's own checkpoints, question then greedy (CoT up to 256 tokens). Test split, file order, batch 1, fp32, greedy.
Bins: ProsQA len(steps) (3-6), ProntoQA all 6, GSM8K number of gold steps.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adapters  # noqa: E402

task, method, dev, out = sys.argv[1:5]; n = int(sys.argv[5]) if len(sys.argv) > 5 else 0
D = adapters.GRID_REPO / 'data'
rows = json.load(open(D / {'prosqa': 'prosqa_test.json', 'prontoqa': 'prontoqa_test.json', 'gsm8k': 'gsm_valid-gold-reasoning-trace_test.json'}[task]))
rows = rows[:n] if n else rows
if task == 'gsm8k':
    sys.path.insert(0, str(adapters.REPO / 'scripts'))
    from coconut_common import num_match
    same = lambda a, g: num_match([a], g) if a else False
    gold = [r['answer'].replace(',', '').strip() for r in rows]
else:
    same = lambda a, g: adapters.norm_text(a) == adapters.norm_text(g)
    gold = [r['answer'] for r in rows]
bins = [len(r['steps']) for r in rows]
if method == 'codi':
    ad = adapters.GridCODI(task, dev); conds = {6: lambda r: ad.finish(*ad_run(r))[0], 0: ad.zero_latent_answer}

    def ad_run(r):
        ctx, past, z = ad.begin([r])
        for s in range(6): past, z = ad.latent(ctx, past, z, s)
        return ctx, past
else:
    ad = adapters.GridCoconut(task, dev, method=method)
    conds = {k: (lambda r, k=k: ad.plain_answer(r, k)) for k in range(7)} if method == 'coconut' else {'plain': ad.plain_answer}
import torch  # noqa: E402
res = {str(k): [] for k in conds}; t0 = time.time()
with torch.no_grad():
    for i, r in enumerate(rows):
        for k, f in conds.items(): res[str(k)].append(f(r))
        if i % 100 == 0: print(time.strftime('%H:%M'), f'{task} {method} {i}/{len(rows)} {time.time() - t0:.0f}s', flush=True)
summary = {}
for k, a in res.items():
    ok = [same(x, g) for x, g in zip(a, gold)]
    by = {}
    for b in sorted(set(bins)):
        ii = [i for i in range(len(rows)) if bins[i] == b]; by[b] = dict(n=len(ii), acc=sum(ok[i] for i in ii) / len(ii))
    summary[k] = dict(acc=sum(ok) / len(ok), by_bin=by, unparseable=sum(not x for x in a) / len(a))
    print(f'{task} {method} latents={k}: acc {summary[k]["acc"]:.3f} unparseable {summary[k]["unparseable"]:.3f} | '
          + ' '.join(f'{b}:{v["acc"]:.2f}(n={v["n"]})' for b, v in by.items()), flush=True)
json.dump(dict(task=task, method=method, n=len(rows), gold=gold, bins=bins, answers=res, summary=summary), open(out, 'w'))
print('DONE')
