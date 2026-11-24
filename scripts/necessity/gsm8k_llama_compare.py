"""t47 / t49 / t50 comparison: released Llama-3.2 CODI 1B and SIM-CODI 1B / 3B against the released GPT-2 checkpoints on GSM8K.

Accuracy overall and by item serial depth (strict rule of gsm8k_depth.py) for Llama from t47/gsm8k_llama_acc.json, next to the GPT-2
models' values in t42/gsm8k_depth_strict.json. Hop counts, steps per pass and within-item slopes come from gsm8k_steps_per_pass.py, which
picks up the t47 hop files automatically; run it first and pass its output here.

Run from ai-capstone/:  ../codi/.venv/bin/python scripts/necessity/gsm8k_llama_compare.py <steps_per_pass.json> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

SPP_PATH, OUT_PATH = sys.argv[1:3]
sys.argv = [sys.argv[0], '/dev/null']
exec(open(Path(__file__).with_name('gsm8k_depth.py')).read().split('qtext = lambda')[0])  # depth_of, load_gsm8k_aug, A

te = load_gsm8k_aug(split='test', n=None, seed=None)
DEP = {e.idx: depth_of(e.question, e.rationale, e.answer)['strict']['depth'] for e in te}
dbin = lambda d: None if d is None else ('1' if d <= 1 else '4+' if d >= 4 else str(d))  # as gsm8k_depth.py
rng = np.random.default_rng(0)


def boot(v):
    v = np.asarray(v, float)
    if len(v) == 0: return dict(n=0, mean=None)
    bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
    return dict(n=int(len(v)), mean=float(v.mean()), ci=[float(x) for x in np.percentile(bs, [2.5, 97.5])])


R = {}
for m, f in (('llama', 't47/gsm8k_llama_acc.json'), ('simllama1b', 't49/gsm8k_simllama1b_acc.json'),  # t49 / t50: SIM-CODI Llama 1B / 3B
             ('simllama1b_bf16', 't50/gsm8k_simllama1b_bf16_acc.json'), ('simllama3b', 't50/gsm8k_simllama3b_acc.json')):
    if not (A / f).exists(): continue
    L = json.load(open(A / f)); it = L['items']; assert len(it) == len(te), (f, len(it))
    R[m] = dict(acc=L['acc'], acc_by_depth={b: boot([x['right'] for x in it if dbin(DEP[x['idx']]) == b]) for b in ('1', '2', '3', '4+')})
G = json.load(open(A / 't42/gsm8k_depth_strict.json'))['models']
for m in ('codi', 'sim', 'coconut', 'simcoconut'):
    R[m] = dict(acc_by_depth=G[m]['acc_by_depth'])
S = json.load(open(SPP_PATH))
for m in R:
    if m in S: R[m]['hops'] = S[m]

print('accuracy by item depth (strict):')
for m, v in R.items():
    print(f"  {m:10s}" + (f" overall {v['acc']:.3f}" if 'acc' in v else ' ' * 14)
          + ''.join(f"  d{b} {w['mean']:.3f} (n{w['n']})" for b, w in v['acc_by_depth'].items() if w.get('mean') is not None))
print('\nhops by serial distance, all qualified pairs | correct pairs:')
for m, v in R.items():
    if 'hops' not in v: continue
    bs = v['hops']['by_s']
    cell = lambda d: '   -  ' if 'hops' not in d else f"{d['hops']:.2f}"
    print(f"  {m:10s} " + '  '.join(f"s{s} {cell(bs[s]['all'])} | {cell(bs[s]['correct'])}" for s in ('2', '3', '4', '5')))
print('\nstart of longest path, correct pairs (crossings at depth 2 / 3 / 4):')
for m, v in R.items():
    if 'hops' not in v: continue
    sp = v['hops']['start_of_path']
    print(f"  {m:10s} " + '  '.join(f"D{D} {sp[D]['hops']:.2f} [{sp[D]['hops_ci'][0]:.2f},{sp[D]['hops_ci'][1]:.2f}] (n{sp[D]['n']})" for D in ('2', '3', '4') if D in sp))
print('\nwithin-item slope (crossings per serial step), all | correct:')
for m, v in R.items():
    if 'hops' not in v: continue
    w = v['hops']['within_item']
    f = lambda d: 'n/a' if d['slope'] is None else f"{d['slope']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] ({d['n_items']} items)"
    print(f"  {m:10s} {f(w['all'])} | {f(w['correct'])}")
json.dump(R, open(OUT_PATH, 'w'), indent=1)
print('\nwrote', OUT_PATH)
