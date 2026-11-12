"""t27b / t27c (post hoc, after t26): RuleChain reference runs. Pre-registered in prereg/t_predictions.md (t27) before running.

  ../codi/.venv/bin/python scripts/necessity/hops_ref.py <model key> <device> <out json>
  model key: startlast0 | stab0 (matched no-latent CODI-0, 0 passes) | startlast | stab (6 latents)

(a) accuracy by level on valid L1-L7 and on 500 train items per level (seed 0): the train-test gap (t27c, memorisation).
(b) CODI-0 keys only: the model's answer on every recipient and donor question of the qualified t26 pairs of its partner models
    (startlast0: gated, sim, startlast; stab0: relay, stab, simsf), for the item-level necessity rescoring (t27b).
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adapters  # noqa: E402
import torch  # noqa: E402


@torch.no_grad()
def nat_run(ad, rows):
    """Natural latent loop (hops.py's nat_run without a feed; hops.py runs its experiment on import)."""
    ctx, past, z = ad.begin(rows)
    for s in range(ad.n_latents):
        past, z = ad.latent(ctx, past, z, s)
    return ad.finish(ctx, past), None


KEY, DEV, OUT = sys.argv[1:4]
BASE = adapters.BASE; T26 = BASE / 'codi_audit_artifacts/t26'
ad = adapters.RuleChainCODI(KEY, DEV)
data = 'synthetic_data_v2_l1to7w_startlast' if KEY.startswith('startlast') else 'synthetic_data_v2_l1to7w'


def answers(qs):
    out = {}; qs = sorted(set(qs), key=len)
    for i in range(0, len(qs), ad.batch_size):
        b = qs[i:i + ad.batch_size]; a, _ = nat_run(ad, [dict(question=q) for q in b])
        out.update(zip(b, a))
    return out


def acc_by_level(rows):
    A = answers([r['question'] for r in rows]); by = defaultdict(list)
    for r in rows: by[int(r['level'])].append(A[r['question']] == int(r['answer']))
    return {L: round(sum(v) / len(v), 4) for L, v in sorted(by.items())}


res = dict(model=KEY)
va = json.load(open(BASE / data / 'depth2/valid.json'))
tr = json.load(open(BASE / data / 'depth2/train.json')); rng = random.Random(0); by = defaultdict(list)
for r in tr: by[int(r['level'])].append(r)
trs = [x for L in sorted(by) for x in rng.sample(by[L], min(500, len(by[L])))]
res['valid_acc'] = acc_by_level(va); res['train_acc'] = acc_by_level(trs)
print(KEY, 'valid', res['valid_acc'], flush=True); print(KEY, 'train', res['train_acc'], flush=True)
if KEY.endswith('0'):
    partners = ('gated', 'sim', 'startlast') if KEY == 'startlast0' else ('relay', 'stab', 'simsf')
    qs = []
    for m in partners:
        for p in json.load(open(T26 / f'{m}.json'))['pairs']:
            if p['qualified']: qs += [p['R']['question'], p['D']['question']]
    res['answers'] = answers(qs); print(KEY, f"answered {len(res['answers'])} pair questions", flush=True)
json.dump(res, open(OUT, 'w'))
print('DONE')
