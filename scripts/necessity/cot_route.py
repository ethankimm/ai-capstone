"""t54: the model's own route. The released CODI checkpoints share weights with their CoT teacher, so the same weights can write the
solution out. This script generates that explicit CoT (greedy) for every GSM8K test item and for every donor question of the t31 pairs
the model answers right on both sides; scoring (cot_route_score.py) parses the model's own <<expr=res>> steps into a dependency graph
and recomputes depth and edit distance on it. Pre-registered in prereg/t_predictions.md (t54).

  ../codi/.venv/bin/python scripts/necessity/cot_route.py <codi|sim> <device> <out json>
  env SMOKE=N: only N questions.

Teacher input as teacher_dependence.py / CODI training: tok(question) followed directly by the CoT tokens, then "The answer is: <a>".
Generation: greedy over the base vocabulary (the added CODI tokens are masked), stop at eos, at a newline after the answer number, or
after 256 tokens.
"""
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adapters  # noqa: E402

M, DEV, OUT = sys.argv[1:4]; SMOKE = int(os.environ.get('SMOKE', '0'))
ad = adapters.GSM8KCODI(DEV, ckpt=str(adapters.BASE / 'grid_checkpoints/simcot_gpt2_codi') if M == 'sim' else None)
tok = ad.tok; lm = ad.m.codi.get_base_model(); V = 50257; A = adapters.BASE / 'codi_audit_artifacts'
from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402

qs = [e.question for e in load_gsm8k_aug(split='test', n=None, seed=None)]
for f in (f't31/gsm8k_{M}.json', f't31/long_gsm8k_{M}.json'):
    for p in json.load(open(A / f))['pairs']:
        if p.get('qualified') and p.get('R_right') and p.get('D_right'):
            qs += [p['R']['question'], p['D']['question']]
seen = set(); Q = [q for q in qs if not (q in seen or seen.add(q))]
if SMOKE: Q = Q[:SMOKE]
print(time.strftime('%H:%M'), f'{M}: {len(Q)} questions', flush=True)
ANS = re.compile(r'The answer is:\s*(-?[\d,]*\.?\d+)')


@torch.no_grad()
def write(q):
    ids = torch.tensor([tok(q).input_ids], device=DEV); past = None; out = []; x = ids
    for _ in range(256):
        o = lm(input_ids=x, past_key_values=past, use_cache=True); past = o.past_key_values
        nid = int(o.logits[0, -1, :V].argmax())
        if nid == tok.eos_token_id: break
        out.append(nid); x = torch.tensor([[nid]], device=DEV)
        if len(out) % 4 == 0 or '\n' in tok.decode([nid]):
            t = tok.decode(out); m = ANS.search(t)
            if m and len(t) > m.end() and not t[m.end()].isdigit() and t[m.end()] not in '.,': break
    return tok.decode(out)


res = []; t0 = time.time()
for n, q in enumerate(Q):
    t = write(q); m = ANS.search(t)
    res.append(dict(question=q, text=t, answer=m.group(1).replace(',', '') if m else None))
    if n % 100 == 0: print(time.strftime('%H:%M'), f'{n}/{len(Q)} {time.time() - t0:.0f}s | {t[:120]!r}', flush=True)
    if n % 200 == 0:
        gc.collect()
        if DEV == 'mps': torch.mps.empty_cache()
json.dump(dict(model=M, n=len(res), rows=res), open(OUT, 'w'))
print('DONE', f'{time.time() - t0:.0f}s')
