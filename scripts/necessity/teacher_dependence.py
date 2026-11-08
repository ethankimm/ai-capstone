"""Teacher-side dependence test: which CoT values does the CODI teacher's answer state depend on?

  ../codi/.venv/bin/python scripts/necessity/teacher_dependence.py <task rulechain|gsm8k|prosqa> <model> <device> <out json> [n=300]
  model: rulechain startlast|relay|stab|sim; gsm8k zen; prosqa grid

The teacher is the CODI model itself reading question + CoT (last step dropped, as in training) + "The answer is: <gold>". Its hidden
state at the ':' of "The answer is:" is the distillation target the latents are trained to reproduce. For each shown CoT value k
(RuleChain v_1..v_{L-1}; GSM8K step results 1..n-1; ProsQA path nodes 1..n-1) every mention of it in the CoT is replaced by another
value of the same token length (RuleChain / GSM8K: value + d, d in 1..3 then -1..-3; ProsQA: another concept name from the question),
leaving the rest of the CoT unchanged. Measures per k (counted from the END of the shown CoT, last = 1):
  flip   P(teacher's first answer token changes); ProsQA: P(the preferred option changes), options scored teacher-forced
  move   ||h_12' - h_12|| / median distance between two different items' h_12 (layer 12 = post ln_f)
Hypothesis under test: the answer state depends on the values the answer cannot recompute from the question, so those are what
the latents are trained to carry (RuleChain: the last shown value only; GSM8K: values the last step uses; ProsQA: last node).
"""
import ast
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adapters  # noqa: E402

task, model, dev, out = sys.argv[1:5]; N = int(sys.argv[5]) if len(sys.argv) > 5 else 300
rng = random.Random(0)
if task == 'rulechain':
    ad = adapters.RuleChainCODI(model, dev)
    DD = adapters.BASE / ('synthetic_data_v2_l1to7w_startlast' if model in ('gated', 'sim', 'startlast') else 'synthetic_data_v2_l1to7w') / 'depth2'
    rows = [r for r in json.load(open(DD / 'valid.json')) if 3 <= int(r['level']) <= 6]; rng.shuffle(rows); rows = rows[:N]
    fmt = lambda r, steps: (r['question'].strip() + '\n', '\n'.join(steps) + '\n')
elif task == 'gsm8k':
    ad = adapters.GSM8KCODI(dev)
    from latentreasoning.data.gsm8k_aug import load_gsm8k_aug
    ex = [e for e in load_gsm8k_aug(split='test', n=None, seed=None)]
    rows = [dict(question=e.question, answer=str(e.answer), steps=re.findall(r'<<[^>]*>>', e.rationale) if hasattr(e, 'rationale') else None) for e in ex]
    rows = [r for r in rows if r['steps'] and len(r['steps']) >= 3]; rng.shuffle(rows); rows = rows[:N]
    fmt = lambda r, steps: (r['question'], ' '.join(steps))
else:
    ad = adapters.GridCODI('prosqa', dev)
    rows = json.load(open(adapters.GRID_REPO / 'data/prosqa_test.json')); rows = [r for r in rows if len(r['steps']) >= 4]; rng.shuffle(rows); rows = rows[:N]
    fmt = lambda r, steps: (r['question'].strip() + '\n', '\n'.join(steps) + '\n')
tok = ad.tok; lm = ad.m.codi.get_base_model(); body = lm.transformer


def steps_of(r):
    s = r['steps']
    return ast.literal_eval(s) if isinstance(s, str) else list(s)


def ids_of(q, cot, ans):
    a = tok(f'The answer is: {ans}').input_ids
    x = tok(q).input_ids + tok(cot).input_ids + a
    return x, len(x) - len(a) + len(tok('The answer is:').input_ids) - 1  # index of ':'


@torch.no_grad()
def state(x, pos, r=None):
    """-> layer-12 state at ':' and the teacher's answer: first answer token, or for ProsQA the preferred option (teacher-forced
    log-prob of " <name> is a <option>." after the ':')."""
    o = body(input_ids=torch.tensor([x], device=dev))
    h = o.last_hidden_state[0, pos]
    if task != 'prosqa':
        return h.float().cpu().numpy(), int(lm.lm_head(h)[:50257].argmax())
    name = r['idx_to_symbol'][r['root']]; lp = []
    for opt in (r['idx_to_symbol'][r['target']], r['idx_to_symbol'][r['neg_target']]):
        y = tok(f' {name} is a {opt}.').input_ids; ids = torch.tensor([x[:pos + 1] + y], device=dev)
        lg = torch.log_softmax(lm.lm_head(body(input_ids=ids).last_hidden_state[0, pos:pos + len(y)])[:, :50257].float(), -1)
        lp.append(float(lg[torch.arange(len(y)), torch.tensor(y, device=dev)].sum()))
    return h.float().cpu().numpy(), int(lp[1] > lp[0])


def values(r, shown):
    """[(k_from_end, index k, original string, candidate replacements)] for each shown value."""
    out = []
    if task in ('rulechain', 'gsm8k'):
        res = [re.search(r'=\s*(-?[\d.]+)\s*>>', s).group(1) for s in shown]
        for k, v in enumerate(res):
            try: fv = float(v)
            except ValueError: continue
            if fv != int(fv): continue
            if task == 'gsm8k' and res.count(v) > 1: continue  # ambiguous mentions
            c = [str(int(fv) + d) for d in (1, 2, 3, -1, -2, -3) if int(fv) + d >= 0]
            out.append((len(res) - k, k, v, c))
    else:
        nodes = [re.match(r'^(?:Every )?(\w+) is an? (\w+)\.$', s.strip()).group(2) for s in shown]
        names = sorted(set(r['idx_to_symbol']) - set(nodes) - {r['idx_to_symbol'][r['root']]})
        for k, v in enumerate(nodes):
            c = names[:]; rng.shuffle(c); out.append((len(nodes) - k, k, v, c))
    return out


def replace(shown, k, v, w):
    """Positional: step k's result (or node) and its mention(s) in later steps."""
    sh = list(shown)
    if task == 'prosqa':
        pat = re.compile(rf'\b{v}\b')
        sh[k] = pat.sub(w, sh[k])
        if k + 1 < len(sh): sh[k + 1] = pat.sub(w, sh[k + 1])
        return sh
    sh[k] = re.sub(rf'=\s*{re.escape(v)}\s*>>', f'={w}>>', sh[k])
    if task == 'rulechain':
        if k + 1 < len(sh): sh[k + 1] = re.sub(rf'^<<{re.escape(v)}(?=\D)', f'<<{w}', sh[k + 1])
    else:
        pat = re.compile(rf'(?<![\w.]){re.escape(v)}(?![\w.])')
        for j in range(k + 1, len(sh)): sh[j] = pat.sub(w, sh[j])
    return sh


t0 = time.time(); recs = []; H0 = []
for i, r in enumerate(rows):
    st = steps_of(r); shown = st[:-1]; q, cot = fmt(r, shown)
    x, pos = ids_of(q, cot, r['answer']); h0, a0 = state(x, pos, r); H0.append(h0)
    rec = dict(i=i, n_shown=len(shown), base_tok=a0, per_k={})
    for k, kk, v, cands in values(r, shown):
        for w in cands:
            sh2 = replace(shown, kk, v, w)
            if sh2 == shown: continue
            q2, cot2 = fmt(r, sh2)
            if len(tok(cot2).input_ids) != len(tok(cot).input_ids): continue
            x2, pos2 = ids_of(q2, cot2, r['answer']); h, a = state(x2, pos2, r)
            rec['per_k'][k] = dict(v=v, w=w, flip=a != a0, d=float(np.linalg.norm(h - h0))); break
    recs.append(rec)
    if i % 50 == 0: print(time.strftime('%H:%M'), f'{task} {model} {i}/{len(rows)} {time.time() - t0:.0f}s', flush=True)
H0 = np.stack(H0); pr = np.random.default_rng(0).integers(0, len(H0), (2000, 2)); pr = pr[pr[:, 0] != pr[:, 1]]
scale = float(np.median(np.linalg.norm(H0[pr[:, 0]] - H0[pr[:, 1]], axis=1)))
summ = {}
for k in sorted({k for r in recs for k in r['per_k']}):
    xs = [r['per_k'][k] for r in recs if k in r['per_k']]
    summ[k] = dict(n=len(xs), flip=float(np.mean([x['flip'] for x in xs])), move=float(np.median([x['d'] for x in xs]) / scale))
print(f'{task} {model}: k counted from the end of the shown CoT (1 = last shown value); scale {scale:.2f}')
print(' | '.join(f"k={k}: flip {v['flip']:.2f} move {v['move']:.2f} (n={v['n']})" for k, v in summ.items()))
json.dump(dict(task=task, model=model, scale=scale, summary=summ, items=recs), open(out, 'w'))
print('DONE')
