"""t21c: does the question pass (prefill) compute the chain on its own, next to the latents? RuleChain CODI models.

  ../codi/.venv/bin/python scripts/necessity/prefill_probe.py <out json> <device> <model> [<model> ...]   (models: simsf stab sim startlast)

For each item, positions in the question: the token holding the period that ends rule m (m = 1..L), and <bot> (the last question
position, whose final state is projected into z_0). Hidden states at layers 4, 8, 12 of the question pass (one forward, no latents).
Linear probes (ridge to 100-way one-hot, standardised features, penalty picked on a held-out 15% of the fit items), one per
(level, layer, position, target v_k), fit on train questions (2000 per level, seed 30), scored exact on all valid items of the level
(L3-L6). Shuffled-label control at every cell. v_k = value after k rules (v_0 = start value).
Reading: start-first, a model that runs the chain while reading the rules holds v_k at the end of rule k (the diagonal). Start-last
cannot (the start value comes after the rules), so its diagonal is the floor; its <bot> cells show what the tail can compute.
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adapters  # noqa: E402

OUT, DEV, MODELS = sys.argv[1], sys.argv[2], sys.argv[3:]
import os
SMOKE = int(os.environ.get('SMOKE', '0'))  # SMOKE=1: L3 only, 300 fit items (code check)
LEVELS = [3] if SMOKE else [3, 4, 5, 6]; LAYERS = [4, 8, 12]; NFIT = 300 if SMOKE else 2000
sys.path.insert(0, str(adapters.REPO))
from latentreasoning.tasks.synthetic_v2 import parse_chain_question, run_chain  # noqa: E402


def chain(q):  # start-last questions get the start value moved to the front before parsing
    m = re.search(r' Start with (\d+)\.', q)
    return parse_chain_question(q if m is None or q.startswith('Start') else 'Start with %s. ' % m.group(1) + q[:m.start()] + q[m.end():])


def values(q):
    c = chain(q); w = run_chain(c.x0, c.rules); return [c.x0] + [s[3] for s in w]


def rule_end_chars(q):
    return [m.end() - 1 for m in re.finditer(r'Rule \d+:[^.]*\.', q)]


def features(ad, items):
    """[n, n_pos, n_layers, d] with positions = rule ends 1..L then <bot>; plus V [n, L+1]."""
    L = int(items[0]['level']); F = np.zeros((len(items), L + 1, len(LAYERS), 768), np.float32); V = np.zeros((len(items), L + 1), int)
    for b in range(0, len(items), 32):
        rows = items[b:b + 32]
        e = ad.h.encode([dict(question=r['question'].strip(), steps=['x'], answer='0') for r in rows], ad.tok, ad.m, ad.dev)
        ids = e['encoder_input_ids']; mask = e['encoder_attention_mask'].long(); W = ids.size(1)
        with torch.no_grad():
            o = ad.body(input_ids=ids, attention_mask=mask, position_ids=ad.qpos(mask), output_hidden_states=True)
        hs = [o.hidden_states[l] for l in LAYERS]
        for i, r in enumerate(rows):
            q = r['question'].strip(); t = ad.tok(q + '\n').input_ids
            n = len(t); pad = W - (n + 1)
            assert ids[i, pad:pad + n].tolist() == t, 'token mismatch'
            ends = rule_end_chars(q); assert len(ends) == L
            tix = [len(ad.tok(q[:c + 1]).input_ids) - 1 for c in ends]  # GPT-2 splits '.' into its own token, so prefix counts line up
            assert all(t[j] == PERIOD for j in tix), 'rule-end token is not a period'
            pos = [j + pad for j in tix] + [W - 1]
            for a, p in enumerate(pos):
                for li, h in enumerate(hs): F[b + i, a, li] = h[i, p].float().cpu().numpy()
            V[b + i] = values(q); assert V[b + i, -1] == int(r['answer'])
    return F, V


def ridge_eval(Xa, Ya, Xb, Yb, rng):
    """exact accuracy on (Xb, Yb) for a ridge probe fit on (Xa, Ya); penalty from a held-out 15% of a."""
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6; Xa = (Xa - mu) / sd; Xb = (Xb - mu) / sd
    def fit(X, y):
        Y = np.eye(100)[y]; ym = Y.mean(0); w, Vv = np.linalg.eigh(X.T @ X); XY = Vv.T @ (X.T @ (Y - ym))
        return lambda lam: (Vv @ (XY / (np.maximum(w, 0) + lam)[:, None]), ym)
    c = int(.85 * len(Xa)); f = fit(Xa[:c], Ya[:c])
    best = max((1e1, 1e2, 1e3, 1e4, 1e5), key=lambda lam: np.mean(((Xa[c:] @ f(lam)[0]) + f(lam)[1]).argmax(1) == Ya[c:]))
    Wt, ym = fit(Xa, Ya)(best); return float(np.mean(((Xb @ Wt) + ym).argmax(1) == Yb))


RES = {}
for mk in MODELS:
    t0 = time.time(); ad = adapters.RuleChainCODI(mk, DEV); PERIOD = ad.tok.encode('.')[0]
    DD = adapters.BASE / ('synthetic_data_v2_l1to7w_startlast' if mk in ('sim', 'startlast', 'gated') else 'synthetic_data_v2_l1to7w') / 'depth2'
    tr_all = json.load(open(DD / 'train.json')); va_all = json.load(open(DD / 'valid.json')); RES[mk] = {}
    rng = np.random.default_rng(0)
    for L in LEVELS:
        tr = [r for r in tr_all if int(r['level']) == L]; tr = [tr[i] for i in np.random.default_rng(30).permutation(len(tr))[:NFIT]]
        va = [r for r in va_all if int(r['level']) == L]
        Fa, Va = features(ad, tr); Fb, Vb = features(ad, va)
        cells = {}
        for li, l in enumerate(LAYERS):
            for a in range(L + 1):
                pname = f'rule{a + 1}' if a < L else 'bot'
                for k in range(1, L):
                    acc = ridge_eval(Fa[:, a, li], Va[:, k], Fb[:, a, li], Vb[:, k], rng)
                    cell = dict(acc=acc)
                    if pname in (f'rule{k}', 'bot'):  # shuffled control on the diagonal and at <bot>
                        cell['shuf'] = ridge_eval(Fa[:, a, li], rng.permutation(Va[:, k]), Fb[:, a, li], Vb[:, k], rng)
                    cells[f'{pname}|{l}|v{k}'] = cell
        RES[mk][L] = dict(n_fit=len(tr), n_test=len(va), mode_share=[float(np.bincount(Vb[:, k]).max() / len(Vb)) for k in range(L + 1)], cells=cells)
        diag = ' '.join(f"v{k}@r{k} {max(cells[f'rule{k}|{l}|v{k}']['acc'] for l in LAYERS):.2f}" for k in range(1, L))
        bot = ' '.join(f"v{k}@bot {max(cells[f'bot|{l}|v{k}']['acc'] for l in LAYERS):.2f}" for k in range(1, L))
        print(time.strftime('%H:%M'), mk, f'L{L}', diag, '|', bot, f'({time.time() - t0:.0f}s)', flush=True)
        json.dump(RES, open(OUT, 'w'))
    del ad; torch.mps.empty_cache() if DEV == 'mps' else None
print('DONE')
