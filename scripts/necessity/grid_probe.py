"""Probes on ProsQA (grid CODI / Coconut): is the path's intermediate node, or the BFS frontier, readable at the question tail or at the latents?

  ../codi/.venv/bin/python scripts/necessity/grid_probe.py <method codi|coconut> <device> <out json> [n_train=3000]

Features (no edits): residual stream at layers 6, 9, 12 (12 = post ln_f) at the last question token ("tail"), at <bot> / <start-latent>
("bot"), and at latent pass j = 0..5 (the hidden state pass j produces, which becomes z_{j+1}; CODI projects it first).
Targets (ProsQA has 55 concept names in all; labels are name indices):
  path_k   the k-th node on the gold path (k = 1..3; node 0 is the person), softmax probe, accuracy = argmax among the item's own graph
           nodes (chance ~ 1/20)
  front_k  BFS frontier at distance k from the person (k = 1..4), multi-label probe, AUROC over the item's graph nodes (chance .5)
Probes: linear, standardised features, trained on n_train train items, tested on the 500 test items; control = same probe trained on
shuffled train labels. Also the Coconut paper's readout: LM head on each latent pass's layer-12 state, probability mass on frontier-k
nodes' first tokens, normalised over all the item's node first tokens (lens_front_k).
"""
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adapters  # noqa: E402
import pairs as PR  # noqa: E402

method, dev, out = sys.argv[1:4]; NTR = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
ad = adapters.GridCODI('prosqa', dev) if method == 'codi' else adapters.GridCoconut('prosqa', dev, 'coconut')
D = adapters.GRID_REPO / 'data'
train = json.load(open(D / 'prosqa_train.json')); random.Random(0).shuffle(train); train = train[:NTR]
test = json.load(open(D / 'prosqa_test.json'))
SYM = sorted({s for r in train + test for s in r['idx_to_symbol']}); SI = {s: i for i, s in enumerate(SYM)}
LAYERS = (6, 9, 12); POS = ['tail', 'bot'] + [f'lat{j}' for j in range(6)]; KP, KF = (1, 2, 3), (1, 2, 3, 4)
tok = ad.tok
first = np.array([tok.encode(' ' + s)[0] for s in SYM])


@torch.no_grad()
def feats(r):
    q = r['question']
    if method == 'codi':
        ids = ad.qids(q).to(dev)
    else:
        ids = torch.tensor([ad.qids(q) + [ad.sl]], device=dev)
    o = ad.body(input_ids=ids, use_cache=True, output_hidden_states=True)
    H = [torch.stack([o.hidden_states[l][0, -2] for l in LAYERS]), torch.stack([o.hidden_states[l][0, -1] for l in LAYERS])]
    past = o.past_key_values; h = o.last_hidden_state[:, -1:, :]; lens = []
    for j in range(6):
        z = ad.m.prj(h) if method == 'codi' else h
        o = ad.body(inputs_embeds=z, use_cache=True, past_key_values=past, output_hidden_states=True)
        past = o.past_key_values; h = o.last_hidden_state[:, -1:, :]
        H.append(torch.stack([o.hidden_states[l][0, -1] for l in LAYERS]))
        lens.append(torch.softmax(ad.head(h[0, -1]).float(), -1)[torch.tensor(first, device=dev)].cpu().numpy())
    return torch.stack(H).float().cpu().numpy().astype(np.float16), np.stack(lens)  # [8, 3, 768], [6, 55]


def labels(r):
    E, name, o1, o2, S = PR.prosqa_parse(r['question']); path = PR.prosqa_path(r)
    nodes = [path[0][0]] + [b for a, b in path]; F = PR.frontiers(E, name, max(KF))
    graph = np.zeros(len(SYM), bool); graph[[SI[s] for s in r['idx_to_symbol'] if s in SI]] = True
    pk = {k: (SI[nodes[k]] if k < len(nodes) - 1 and nodes[k] in SI else -1) for k in KP}  # intermediate nodes only
    fk = {}
    for k in KF:
        v = np.zeros(len(SYM), bool); v[[SI[s] for s in F[k] if s in SI]] = True; fk[k] = v
    return graph, pk, fk


t0 = time.time()
def collect(rows, tag):
    X, LN, G, PK, FK = [], [], [], [], []
    for i, r in enumerate(rows):
        if i % 20 == 0: gc.collect()
        x, ln = feats(r); g, pk, fk = labels(r); X.append(x); LN.append(ln); G.append(g); PK.append(pk); FK.append(fk)
        if i % 500 == 0: print(time.strftime('%H:%M'), f'{tag} {i}/{len(rows)} {time.time() - t0:.0f}s', flush=True)
    return np.stack(X), np.stack(LN), np.stack(G), PK, FK


Xtr, _, Gtr, PKtr, FKtr = collect(train, 'train'); Xte, LNte, Gte, PKte, FKte = collect(test, 'test')


def fit(X, Y, kind, epochs=300, wd=1e-3, seed=0):
    torch.manual_seed(seed)
    X = torch.tensor(X, dtype=torch.float32); mu, sd = X.mean(0), X.std(0) + 1e-5; X = (X - mu) / sd
    W = torch.nn.Linear(X.size(1), len(SYM)); opt = torch.optim.Adam(W.parameters(), lr=1e-2, weight_decay=wd)
    Y = torch.tensor(Y)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(W(X), Y) if kind == 'ce' else torch.nn.functional.binary_cross_entropy_with_logits(W(X), Y.float())
        loss.backward(); opt.step()
    return lambda Z: W((torch.tensor(Z, dtype=torch.float32) - mu) / sd).detach().numpy()


def auroc(s, y):
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0: return None
    return float((pos[:, None] > neg[None, :]).mean() + .5 * (pos[:, None] == neg[None, :]).mean())


res = {}
rng = np.random.default_rng(0)
for pi, p in enumerate(POS):
    for li, l in enumerate(LAYERS):
        Ftr, Fte = Xtr[:, pi, li].astype(np.float32), Xte[:, pi, li].astype(np.float32)
        for k in KP:
            itr = [i for i, d in enumerate(PKtr) if d[k] >= 0]; ite = [i for i, d in enumerate(PKte) if d[k] >= 0]
            ytr = np.array([PKtr[i][k] for i in itr]); yte = np.array([PKte[i][k] for i in ite])
            for ctl in (False, True):
                f = fit(Ftr[itr], rng.permutation(ytr) if ctl else ytr, 'ce')
                s = f(Fte[ite]); s[~Gte[ite]] = -1e9
                res[f'{p}|L{l}|path_{k}' + ('|shuf' if ctl else '')] = dict(acc=float((s.argmax(1) == yte).mean()), n=len(ite))
        for k in KF:
            Ytr = np.stack([d[k] for d in FKtr]); Yte = np.stack([d[k] for d in FKte])
            for ctl in (False, True):
                f = fit(Ftr, Ytr[rng.permutation(len(Ytr))] if ctl else Ytr, 'bce')
                s = f(Fte); a = [auroc(s[i][Gte[i]], Yte[i][Gte[i]]) for i in range(len(Yte))]; a = [x for x in a if x is not None]
                res[f'{p}|L{l}|front_{k}' + ('|shuf' if ctl else '')] = dict(auroc=float(np.mean(a)) if a else None, n=len(a))
    print(time.strftime('%H:%M'), f'probes at {p} done', flush=True)
for j in range(6):  # the paper's readout
    for k in KF:
        m = []
        for i in range(len(test)):
            g = Gte[i]; y = FKte[i][k] & g
            if y.any() and (g & ~y).any(): m.append(LNte[i, j][y].sum() / LNte[i, j][g].sum())
        res[f'lens|lat{j}|front_{k}'] = dict(mass=float(np.mean(m)), n=len(m),
                                             base=float(np.mean([(FKte[i][k] & Gte[i]).sum() / Gte[i].sum() for i in range(len(test))])))
f3 = lambda v: '  - ' if v is None else f'{v:.2f}'
for l in LAYERS:
    print(f'L{l} path_k acc (shuf):', ' | '.join(f"{p}: " + ' '.join(f"{f3(res[f'{p}|L{l}|path_{k}']['acc'])}({f3(res[f'{p}|L{l}|path_{k}|shuf']['acc'])})" for k in KP) for p in POS))
    print(f'L{l} front_k auroc (shuf):', ' | '.join(f"{p}: " + ' '.join(f"{f3(res[f'{p}|L{l}|front_{k}']['auroc'])}({f3(res[f'{p}|L{l}|front_{k}|shuf']['auroc'])})" for k in KF) for p in POS))
print('lens front_k mass (base):', ' | '.join(f"lat{j}: " + ' '.join(f"{res[f'lens|lat{j}|front_{k}']['mass']:.2f}({res[f'lens|lat{j}|front_{k}']['base']:.2f})" for k in KF) for j in range(6)))
json.dump(dict(method=method, n_train=NTR, symbols=SYM, res=res), open(out, 'w'))
print('DONE')
