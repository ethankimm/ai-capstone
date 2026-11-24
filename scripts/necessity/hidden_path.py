"""t34 hidden-path detection: can a monitor reading the latents tell a property of the model's reasoning path that a monitor seeing only
the question pass and the answer cannot? Pre-registered in scripts/codi_null_audit/prereg/t_predictions.md (t34).

  ../codi/.venv/bin/python scripts/necessity/hidden_path.py <model> <device> <out json>     (gated sim simsf startlast relay stab)
  simcoconut_rc (t37): Kevin's RuleChain SIM-Coconut (Coconut stack, batch 1); fit on synthetic_data_v2/depth2 train, scored on its test
  split (the split its run record used; valid has only 167 items per level). Q = the <start-latent> position, Z = pass outputs fed back.
  env SMOKE=1: L4 only, 300 fit / 200 test items (code check).
  env LEVELS (default 4,5,6). env SAVE_ITEMS=1 (t40): also store per-item ids, labels and predictions (QA, QAZ, QAH) for joins.
  env PER_POS=1 (t38c): also fit each latent position alone: Z1..Z5 (z_j fed after pass j-1) and H0..H4 (pass j, layer 8).

Target: the branch taken at rule k (condition held or not), for the middle rules k = 2 .. L-1 of RuleChain L4-L6. Rule 1 depends only on
the start value (in the question) and rule L is close to the answer, so both are left out. Labels are the gold walk; scored on all
valid items and on items the model answers right (where its path should be the gold one).
Feature sets, per item (one forward of the model's own natural run):
  Q   question pass at <bot> (the last question position), layers 4, 8, 12, concatenated (2,304)
  A   the model's own answer, one-hot (100)
  QA  Q + A = the baseline: what a monitor sees without the latents
  Z   the latent inputs z_1 .. z_5 (fed back after 1-5 passes), concatenated (3,840)
  H   latent passes 0-4 at layer 8, concatenated (3,840): where keys/values carry values in models that use attention entries
  QAZ, QAH  the baseline plus each latent set
Probe: ridge regression to +-1 (standardised features), penalty from a held-out 15% of the fit items, one eigendecomposition per
(level, feature set) shared by every target. Score = balanced accuracy on valid (chance .5). Shuffled-label control on QAZ.
Fit on 2,000 train items per level (seed 30); valid = all items of the level.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adapters  # noqa: E402
sys.path.insert(0, str(adapters.REPO / 'scripts/codi_null_audit'))
import rc as RC  # noqa: E402

MODEL, DEV, OUT = sys.argv[1:4]
SMOKE = int(os.environ.get('SMOKE', '0'))
LEVELS = [4] if SMOKE else [int(x) for x in os.environ.get('LEVELS', '4,5,6').split(',')]; NFIT = 300 if SMOKE else 2000; NTEST = 200 if SMOKE else None
QL = [4, 8, 12]; HL = 8; ZS = [1, 2, 3, 4, 5]; HS = [0, 1, 2, 3, 4]
COC = MODEL == 'simcoconut_rc'
if COC:
    ad = adapters.GridCoconut('rulechain', DEV, 'coconut', max_new=16,
                              path=str(adapters.REPO / 'rulechain-checkpoints/pilot_k8_simcot_c1m6/checkpoint_26'))
    DD = adapters.BASE / 'synthetic_data_v2/depth2'; VA_SPLIT = 'test.json'
else:
    ad = adapters.RuleChainCODI(MODEL, DEV); VA_SPLIT = 'valid.json'
    DD = adapters.BASE / ('synthetic_data_v2_l1to7w_startlast' if MODEL in ('sim', 'startlast', 'gated') else 'synthetic_data_v2_l1to7w') / 'depth2'


@torch.no_grad()
def run_coconut(rows):
    """Coconut latent loop (GridCoconut begin / latent / finish), one item at a time, hidden states kept."""
    Qs, Zs, Hs, ans = [], [], [], []
    for r in rows:
        ids = torch.tensor([ad.qids(r['question']) + [ad.sl]], device=ad.dev)
        o = ad.body(input_ids=ids, use_cache=True, output_hidden_states=True)
        Qs.append(torch.cat([o.hidden_states[l][:, -1] for l in QL], -1)); past = o.past_key_values
        z = o.last_hidden_state[:, -1]; Z = [z]; H = []
        for s in range(ad.n_latents):
            o = ad.body(inputs_embeds=z[:, None], use_cache=True, past_key_values=past, output_hidden_states=True)
            past = o.past_key_values; H.append(o.hidden_states[HL][:, -1]); z = o.last_hidden_state[:, -1]; Z.append(z)
        ans += ad.finish(dict(Q=ids.size(1)), past)
        Zs.append(torch.cat([Z[s] for s in ZS], -1)); Hs.append(torch.cat([H[s] for s in HS], -1))
    f = lambda t: torch.cat(t).float().cpu().numpy()
    return f(Qs), f(Zs), f(Hs), ans


@torch.no_grad()
def run(rows):
    if COC: return run_coconut(rows)
    """The adapter's begin / latent / finish (RuleChainCODI), with hidden states kept. Returns Q, Z, H, answers."""
    b = ad.h.encode([dict(question=r['question'].strip(), steps=['x'], answer='0') for r in rows], ad.tok, ad.m, ad.dev)
    ids = b['encoder_input_ids']; mask = b['encoder_attention_mask'].long(); W = ids.size(1)
    g = ad.RuleGate(ids, mask) if ad.gate else None
    o = ad.body(input_ids=ids, use_cache=True, attention_mask=g.question(torch.float32) if ad.gate else mask,
                position_ids=ad.qpos(mask), output_hidden_states=True)
    Q = torch.cat([o.hidden_states[l][:, -1] for l in QL], -1)
    ctx = dict(W=W, g=g, pi=ad.PadInvariant(mask), B=ids.size(0)); past = o.past_key_values
    z = ad.m.prj(o.last_hidden_state[:, -1:, :])[:, 0]; Z = [z]; H = []
    for s in range(ad.n_latents):
        if ad.gate: ctx['pi'].mask[:, :ctx['W']] = ctx['g'].latent(s)
        o = ad.body(inputs_embeds=z[:, None], use_cache=True, past_key_values=past, output_hidden_states=True, **ctx['pi'].next(1))
        past = o.past_key_values; H.append(o.hidden_states[HL][:, -1]); z = ad.m.prj(o.last_hidden_state[:, -1:, :])[:, 0]; Z.append(z)
    ans = ad.finish(ctx, past)
    f = lambda t: t.float().cpu().numpy()
    return f(Q), f(torch.cat([Z[s] for s in ZS], -1)), f(torch.cat([H[s] for s in HS], -1)), ans


def feats(items):
    Qs, Zs, Hs, As = [], [], [], []
    order = sorted(range(len(items)), key=lambda i: len(items[i]['question']))
    for i in range(0, len(order), 32):
        q, z, h, a = run([items[k] for k in order[i:i + 32]]); Qs.append(q); Zs.append(z); Hs.append(h); As += a
    inv = np.argsort(order)
    Q, Z, H = (np.concatenate(x)[inv] for x in (Qs, Zs, Hs)); A = np.array(As)[inv]
    Aoh = np.zeros((len(items), 100), np.float32); ok = (A >= 0) & (A < 100); Aoh[np.arange(len(items))[ok], A[ok]] = 1
    BR = np.array([RC.walk(*RC.parse(r['question']))[1] for r in items], bool)
    right = A == np.array([int(r['answer']) for r in items])
    return dict(Q=Q, A=Aoh, Z=Z, H=H), BR, right


def ridge(Xa, Ya, Xb):
    """Ya [n, t] in {-1, +1}; returns predictions on Xb [m, t]; penalty per target from a held-out 15% of a."""
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6; Xa = (Xa - mu) / sd; Xb = (Xb - mu) / sd
    def solver(X, Y):
        w, V = np.linalg.eigh(X.T @ X); ym = Y.mean(0); XY = V.T @ (X.T @ (Y - ym))
        return lambda lam: (V @ (XY / (np.maximum(w, 0) + lam)[:, None]), ym)
    c = int(.85 * len(Xa)); f = solver(Xa[:c], Ya[:c]); lams = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5)
    held = np.array([[bacc(Ya[c:, t] > 0, (Xa[c:] @ f(l)[0][:, t] + f(l)[1][t]) > 0) for t in range(Ya.shape[1])] for l in lams])
    best = [lams[i] for i in held.argmax(0)]; F = solver(Xa, Ya)
    return np.stack([(Xb @ F(l)[0][:, t] + F(l)[1][t]) > 0 for t, l in enumerate(best)], 1)


def bacc(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    if y.all() or (~y).all(): return float('nan')
    return float(((p & y).sum() / y.sum() + (~p & ~y).sum() / (~y).sum()) / 2)


SETS = dict(Q=['Q'], A=['A'], QA=['Q', 'A'], Z=['Z'], H=['H'], QAZ=['Q', 'A', 'Z'], QAH=['Q', 'A', 'H'])
PER_POS = int(os.environ.get('PER_POS', '0'))
if PER_POS:
    SETS.update({f'Z{z}': [f'Z{z}'] for z in ZS}); SETS.update({f'H{h}': [f'H{h}'] for h in HS})


def split_pos(F):
    if PER_POS:
        d = F['Z'].shape[1] // len(ZS)
        for c, z in enumerate(ZS): F[f'Z{z}'] = F['Z'][:, c * d:(c + 1) * d]
        for c, h in enumerate(HS): F[f'H{h}'] = F['H'][:, c * d:(c + 1) * d]
    return F
tr_all = json.load(open(DD / 'train.json')); va_all = json.load(open(DD / VA_SPLIT)); RES = dict(model=MODEL, levels={})
rng = np.random.default_rng(0); t0 = time.time()
for L in LEVELS:
    tr = [r for r in tr_all if int(r['level']) == L]; tr = [tr[i] for i in np.random.default_rng(30).permutation(len(tr))[:NFIT]]
    va = [r for r in va_all if int(r['level']) == L][:NTEST]
    Fa, Ba, _ = feats(tr); Fb, Bb, right = feats(va); Fa, Fb = split_pos(Fa), split_pos(Fb)
    ks = list(range(2, L)); Ya = np.where(Ba[:, [k - 1 for k in ks]], 1.0, -1.0); Yb = Bb[:, [k - 1 for k in ks]]
    cells = {}; SAVE = int(os.environ.get('SAVE_ITEMS', '0')); items = {}
    for name, parts in SETS.items():
        Xa = np.concatenate([Fa[p] for p in parts], 1); Xb = np.concatenate([Fb[p] for p in parts], 1)
        P = ridge(Xa, Ya, Xb)
        if SAVE and name in ('QA', 'QAZ', 'QAH'): items[name] = P.astype(int).tolist()
        for j, k in enumerate(ks):
            cells[f'{name}|r{k}'] = dict(all=bacc(Yb[:, j], P[:, j]), right=bacc(Yb[right, j], P[right, j]))
        if name == 'QAZ':
            Ps = ridge(Xa, Ya[rng.permutation(len(Ya))], Xb)
            for j, k in enumerate(ks): cells[f'QAZ_shuf|r{k}'] = dict(all=bacc(Yb[:, j], Ps[:, j]))
    base_rate = {f'r{k}': float(Bb[:, k - 1].mean()) for k in ks}
    RES['levels'][L] = dict(n_fit=len(tr), n_test=len(va), acc=float(right.mean()), base_rate=base_rate, cells=cells)
    if SAVE: RES['levels'][L]['items'] = dict(ids=[r['id'] for r in va], ks=ks, y=Yb.astype(int).tolist(), right=right.tolist(), pred=items)
    show = lambda n: ' '.join(f"{cells[f'{n}|r{k}']['all']:.2f}" for k in ks)
    print(time.strftime('%H:%M'), MODEL, f'L{L} acc {right.mean():.3f} | QA {show("QA")} | Z {show("Z")} | H {show("H")} | QAZ {show("QAZ")} '
          f'| QAH {show("QAH")} | shuf {show("QAZ_shuf")} ({time.time() - t0:.0f}s)', flush=True)
    json.dump(RES, open(OUT, 'w'))
print('DONE')
