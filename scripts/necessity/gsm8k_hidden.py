"""t39 GSM8K hidden-step monitor, crossing-targeted. Pre-registered in scripts/codi_null_audit/prereg/t_predictions.md (t39).

  ../codi/.venv/bin/python scripts/necessity/gsm8k_hidden.py <model> <device> <out json>
  model: gsm8k_codi | gsm8k_sim | gsm8k_coconut | gsm8k_simcoconut
  env SMOKE=1: 150 fit / 120 test items (code check). env NFIT (default 2000).

Question: does reading the latent loop tell a monitor something about the model's intermediate steps that the question pass and
the answer do not, and is that extra signal concentrated where the hop count (t31 / t36) says the answer depends on crossings?
Targets per item, from the GSM8K-Aug calculator rationale (<<a*b=c>> per step), for steps k = 1, 2, 3:
  op_k   step k uses * or / (vs only + and -)
  big_k  step k's result >= the median step-k result of the fit set
  wrong  the model's own final answer is wrong
Features, one natural run per item (fp32):
  Q   last question position (the position whose output starts the loop), layers 4, 8, 12 (2,304)
  A   the model's own answer: log10(1+|a|), sign, integer flag, a mod 2, last digit one-hot, number of digits (16)
  Z   the latent inputs z_1 .. z_{K-1} fed back after passes 0 .. K-2
  H   every latent pass at layer 8 (where values sit in keys / values in released CODI)
  QA (baseline), QAZ, QAH; and each position alone: Z1.., H0..
Probe: ridge in dual form (n << d), standardised, penalty from a held-out 15% of the fit set; balanced accuracy (chance .5).
Fit on NFIT GSM8K-Aug train items (seed 39), score on all 1,319 test items. Shuffled-label control on QAH.
Per-item test predictions are stored for the hop join: items are matched to t31 / t36 pairs by recipient question text, and the
item's hop count is the mean over its qualified pairs (both twin sets).
"""
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adapters  # noqa: E402
sys.path.insert(0, str(adapters.REPO))
from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.eval.metrics import is_correct  # noqa: E402

MODEL, DEV, OUT = sys.argv[1:4]
SMOKE = int(os.environ.get('SMOKE', '0')); NFIT = 150 if SMOKE else int(os.environ.get('NFIT', '2000')); NTEST = 120 if SMOKE else None
QL = [4, 8, 12]; HL = 8
ART = adapters.BASE / 'codi_audit_artifacts'
HOPS = {'gsm8k_simcoconut': [ART / 't36/gsm8k_simcoconut.json', ART / 't36/long_gsm8k_simcoconut.json']}.get(
    MODEL, [ART / f't31/{MODEL}.json', ART / f't31/long_{MODEL}.json'])

# data first: the adapters switch the HF hub to offline mode
tr = load_gsm8k_aug(split='train', n=NFIT, seed=39); te = load_gsm8k_aug(split='test', n=None, seed=None)
COC = MODEL in ('gsm8k_coconut', 'gsm8k_simcoconut')
if MODEL == 'gsm8k_coconut':
    ad = adapters.GridCoconut('gsm8k', DEV, 'coconut')
elif MODEL == 'gsm8k_simcoconut':
    ad = adapters.GridCoconut('gsm8k', DEV, 'coconut', path=str(adapters.BASE / 'grid_checkpoints/simcot_gpt2_coconut/checkpoint_28'))
    ad.n_latents = 10  # as hops.py (t36)
else:
    ad = adapters.GSM8KCODI(DEV, ckpt=str(adapters.BASE / 'grid_checkpoints/simcot_gpt2_codi') if MODEL == 'gsm8k_sim' else None)
K = ad.n_latents; ZS = list(range(1, K)); HS = list(range(K))


@torch.no_grad()
def run(r):
    """One natural run. Returns Q [2304], Z [K-1, 768], H [K, 768], answer."""
    if COC:
        ids = torch.tensor([ad.qids(r['question']) + [ad.sl]], device=ad.dev)
        o = ad.body(input_ids=ids, use_cache=True, output_hidden_states=True); ctx = dict(Q=ids.size(1))
        nxt = lambda o: o.last_hidden_state[:, -1]
        step = lambda z, past: ad.body(inputs_embeds=z[:, None], use_cache=True, past_key_values=past, output_hidden_states=True)
    else:
        ids, attn = ad.enc(ad.m, ad.tok, r['question'].strip().replace('  ', ' '), ad.dev); ctx = dict(Q=ids.size(1))
        o = ad.m.codi(input_ids=ids, use_cache=True, output_hidden_states=True, attention_mask=attn)
        nxt = lambda o: ad.m.prj(o.hidden_states[-1][:, -1, :].unsqueeze(1))[:, 0]

        def step(z, past, s=[0]):
            m = torch.ones((1, ctx['Q'] + step.s + 1), dtype=torch.long, device=ad.dev); step.s += 1
            return ad.m.codi(inputs_embeds=z[:, None], use_cache=True, output_hidden_states=True, past_key_values=past, attention_mask=m)
        step.s = 0
    Q = torch.cat([o.hidden_states[l][0, -1] for l in QL]); past = o.past_key_values; z = nxt(o); Z = []; H = []
    for s in range(K):
        o = step(z, past); past = o.past_key_values; H.append(o.hidden_states[HL][0, -1]); z = nxt(o)
        if s < K - 1: Z.append(z[0])
    a = ad.finish(ctx, past)[0]
    f = lambda t: t.float().cpu().numpy()
    return f(Q), f(torch.stack(Z)), f(torch.stack(H)), a


def num(a):
    try: return float(str(a).replace(',', ''))
    except (TypeError, ValueError): return None


def afeat(a):
    v = num(a); x = np.zeros(16, np.float32)
    if v is None or not math.isfinite(v): x[15] = 1; return x
    iv = int(round(abs(v))) if abs(v) < 1e12 else 0
    x[0] = math.log10(1 + abs(v)); x[1] = float(v < 0); x[2] = float(v == round(v)); x[3] = iv % 2; x[4 + iv % 10] = 1
    x[14] = len(str(iv)); return x


STEP = re.compile(r'<<([^<>=]*)=([^<>]*)>>')


def steps(e):
    out = []
    for expr, res in STEP.findall(e.rationale):
        out.append((bool(re.search(r'[*/]', expr)), num(res)))
    return out


def feats(exs):
    Q, Z, H, A, ans = [], [], [], [], []; t0 = time.time()
    for i, e in enumerate(exs):
        q, z, h, a = run(dict(question=e.question)); Q.append(q); Z.append(z); H.append(h); A.append(afeat(a)); ans.append(a)
        if i % 250 == 0: print(time.strftime('%H:%M'), f'{MODEL} feats {i}/{len(exs)} ({time.time() - t0:.0f}s)', flush=True)
    Z, H = np.stack(Z), np.stack(H)
    F = dict(Q=np.stack(Q), A=np.stack(A), Z=Z.reshape(len(exs), -1), H=H.reshape(len(exs), -1))
    for j, zj in enumerate(ZS): F[f'Z{zj}'] = Z[:, j]
    for j, hj in enumerate(HS): F[f'H{hj}'] = H[:, j]
    return F, ans


def labels(exs, ans, med):
    """Y [n, 7] in {+1, -1, 0 = missing}: op_1..3, big_1..3, wrong."""
    Y = np.zeros((len(exs), 7), np.float32)
    for i, (e, a) in enumerate(zip(exs, ans)):
        st = steps(e)
        for k in range(3):
            if k < len(st):
                Y[i, k] = 1 if st[k][0] else -1
                if st[k][1] is not None: Y[i, 3 + k] = 1 if st[k][1] >= med[k] else -1
        Y[i, 6] = -1 if (a not in (None, '') and is_correct(str(a), e.answer)) else 1
    return Y


TARGETS = ['op_1', 'op_2', 'op_3', 'big_1', 'big_2', 'big_3', 'wrong']


def bacc(y, p):
    y, p = np.asarray(y) > 0, np.asarray(p) > 0
    if y.all() or (~y).all(): return float('nan')
    return float(((p & y).sum() / y.sum() + (~p & ~y).sum() / (~y).sum()) / 2)


def ridge(Xa, Ya, Xb):
    """Dual ridge per target on the rows where that target is defined; returns scores on Xb [m, t]."""
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6; Xa = (Xa - mu) / sd; Xb = (Xb - mu) / sd; Xa /= math.sqrt(Xa.shape[1]); Xb /= math.sqrt(Xa.shape[1])
    lams = (1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2); out = np.zeros((len(Xb), Ya.shape[1]), np.float32)
    for t in range(Ya.shape[1]):
        ok = Ya[:, t] != 0; X, y = Xa[ok], Ya[ok, t]; c = int(.85 * len(X))
        def fit(Xf, yf, lam):
            G = Xf @ Xf.T; ym = yf.mean(); al = np.linalg.solve(G + lam * np.eye(len(G)), yf - ym); return Xf.T @ al, ym
        sc = [bacc(y[c:], X[c:] @ w + b) for w, b in (fit(X[:c], y[:c], l) for l in lams)]
        lam = lams[int(np.nanargmax(sc))]; w, b = fit(X, y, lam); out[:, t] = Xb @ w + b
    return out


# ---------------------------------------------------------------- data
te = te[:NTEST] if NTEST else te
print(time.strftime('%H:%M'), f'{MODEL}: K={K}, fit {len(tr)}, test {len(te)}', flush=True)
Fa, ansa = feats(tr); Fb, ansb = feats(te)
med = [float(np.median([s[k][1] for e in tr if len(s := steps(e)) > k and s[k][1] is not None])) for k in range(3)]
Ya, Yb = labels(tr, ansa, med), labels(te, ansb, med)

# hop counts per test item (mean over qualified pairs where the item is the recipient)
qtext = lambda q: ' '.join(q.split())
hop = defaultdict(list); slen = {}
for f in HOPS:
    if not f.exists(): continue
    for p in json.load(open(f))['pairs']:
        if p.get('qualified') and p.get('hops') is not None:
            k = qtext(p['R']['question']); hop[k].append(p['hops']); slen[k] = p['level']
H_item = [float(np.mean(hop[qtext(e.question)])) if hop.get(qtext(e.question)) else None for e in te]
print(f'hop counts joined for {sum(h is not None for h in H_item)} / {len(te)} test items', flush=True)

SETS = dict(Q=['Q'], A=['A'], QA=['Q', 'A'], Z=['Z'], H=['H'], QAZ=['Q', 'A', 'Z'], QAH=['Q', 'A', 'H'])
SETS.update({f'Z{z}': [f'Z{z}'] for z in ZS}); SETS.update({f'H{h}': [f'H{h}'] for h in HS})
RES = dict(model=MODEL, K=K, n_fit=len(tr), n_test=len(te), median=med, targets=TARGETS, base_rate={}, cells={},
           items=dict(idx=[e.idx for e in te], hops=H_item, steps=[len(steps(e)) for e in te], y=Yb.astype(int).tolist(), pred={}),
           acc=float(np.mean(Yb[:, 6] < 0)))
for j, t in enumerate(TARGETS): RES['base_rate'][t] = float(np.mean(Yb[Yb[:, j] != 0, j] > 0))
rng = np.random.default_rng(0)
for name, parts in SETS.items():
    Xa = np.concatenate([Fa[p] for p in parts], 1); Xb = np.concatenate([Fb[p] for p in parts], 1)
    P = ridge(Xa, Ya, Xb)
    for j, t in enumerate(TARGETS):
        ok = Yb[:, j] != 0; RES['cells'][f'{name}|{t}'] = bacc(Yb[ok, j], P[ok, j])
    if name in ('QA', 'QAZ', 'QAH') or name.startswith(('Z', 'H')): RES['items']['pred'][name] = (P > 0).astype(int).tolist()
    if name == 'QAH':
        Ys = Ya.copy()
        for j in range(Ya.shape[1]): ok = Ya[:, j] != 0; Ys[ok, j] = rng.permutation(Ya[ok, j])
        Ps = ridge(Xa, Ys, Xb)
        for j, t in enumerate(TARGETS): ok = Yb[:, j] != 0; RES['cells'][f'QAH_shuf|{t}'] = bacc(Yb[ok, j], Ps[ok, j])
show = lambda n: ' '.join(f"{RES['cells'][f'{n}|{t}']:.2f}" for t in TARGETS)
print('targets', TARGETS, f"acc {RES['acc']:.3f}")
for n in ('QA', 'Q', 'A', 'Z', 'H', 'QAZ', 'QAH', 'QAH_shuf'): print(f'{n:9s}', show(n))
json.dump(RES, open(OUT, 'w'))
print('DONE')
