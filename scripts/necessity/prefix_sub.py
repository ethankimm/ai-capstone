"""t53: are the latents before an edit really the same for A and B, and where does the edit enter the loop?
Pre-registered in scripts/codi_null_audit/prereg/t_predictions.md (t53).

  ../codi/.venv/bin/python scripts/necessity/prefix_sub.py <model> <device> <out json>
  model: gated | startlast            (RuleChain valid L2-L6; pairs of t31/<model>.json)
         gsm8k_codi | gsm8k_sim       (released GPT-2 CODI / SIM-CODI; pairs of t31/gsm8k_<m>.json + t31/long_gsm8k_<m>.json)
  env CAP (default 25): RuleChain pairs per (level, edit) cell.  env SMOKE=1: a few pairs.

Pairs: the t31 pairs (QUAL=own) on which the model answers both questions right (R = recipient A, D = donor B). For each pair:
  (a) latent distance: d_j = ||z_j(R) - z_j(D)|| for the natural inputs z_0 (question pass output) .. z_K; the reference is
      ||z_j(R) - z_j(X)|| for X = the recipient of another pair (another item, same model), saved for normalising offline.
  (b) prefix substitution: for c = 0..K the start run M'_0 gets D's natural z_0..z_{c-1} and R's natural z_c..z_{K-1}; later runs
      refresh exactly as hops.py (M'_n inputs = the natural inputs produced inside M'_{n-1}). Since the first c + n inputs of M'_n are
      then D's own, M'_n for c + n >= K is D's own run. h_c = first n from which the answer is D's own and stays. h_0 is the t31 hop
      count (checked). S_c = h_0 - h_c is the refreshes saved by installing D's first c latents.
If D's latents before the edit's entry pass equal R's in effect, S_c = 0 up to the entry and S_c = c - entry after it.
"""
from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adapters  # noqa: E402

MODEL, DEV, OUT = sys.argv[1:4]
SMOKE = int(os.environ.get('SMOKE', '0')); CAP = int(os.environ.get('CAP', '25'))
A = adapters.BASE / 'codi_audit_artifacts'
RC = MODEL in ('gated', 'startlast')

if RC:
    ad = adapters.RuleChainCODI(MODEL, DEV)
    files = [A / f't31/{MODEL}.json']
    same = lambda a, b: a == b
else:
    from latentreasoning.eval.metrics import is_correct  # noqa: E402
    ad = adapters.GSM8KCODI(DEV, ckpt=str(adapters.BASE / 'grid_checkpoints/simcot_gpt2_codi') if MODEL == 'gsm8k_sim' else None)
    m = MODEL.split('_')[1]
    files = [A / f't31/gsm8k_{m}.json', A / f't31/long_gsm8k_{m}.json']
    same = lambda a, b: a is not None and a != '' and is_correct(str(a), str(b))
K = ad.n_latents

P = []
for f in files:
    for p in json.load(open(f))['pairs']:
        if p.get('qualified') and p.get('R_right') and p.get('D_right') and p.get('hops') is not None:
            P.append(dict(p, src=f.name))
if RC:  # CAP per (level, edit) cell, fixed seed
    by = defaultdict(list)
    for p in P: by[(p['level'], p['edit'])].append(p)
    rng = random.Random(53); P = []
    for k in sorted(by):
        v = by[k][:]; rng.shuffle(v); P += v[:CAP]
if SMOKE: P = P[:8 if RC else 4]
print(time.strftime('%H:%M'), f'{MODEL}: {len(P)} both-right pairs from {[f.name for f in files]}', flush=True)


@torch.no_grad()
def nat_run(rows, feed=None):
    """As hops.py: latent loop with optional fed inputs; returns (answers, natural inputs z_0..z_K)."""
    ctx, past, z = ad.begin(rows)
    nat = [z]
    for s in range(K):
        zin = feed[s].to(device=z.device, dtype=z.dtype) if feed is not None else z
        past, z = ad.latent(ctx, past, zin, s)
        nat.append(z)
    return list(ad.finish(ctx, past)), [x.detach() for x in nat]


@torch.no_grad()
def prefix_runs(R, D):
    aR, natR = nat_run(R); aD, natD = nat_run(D)
    ans = {}  # ans[c] = answers of M'_0 .. M'_{K-c} (list over n of list over batch)
    for c in range(K + 1):
        feed = [natD[j] if j < c else natR[j] for j in range(K)]; rows = []
        for n in range(K - c + 1):
            a, nat = nat_run(D, feed); rows.append(a); feed = nat[:K]
        ans[c] = rows
    return aR, aD, natR, natD, ans


res = []; ZR, ZD = [], []; t0 = time.time(); B = ad.batch_size
for i in range(0, len(P), B):
    pp = P[i:i + B]
    aR, aD, natR, natD, ans = prefix_runs([p['R'] for p in pp], [p['D'] for p in pp])
    for b, p in enumerate(pp):
        zr = torch.stack([z[b].float().cpu() for z in natR]); zd = torch.stack([z[b].float().cpu() for z in natD])
        ZR.append(zr.numpy()); ZD.append(zd.numpy())
        h, fol = [], []
        for c in range(K + 1):
            f = [bool(same(ans[c][n][b], aD[b])) for n in range(K - c + 1)] + [True] * c  # n > K - c: D's own run
            fol.append(f); h.append(next(n for n in range(K + 1) if all(f[n:])))
        rec = {k: p[k] for k in ('level', 'edit', 's', 'id', 'R', 'D', 'src')}
        rec.update(aR=aR[b], aD=aD[b], t31_hops=p['hops'], h=h, follows=fol,
                   own_ok=bool(same(aR[b], p['aR']) and same(aD[b], p['aD'])),
                   ck_last=bool(same(ans[K][0][b], aD[b])),
                   dist=[float(x) for x in (zr - zd).norm(dim=1)], normR=[float(x) for x in zr.norm(dim=1)])
        res.append(rec)
    if (i // B) % (10 if B == 1 else 2) == 0:
        print(time.strftime('%H:%M'), f'pair {i + len(pp)}/{len(P)} {time.time() - t0:.0f}s', flush=True)
    gc.collect()
    if DEV == 'mps': torch.mps.empty_cache()

# reference distance: each recipient vs the recipient of another pair with a different item (fixed permutation)
ZR = np.stack(ZR); ids = [r['id'] for r in res]; rng = np.random.default_rng(53); ref = []
for j in range(len(res)):
    cand = [x for x in rng.permutation(len(res))[:20] if ids[x] != ids[j]]
    ref.append([float(v) for v in np.linalg.norm(ZR[j] - ZR[cand[0]], axis=1)] if cand else None)
for r, x in zip(res, ref): r['ref_dist'] = x

h0_match = float(np.mean([r['h'][0] == r['t31_hops'] for r in res])) if res else None
print(f'{MODEL}: {len(res)} pairs | h_0 = t31 hops on {h0_match:.3f} | own answers reproduced {np.mean([r["own_ok"] for r in res]):.3f}'
      f' | c=K run = D on {np.mean([r["ck_last"] for r in res]):.3f}', flush=True)
mean_h = np.mean([r['h'] for r in res], axis=0)
print('mean h_c, c = 0..K:', ' '.join(f'{v:.2f}' for v in mean_h))
rel = np.nanmedian([[d / x if x else np.nan for d, x in zip(r['dist'], r['ref_dist'])] for r in res if r['ref_dist']], axis=0)
print('median d_j / ref_j, j = 0..K:', ' '.join(f'{v:.2f}' for v in rel))
json.dump(dict(model=MODEL, K=K, n=len(res), h0_match=h0_match, pairs=res), open(OUT, 'w'), default=str)
print('DONE', f'{time.time() - t0:.0f}s')
