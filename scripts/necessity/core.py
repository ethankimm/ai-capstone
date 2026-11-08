"""Model-agnostic route split / necessity check.

The check (HANDOFF_2026-11-08_NECESSITY.md): per latent site j, put a donor's value into that latent's INPUT vector only (its keys/values
reset to the recipient's own) and see whether the answer follows the donor. Only the recurrent input (final layer of pass j -> input of
pass j+1) adds serial depth; attention across positions does not.

An adapter wraps one model family and exposes a latent loop over a batch of rows:

    ctx, past, z = ad.begin(rows)            # encode the question(s); z [B, d] = input of latent pass 0
    past, z_next = ad.latent(ctx, past, z, s) # latent pass s with input z; appends its keys/values at ad.pos(ctx, s)
    answers = ad.finish(ctx, past, hide)      # answer from the cache; hide = latent sites the answer may not attend to
    ad.n_latents, ad.batch_size

`run` drives the loop with three kinds of edits (feed = replace an input, kv = overwrite a pass's stored keys/values after it ran,
hide = mask a site from the answer). `route_split` runs the four conditions per site plus the all-sites swaps on aligned
(recipient, donor, target) triples. Donor questions must have the same token length as their recipient so cache positions line up;
the pair generators in pairs.py enforce this.
"""
from __future__ import annotations

import numpy as np
import torch


def _layers(past):
    """Per-layer (key, value) tensors of a legacy tuple cache or a transformers DynamicCache (4.4x-4.5x and 4.56+ layouts)."""
    if hasattr(past, 'layers'):
        return [(l.keys, l.values) for l in past.layers]
    if hasattr(past, 'key_cache'):
        return list(zip(past.key_cache, past.value_cache))
    return [(k, v) for k, v in past]


def kv_get(past, pos):
    return [(k[:, :, pos].clone(), v[:, :, pos].clone()) for k, v in _layers(past)]


def kv_set(past, pos, src):
    for (k, v), (sk, sv) in zip(_layers(past), src):
        k[:, :, pos] = sk.to(k.device)
        v[:, :, pos] = sv.to(v.device)


@torch.no_grad()
def run(ad, rows, feed=None, kv=None, hide=()):
    """One forward of the latent loop with optional edits. feed {s: [B, d]} replaces the input of pass s; kv {s: kv list} overwrites
    pass s's keys/values after it ran (so pass s's own output, the next input, is unaffected); hide = sites the answer cannot see.
    Returns (answers list, Z {s: [B, d]}, KV {s: kv list})."""
    ctx, past, z = ad.begin(rows)
    Z, KV = {}, {}
    for s in range(ad.n_latents):
        if feed and s in feed:
            z = feed[s].to(device=z.device, dtype=z.dtype)
        Z[s] = z
        past, z = ad.latent(ctx, past, z, s)
        if kv and s in kv:
            kv_set(past, ad.pos(ctx, s), kv[s])
        KV[s] = kv_get(past, ad.pos(ctx, s))
    return ad.finish(ctx, past, hide), Z, KV


def to_cpu(res):
    """A run's (answers, Z, KV) with tensors on the CPU: tiny MPS tensors each take ~1 MB of allocator space, so stored runs go here."""
    a, Z, KV = res
    return a, {s: z.cpu() for s, z in Z.items()}, {s: [(k.cpu(), v.cpu()) for k, v in kv] for s, kv in KV.items()}


def _batches(n, bs, order=None):
    order = np.arange(n) if order is None else order
    for i in range(0, n, bs):
        yield order[i:i + bs]


def answers(ad, rows, order=None, **kw):
    out = [None] * len(rows)
    for ii in _batches(len(rows), ad.batch_size, order):
        a, _, _ = run(ad, [rows[k] for k in ii], **kw)
        for k, x in zip(ii, a):
            out[k] = x
    return out


SITE_CONDS = ('both', 'input', 'kv', 'kv_noans')


def route_split(ad, rows, donors, targets, sites=None, all_sites=True, conds=SITE_CONDS, order=None, after_batch=None, check_own=False):
    """rows: recipients. donors[j] / targets[j]: per site j (int, or 'all') a list aligned with rows giving the donor row (or None) and the
    donor-implied answer. Returns {cond_j: list of answers (None where no donor)}. order: batch order (e.g. sorted by length, for padding).
    check_own: on the first batch, assert that writing back a run's own input / own keys/values changes no answer."""
    sites = list(range(ad.n_latents)) if sites is None else list(sites)
    keys = [j for j in sites] + (['all'] if all_sites else [])
    out = {}
    for j in keys:
        cs = ('both', 'kv') if j == 'all' else conds
        for c in cs:
            out[f'{c}_{j}'] = [None] * len(rows)
        for bi, ii in enumerate(_batches(len(rows), ad.batch_size, order)):
            ii = [k for k in ii if donors[j][k] is not None]
            if not ii:
                continue
            rr = [rows[k] for k in ii]; dd = [donors[j][k] for k in ii]
            a0, Zr, KVr = run(ad, rr)
            _, Zd, KVd = run(ad, dd)
            if check_own and bi == 0 and j == keys[0]:
                s0 = 0 if j == 'all' else j
                a1, _, _ = run(ad, rr, kv={s0: KVr[s0]}); a2, _, _ = run(ad, rr, feed={s0: Zr[s0]})
                assert list(a1) == list(a0) and list(a2) == list(a0), 'own write-back changed an answer'
                print('  own write-back checks pass', flush=True)
            res = {}
            if j == 'all':
                S = range(ad.n_latents)
                res['both'] = run(ad, rr, feed={s: Zd[s] for s in S})[0]
                res['kv'] = run(ad, rr, kv={s: KVd[s] for s in S})[0]
            else:
                if 'both' in cs: res['both'] = run(ad, rr, feed={j: Zd[j]})[0]
                if 'input' in cs: res['input'] = run(ad, rr, feed={j: Zd[j]}, kv={j: KVr[j]})[0]
                if 'kv' in cs: res['kv'] = run(ad, rr, kv={j: KVd[j]})[0]
                if 'kv_noans' in cs: res['kv_noans'] = run(ad, rr, kv={j: KVd[j]}, hide=(j,))[0]
            for c, a in res.items():
                for k, x in zip(ii, a):
                    out[f'{c}_{j}'][k] = x
            if after_batch:
                after_batch()
    return out


def summarise(out, gold, clean, targets, same=lambda a, b: a == b):
    """Follow / keep rates per condition on recipients answered right with a valid donor."""
    S = {}
    for c, ans in out.items():
        j = c.rsplit('_', 1)[1]; j = j if j == 'all' else int(j)
        ok = [i for i in range(len(gold)) if same(clean[i], gold[i]) and targets[j][i] is not None and ans[i] is not None]
        S[c] = dict(n=len(ok), follows=float(np.mean([same(ans[i], targets[j][i]) for i in ok])) if ok else None,
                    keeps=float(np.mean([same(ans[i], gold[i]) for i in ok])) if ok else None)
    return S


def fmt(S, n_latents):
    f = lambda x: '  - ' if x is None else f'{x:.2f}'
    parts = []
    for j in range(n_latents):
        if f'both_{j}' in S:
            parts.append(f"j{j}: " + ' '.join(f"{c[0] if c != 'kv_noans' else 'na'} {f(S[f'{c}_{j}']['follows'])}" for c in SITE_CONDS if f'{c}_{j}' in S))
    if 'both_all' in S:
        parts.append(f"all: b {f(S['both_all']['follows'])} kv {f(S['kv_all']['follows'])} (n={S['both_all']['n']})")
    return ' | '.join(parts)
