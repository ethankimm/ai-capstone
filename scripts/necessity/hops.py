"""t26 hop count: how many recurrent edges does the information that changes the answer cross? Pre-registered in
scripts/codi_null_audit/prereg/t_predictions.md (t26; t31-t33 for the QUAL / FIRST / coconut_rc options).

  ../codi/.venv/bin/python scripts/necessity/hops.py <model> <device> <out json>
  model: gated | sim | simsf | startlast | relay | stab   (RuleChain valid L2-L6)
         coconut_rc                                      (our Coconut-stack RuleChain c1m6, synthetic_data_v2/depth2 test L2-L6)
         simcoconut_rc                                   (Kevin's RuleChain SIM-Coconut pilot_k8_simcot_c1m6/checkpoint_26, same data, t37)
         gsm8k_codi | gsm8k_sim | gsm8k_coconut          (GSM8K test, the 1471 twin candidates of t22)
         gsm8k_simcoconut                                (released internlm/SIM_COT-GPT2-Coconut checkpoint_28, t36)
         gsm8k_llama                                     (released zen-E/CODI-llama3.2-1b-Instruct, t47; run with FIRST=0)
         gsm8k_simllama1b                                (released internlm/SIM_COT-LLaMA3-CODI-1B, t49; FIRST=0)
         gsm8k_simllama3b                                (released internlm/SIM_COT-LLaMA3-CODI-3B, t50, bf16; FIRST=0)
  env NPAIRS=N (GSM8K): only the first N candidate pairs (t50 dtype control).
  env SMOKE=1: a few pairs plus the write-back checks. CAP (default 80) recipients per (level, edit) cell on RuleChain.
  env LEVELS (default 2,3,4,5,6; t41: 7) RuleChain levels.
  env QUAL=gold (default, t26): a pair counts only if the model answers both questions right; hops = first n from which the answer is
      D's gold. QUAL=own (t31): recipients need not be right; a pair counts if the model's own answers to R and D differ (both parse);
      hops = first n from which the answer is the model's own D answer. Right/wrong is recorded per pair.
  env FIRST=1 (t32, GSM8K only): also score the first answer token, teacher-forced after the answer delimiter on a copy of the cache, so
      no generated answer token can carry information (answer tokens are discrete loop crossings of their own).

Recurrent edges: question pass -> z_0, pass s -> z_{s+1}. Attention reads other positions at the same layer and adds no layers, so the
only way past one pass of depth is across these edges. Hop-limited recomputation on a minimal pair (recipient R, donor D):
  M_0: D's question, every latent input z_s set to R's natural z_s        -> donor info reaches the answer with 0 hops
  M_n: D's question, z_s set to the natural z_s produced inside M_{n-1}   -> <= n hops
  M_K (K = 6) is D's own run (checked exactly). Only z vectors move between runs, never keys/values, so token lengths need not match.
F_n = share of qualified pairs whose M_n answer is D's. hops* = smallest n with D's answer from n on.
"""
from __future__ import annotations

import copy
import gc
import json
import os
import random
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

MODEL, DEV, OUT = sys.argv[1:4]
SMOKE = int(os.environ.get('SMOKE', '0')); CAP = int(os.environ.get('CAP', '80'))
QUAL = os.environ.get('QUAL', 'gold'); FIRST = int(os.environ.get('FIRST', '0'))
LV = [int(x) for x in os.environ.get('LEVELS', '2,3,4,5,6').split(',')]  # t41: LEVELS=7
assert QUAL in ('gold', 'own'), QUAL
RC = MODEL in ('gated', 'sim', 'simsf', 'startlast', 'relay', 'stab', 'coconut_rc', 'simcoconut_rc')
assert not (FIRST and RC), 'FIRST is for GSM8K (RuleChain answers are already one token at one position)'
BASE = adapters.BASE


# ---------------------------------------------------------------- first answer token (t32)
def _first_prefix(ad):
    """Token ids teacher-forced after the last latent, up to the answer: GSM8K CODI <eot> 'The answer is:'; Coconut <end-latent> '###'."""
    if isinstance(ad, adapters.GSM8KCODI):
        return [ad.m.eot_id] + ad.tok.encode('The answer is:')
    return [ad.el] + ad.tok.encode('###')


@torch.no_grad()
def first_token(ad, past):
    ids = torch.tensor(_first_prefix(ad), device=ad.dev); p = copy.deepcopy(past)
    if isinstance(ad, adapters.GSM8KCODI):
        o = ad.m.codi(inputs_embeds=ad.emb(ids)[None], use_cache=True, past_key_values=p)
        return int(o.logits[0, -1, :ad.m.codi.config.vocab_size - 1].argmax())
    o = ad.body(inputs_embeds=ad.embed(ids)[None], use_cache=True, past_key_values=p)
    return int(ad.head(o.last_hidden_state[0, -1])[:ad.vocab].argmax())


# ---------------------------------------------------------------- hop-limited recomputation
@torch.no_grad()
def nat_run(ad, rows, feed=None):
    """Latent loop; feed = list of K input vectors [B, d] (None: natural). Returns (answers, nat, first) with nat[s] = the input pass s
    would get naturally (nat[0] = question pass output; nat[s+1] = pass s output); first = first answer token id (FIRST) or None."""
    ctx, past, z = ad.begin(rows)
    nat = [z]
    for s in range(ad.n_latents):
        zin = feed[s].to(device=z.device, dtype=z.dtype) if feed is not None else z
        past, z = ad.latent(ctx, past, zin, s)
        nat.append(z)
    ft = first_token(ad, past) if FIRST else None
    return ad.finish(ctx, past), [x.detach() for x in nat], ft


@torch.no_grad()
def hop_runs(ad, R, D):
    aR, natR, fR = nat_run(ad, R)
    aD, natD, fD = nat_run(ad, D)
    K = ad.n_latents; feed = natR[:K]; A = []; FT = []
    for n in range(K + 1):
        a, nat, ft = nat_run(ad, D, feed)
        A.append(list(a)); FT.append(ft); feed = nat[:K]
    return list(aR), list(aD), A, (fR, fD, FT)


# ---------------------------------------------------------------- RuleChain pairs
sys.path.insert(0, str(adapters.REPO / 'scripts/codi_null_audit'))
NUM = re.compile(r'\d+')


def rc_edit(q, pos, rng, parse, walk):
    """Edit the start value (pos 0) or one number of rule pos (1..L), same digit count; v_pos must change, all values in 0..99 and the
    answer must change. Returns (new question, new final) or None."""
    x0, rules = parse(q); old = walk(x0, rules)[0]
    if pos == 0:
        m = re.search(r'Start with (\d+)\.', q); spans = [m.span(1)]
    else:
        m = re.search(rf'Rule {pos}: (.*?)(?= Rule {pos + 1}: | Start with | What is)', q)
        if m is None: return None
        spans = [(m.start(1) + x.start(), m.start(1) + x.end()) for x in NUM.finditer(m.group(1))]
    for _ in range(60):
        a, b = rng.choice(spans); s = q[a:b]; nd = len(s)
        v = rng.randint(10 ** (nd - 1) if nd > 1 else 1, 10 ** nd - 1)
        if v == int(s): continue
        nq = q[:a] + str(v) + q[b:]
        try:
            nx, nr = parse(nq)
        except Exception:
            continue
        new = walk(nx, nr)[0]
        if min(new) < 0 or max(new) > 99 or new[-1] == old[-1] or new[pos] == old[pos] or new[:pos] != old[:pos]: continue
        return nq, new[-1]
    return None


def rc_pairs(ad, rows, parse, walk):
    """Recipients (gold: answered right; own: all), then one edit per (recipient, position); CAP recipients per (level, position)."""
    ans = [None] * len(rows); order = sorted(range(len(rows)), key=lambda i: len(rows[i]['question']))
    for i in range(0, len(order), ad.batch_size):
        ii = order[i:i + ad.batch_size]; a, _, _ = nat_run(ad, [rows[k] for k in ii])
        for k, x in zip(ii, a): ans[k] = x
    by = defaultdict(list); right = defaultdict(int)
    for i, r in enumerate(rows):
        ok = ans[i] == int(r['answer']); right[int(r['level'])] += ok
        if ok or QUAL == 'own': by[int(r['level'])].append(i)
    acc = {L: right[L] / sum(int(r['level']) == L for r in rows) for L in sorted(right)}
    print('clean acc by level', {L: round(a, 3) for L, a in acc.items()}, flush=True)
    P = []
    for L in sorted(by):
        for pos in range(L + 1):
            rng = random.Random(1000 * L + pos); idx = by[L][:]; rng.shuffle(idx); n = 0
            for i in idx:
                e = rc_edit(rows[i]['question'], pos, rng, parse, walk)
                if e is None: continue
                P.append(dict(level=L, edit='start' if pos == 0 else f'rule{pos}', s=L if pos == 0 else L - pos + 1, id=rows[i]['id'],
                              R=dict(question=rows[i]['question'], answer=int(rows[i]['answer'])), D=dict(question=e[0], answer=e[1])))
                n += 1
                if n >= (4 if SMOKE else CAP): break
    return P, acc


# ---------------------------------------------------------------- models
if RC:
    import rc as RCmod  # noqa: E402  (parse / walk)
    if MODEL in ('coconut_rc', 'simcoconut_rc'):
        ck = 'gpt2_depth2_coconut_c1m6/checkpoint_10' if MODEL == 'coconut_rc' else 'pilot_k8_simcot_c1m6/checkpoint_26'
        ad = adapters.GridCoconut('rulechain', DEV, 'coconut', max_new=16, path=str(adapters.REPO / 'rulechain-checkpoints' / ck))
        rows = [r for r in json.load(open(BASE / 'synthetic_data_v2/depth2/test.json')) if int(r['level']) in LV]
    else:
        ad = adapters.RuleChainCODI(MODEL, DEV)
        data = 'synthetic_data_v2_l1to7w_startlast' if MODEL in ('gated', 'sim', 'startlast') else 'synthetic_data_v2_l1to7w'
        rows = [r for r in json.load(open(BASE / data / 'depth2/valid.json')) if int(r['level']) in LV]
    if SMOKE: rows = [r for r in rows if int(r['level']) in (3, 4)][:120]
    same = lambda a, b: a == b
    parses = lambda a: isinstance(a, int) and a != -999
    P, clean_acc = rc_pairs(ad, rows, RCmod.parse, RCmod.walk)
else:
    from latentreasoning.eval.metrics import is_correct  # noqa: E402
    if MODEL == 'gsm8k_coconut':
        ad = adapters.GridCoconut('gsm8k', DEV, 'coconut')
    elif MODEL == 'gsm8k_simcoconut':  # same Coconut format (base_causallm.*); SIM-CoT's step decoder (expainable_llm.*) is dropped
        ad = adapters.GridCoconut('gsm8k', DEV, 'coconut', path=str(BASE / 'grid_checkpoints/simcot_gpt2_coconut/checkpoint_28'))
        ad.n_latents = 10  # SIM-CoT gsm_simcot.yaml: c_thought 2 x max_latent_stage 5 (200-item check: .39 at 10, .295 at 6)
    elif MODEL in adapters.LLAMA_GSM8K:  # t47: released zen-E/CODI-llama3.2-1b-Instruct; t49 / t50: internlm SIM-CODI 1B / 3B
        ad = adapters.LLAMA_GSM8K[MODEL][1](DEV)
    else:
        ad = adapters.GSM8KCODI(DEV, ckpt=str(BASE / 'grid_checkpoints/simcot_gpt2_codi') if MODEL == 'gsm8k_sim' else None)
    cand = json.load(open(os.environ.get('CANDS', BASE / 'codi_audit_artifacts/t22/gsm8k_twin_candidates.json')))  # t27a: CANDS = long twins
    if SMOKE: cand = cand[:30]
    if int(os.environ.get('NPAIRS', '0')): cand = cand[:int(os.environ['NPAIRS'])]  # t50 dtype control: first N pairs
    same = lambda a, b: a is not None and a != '' and is_correct(str(a), str(b))
    parses = lambda a: a is not None and a != ''
    P = [dict(level=c['chain_len'], edit=f"step{c['step']}", s=c['chain_len'] - c['step'], id=c['recipient_idx'],
              R=dict(question=c['recipient_question'], answer=c['recipient_gold']), D=dict(question=c['twin_question'], answer=c['twin_answer']))
         for c in cand]
    clean_acc = None
print(time.strftime('%H:%M'), f'{MODEL}: {len(P)} candidate pairs, QUAL={QUAL} FIRST={FIRST}', flush=True)

# ---------------------------------------------------------------- smoke checks
if SMOKE:
    r0 = [P[0]['R']]; a0, nat0, _ = nat_run(ad, r0); a1, _, _ = nat_run(ad, r0, nat0[:ad.n_latents])
    assert list(a0) == list(a1), 'own natural z write-back changed the answer'
    print('  own write-back check passes', flush=True)


def qualifies(aR, aD, p):
    if QUAL == 'gold': return bool(same(aR, p['R']['answer']) and same(aD, p['D']['answer']))
    return bool(parses(aR) and parses(aD) and not same(aR, aD))


def first_agrees(a, ft):
    """Teacher-forced first token is the start of the greedy answer (sanity check for FIRST)."""
    t = ad.tok.decode([ft]).strip().replace(',', '')
    return bool(t) and str(a).replace(',', '').startswith(t)


# ---------------------------------------------------------------- main loop
res = []; t0 = time.time(); B = ad.batch_size; memo = {}
K = ad.n_latents
for i in range(0, len(P), B):
    pp = P[i:i + B]
    if B == 1:
        p = pp[0]; key = p['R']['question']
        if key not in memo: memo[key] = list(nat_run(ad, [p['R']])[0])
        aR0 = memo[key][0]
        if QUAL == 'gold' and not same(aR0, p['R']['answer']):
            res.append(dict(p, qualified=False, aR=aR0)); continue
        dk = p['D']['question']
        if dk not in memo: memo[dk] = list(nat_run(ad, [p['D']])[0])
        aD0 = memo[dk][0]
        if not qualifies(aR0, aD0, p):
            res.append(dict(p, qualified=False, aR=aR0, aD=aD0)); continue
    aR, aD, A, (fR, fD, FT) = hop_runs(ad, [p['R'] for p in pp], [p['D'] for p in pp])
    for b, p in enumerate(pp):
        rec = dict(p, aR=aR[b], aD=aD[b], A=[A[n][b] for n in range(K + 1)])
        rec['qualified'] = qualifies(aR[b], aD[b], p)
        rec['R_right'] = bool(same(aR[b], p['R']['answer'])); rec['D_right'] = bool(same(aD[b], p['D']['answer']))
        rec['mk_matches_donor'] = A[K][b] == aD[b]
        tgtD, tgtR = (p['D']['answer'], p['R']['answer']) if QUAL == 'gold' else (aD[b], aR[b])
        fol = [bool(same(A[n][b], tgtD)) for n in range(K + 1)]
        rec['follows'] = fol; rec['keeps'] = [bool(same(A[n][b], tgtR)) for n in range(K + 1)]
        rec['hops'] = next((n for n in range(K + 1) if all(fol[n:])), None)
        if FIRST:
            rec['ftR'], rec['ftD'], rec['FT'] = fR, fD, FT
            rec['first_agree'] = [first_agrees(aR[b], fR), first_agrees(aD[b], fD)]
            rec['first_qual'] = fR != fD
            ff = [FT[n] == fD for n in range(K + 1)]
            rec['follows_first'] = ff; rec['hops_first'] = next((n for n in range(K + 1) if all(ff[n:])), None)
        res.append(rec)
    if (i // B) % (25 if B == 1 else 5) == 0:
        nq = sum(r['qualified'] for r in res)
        print(time.strftime('%H:%M'), f'pair {i}/{len(P)} qualified {nq} {time.time() - t0:.0f}s', flush=True)
    if len(memo) > 600: memo.clear()
    gc.collect()
    if DEV == 'mps': torch.mps.empty_cache()
    if SMOKE and sum(r['qualified'] for r in res) >= 8: break

# ---------------------------------------------------------------- summary
Q = [r for r in res if r['qualified']]
bad = [r for r in res if 'mk_matches_donor' in r and not r['mk_matches_donor']]
ran = sum('mk_matches_donor' in r for r in res)
print(f'{MODEL}: qualified {len(Q)} / {len(res)}; M_K = donor run on {ran - len(bad)} of {ran} pairs run', flush=True)


def curve(xs, key='hops', fkey='follows'):
    xs = [x for x in xs if x.get(key) is not None]
    if not xs: return None
    out = dict(n=len(xs), F=[float(np.mean([x[fkey][n] for x in xs])) for n in range(K + 1)],
               hops_mean=float(np.mean([x[key] for x in xs])), hops_hist=[sum(x[key] == n for x in xs) for n in range(K + 1)])
    if fkey == 'follows': out['Rk'] = [float(np.mean([x['keeps'][n] for x in xs])) for n in range(K + 1)]
    return out


def split(xs, **kw):
    return dict(all=curve(xs, **kw), by_s={s: curve([x for x in xs if x['s'] == s], **kw) for s in sorted({x['s'] for x in xs})},
                by_level={L: curve([x for x in xs if x['level'] == L], **kw) for L in sorted({x['level'] for x in xs})})


right = [x for x in Q if x['R_right'] and x['D_right']]; wrong = [x for x in Q if not (x['R_right'] and x['D_right'])]
summ = dict(split(Q), by_level_edit={f"L{L}_{e}": curve([x for x in Q if x['level'] == L and x['edit'] == e])
                                     for L, e in sorted({(x['level'], x['edit']) for x in Q})},
            both_right=split(right), any_wrong=split(wrong))
if FIRST:
    Qf = [x for x in Q if x['first_qual']]
    summ['first'] = dict(split(Qf, key='hops_first', fkey='follows_first'),
                         agreement=float(np.mean([a for x in Q for a in x['first_agree']])) if Q else None,
                         n_first_qual=len(Qf), full_vs_first_equal=float(np.mean([x['hops'] == x['hops_first'] for x in Qf])) if Qf else None)
f = lambda c: '-' if c is None else f"n={c['n']:4d} F " + ' '.join(f'{v:.2f}' for v in c['F']) + f" | hops {c['hops_mean']:.2f}"
print('all  ', f(summ['all']))
for s, c in summ['by_s'].items(): print(f's={s:<3}', f(c))
for L, c in summ['by_level'].items(): print(f'L{L:<3} ', f(c))
if QUAL == 'own':
    print(f'both right n={len(right)}, any wrong n={len(wrong)}')
    for s, c in summ['any_wrong']['by_s'].items(): print(f'wrong s={s:<3}', f(c))
if FIRST:
    print(f"first token: agreement {summ['first']['agreement']}, n {summ['first']['n_first_qual']}, "
          f"hops equal to full-answer hops {summ['first']['full_vs_first_equal']}")
    for s, c in summ['first']['by_s'].items(): print(f'first s={s:<3}', f(c))
json.dump(dict(model=MODEL, qual=QUAL, first=FIRST, clean_acc=clean_acc, n_candidates=len(P), pairs=res, summary=summ,
               mk_mismatch=len(bad)), open(OUT, 'w'), default=str)
print('DONE', f'{time.time() - t0:.0f}s')
