"""Route split on RuleChain CODI models (t12's design on the RuleChain harness): does a latent's value reach the answer through the recurrent
input or through its own keys/values (attention)? Usage: t15_rc_route_split.py <codi_null_audit dir> <device> <out dir>; env MODEL =
gated | sim | startlast | relay | stab. SMOKE=1: L4, 40 items, with own-cache and own-input write-back checks.
Full: valid L3-L5 (start-first valid for relay / stab).
Donors: t1's rule (same level, seeded Random(1000 L + j), differs at v_ks, predicted answer in 0..99 and != gold), restricted to donors whose
question has the same token length as the recipient's, so latent keys/values line up position by position (pad-invariant positions).
ks per model as t1: gated min(j, L-1); sim min(j+1, L-1); startlast / relay / stab L-1 (latents carry the value before the last rule).
Conditions, site j = latent pass j (input z_j; keys/values at padded position W + j):
  both_j     donor's z_j fed into pass j, later passes recomputed (t1's swap)
  input_j    donor's z_j fed in, then pass j's keys/values reset to the recipient's own: the donor value can only travel on through the next
             input z_{j+1} (by construction ~0 at j = 5, which no pass reads)
  kv_j       recipient's own z_j, pass j's keys/values set to the donor's: only attention (later passes, answer) can carry it
  kv_noans_j as kv_j with the answer tokens masked from position W + j: only later passes reading it by attention can pass it on
  both_all / kv_all: the same for all six sites at once (pooling shows only here)
Scored on recipients answered right: follows donor (the predicted answer) / keeps own. Gate on for gated in every forward."""
import sys, json, time, os, random; sys.path.insert(0, sys.argv[1])
from harness import *
import numpy as np
from rc import *
from src.model import RuleGate
from eval_codi import PadInvariant, question_position_ids
DEV = sys.argv[2]; OUT = sys.argv[3]; SMOKE = int(os.environ.get('SMOKE', '0')); MODEL = os.environ['MODEL']; NL = 6
LV = [4] if SMOKE else [3, 4, 5]
SL = BASE + '/synthetic_data_v2_l1to7w_startlast/depth2/'; SF = BASE + '/synthetic_data_v2_l1to7w/depth2/'
CKR, DL = dict(gated=('codi_runs_l1to7w_startlast_gated/codi/run/checkpoint-17820', SL), sim=('codi_runs_l1to7w_startlast_sim/codi/run/checkpoint-16335', SL),
              startlast=('codi_runs_l1to7w_startlast/codi/run/checkpoint-17820', SL), relay=(None, SF),
              stab=('codi_runs_l1to7w_stab/codi/run/checkpoint-17820', SF))[MODEL]
GATE = MODEL == 'gated'


def ks_of(j, L):
    if MODEL == 'gated': return min(j, L - 1)
    if MODEL == 'sim': return min(j + 1, L - 1)
    return L - 1


def kv_get(past, pos):
    return [(k[:, :, pos].clone(), v[:, :, pos].clone()) for k, v in past]
def kv_set(past, pos, src):
    for (k, v), (sk, sv) in zip(past, src): k[:, :, pos] = sk; v[:, :, pos] = sv


@torch.no_grad()
def run(m, tok, rows_, dec, num, feed=None, kv=None, hide=()):
    """rows_ in batch order (no re-sorting: callers align recipients and donors). feed {j: [B,768]} replaces z_j; kv {j: kv list} overwrites
    pass j's keys/values after it runs; hide = passes the answer cannot see. Returns answers [B], Z {j: [B,768]}, KV {j: kv list}."""
    body = body_of(m)
    b = encode([dict(question=r['question'].strip(), steps=['x'], answer='0') for r in rows_], tok, m, DEV)
    ids = b['encoder_input_ids']; mask = b['encoder_attention_mask'].long(); W = ids.size(1)
    g = RuleGate(ids, mask) if GATE else None
    o = body(input_ids=ids, use_cache=True, attention_mask=g.question(torch.float32) if GATE else mask, position_ids=question_position_ids(mask))
    past = o.past_key_values; pi = PadInvariant(mask); z = m.prj(o.last_hidden_state[:, -1:, :]); Z = {}; KV = {}
    for s in range(NL):
        if feed and s in feed: z = feed[s][:, None]
        Z[s] = z[:, 0]
        if GATE: pi.mask[:, :W] = g.latent(s)
        o = body(inputs_embeds=z, use_cache=True, past_key_values=past, **pi.next(1))
        past = o.past_key_values; z = m.prj(o.last_hidden_state[:, -1:, :])
        if kv and s in kv: kv_set(past, W + s, kv[s])
        KV[s] = kv_get(past, W + s)
    if GATE: pi.mask[:, :W] = g.answer()
    for s in hide: pi.mask[:, W + s] = 0
    emb = m.get_embd(m.codi, m.model_name)(dec)[None].expand(ids.size(0), -1, -1)
    o = body(inputs_embeds=emb, use_cache=True, past_key_values=past, **pi.next(emb.size(1)))
    A = [num.get(t, -999) for t in m.codi.get_base_model().lm_head(o.last_hidden_state[:, -1, :])[:, :50257].argmax(-1).tolist()]
    return np.array(A), Z, KV


ckpt = CK['lr3'] if MODEL == 'relay' else BASE + '/' + CKR  # CK = harness's checkpoint table
m, tok, ta = load(ckpt, NL, DEV); m.eval()
dec = torch.tensor([m.eot_id] + tok.encode('The answer is:'), device=DEV)
num = {tok.encode(f"The answer is: {n}")[-1]: n for n in range(100)}; assert len(num) == 100
va = json.load(open(DL + 'valid.json')); RES = {}
COND = [f'{c}_{j}' for j in range(NL) for c in ('both', 'input', 'kv', 'kv_noans')] + ['both_all', 'kv_all']
for L in LV:
    ev = [r for r in va if int(r['level']) == L]; ev = ev[:40] if SMOKE else ev
    P = [parse(r['question']) for r in ev]; V = np.array([walk(*p)[0] for p in P]); gold = V[:, -1]
    tl = np.array([len(tok(r['question'].strip() + '\n').input_ids) for r in ev])
    order = np.argsort(tl, kind='stable')
    A0 = np.zeros(len(ev), int)
    for i in range(0, len(ev), 32):
        ii = order[i:i + 32]; A0[ii], _, _ = run(m, tok, [ev[k] for k in ii], dec, num)
    right = A0 == gold
    print(time.strftime('%H:%M'), f'{MODEL} L{L} clean acc {right.mean():.3f}', flush=True)
    out = {c: np.zeros(len(ev), int) - 999 for c in COND}; donors = {}; targets = {}
    for j in list(range(NL)) + ['all']:
        k = ks_of(L - 1, L) if j == 'all' else ks_of(j, L); rng = random.Random(1000 * L + (7 if j == 'all' else j)); dn = []; tg = []
        for i in range(len(ev)):
            cand = [d for d in range(len(ev)) if d != i and tl[d] == tl[i]]; rng.shuffle(cand); pick = None
            for d in cand:
                if V[d, k] == V[i, k]: continue
                vals = walk(int(V[d, k]), P[i][1][k:])[0]
                if min(vals) < 0 or max(vals) > 99 or vals[-1] == gold[i]: continue
                pick = (d, vals[-1]); break
            dn.append(pick[0] if pick else i); tg.append(pick[1] if pick else -1)
        donors[j] = np.array(dn); targets[j] = np.array(tg)
        for i0 in range(0, len(ev), 32):
            ii = order[i0:i0 + 32]; rr = [ev[x] for x in ii]; dd = [ev[x] for x in donors[j][ii]]
            _, Zr, KVr = run(m, tok, rr, dec, num); _, Zd, KVd = run(m, tok, dd, dec, num)
            if SMOKE and i0 == 0 and j == 0:
                a1, _, _ = run(m, tok, rr, dec, num, kv={0: KVr[0]}); a2, _, _ = run(m, tok, rr, dec, num, feed={0: Zr[0]})
                assert (a1 == A0[ii]).all() and (a2 == A0[ii]).all(), 'own write-back changed an answer'; print('  own write-back checks pass', flush=True)
            if j == 'all':
                out['both_all'][ii], _, _ = run(m, tok, rr, dec, num, feed={s: Zd[s] for s in range(NL)})
                out['kv_all'][ii], _, _ = run(m, tok, rr, dec, num, kv={s: KVd[s] for s in range(NL)})
            else:
                out[f'both_{j}'][ii], _, _ = run(m, tok, rr, dec, num, feed={j: Zd[j]})
                out[f'input_{j}'][ii], _, _ = run(m, tok, rr, dec, num, feed={j: Zd[j]}, kv={j: KVr[j]})
                out[f'kv_{j}'][ii], _, _ = run(m, tok, rr, dec, num, kv={j: KVd[j]})
                out[f'kv_noans_{j}'][ii], _, _ = run(m, tok, rr, dec, num, kv={j: KVd[j]}, hide=(j,))
            if DEV == 'mps': torch.mps.empty_cache()
    o = dict(n=len(ev), clean=float(right.mean()), gold=gold.tolist(), V=V.tolist(), clean_ans=A0.tolist(), ids=[r['id'] for r in ev], res={})
    for c in COND:
        j = 'all' if c.endswith('all') else int(c.split('_')[-1]); ok = right & (targets[j] >= 0)
        o['res'][c] = dict(n=int(ok.sum()), follows=float(np.mean(out[c][ok] == targets[j][ok])) if ok.any() else None,
                           keeps=float(np.mean(out[c][ok] == gold[ok])) if ok.any() else None, ans=out[c].tolist(), donor=donors[j].tolist(), target=targets[j].tolist())
    RES[L] = o
    print(time.strftime('%H:%M'), f'{MODEL} L{L} ' + ' | '.join(f"j{j}: b {o['res'][f'both_{j}']['follows']:.2f} i {o['res'][f'input_{j}']['follows']:.2f} "
          f"kv {o['res'][f'kv_{j}']['follows']:.2f} kvna {o['res'][f'kv_noans_{j}']['follows']:.2f}" for j in range(NL))
          + f" | all: b {o['res']['both_all']['follows']:.2f} kv {o['res']['kv_all']['follows']:.2f} (n={o['res']['both_all']['n']})", flush=True)
    json.dump(RES, open(OUT + (f'/t15_{MODEL}_smoke.json' if SMOKE else f'/t15_{MODEL}.json'), 'w'))
print('DONE')
