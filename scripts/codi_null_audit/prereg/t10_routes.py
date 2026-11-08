"""Which route carries the value between latent passes: the recurrent input z_s or attention back to earlier passes? Usage: t10_routes.py
<codi_null_audit dir> <device> <out dir>; env MODEL = startlast | gated | sim. SMOKE=1: L4, 40 items. Full: valid L2-L5.
noback: every latent pass s >= 1 is masked from attending to earlier latent passes 0..s-1 (for that pass only; the answer tokens still see every
pass, and the question stays visible, through the gate for gated). Conditions: clean, noback (accuracy); swap_j and noback_swap_j, j = 1, 2:
z_j replaced by a same-level donor's (t1's donors and ks), later latents recomputed; follows donor / keeps own on items answered right.
Prediction for a chain: noback keeps accuracy and a swapped z_j still moves the answer (the value travels through z). For start-last CODI,
b21 already showed noback costs .30 at L2 (valid); its single swaps follow .00. Copy of t1_single_use.py (same run(), donors, seeds). t1 docstring:
Readable x used, use half (HANDOFF_2026-11-07 step 1: 1a, 1b, 1c in one script). Usage: t1_single_use.py <codi_null_audit dir> <device> <out dir>.
env MODEL = plain | gated | sim. SMOKE=1: L4, 40 items, plus an identity check. Full: valid L3-L6 (286/286/286/285).

For every latent input z_j, j = 0..5, of one start-last model, one at a time:
  swap : replace z_j with a same-level donor's z_j. ks = the value the swap moves:
           gated  ks = min(j, L-1)      (z_j carries v_j; passes after L-2 read no rule and carry v_{L-1}); j = 0: start value, target =
                                         the recipient's rules 1..L on the donor's v_0 (1c)
           sim    ks = min(j+1, L-1)    (z_j, j <= L-2, sets v_{j+1}, s1; from z_{L-1} on, v_{L-1}: 1a)
           plain  ks = L-1              (latents carry everything before the last rule, near copies; 1b)
         Donor: first in a seeded shuffle (random.Random(1000 L + j), as b28 / s1, so overlapping cells reproduce them) with
         v_ks != the recipient's v_ks whose predicted answer, the recipient's rules ks+1..L on the donor's v_ks, is in 0..99 and != the
         recipient's gold. Recipients: answered right. Outcomes: follows donor, keeps own, donor's own answer.
  mean : z_j set to its mean over 2,000 train items of the same level (seed 21), never pooled. Accuracy on all items.
Hide (b30's conditions, the answer tokens' attention to latent passes only; L3-L5, levels with passes after L-1):
  block_used (pass L-1) / block_upto (passes 0..L-1) / block_idle (passes L..5) / block_all. Run for all three models; gated repeats b30.
Gate on for gated in every forward (RuleGate, as b28)."""
import sys, json, time, os, random; sys.path.insert(0, sys.argv[1])
from harness import *
import numpy as np
from rc import *
from src.model import RuleGate
from eval_codi import PadInvariant, question_position_ids
DEV = sys.argv[2]; OUT = sys.argv[3]; SMOKE = int(os.environ.get('SMOKE', '0')); MODEL = os.environ['MODEL']; MODEL = 'plain' if MODEL == 'startlast' else MODEL
LV = [4] if SMOKE else [2, 3, 4, 5]; NL = 6
DL = BASE + '/synthetic_data_v2_l1to7w_startlast/depth2/'
CK = dict(plain='codi_runs_l1to7w_startlast/codi/run/checkpoint-17820', gated='codi_runs_l1to7w_startlast_gated/codi/run/checkpoint-17820',
          sim='codi_runs_l1to7w_startlast_sim/codi/run/checkpoint-16335')[MODEL]
GATE = MODEL == 'gated'


def ks_of(j, L):
    if MODEL == 'gated': return min(j, L - 1)
    if MODEL == 'sim': return min(j + 1, L - 1)
    return L - 1


@torch.no_grad()
def run(m, tok, rows_, dec, num, patch=None, hide=(), noback=False):
    """Answers [n], latent inputs Z [n, NL, 768]. patch = {j: Zrep [n, 768]} replaces z_j before pass j; hide = passes the answer cannot see."""
    body = body_of(m); n = len(rows_); A = np.zeros(n, int) - 999; Z = np.zeros((n, NL, 768), np.float32)
    order = sorted(range(n), key=lambda i: len(rows_[i]['question']))
    for i in range(0, n, 32):
        ii = order[i:i + 32]
        b = encode([dict(question=rows_[q]['question'].strip(), steps=['x'], answer='0') for q in ii], tok, m, DEV)
        ids = b['encoder_input_ids']; mask = b['encoder_attention_mask'].long(); W = ids.size(1)
        g = RuleGate(ids, mask) if GATE else None
        o = body(input_ids=ids, use_cache=True, attention_mask=g.question(torch.float32) if GATE else mask, position_ids=question_position_ids(mask))
        past = o.past_key_values; pi = PadInvariant(mask); z = m.prj(o.last_hidden_state[:, -1:, :]); zs = []
        for s in range(NL):
            if patch and s in patch: z = torch.tensor(patch[s][ii], device=DEV)[:, None]
            zs.append(z[:, 0])
            if GATE: pi.mask[:, :W] = g.latent(s)
            saved = None
            if noback and s > 0: saved = pi.mask[:, W:W + s].clone(); pi.mask[:, W:W + s] = 0
            o = body(inputs_embeds=z, use_cache=True, past_key_values=past, **pi.next(1))
            if saved is not None: pi.mask[:, W:W + s] = saved
            past = o.past_key_values; z = m.prj(o.last_hidden_state[:, -1:, :])
        if GATE: pi.mask[:, :W] = g.answer()
        for s in hide: pi.mask[:, W + s] = 0
        emb = m.get_embd(m.codi, m.model_name)(dec)[None].expand(ids.size(0), -1, -1)
        o = body(inputs_embeds=emb, use_cache=True, past_key_values=past, **pi.next(emb.size(1)))
        A[ii] = [num.get(t, -999) for t in m.codi.get_base_model().lm_head(o.last_hidden_state[:, -1, :])[:, :50257].argmax(-1).tolist()]
        Z[ii] = torch.stack(zs, 1).cpu().numpy()
        del b, o, past
        if DEV == 'mps': torch.mps.empty_cache()
    return A, Z


m, tok, ta = load(BASE + '/' + CK, NL, DEV); m.eval()
dec = torch.tensor([m.eot_id] + tok.encode('The answer is:'), device=DEV)
num = {tok.encode(f"The answer is: {n}")[-1]: n for n in range(100)}; assert len(num) == 100
va = json.load(open(DL + 'valid.json')); RES = {}
for L in LV:
    ev = [r for r in va if int(r['level']) == L]; ev = ev[:40] if SMOKE else ev
    P = [parse(r['question']) for r in ev]; V = np.array([walk(*p)[0] for p in P]); gold = V[:, -1]
    assert all(int(r['answer']) == g for r, g in zip(ev, gold))
    A0, Z0 = run(m, tok, ev, dec, num); right = A0 == gold
    print(time.strftime('%H:%M'), f"{MODEL} L{L} clean acc {right.mean():.3f}", flush=True)
    if SMOKE:
        A_id, _ = run(m, tok, ev, dec, num, patch={2: Z0[:, 2]}); print('  identity patch keeps every answer:', bool((A_id == A0).all()), flush=True)
        assert (A_id == A0).all()
    An, _ = run(m, tok, ev, dec, num, noback=True)
    o = dict(n=len(ev), clean=float(right.mean()), noback=dict(acc=float(np.mean(An == gold)), kept_right=float(np.mean(An[right] == gold[right])), ans=An.tolist()), swap={})
    if SMOKE:
        Ac, _ = run(m, tok, ev, dec, num, noback=False); assert (Ac == A0).all()
    for j in (1, 2):
        k = ks_of(j, L); rng = random.Random(1000 * L + j); donors = []; targets = []
        for i in range(len(ev)):
            cand = [d for d in range(len(ev)) if d != i]; rng.shuffle(cand); pick = None
            for d in cand:
                if V[d, k] == V[i, k]: continue
                vals = walk(int(V[d, k]), P[i][1][k:])[0]
                if min(vals) < 0 or max(vals) > 99 or vals[-1] == gold[i]: continue
                pick = (d, vals[-1]); break
            donors.append(pick[0] if pick else i); targets.append(pick[1] if pick else -1)
        donors = np.array(donors); targets = np.array(targets); ok = (targets >= 0) & right
        for nb in (False, True):
            As, _ = run(m, tok, ev, dec, num, patch={j: Z0[donors, j]}, noback=nb)
            o['swap'][f"{'noback_' if nb else ''}swap_{j}"] = dict(ks=k, n=int(ok.sum()), follows_donor=float(np.mean(As[ok] == targets[ok])), keeps_own=float(np.mean(As[ok] == gold[ok])), ans=As.tolist(), donor=donors.tolist(), target=targets.tolist())
    print(time.strftime('%H:%M'), f"{MODEL} L{L} clean {o['clean']:.3f} noback {o['noback']['acc']:.3f} (right kept {o['noback']['kept_right']:.3f}) | " +
          ' | '.join(f"{c} follows {v['follows_donor']:.3f} keeps {v['keeps_own']:.3f} (n={v['n']})" for c, v in o['swap'].items()), flush=True)
    o.update(gold=gold.tolist(), V=V.tolist(), clean_ans=A0.tolist(), ids=[r['id'] for r in ev]); RES[L] = o
    json.dump(RES, open(OUT + (f'/t10_{MODEL}_smoke.json' if SMOKE else f'/t10_{MODEL}.json'), 'w'))
print('DONE')
