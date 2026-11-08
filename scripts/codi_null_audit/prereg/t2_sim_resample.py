"""HANDOFF_2026-11-07 step 2: copy of b29_ungated_resample.py on SIM-CODI start-last checkpoint-16335 (no gate; decoder weights dropped by load). Usage: t2_sim_resample.py <codi_null_audit dir> <device> <out dir>.
SMOKE=1: L2, 40 items, plus an identity check. Test L1-L7 (500/level), so mean ablation compares with 20261106-105537.

All six latent inputs z_0..z_5 are replaced at once (the later ones are near copies, so single swaps say nothing here):
  diff donor : same level, a different v_{L-1} (latents carry everything before the last rule, 20261106-105634); predicted
               answer = the recipient's last rule applied to the donor's v_{L-1}, in 0..99 and != the recipient's gold
  same donor : same level, the same v_{L-1}, a different start value where possible (format control)
  mean       : each z_s set to its mean over 2,000 train items of the same level (seed 21)
Recipients: items the model answers right. Outcomes: follows donor (the predicted answer), keeps own, donor's own answer, other."""
import sys, json, time, os, random; sys.path.insert(0, sys.argv[1])
from harness import *
import numpy as np
from rc import *
from eval_codi import PadInvariant, question_position_ids
DEV = sys.argv[2]; OUT = sys.argv[3]; SMOKE = int(os.environ.get('SMOKE', '0'))
LV = [2] if SMOKE else [1, 2, 3, 4, 5, 6, 7]; NL = 6
DL = BASE + '/synthetic_data_v2_l1to7w_startlast/depth2/'
CK_ = BASE + '/codi_runs_l1to7w_startlast_sim/codi/run/checkpoint-16335'


@torch.no_grad()
def run(m, tok, rows_, dec, num, Zrep=None):
    """Answers [n], latent inputs Z [n, NL, 768]. Zrep [n, NL, 768] replaces every latent input."""
    body = body_of(m); n = len(rows_); A = np.zeros(n, int) - 999; Z = np.zeros((n, NL, 768), np.float32)
    order = sorted(range(n), key=lambda i: len(rows_[i]['question']))
    for i in range(0, n, 32):
        ii = order[i:i + 32]
        b = encode([dict(question=rows_[q]['question'].strip(), steps=['x'], answer='0') for q in ii], tok, m, DEV)
        ids = b['encoder_input_ids']; mask = b['encoder_attention_mask'].long()
        o = body(input_ids=ids, use_cache=True, attention_mask=mask, position_ids=question_position_ids(mask))
        past = o.past_key_values; pi = PadInvariant(mask); z = m.prj(o.last_hidden_state[:, -1:, :]); zs = []
        for s in range(NL):
            if Zrep is not None: z = torch.tensor(Zrep[ii, s], device=DEV)[:, None]
            zs.append(z[:, 0])
            o = body(inputs_embeds=z, use_cache=True, past_key_values=past, **pi.next(1))
            past = o.past_key_values; z = m.prj(o.last_hidden_state[:, -1:, :])
        emb = m.get_embd(m.codi, m.model_name)(dec)[None].expand(ids.size(0), -1, -1)
        o = body(inputs_embeds=emb, use_cache=True, past_key_values=past, **pi.next(emb.size(1)))
        A[ii] = [num.get(t, -999) for t in m.codi.get_base_model().lm_head(o.last_hidden_state[:, -1, :])[:, :50257].argmax(-1).tolist()]
        Z[ii] = torch.stack(zs, 1).cpu().numpy()
        del b, o, past
        if DEV == 'mps': torch.mps.empty_cache()
    return A, Z


m, tok, ta = load(CK_, NL, DEV); m.eval()
dec = torch.tensor([m.eot_id] + tok.encode('The answer is:'), device=DEV)
num = {tok.encode(f"The answer is: {n}")[-1]: n for n in range(100)}; assert len(num) == 100
te = json.load(open(DL + 'test.json')); RES = {}
for L in LV:
    ev = [r for r in te if int(r['level']) == L]; ev = ev[:40] if SMOKE else ev
    P = [parse(r['question']) for r in ev]; V = np.array([walk(*p)[0] for p in P]); gold = V[:, -1]; k = L - 1
    A0, Z0 = run(m, tok, ev, dec, num); right = A0 == gold
    if SMOKE:
        Aid, _ = run(m, tok, ev, dec, num, Zrep=Z0); print('  identity keeps every answer:', bool((Aid == A0).all()), flush=True); assert (Aid == A0).all()
    tr = rows(DL + 'train.json', per_level=300 if SMOKE else 2000, seed=21, levels=[L]); _, Zt = run(m, tok, tr, dec, num)
    rng = random.Random(77 + L); dd = []; dt = []; sd = []
    for i in range(len(ev)):
        cand = [d for d in range(len(ev)) if d != i]; rng.shuffle(cand); pick = None; same = None
        for d in cand:
            if V[d, k] == V[i, k]:
                if same is None or (V[same, 0] == V[i, 0] and V[d, 0] != V[i, 0]): same = d
                continue
            t = walk(int(V[d, k]), P[i][1][k:])[0]
            if pick is None and min(t) >= 0 and max(t) <= 99 and t[-1] != gold[i]: pick = (d, t[-1])
        dd.append(pick[0] if pick else i); dt.append(pick[1] if pick else -1); sd.append(same if same is not None else -1)
    dd = np.array(dd); dt = np.array(dt); sd = np.array(sd)
    Ad, _ = run(m, tok, ev, dec, num, Zrep=Z0[dd]); Am, _ = run(m, tok, ev, dec, num, Zrep=np.repeat(Zt.mean(0)[None], len(ev), 0))
    okd = right & (dt >= 0); oks = right & (sd >= 0)
    As, _ = run(m, tok, ev, dec, num, Zrep=Z0[np.where(sd >= 0, sd, np.arange(len(ev)))])
    o = dict(n=len(ev), right=int(right.sum()), clean=float(right.mean()),
             diff=dict(n=int(okd.sum()), follows_donor=float(np.mean(Ad[okd] == dt[okd])), keeps_own=float(np.mean(Ad[okd] == gold[okd])),
                       donor_answer=float(np.mean(Ad[okd] == gold[dd][okd]))),
             same=dict(n=int(oks.sum()), keeps_own=float(np.mean(As[oks] == gold[oks])), same_start_share=float(np.mean(V[sd[oks], 0] == V[oks, 0]))),
             mean=dict(acc_all=float(np.mean(Am == gold)), keeps_own_right=float(np.mean(Am[right] == gold[right]))),
             gold=gold.tolist(), clean_ans=A0.tolist(), diff_ans=Ad.tolist(), diff_donor=dd.tolist(), diff_target=dt.tolist(),
             same_ans=As.tolist(), same_donor=sd.tolist(), mean_ans=Am.tolist(), ids=[r['id'] for r in ev])
    RES[L] = o
    print(time.strftime('%H:%M'), f"L{L} clean {o['clean']:.3f} | diff donor n={o['diff']['n']} follows {o['diff']['follows_donor']:.3f} keeps own {o['diff']['keeps_own']:.3f} "
          f"donor ans {o['diff']['donor_answer']:.3f} | same-value donor n={o['same']['n']} keeps own {o['same']['keeps_own']:.3f} | mean: acc {o['mean']['acc_all']:.3f}, "
          f"right kept {o['mean']['keeps_own_right']:.3f}", flush=True)
    json.dump(RES, open(OUT + ('/t2_smoke.json' if SMOKE else '/t2_sim_resample.json'), 'w'))
print('DONE')
