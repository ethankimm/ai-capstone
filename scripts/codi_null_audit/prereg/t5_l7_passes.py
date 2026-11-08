"""HANDOFF_2026-11-07 step 5, second check: why SIM-CODI start-last falls to .02-.06 at L7 with 7+ latent passes (s3_sim_passes.json).
Usage: t5_l7_passes.py <codi_null_audit dir> <device> <out dir>. SMOKE=1: 40 items, k = 6 only, must equal t1-style clean answers' accuracy (>= .98).
Valid L7 (285) and L6 (286, control: an idle pass is available, s3 says L6 stays 1.000), answer after k = 6..9 passes (truly running k passes).
Guess fixed before the run: at k = 7 the extra pass applies rule 7 and the answer tokens apply it again, so A = r7(r7(v_6)) = r7(v_7).
Candidates scored per k: gold v_L; r_L(v_L) = r_L(r_L(v_{L-1})) (last rule twice); r_L(r_L(v_L)) (three times); r_L(r_j(v_{L-1})) for each
rule j < L (the extra pass applies rule j to v_{L-1}, then the answer applies the last rule); v_{L-1}. Shares are over all items; candidates can coincide, so each is
scored separately and a 'unique' share counts items where only that candidate matches."""
import sys, json, time, os; sys.path.insert(0, sys.argv[1])
from harness import *
import numpy as np
from rc import *
from eval_codi import PadInvariant, question_position_ids
DEV = sys.argv[2]; OUT = sys.argv[3]; SMOKE = int(os.environ.get('SMOKE', '0'))
DL = BASE + '/synthetic_data_v2_l1to7w_startlast/depth2/'; CK = BASE + '/codi_runs_l1to7w_startlast_sim/codi/run/checkpoint-16335'
KS = [6] if SMOKE else [6, 7, 8, 9]; LV = [7] if SMOKE else [7, 6]


@torch.no_grad()
def run(m, tok, rows_, dec, num, npass):
    body = body_of(m); n = len(rows_); A = np.zeros(n, int) - 999
    order = sorted(range(n), key=lambda i: len(rows_[i]['question']))
    for i in range(0, n, 32):
        ii = order[i:i + 32]
        b = encode([dict(question=rows_[q]['question'].strip(), steps=['x'], answer='0') for q in ii], tok, m, DEV)
        ids = b['encoder_input_ids']; mask = b['encoder_attention_mask'].long()
        o = body(input_ids=ids, use_cache=True, attention_mask=mask, position_ids=question_position_ids(mask))
        past = o.past_key_values; pi = PadInvariant(mask); z = m.prj(o.last_hidden_state[:, -1:, :])
        for s in range(npass):
            o = body(inputs_embeds=z, use_cache=True, past_key_values=past, **pi.next(1))
            past = o.past_key_values; z = m.prj(o.last_hidden_state[:, -1:, :])
        emb = m.get_embd(m.codi, m.model_name)(dec)[None].expand(ids.size(0), -1, -1)
        o = body(inputs_embeds=emb, use_cache=True, past_key_values=past, **pi.next(emb.size(1)))
        A[ii] = [num.get(t, -999) for t in m.codi.get_base_model().lm_head(o.last_hidden_state[:, -1, :])[:, :50257].argmax(-1).tolist()]
        del b, o, past
        if DEV == 'mps': torch.mps.empty_cache()
    return A


m, tok, ta = load(CK, 6, DEV); m.eval()
dec = torch.tensor([m.eot_id] + tok.encode('The answer is:'), device=DEV)
num = {tok.encode(f"The answer is: {n}")[-1]: n for n in range(100)}; assert len(num) == 100
va = json.load(open(DL + 'valid.json')); RES = {}
for L in LV:
    ev = [r for r in va if int(r['level']) == L]; ev = ev[:40] if SMOKE else ev
    P = [parse(r['question']) for r in ev]; V = np.array([walk(*p)[0] for p in P]); gold = V[:, -1]
    rl = lambda i, j, x: apply(P[i][1][j], x)
    cands = {'gold': gold, 'last_twice': np.array([rl(i, L - 1, gold[i]) for i in range(len(ev))]),
             'last_thrice': np.array([rl(i, L - 1, rl(i, L - 1, gold[i])) for i in range(len(ev))]), 'v_{L-1}': V[:, L - 1]}
    for j in range(L - 1): cands[f'r{L}(r{j + 1}(v_{L-1}))'] = np.array([rl(i, L - 1, rl(i, j, V[i, L - 1])) for i in range(len(ev))])
    o = dict(n=len(ev), gold=gold.tolist(), V=V.tolist(), ids=[r['id'] for r in ev], by_k={})
    for k in KS:
        A = run(m, tok, ev, dec, num, k); hit = {c: A == v for c, v in cands.items()}
        tot = np.sum([h for h in hit.values()], 0)
        o['by_k'][k] = dict(ans=A.tolist(), share={c: float(h.mean()) for c, h in hit.items()}, unique={c: float((h & (tot == 1)).mean()) for c, h in hit.items()},
                            none=float((tot == 0).mean()))
        s = o['by_k'][k]
        print(time.strftime('%H:%M'), f"L{L} k={k}: " + ' | '.join(f"{c} {s['share'][c]:.3f} ({s['unique'][c]:.3f})" for c in cands) + f" | none {s['none']:.3f}", flush=True)
        if SMOKE: assert s['share']['gold'] >= .95
    RES[L] = o
    json.dump(RES, open(OUT + ('/t5_smoke.json' if SMOKE else '/t5_l7_passes.json'), 'w'))
print('DONE')
