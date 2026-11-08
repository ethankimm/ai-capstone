"""Latent dynamics, plain vs gated vs SIM start-last CODI (HANDOFF_2026-11-07 step 3), in the metrics of 2607.09698 (Eqs. 1-3).
Usage: t3_dynamics.py <codi_null_audit dir> <device> <out dir>. SMOKE=1: L4, 40 items, plain only, plus a check that the collected
latents reproduce the clean answers of t1. Full: valid L3-L6 (286/286/286/285), all three models (gate on for gated).

Two trajectories per item, t = 0..6:
  z : latent inputs z_t = prj(h_t) (z_0 from <bot>, z_t the input of pass t; z_6 = prj of the last pass, computed but never read)
  h : last-layer states before prj, h_0 = <bot>, h_{t+1} = output of pass t
Per item: D_t = x_{t+1} - x_t (t = 0..5); step size ||D_t||; relative step ||D_t|| / ||x_t||; direction cosine cos(D_t, D_{t-1});
sensitivity surrogate log(||D_t|| / ||D_{t-1}||); centred consecutive cosine cos(x_t - mu, x_{t+1} - mu), mu = mean over the level's items
and positions 0..5 (as the earlier collapse statistic: plain was .58, .97, .99, .996 on test). Saved per item, as lists."""
import sys, json, time, os; sys.path.insert(0, sys.argv[1])
from harness import *
import numpy as np
from rc import *
from src.model import RuleGate
from eval_codi import PadInvariant, question_position_ids
DEV = sys.argv[2]; OUT = sys.argv[3]; SMOKE = int(os.environ.get('SMOKE', '0'))
LV = [4] if SMOKE else [3, 4, 5, 6]; NL = 6
DL = BASE + '/synthetic_data_v2_l1to7w_startlast/depth2/'
CKS = dict(plain='codi_runs_l1to7w_startlast/codi/run/checkpoint-17820', gated='codi_runs_l1to7w_startlast_gated/codi/run/checkpoint-17820',
           sim='codi_runs_l1to7w_startlast_sim/codi/run/checkpoint-16335')
MODELS = ['plain'] if SMOKE else ['plain', 'gated', 'sim']


@torch.no_grad()
def run(m, tok, rows_, dec, num, gate):
    """Answers [n], Z [n, NL+1, 768] latent inputs z_0..z_6, H [n, NL+1, 768] last-layer states h_0..h_6."""
    body = body_of(m); n = len(rows_); A = np.zeros(n, int) - 999
    Z = np.zeros((n, NL + 1, 768), np.float32); H = np.zeros((n, NL + 1, 768), np.float32)
    order = sorted(range(n), key=lambda i: len(rows_[i]['question']))
    for i in range(0, n, 32):
        ii = order[i:i + 32]
        b = encode([dict(question=rows_[q]['question'].strip(), steps=['x'], answer='0') for q in ii], tok, m, DEV)
        ids = b['encoder_input_ids']; mask = b['encoder_attention_mask'].long(); W = ids.size(1)
        g = RuleGate(ids, mask) if gate else None
        o = body(input_ids=ids, use_cache=True, attention_mask=g.question(torch.float32) if gate else mask, position_ids=question_position_ids(mask))
        past = o.past_key_values; pi = PadInvariant(mask); h = o.last_hidden_state[:, -1:, :]; z = m.prj(h); zs = [z[:, 0]]; hs = [h[:, 0]]
        for s in range(NL):
            if gate: pi.mask[:, :W] = g.latent(s)
            o = body(inputs_embeds=z, use_cache=True, past_key_values=past, **pi.next(1))
            past = o.past_key_values; h = o.last_hidden_state[:, -1:, :]; z = m.prj(h); zs.append(z[:, 0]); hs.append(h[:, 0])
        if gate: pi.mask[:, :W] = g.answer()
        emb = m.get_embd(m.codi, m.model_name)(dec)[None].expand(ids.size(0), -1, -1)
        o = body(inputs_embeds=emb, use_cache=True, past_key_values=past, **pi.next(emb.size(1)))
        A[ii] = [num.get(t, -999) for t in m.codi.get_base_model().lm_head(o.last_hidden_state[:, -1, :])[:, :50257].argmax(-1).tolist()]
        Z[ii] = torch.stack(zs, 1).cpu().numpy(); H[ii] = torch.stack(hs, 1).cpu().numpy()
        del b, o, past
        if DEV == 'mps': torch.mps.empty_cache()
    return A, Z, H


def metrics(X):
    X = X.astype(np.float64); D = X[:, 1:] - X[:, :-1]; nD = np.linalg.norm(D, axis=-1); nX = np.linalg.norm(X, axis=-1)
    cosD = (D[:, 1:] * D[:, :-1]).sum(-1) / (nD[:, 1:] * nD[:, :-1] + 1e-12)
    mu = X[:, :NL].mean((0, 1)); C = X - mu; nC = np.linalg.norm(C, axis=-1)
    ccos = (C[:, 1:] * C[:, :-1]).sum(-1) / (nC[:, 1:] * nC[:, :-1] + 1e-12)
    r = lambda a: np.round(a, 5).tolist()
    return dict(step=r(nD), rel_step=r(nD / (nX[:, :-1] + 1e-12)), norm=r(nX), dir_cos=r(cosD), log_ratio=r(np.log((nD[:, 1:] + 1e-12) / (nD[:, :-1] + 1e-12))), centred_cos=r(ccos))


va = json.load(open(DL + 'valid.json')); RES = {}
for name in MODELS:
    m, tok, ta = load(BASE + '/' + CKS[name], NL, DEV); m.eval()
    dec = torch.tensor([m.eot_id] + tok.encode('The answer is:'), device=DEV)
    num = {tok.encode(f"The answer is: {n}")[-1]: n for n in range(100)}; assert len(num) == 100
    RES[name] = {}
    for L in LV:
        ev = [r for r in va if int(r['level']) == L]; ev = ev[:40] if SMOKE else ev
        gold = np.array([walk(*parse(r['question']))[0][-1] for r in ev])
        A, Z, H = run(m, tok, ev, dec, num, name == 'gated')
        if os.path.exists(OUT + f'/t1_{name}.json'):
            ref = json.load(open(OUT + f'/t1_{name}.json')).get(str(L))
            if ref: assert A.tolist() == ref['clean_ans'][:len(A)], 'collector answers differ from t1'; print('  answers match t1', flush=True)
        o = dict(acc=float(np.mean(A == gold)), z=metrics(Z), h=metrics(H), ids=[r['id'] for r in ev]); RES[name][L] = o
        mz = {k: np.round(np.mean(v, 0), 3).tolist() for k, v in o['z'].items() if k != 'norm'}
        print(time.strftime('%H:%M'), f"{name} L{L} acc {o['acc']:.3f} | z step {mz['step']} | dir cos {mz['dir_cos']} | centred cos {mz['centred_cos']}", flush=True)
        json.dump(RES, open(OUT + ('/t3_smoke.json' if SMOKE else '/t3_dynamics.json'), 'w'))
    del m
    if DEV == 'mps': torch.mps.empty_cache()
print('DONE')
