"""What do start-last CODI's latents hold, if not the exact value? Usage: t7_partial_probe.py <codi_null_audit dir> <device> <out dir>.
SMOKE=1: L4, 300 fit / 40 eval items, CODI only. Full: valid L3-L5, fit on 8,000 train items per level (seed 20), as b23.
Collector = b23_right_wrong.py's (same sites, same encode path), models start-last CODI (6 latents) and start-last CODI-0 (0 latents, the
single-pass control: same data and recipe, no latents).
Sites: t10 = the start-value token (" Start with N. What is the final number?\\n<bot>": t1 = <bot>, t10 = N), t9 = the "." after it, t1 = <bot>,
h1..h6 = latent pass s-1 (h{s+1} = pass s, input z_s; layer 0 = z_s itself), colon = last answer-prompt token. Layers 0, 1, 2, 3, 4, 8, 12.
Targets, for every step value v_j (j = 0..L): tens digit (10 classes), ones digit (10), parity (2), exact (tens and ones both right),
a scalar ridge readout of the value (MAE, share within 2); for every rule j = 1..L: the branch taken (2).
Probe: ridge on standardised features, one-hot (or scalar) targets, penalty chosen per target on a held-out 15% of fit items; a shuffled-label
fit (rows permuted) gives each target's floor. Scored on all valid items and on the items the model answers right.
Question this answers: is the value before the last rule held coarsely (tens digit / magnitude right, ones digit and parity wrong), as the
all-six donor swaps suggest (b29: exact .37, within 2 .82 at L3), and does any latent site hold more than CODI-0's single pass does?"""
import sys, json, time, os; sys.path.insert(0, sys.argv[1])
from harness import *
import numpy as np
from rc import *
DEV = sys.argv[2]; OUT = sys.argv[3]; SMOKE = int(os.environ.get('SMOKE', '0'))
FIT = 300 if SMOKE else 8000; LV = [4] if SMOKE else [3, 4, 5]
SL = BASE + '/codi_runs_l1to7w_startlast'; DL = BASE + '/synthetic_data_v2_l1to7w_startlast/depth2/'
MODELS = [('codi', f'{SL}/codi/run/checkpoint-17820', 6), ('codi0', f'{SL}/codi0/run/checkpoint-17820', 0)]
if SMOKE: MODELS = MODELS[:1]
NT = 13; DECN = ['eot', 'The', 'answer', 'is', 'colon']; LAY = [0, 1, 2, 3, 4, 8, 12]
def names(nl): return [f't{NT - i}' for i in range(NT)] + [f'h{s + 1}' for s in range(nl)] + DECN
def keep(nl): return ['t10', 't9', 't1'] + [f'h{s + 1}' for s in range(nl)] + ['colon']


@torch.no_grad()
def collect(m, tok, rows_, nl, dec, num):
    N = names(nl); K = [N.index(k) for k in keep(nl)]; body = body_of(m); X = np.zeros((len(rows_), len(K), len(LAY), 768), np.float16); A = np.zeros(len(rows_), int) - 999
    order = sorted(range(len(rows_)), key=lambda i: len(rows_[i]['question']))
    for i in range(0, len(order), 32):
        ii = order[i:i + 32]; b = encode([dict(question=rows_[j]['question'].strip(), steps=['x'], answer='0') for j in ii], tok, m, DEV)
        if i == 0:  # the start-value token must sit at t10 (" N" is one GPT-2 token for 0..99)
            ids = b['encoder_input_ids'][0].tolist(); x0 = parse(rows_[ii[0]]['question'])[0]
            assert tok.decode([ids[-10]]) == f' {x0}', (tok.decode(ids[-13:]), x0)
        mask = b['encoder_attention_mask'].long(); pos = (mask.cumsum(-1) - 1).clamp(min=0)
        o = body(input_ids=b['encoder_input_ids'], use_cache=True, attention_mask=mask, position_ids=pos, output_hidden_states=True)
        cols = [torch.stack(o.hidden_states, 2)[:, -NT:]]; past = o.past_key_values; npos = mask.sum(-1, keepdim=True); z = m.prj(o.last_hidden_state[:, -1:, :]) if nl else None
        for s in range(nl):
            mask = torch.cat((mask, torch.ones(mask.size(0), 1, dtype=mask.dtype, device=mask.device)), 1)
            o = body(inputs_embeds=z, use_cache=True, past_key_values=past, attention_mask=mask, position_ids=npos, output_hidden_states=True)
            cols.append(torch.stack(o.hidden_states, 2)); past = o.past_key_values; npos = npos + 1; z = m.prj(o.last_hidden_state[:, -1:, :])
        emb = m.get_embd(m.codi, m.model_name)(dec)[None].expand(mask.size(0), -1, -1); T = emb.size(1)
        mask = torch.cat((mask, torch.ones(mask.size(0), T, dtype=mask.dtype, device=mask.device)), 1)
        o = body(inputs_embeds=emb, use_cache=True, past_key_values=past, attention_mask=mask, position_ids=npos + torch.arange(T, device=mask.device)[None], output_hidden_states=True)
        cols.append(torch.stack(o.hidden_states, 2)); C = torch.cat(cols, 1)[:, K][:, :, LAY]; X[ii] = C.to('cpu', torch.float16).numpy()
        A[ii] = [num.get(t, -999) for t in m.codi.get_base_model().lm_head(o.last_hidden_state[:, -1, :])[:, :50257].argmax(-1).tolist()]
        del b, o, cols, C
        if DEV == 'mps': torch.mps.empty_cache()
    return X, A


def targets(rows_):
    """Y columns and a spec of (name, kind, columns): kind 'cls' = argmax over one-hot block, 'reg' = scalar."""
    W = [walk(*parse(r['question'])) for r in rows_]; V = np.array([w[0] for w in W]); B = np.array([w[1] for w in W]).astype(int)
    n, T = V.shape; cols = []; spec = []; truth = {}
    def add(name, kind, y, k=None):
        c0 = sum(c.shape[1] for c in cols)
        if kind == 'cls': M = np.zeros((n, k), np.float32); M[np.arange(n), y] = 1
        else: M = y[:, None].astype(np.float32)
        cols.append(M); spec.append((name, kind, list(range(c0, c0 + M.shape[1])))); truth[name] = y
    for j in range(T):
        add(f'v{j}_tens', 'cls', V[:, j] // 10, 10); add(f'v{j}_ones', 'cls', V[:, j] % 10, 10); add(f'v{j}_par', 'cls', V[:, j] % 2, 2); add(f'v{j}_val', 'reg', V[:, j])
    for j in range(B.shape[1]): add(f'b{j + 1}', 'cls', B[:, j], 2)
    return V, np.concatenate(cols, 1), spec, truth


def readout(P, kind, cols): return P[:, cols].argmax(1) if kind == 'cls' else P[:, cols[0]]
def score(name, kind, pred, y):
    if kind == 'cls': return float(np.mean(pred == y))
    return float(np.mean(np.abs(pred - y) <= 2))  # within 2


def probe(Xa, Ya, spec, truth_a, Xb):
    """Per-target penalty on a held-out 15%; returns dict target -> prediction on Xb."""
    Xa = Xa.astype(np.float64); Xb = Xb.astype(np.float64); mu, sd = Xa.mean(0), Xa.std(0) + 1e-6; Xa = (Xa - mu) / sd; Xb = (Xb - mu) / sd
    ym = Ya.mean(0); Yc = (Ya - ym).astype(np.float64); c = int(len(Xa) * .85); LAM = (1e1, 1e2, 1e3, 1e4, 1e5)
    def fit(X, Y):
        w, Vv = np.linalg.eigh(X.T @ X); XY = Vv.T @ (X.T @ Y); return lambda l: Vv @ (XY / (np.maximum(w, 0) + l)[:, None])
    f = fit(Xa[:c], Yc[:c]); Ph = {l: Xa[c:] @ f(l) + ym for l in LAM}; g = fit(Xa, Yc); Pb = {l: Xb @ g(l) + ym for l in LAM}; out = {}
    for name, kind, cols in spec:
        best = max(LAM, key=lambda l: score(name, kind, readout(Ph[l], kind, cols), truth_a[name][c:]))
        out[name] = readout(Pb[best], kind, cols)
    return out


def measure(name, ck, nl):
    m, tok, _ = load(ck, nl, DEV); m.eval(); dec = torch.tensor([m.eot_id] + tok.encode('The answer is:'), device=DEV)
    num = {tok.encode(f"The answer is: {n}")[-1]: n for n in range(100)}; assert len(num) == 100
    va = json.load(open(DL + 'valid.json')); out = {}
    for L in LV:
        fit = rows(DL + 'train.json', per_level=FIT, seed=20, levels=[L]); ev = [r for r in va if int(r['level']) == L]; ev = ev[:40] if SMOKE else ev
        Xa, _ = collect(m, tok, fit, nl, dec, num); Xb, Ab = collect(m, tok, ev, nl, dec, num)
        Va, Ya, spec, ta = targets(fit); Vb, _, _, tb = targets(ev); right = Ab == Vb[:, -1]
        pm = np.random.default_rng(0).permutation(len(Xa)); tsh = {k: v[pm] for k, v in ta.items()}
        o = dict(n=len(ev), right=int(right.sum()), ans=Ab.tolist(), V=Vb.tolist(), cells={})
        for si, s in enumerate(keep(nl)):
            for li, l in enumerate(LAY):
                a = Xa[:, si, li]; b = Xb[:, si, li]
                pr = probe(a, Ya, spec, ta, b); ps = probe(a, Ya[pm], spec, tsh, b); cell = {}
                for tn, kind, _ in spec:
                    cell[tn] = dict(all=score(tn, kind, pr[tn], tb[tn]), right=score(tn, kind, pr[tn][right], tb[tn][right]) if right.any() else None,
                                    shuf=score(tn, kind, ps[tn], tb[tn]))
                for j in range(Vb.shape[1]):  # exact = both digits right
                    ex = (pr[f'v{j}_tens'] == tb[f'v{j}_tens']) & (pr[f'v{j}_ones'] == tb[f'v{j}_ones']); exs = (ps[f'v{j}_tens'] == tb[f'v{j}_tens']) & (ps[f'v{j}_ones'] == tb[f'v{j}_ones'])
                    cell[f'v{j}_exact'] = dict(all=float(ex.mean()), right=float(ex[right].mean()) if right.any() else None, shuf=float(exs.mean()))
                o['cells'][f'{s}|{l}'] = cell
        o['majority'] = {tn: float(np.max(np.bincount(tb[tn].astype(int))) / len(tb[tn])) for tn, kind, _ in spec if kind == 'cls'}
        out[L] = o; del Xa, Xb
        print(time.strftime('%H:%M'), f'{name} L{L} done right={int(right.sum())}/{len(ev)}', flush=True)
    del m
    if DEV == 'mps': torch.mps.empty_cache()
    return out


PATH = OUT + ('/t7_smoke.json' if SMOKE else '/t7_partial_probe.json'); RES = json.load(open(PATH)) if os.path.exists(PATH) and not SMOKE else {}
for name, ck, nl in MODELS:
    if name in RES: continue
    t0 = time.time(); RES[name] = measure(name, ck, nl); json.dump(RES, open(PATH, 'w')); print('##', name, f'{time.time() - t0:.0f}s', flush=True)
print('DONE')
