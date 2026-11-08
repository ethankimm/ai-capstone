"""Where does each latent pass read from: the question or earlier latents? Usage: t9_attn_flow.py <codi_null_audit dir> <device> <out dir>.
SMOKE=1: L4, 32 items, start-last CODI only. Full: valid L3-L5 (286 per level), four models:
  startlast  start-last CODI (20261106-083843), the model under study
  gated      rule-gated start-last CODI, gate on (20261107-204814): a known chain
  sim        SIM-CODI start-last (20261107-225603): a known chain, no gate
  relay      lr-3e-3 CODI, start value first (20261103-193425): a known relay
For the query of latent pass s (input z_s), every layer, two measures per key position:
  raw   attention weight, mean over heads
  norm  ||sum_h a_hk v_hk W_O,h||, the size of what key k adds to the attention output (Kobayashi et al. 2020); W_O taken from the c_proj
        module itself (LoRA included) as c_proj(I) - c_proj(0)
summed over key sets: rules, start sentence, rest of the question (incl. <bot>), earlier latent passes (also split by distance d = 1..s),
self, attention sink (first real token). Set spans from scripts/codi_null_audit/b21_reads.py. Shares are of the total, per item, then averaged.
Descriptive: attention and norms are not causal (b21's blocking and t6's swaps are the causal tests)."""
import sys, json, time, os; sys.path.insert(0, sys.argv[1])
from harness import *
import numpy as np
from rc import *
from src.model import RuleGate
from eval_codi import PadInvariant, question_position_ids
DEV = sys.argv[2]; OUT = sys.argv[3]; SMOKE = int(os.environ.get('SMOKE', '0')); NL = 6
LV = [4] if SMOKE else [3, 4, 5]
DSL = BASE + '/synthetic_data_v2_l1to7w_startlast/depth2/'; DSF = BASE + '/synthetic_data_v2_l1to7w/depth2/'
MODELS = [('startlast', BASE + '/codi_runs_l1to7w_startlast/codi/run/checkpoint-17820', DSL, False),
          ('gated', BASE + '/codi_runs_l1to7w_startlast_gated/codi/run/checkpoint-17820', DSL, True),
          ('sim', BASE + '/codi_runs_l1to7w_startlast_sim/codi/run/checkpoint-16335', DSL, False),
          ('relay', CK['lr3'], DSF, False)]
if SMOKE: MODELS = MODELS[:1]
SETS = ['rules', 'start', 'rest', 'prev', 'self', 'sink']


def spans(tok, q):
    """b21's: token indices in q+'\\n'+<bot>, unpadded: one range per rule, the start sentence, everything else."""
    ti = lambda p: len(tok(q[:p]).input_ids); n = q.count('Rule ')
    ps = [q.index(f'Rule {k + 1}:') for k in range(n)]; ps = [p - 1 if p else 0 for p in ps]
    s = q.index('Start with'); s = s - 1 if s else 0; e = q.index('.', s) + 1; w = q.index(' What is')
    end = w if s < ps[0] else s
    rule = [(ti(ps[k]), ti(ps[k + 1]) if k + 1 < n else ti(end)) for k in range(n)]
    total = len(tok(q + '\n').input_ids) + 1
    start = (ti(s), ti(e)); used = set(range(start[0], start[1])) | {i for a, b in rule for i in range(a, b)}
    return rule, start, [i for i in range(total) if i not in used], total


@torch.no_grad()
def measure(name, ck, D, gate):
    m, tok, _ = load(ck, NL, DEV); m.eval(); body = body_of(m)
    WO = []
    for blk in body.h:  # effective output projection per layer, [12 heads, 64, 768]
        I = torch.eye(768, device=DEV); WO.append((blk.attn.c_proj(I) - blk.attn.c_proj(torch.zeros(1, 768, device=DEV))).reshape(12, 64, 768))
    va = json.load(open(D + 'valid.json')); out = {}
    for L in LV:
        ev = [r for r in va if int(r['level']) == L]; ev = ev[:32] if SMOKE else ev; ev.sort(key=lambda r: len(r['question']))
        acc = {k: np.zeros((NL, 12, len(SETS))) for k in ('raw', 'norm')}; dist = {k: np.zeros((NL, 12, NL)) for k in ('raw', 'norm')}; n = 0
        for i in range(0, len(ev), 32):
            rs = ev[i:i + 32]; qs = [r['question'].strip() for r in rs]; sps = [spans(tok, q) for q in qs]
            b = encode([dict(question=q, steps=['x'], answer='0') for q in qs], tok, m, DEV); ids = b['encoder_input_ids']; W = ids.size(1)
            if i == 0:
                u = tok(qs[0] + '\n').input_ids; a0, b0 = sps[0][0][0]; assert tok.decode(u[a0:b0]).strip().startswith('Rule 1')
            mask = b['encoder_attention_mask'].long(); g = RuleGate(ids, mask) if gate else None
            o = body(input_ids=ids, use_cache=True, attention_mask=g.question(torch.float32) if gate else mask, position_ids=question_position_ids(mask))
            past = o.past_key_values; pi = PadInvariant(mask); z = m.prj(o.last_hidden_state[:, -1:, :]); B = ids.size(0)
            sel = np.zeros((B, len(SETS), W + NL), bool)
            for r, (rule, start, rest, total) in enumerate(sps):
                off = W - total
                sel[r, 0, [off + j for a, c in rule for j in range(a, c) if j > 0]] = True
                sel[r, 1, [off + j for j in range(*start) if j > 0]] = True
                sel[r, 2, [off + j for j in rest if j > 0]] = True
                sel[r, 5, off] = True
            for s in range(NL):
                if gate: pi.mask[:, :W] = g.latent(s)
                o = body(inputs_embeds=z, use_cache=True, past_key_values=past, output_attentions=True, **pi.next(1))
                past = o.past_key_values; z = m.prj(o.last_hidden_state[:, -1:, :]); K = W + s + 1
                S = sel[:, :, :K].copy(); S[:, 3, W:W + s] = True; S[:, 4, W + s] = True; St = torch.tensor(S, device=DEV, dtype=torch.float32)
                for l in range(12):
                    a = o.attentions[l][:, :, 0, :K]  # [B, H, K]
                    v = past[l][1][:, :, :K, :] if isinstance(past[l], (tuple, list)) else past.value_cache[l][:, :, :K, :]
                    c = torch.einsum('bhk,bhkd,hde->bke', a, v, WO[l]).norm(dim=-1)  # [B, K]
                    for key, val in (('raw', a.mean(1)), ('norm', c)):
                        val = val / val.sum(-1, keepdim=True)
                        acc[key][s, l] += torch.einsum('bk,bsk->s', val, St).cpu().numpy()
                        for d in range(1, s + 1): dist[key][s, l, d - 1] += val[:, W + s - d].sum().item()
            n += B; del o, past, b
            if DEV == 'mps': torch.mps.empty_cache()
        out[L] = dict(n=n, sets=SETS, **{f'{k}_share': (acc[k] / n).tolist() for k in acc}, **{f'{k}_by_distance': (dist[k] / n).tolist() for k in dist})
        print(time.strftime('%H:%M'), f'{name} L{L} n={n} | pass 3 mean over layers, norm share: ' + ' '.join(f'{SETS[j]} {np.mean(acc["norm"][3, :, j]) / n:.2f}' for j in range(len(SETS))), flush=True)
    del m
    if DEV == 'mps': torch.mps.empty_cache()
    return out


PATH = OUT + ('/t9_smoke.json' if SMOKE else '/t9_attn_flow.json'); RES = json.load(open(PATH)) if os.path.exists(PATH) and not SMOKE else {}
for name, ck, D, gate in MODELS:
    if name in RES: continue
    t0 = time.time(); RES[name] = measure(name, ck, D, gate); json.dump(RES, open(PATH, 'w')); print('##', name, f'{time.time() - t0:.0f}s', flush=True)
print('DONE')
