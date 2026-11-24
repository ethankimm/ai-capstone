"""t39 / t40 scoring: is the latents' extra monitor signal concentrated on items whose answers depend on more loop crossings?
Offline on the t40 (RuleChain hidden_path.py SAVE_ITEMS=1) and t39 (gsm8k_hidden.py) outputs, joined to the t31 / t36 / t41 hop counts.
Pre-registered in scripts/codi_null_audit/prereg/t_predictions.md (t39-t41).

  python3 scripts/necessity/crossing_split.py <out json>

Per-item gain = correct(best latent set) - correct(QA); best = QAZ or QAH, whichever has the higher balanced accuracy for that
(level, rule) on RuleChain / that target on GSM8K. Item hops = mean over qualified pairs where the item is the recipient.
RuleChain: groups low (hops <= 1.5) / high (>= 2), difference within level, averaged over levels weighted by the items in both groups.
GSM8K: rationale length 3 or 4, groups low (<= 1.5) / high (>= 2.5), stratified by length; targets op_2, op_3, big_2, big_3.
95% CIs: 2,000 bootstrap resamples of items within each stratum.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

A = Path('/Users/henninglindig/Documents/Penn/CIS5980/codi_audit_artifacts'); OUT = sys.argv[1]
rng = np.random.default_rng(0); NB = 2000


def bacc(y, p):
    y, p = np.asarray(y) > 0, np.asarray(p) > 0
    if y.all() or (~y).all(): return float('nan')
    return float(((p & y).sum() / y.sum() + (~p & ~y).sum() / (~y).sum()) / 2)


def strat_diff(strata):
    """strata: list of (gain_low [n_l, t], gain_high [n_h, t]); returns weighted mean diff, CI, and per-group means."""
    def stat(S):
        w = np.array([len(l) + len(h) for l, h in S], float)
        d = np.array([np.nanmean(h) - np.nanmean(l) for l, h in S])
        return float((w * d).sum() / w.sum())
    S = [(l, h) for l, h in strata if len(l) >= 5 and len(h) >= 5]
    if not S: return None
    est = stat(S); bs = []
    for _ in range(NB):
        bs.append(stat([(l[rng.integers(0, len(l), len(l))], h[rng.integers(0, len(h), len(h))]) for l, h in S]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(diff=est, ci=[float(lo), float(hi)], n_low=int(sum(len(l) for l, _ in S)), n_high=int(sum(len(h) for _, h in S)),
                gain_low=float(np.mean(np.concatenate([np.nanmean(l, 1) for l, _ in S]))),
                gain_high=float(np.mean(np.concatenate([np.nanmean(h, 1) for _, h in S]))))


RES = dict(rulechain={}, gsm8k={})
# ---------------------------------------------------------------- RuleChain (t40)
for m in ['gated', 'sim', 'simsf', 'startlast', 'relay', 'stab']:
    hop = defaultdict(list)
    for f in (A / f't31/{m}.json', A / f't41/{m}.json'):
        for p in json.load(open(f))['pairs']:
            if p.get('qualified') and p.get('hops') is not None: hop[p['id']].append(p['hops'])
    R = json.load(open(A / f't40/{m}.json')); strata = []; gains = {}; ncov = 0
    for L, lv in R['levels'].items():
        it = lv['items']; ks = it['ks']; y = np.array(it['y']); P = {k: np.array(v) for k, v in it['pred'].items()}
        G = np.zeros((len(it['ids']), len(ks)))
        for j, k in enumerate(ks):
            best = max(('QAZ', 'QAH'), key=lambda s: bacc(y[:, j], P[s][:, j]))
            cQA = P['QA'][:, j] == y[:, j]; cB = P[best][:, j] == y[:, j]; G[:, j] = cB.astype(float) - cQA
            gains[f'L{L}_r{k}'] = bacc(y[:, j], P[best][:, j]) - bacc(y[:, j], P['QA'][:, j])
        h = np.array([np.mean(hop[i]) if hop.get(i) else np.nan for i in it['ids']]); ncov += int(np.isfinite(h).sum())
        strata.append((G[h <= 1.5], G[h >= 2]))
    g456 = [v for k, v in gains.items() if k[1] in '456']
    RES['rulechain'][m] = dict(split=strat_diff(strata), mean_gain_L4_6=float(np.mean(g456)),
                               gain_L7={k: v for k, v in gains.items() if k.startswith('L7')}, n_with_hops=ncov)
    s = RES['rulechain'][m]['split']
    print(f"RC {m:9s} mean gain L4-6 {np.mean(g456):+.3f} | hops join {ncov} | "
          + ('split -' if s is None else f"low {s['gain_low']:+.3f} (n {s['n_low']}) high {s['gain_high']:+.3f} (n {s['n_high']}) "
             f"diff {s['diff']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]"))

# ---------------------------------------------------------------- GSM8K (t39)
TG = ['op_2', 'op_3', 'big_2', 'big_3']
for m in ['gsm8k_codi', 'gsm8k_sim', 'gsm8k_coconut', 'gsm8k_simcoconut']:
    R = json.load(open(A / f't39/{m}.json')); it = R['items']; T = R['targets']
    y = np.array(it['y']); P = {k: np.array(v) for k, v in it['pred'].items()}
    h = np.array([np.nan if x is None else x for x in it['hops']], float); L = np.array(it['steps'])
    G = np.full((len(L), len(TG)), np.nan); cell = R['cells']
    for j, t in enumerate(TG):
        c = T.index(t); best = max(('QAZ', 'QAH'), key=lambda s: cell[f'{s}|{t}']); ok = y[:, c] != 0
        G[ok, j] = (P[best][ok, c] == y[ok, c]).astype(float) - (P['QA'][ok, c] == y[ok, c])
    strata = [(G[(L == n) & (h <= 1.5)], G[(L == n) & (h >= 2.5)]) for n in (3, 4)]
    gain = {t: max(cell[f'QAZ|{t}'], cell[f'QAH|{t}']) - cell[f'QA|{t}'] for t in T}
    pos = {}
    for t in T:
        sc = {k.split('|')[0]: v for k, v in cell.items() if k.endswith('|' + t) and k.split('|')[0][0] in 'ZH' and k.split('|')[0][1:].isdigit()}
        pos[t] = dict(bestH=max((k for k in sc if k[0] == 'H'), key=sc.get), bestZ=max((k for k in sc if k[0] == 'Z'), key=sc.get),
                      scores=sc)
    RES['gsm8k'][m] = dict(gain=gain, mean_gain=float(np.mean([gain[t] for t in TG])), split=strat_diff(strata), positions=pos,
                           shuf_max=max(v for k, v in cell.items() if k.startswith('QAH_shuf')), acc=R['acc'],
                           hop_hist=dict(low=int(np.sum(h <= 1.5)), mid=int(np.sum((h > 1.5) & (h < 2.5))), high=int(np.sum(h >= 2.5))))
    s = RES['gsm8k'][m]['split']
    print(f"GS {m:16s} mean gain {RES['gsm8k'][m]['mean_gain']:+.3f} wrong {gain['wrong']:+.3f} | "
          + ('split -' if s is None else f"low {s['gain_low']:+.3f} (n {s['n_low']}) high {s['gain_high']:+.3f} (n {s['n_high']}) "
             f"diff {s['diff']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]")
          + ' | best H for op_1..3 ' + ','.join(pos[f'op_{k}']['bestH'] for k in (1, 2, 3))
          + ' big_1..3 ' + ','.join(pos[f'big_{k}']['bestH'] for k in (1, 2, 3)))
json.dump(RES, open(OUT, 'w'), indent=1)
