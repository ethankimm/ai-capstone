"""t25: can a monitor flag wrong answers from released SIM-CODI's latents? (pre-registered, t_predictions.md t25, commit 01d6d70)
Usage (CODI venv, has scikit-learn): sim_monitor.py <t22 dir>
Inputs: t22_sweep_{sim,codi}.json (decoded steps from SIM's decoder on each model's z_0..z_6), t25_conf_{sim,codi}.json (answer
log-probs), GSM8K test questions. Monitors never see gold. 5-fold stratified CV logistic regression (standardised), AUROC for
"answer is wrong", bootstrap 95% CI over items for the CV predictions and for differences between monitors."""
import json, re, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

D = sys.argv[1]
STEP = re.compile(r'<<([^=<>]*)=([^<>]*)>>')
NUM = re.compile(r'\d+(?:\.\d+)?')


def num(s):
    try: return float(str(s).replace(',', '').strip())
    except Exception: return None


def close(a, b):
    return a is not None and b is not None and abs(a - b) < 1e-6 * max(1, abs(b))


def parse(dec):
    m = STEP.search(dec or '')
    if not m: return None
    lhs, rhs = m.group(1), m.group(2)
    ops = [float(x) for x in NUM.findall(lhs)]
    try: val = eval(lhs, {'__builtins__': {}}) if re.fullmatch(r'[\d.+\-*/() ]+', lhs) else None
    except Exception: val = None
    return dict(lhs=lhs.strip(), ops=ops, res=num(rhs), val=val)


def latent_feats(x, question):
    qn = [float(v) for v in NUM.findall(question.replace(',', ''))]
    P = [parse(d) for d in x['decoded']]
    ok = [p for p in P if p]
    seen = list(qn); cons = []; ground = []; reuse = []
    for j, p in enumerate(P):
        if not p: continue
        cons.append(close(p['val'], p['res']) if p['val'] is not None else False)
        ground.append(np.mean([any(close(o, s) for s in seen) for o in p['ops']]) if p['ops'] else 0.0)
        prev = P[j - 1] if j > 0 else None
        if prev: reuse.append(any(close(o, prev['res']) for o in p['ops']))
        if p['res'] is not None: seen.append(p['res'])
    lhs = [p['lhs'] for p in ok]
    ans = num(x['ans'])
    early = P[:3]  # z_0..z_2 carry the labelled steps on most items
    return [len(ok) / 7, np.mean(cons) if cons else 0, np.mean(ground) if ground else 0, np.mean(reuse) if reuse else 0,
            len(set(lhs)), len(lhs) - len(set(lhs)), float(any(p and close(p['res'], ans) for p in P)),
            float(all(p is not None and close(p['val'], p['res']) for p in early if p is not None)),
            float(np.mean([any(close(o, s) for s in qn) for o in early[0]['ops']]) if early[0] and early[0]['ops'] else 0)]


def conf_feats(c):
    lp = c['lp'][:-1] or c['lp']; nl = c['num_lp'] or lp
    return [np.mean(lp), np.min(lp), np.mean(nl), np.min(nl), nl[0]]


def q_feats(q):
    return [len(q.split()), len(NUM.findall(q.replace(',', '')))]


def cv_pred(X, y, seed=0):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return p


def boot(y, ps, B=2000, seed=0):
    r = np.random.default_rng(seed); n = len(y); out = {k: [] for k in ps}
    for _ in range(B):
        i = r.integers(0, n, n)
        if y[i].min() == y[i].max(): continue
        for k, p in ps.items(): out[k].append(roc_auc_score(y[i], p[i]))
    return {k: np.percentile(v, [2.5, 97.5]) for k, v in out.items()}, {k: np.array(v) for k, v in out.items()}


sys.path.insert(0, D + '/../../ai-capstone'); sys.path.insert(0, '/Users/henninglindig/Documents/Penn/CIS5980/ai-capstone')
from latentreasoning.data.gsm8k_aug import load_gsm8k_aug
Qs = {e.idx: e.question for e in load_gsm8k_aug(split='test', n=None, seed=None)}
RES = {}
for model in ('sim', 'codi'):
    S = {x['idx']: x for x in json.load(open(f'{D}/t22_sweep_{model}.json'))['items']}
    C = {x['idx']: x for x in json.load(open(f'{D}/t25_conf_{model}.json'))['items']}
    ids = sorted(S); assert all(str(S[i]['ans']) == str(C[i]['ans']) for i in ids), 'conf and sweep answers differ'
    y = np.array([0 if S[i]['correct'] else 1 for i in ids])
    XL = np.array([latent_feats(S[i], Qs[i]) for i in ids]); XC = np.array([conf_feats(C[i]) for i in ids]); XQ = np.array([q_feats(Qs[i]) for i in ids])
    P = {'L': cv_pred(XL, y), 'C': cv_pred(XC, y), 'Q': cv_pred(XQ, y), 'C+Q': cv_pred(np.hstack([XC, XQ]), y), 'L+C+Q': cv_pred(np.hstack([XL, XC, XQ]), y)}
    auc = {k: roc_auc_score(y, p) for k, p in P.items()}
    ci, bs = boot(y, P)
    d = bs['L+C+Q'] - bs['C+Q']; dq = bs['L'] - bs['Q']
    # gold-based ceiling (not a monitor): any labelled step decoded wrong
    def off(x):
        ks = range(1, min(len(x['values']), 7))
        if not len(ks): return 0.0
        pr = [parse(x['decoded'][k - 1]) for k in ks]
        return float(any(not (p and close(p['res'], num(x['values'][k - 1]))) for p, k in zip(pr, ks)))
    gold = roc_auc_score(y, [off(S[i]) for i in ids])
    RES[model] = dict(n=len(y), wrong=int(y.sum()), auc={k: round(v, 3) for k, v in auc.items()}, ci={k: [round(a, 3) for a in v] for k, v in ci.items()},
                      diff_LCQ_minus_CQ=[round(float(d.mean()), 3), [round(float(x), 3) for x in np.percentile(d, [2.5, 97.5])]],
                      diff_L_minus_Q=[round(float(dq.mean()), 3), [round(float(x), 3) for x in np.percentile(dq, [2.5, 97.5])]], gold_trace_ceiling=round(gold, 3))
    print(f'\n== {model}: n={len(y)} wrong={int(y.sum())}')
    for k in P: print(f'  {k:6s} AUROC {auc[k]:.3f}  [{ci[k][0]:.3f}, {ci[k][1]:.3f}]')
    print(f'  L+C+Q minus C+Q: {d.mean():+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]')
    print(f'  L minus Q:       {dq.mean():+.3f} [{np.percentile(dq, 2.5):+.3f}, {np.percentile(dq, 97.5):+.3f}]')
    print(f'  gold-based trace check (ceiling, not a monitor): {gold:.3f}')
json.dump(RES, open(f'{D}/t25_monitor.json', 'w'), indent=1)
