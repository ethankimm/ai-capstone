"""t25 post hoc (not pre-registered; added after t25's decoded-feature result): a learned probe on the raw latents as the latent
monitor, against a probe on the question's last hidden state (no latents) and answer confidence. Usage (CODI venv): sim_monitor_probe.py <t22 dir>
Inputs: t25_z_{sim,codi}.npz (Z [n, 7, 768] = z_0..z_6, H [n, 768] = last question-token state), t25_conf_{model}.json, t22_sweep_{model}.json.
5-fold stratified CV; standardise -> PCA 128 -> logistic regression (C = 0.1); AUROC for "answer is wrong"; bootstrap CI for differences."""
import json, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = sys.argv[1]


def conf_feats(c):
    lp = c['lp'][:-1] or c['lp']; nl = c['num_lp'] or lp
    return [np.mean(lp), np.min(lp), np.mean(nl), np.min(nl), nl[0]]


def cv(X, y, pca=True):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        steps = [StandardScaler()] + ([PCA(128, random_state=0)] if pca and X.shape[1] > 128 else []) + [LogisticRegression(C=0.1, max_iter=3000)]
        m = make_pipeline(*steps).fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return p


def diff_ci(y, a, b, B=2000):
    r = np.random.default_rng(0); n = len(y); d = []
    for _ in range(B):
        i = r.integers(0, n, n)
        if y[i].min() != y[i].max(): d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return float(np.mean(d)), [float(x) for x in np.percentile(d, [2.5, 97.5])]


RES = {}
for model in ('sim', 'codi'):
    z = np.load(f'{D}/t25_z_{model}.npz'); idx = list(z['idx'])
    S = {x['idx']: x for x in json.load(open(f'{D}/t22_sweep_{model}.json'))['items']}
    C = {x['idx']: x for x in json.load(open(f'{D}/t25_conf_{model}.json'))['items']}
    y = np.array([0 if S[i]['correct'] else 1 for i in idx])
    Z = z['Z'].astype(np.float32).reshape(len(idx), -1); H = z['H'].astype(np.float32); XC = np.array([conf_feats(C[i]) for i in idx])
    P = {'H (question state, no latents)': cv(H, y), 'Z (all latents)': cv(Z, y), 'C (confidence)': cv(XC, y, pca=False),
         'H+C': cv(np.hstack([H, XC]), y), 'Z+H+C': cv(np.hstack([Z, H, XC]), y)}
    auc = {k: round(roc_auc_score(y, p), 3) for k, p in P.items()}
    d1 = diff_ci(y, P['Z (all latents)'], P['H (question state, no latents)']); d2 = diff_ci(y, P['Z+H+C'], P['H+C'])
    RES[model] = dict(auc=auc, Z_minus_H=d1, ZHC_minus_HC=d2)
    print(f'\n== {model} (n={len(y)}, wrong={int(y.sum())})')
    for k, v in auc.items(): print(f'  {k:32s} {v:.3f}')
    print(f'  Z minus H:       {d1[0]:+.3f} [{d1[1][0]:+.3f}, {d1[1][1]:+.3f}]')
    print(f'  Z+H+C minus H+C: {d2[0]:+.3f} [{d2[1][0]:+.3f}, {d2[1][1]:+.3f}]')
json.dump(RES, open(f'{D}/t25_probe.json', 'w'), indent=1)

# Stacked combination (the raw concatenations above dilute the 5 confidence features among 768+ dims under PCA / C = 0.1, so H+C
# scores below C alone): out-of-fold probe logits joined with the confidence features in a second 5-fold logistic regression.
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
ST = {}
for model in ('sim', 'codi'):
    z = np.load(f'{D}/t25_z_{model}.npz'); idx = list(z['idx'])
    S = {x['idx']: x for x in json.load(open(f'{D}/t22_sweep_{model}.json'))['items']}
    C = {x['idx']: x for x in json.load(open(f'{D}/t25_conf_{model}.json'))['items']}
    y = np.array([0 if S[i]['correct'] else 1 for i in idx])
    Z = z['Z'].astype(np.float32).reshape(len(idx), -1); H = z['H'].astype(np.float32); XC = np.array([conf_feats(C[i]) for i in idx])
    pZ, pH = cv(Z, y), cv(H, y)
    P = {'C': cv(XC, y, pca=False), 'C + H-probe': cv(np.c_[XC, lg(pH)], y, pca=False), 'C + H-probe + Z-probe': cv(np.c_[XC, lg(pH), lg(pZ)], y, pca=False)}
    a = {k: round(roc_auc_score(y, p), 3) for k, p in P.items()}; d = diff_ci(y, P['C + H-probe + Z-probe'], P['C + H-probe'])
    ST[model] = dict(auc=a, latents_add=d); print(model, 'stacked', a, 'latents add', round(d[0], 3), [round(x, 3) for x in d[1]])
json.dump(ST, open(f'{D}/t25_stack.json', 'w'), indent=1)
