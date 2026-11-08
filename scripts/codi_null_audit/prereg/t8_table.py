"""Tables for t8_blend.json. Usage: t8_table.py <audit dir>. Pairs with the edit 1+ rules before the last unless stated.
blend prediction = the shared last rule applied to round((1-a) v_{L-1}^rec + a v_{L-1}^don), the analog-magnitude account."""
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1] + '/../ai-capstone/scripts/codi_null_audit')
from importlib import util
spec = util.spec_from_file_location('rc', sys.argv[1] + '/../ai-capstone/scripts/codi_null_audit/rc.py'); rc = util.module_from_spec(spec); spec.loader.exec_module(rc)
W = sys.argv[1]; D = json.load(open(W + '/t8_blend.json')); P = D['P']; st = D['store']['startlast CODI']; R = st['res']
AL = [round(0.1 * k, 1) for k in range(11)]
diff = [i for i in st['keep'] if P[i]['kind'] == 'diff']
assert all(R['blend_0.0'][str(i)] == P[i]['gold'] for i in diff), 'a = 0 control failed'
def info(i):
    x0, ru = rc.parse(P[i]['rq']); y0, rd = rc.parse(P[i]['dq']); va = rc.walk(x0, ru)[0]; vb = rc.walk(y0, rd)[0]; L = len(ru)
    assert ru[-1] == rd[-1] and vb[-1] == P[i]['dgold']
    return va[L - 1], vb[L - 1], ru[-1]
for name, sel in (('edit 1+ rules before the last', lambda p: p['left'] >= 1), ('edit 1 rule before the last', lambda p: p['left'] == 1), ('edit 2+ before', lambda p: p['left'] >= 2), ('edit at the last rule (control)', lambda p: p['left'] == 0)):
    ii = [i for i in diff if sel(P[i])]
    print(f'\n## {name}, n = {len(ii)} pairs\n')
    print('| a | follows donor | keeps own | strictly between | = blend prediction | within 2 of blend prediction |'); print('|---|---|---|---|---|---|')
    I = {i: info(i) for i in ii} if name != 'edit at the last rule (control)' else None
    for a in AL:
        A = np.array([R[f'blend_{a}'][str(i)] for i in ii]); g = np.array([P[i]['gold'] for i in ii]); d = np.array([P[i]['dgold'] for i in ii])
        btw = np.mean((A > np.minimum(g, d)) & (A < np.maximum(g, d)))
        if I:
            pr = np.array([rc.apply(I[i][2], int(round((1 - a) * I[i][0] + a * I[i][1]))) for i in ii])
            ex = f'{np.mean(A == pr):.2f} | {np.mean(np.abs(A - pr) <= 2):.2f}'
        else: ex = '- | -'
        print(f'| {a} | {np.mean(A == d):.2f} | {np.mean(A == g):.2f} | {btw:.2f} | {ex} |')
    if I:
        seqs = np.array([[R[f'blend_{a}'][str(i)] for a in AL] for i in ii]); dif = np.diff(seqs, axis=1)
        mono = np.mean([(np.all(r >= 0) or np.all(r <= 0)) for r in dif]); nd = np.mean([len(set(r)) for r in seqs])
        print(f'\nper pair over the 11 a values: monotone answer sequence {mono:.2f}; distinct answers visited, mean {nd:.1f}; '
              f'|v_rec - v_don| mean {np.mean([abs(I[i][0] - I[i][1]) for i in ii]):.1f}')
