"""Log t8 (start-last CODI latent blends). Usage: uv run python record15.py <audit dir>"""
import json, sys
import numpy as np
from importlib import util
from latentreasoning.runlog.manifest import RunRecord, ModelInfo, DatasetInfo, new_run_id
spec = util.spec_from_file_location('rc', 'scripts/codi_null_audit/rc.py'); rc = util.module_from_spec(spec); spec.loader.exec_module(rc)
W = sys.argv[1]; r3 = lambda x: round(float(x), 3)
D = json.load(open(W + '/t8_blend.json')); P = D['P']; st = D['store']['startlast CODI']; R = st['res']; AL = [round(0.1 * k, 1) for k in range(11)]
diff = [i for i in st['keep'] if P[i]['kind'] == 'diff']; d1 = [i for i in diff if P[i]['left'] >= 1]
def pre(rule, ans): return [u for u in range(100) if rc.apply(rule, u) == ans]
ex = {}
for a in AL:
    A = np.array([R[f'blend_{a}'][str(i)] for i in d1]); g = np.array([P[i]['gold'] for i in d1]); d = np.array([P[i]['dgold'] for i in d1]); inb = []
    for i, x in zip(d1, A):
        x0, ru = rc.parse(P[i]['rq']); y0, rd = rc.parse(P[i]['dq']); L = len(ru); va = rc.walk(x0, ru)[0][L - 1]; vb = rc.walk(y0, rd)[0][L - 1]
        inb.append(any(min(va, vb) <= u <= max(va, vb) for u in pre(ru[-1], int(x))))
    ex[str(a)] = dict(follows=r3(np.mean(A == d)), keeps=r3(np.mean(A == g)), strictly_between=r3(np.mean((A > np.minimum(g, d)) & (A < np.maximum(g, d)))), preimage_in_range=r3(np.mean(inb)))
rows = []
for i in diff:
    p = P[i]; rows.append(dict(idx=len(rows), id=p['item'], level=p['level'], edited_rule=p['rule'], rules_left=p['left'], gold_answer=str(p['gold']), donor_gold=p['dgold'],
                               predicted_answer=str(R['blend_0.0'][str(i)]), correct=R['blend_0.0'][str(i)] == p['gold'], blend={str(a): R[f'blend_{a}'][str(i)] for a in AL}))
rec = RunRecord(run_id=new_run_id('codi', 'v2-l1to7w-startlast-latent-blend'), mechanism='codi', stage='full_run',
    model=ModelInfo(backbone='gpt2', checkpoint='start-last CODI checkpoint-17820 (20261106-083843)', n_params=144499200),
    dataset=DatasetInfo(name='synthetic-depth2', split='test', n_examples=len(rows), seed=61),
    metrics=dict(final_answer_accuracy=1.0, unparseable_rate=0.0, compute_steps=6, extra=dict(accuracy_kind='diff pairs both answered right clean (1.0 by construction)', edits_before_last_by_a=ex)),
    hyperparams=dict(script='codi_audit_artifacts/t8_blend.py (copy of t6_prefix_hide.py), t8_table.py', blend='every latent input z_s = (1-a) z_rec + a z_don, a = 0..1 step .1, recipient question', dtype='float32'),
    seed=16, hardware='MacBook M4 Pro 24 GB, MPS', notes='Blend two items\' latents: does the answer move through in-between values (analog) or jump (discrete)?')
rec.save(predictions=rows); print(rec.run_id)
