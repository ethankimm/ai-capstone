"""Log b27 (decoder control for rule-gated CODI). Usage: uv run python record6.py <audit dir>"""
import json, sys
import numpy as np
from latentreasoning.runlog.manifest import RunRecord, ModelInfo, DatasetInfo, new_run_id
W = sys.argv[1]; D = json.load(open(W + '/b27_gated_decoder.json'))
va = json.load(open(W + '/../synthetic_data_v2_l1to7w_startlast/depth2/valid.json'))
SITES = ['t1'] + [f'h{s+1}' for s in range(6)] + ['colon']
r3 = lambda x: round(float(x), 3)
ex = {}; rows = []
for m, R in D.items():
    ex[m] = {}
    for L, o in R.items():
        L_ = int(L); V = np.array(o['V']); A = np.array(o['ans']); LLn = np.array(o['lens']); items = [r for r in va if int(r['level']) == L_]
        assert len(items) == len(V) and all(int(r['answer']) == v for r, v in zip(items, V[:, -1]))
        z = {f'v{j}': dict(probe=r3(np.mean(np.array(o['cells'][f'h{j+1}|0']['full'])[:, j] == V[:, j])),
                           shuffled=r3(np.mean(np.array(o['cells'][f'h{j+1}|0']['shuf'])[:, j] == V[:, j])),
                           logit_lens=r3(np.mean(LLn[:, SITES.index(f'h{j+1}'), 0] == V[:, j]))) for j in range(1, L_)}
        ex[m][L] = dict(n=len(V), answer_exact=r3(np.mean(A == V[:, -1])), z_j_reads=z,
                        value_before_last_rule_h4_layer2=r3(np.mean(np.array(o['cells']['h4|2']['full'])[:, L_ - 1] == V[:, L_ - 1])))
        for i, r in enumerate(items):
            rows.append(dict(idx=len(rows), id=f"{m}|{r['id']}", model=m, level=L_, gold_answer=str(V[i, -1]), predicted_answer=str(A[i]),
                             correct=bool(A[i] == V[i, -1]), values=V[i].tolist(),
                             probe_z=[int(np.array(o['cells'][f'h{j+1}|0']['full'])[i, j]) for j in range(1, L_)],
                             lens_z=[int(LLn[i, SITES.index(f'h{j+1}'), 0]) for j in range(1, L_)]))
acc = np.mean([x['correct'] for x in rows if x['model'] == 'gated'])
rec = RunRecord(run_id=new_run_id('codi', 'v2-l1to7w-startlast-gated-decoder-control'), mechanism='codi', stage='full_run',
    model=ModelInfo(backbone='gpt2', checkpoint='rule-gated start-last CODI checkpoint-17820 (20261107-204814); ungated start-last CODI checkpoint-17820 (20261106-083843) as comparison', n_params=144499200),
    dataset=DatasetInfo(name='synthetic-depth2', split='validation', n_examples=len(rows) // 2, seed=20),
    metrics=dict(final_answer_accuracy=r3(acc), unparseable_rate=0.0, compute_steps=6,
                 extra=dict(accuracy_kind='rule-gated CODI exact on valid L3-L6 from the probe collector, not test', reads=ex)),
    hyperparams=dict(script='codi_audit_artifacts/b27_gated_decoder.py', analysis='codi_audit_artifacts/a29_gated_decoder.py',
                     probe='ridge to digit one-hots, penalty picked on a held-out 15% of fit items (as b23)', fit_items_per_level=8000,
                     sites='t1, h1-h6, colon at layers 0, 2, 4, 8, 12', logit_lens='ln_f then LM head on the 100 number tokens, 13 layers',
                     headline='v_j at z_j (h{j+1} layer 0) and h{j} layer 12, fixed before any read', dtype='float32'),
    seed=20, hardware='MacBook M4 Pro 24 GB, MPS', notes='Decoder control (TRAINING_PROTOCOL 12.3): can probes and the logit lens read a known rollout?')
rec.save(predictions=rows); print(rec.run_id)
