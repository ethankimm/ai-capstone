"""Log b29 + b29b (ungated start-last CODI: all-latent donors vs mean ablation) and b30 (gated backup). Usage: uv run python record8.py <audit dir>"""
import json, sys
import numpy as np
from latentreasoning.runlog.manifest import RunRecord, ModelInfo, DatasetInfo, new_run_id
W = sys.argv[1]; r3 = lambda x: round(float(x), 3)
R = json.load(open(W + '/b29_ungated_resample.json')); P = json.load(open(W + '/b29b_pooled_mean.json'))
rows = []; ex = {}
for L, o in R.items():
    ex[L] = dict(clean=r3(o['clean']), right=o['right'], diff_donor={k: (r3(v) if isinstance(v, float) else v) for k, v in o['diff'].items()},
                 same_value_donor={k: (r3(v) if isinstance(v, float) else v) for k, v in o['same'].items()},
                 same_level_mean={k: r3(v) for k, v in o['mean'].items()}, pooled_mean_acc=r3(P[L]['pooled L1-L7 mean']) if L in P else None)
    for i, iid in enumerate(o['ids']):
        rows.append(dict(idx=len(rows), id=iid, level=int(L), gold_answer=str(o['gold'][i]), predicted_answer=str(o['clean_ans'][i]), correct=o['clean_ans'][i] == o['gold'][i],
                         diff_donor=o['ids'][o['diff_donor'][i]], diff_target=o['diff_target'][i], diff_answer=o['diff_ans'][i],
                         same_donor=o['ids'][o['same_donor'][i]] if o['same_donor'][i] >= 0 else None, same_answer=o['same_ans'][i], mean_answer=o['mean_ans'][i]))
rec = RunRecord(run_id=new_run_id('codi', 'v2-l1to7w-startlast-content-reliance'), mechanism='codi', stage='full_run',
    model=ModelInfo(backbone='gpt2', checkpoint='start-last CODI checkpoint-17820 (20261106-083843), ungated', n_params=144499200),
    dataset=DatasetInfo(name='synthetic-depth2', split='test', n_examples=len(rows), seed=None),
    metrics=dict(final_answer_accuracy=r3(np.mean([x['correct'] for x in rows])), unparseable_rate=0.0, compute_steps=6,
                 extra=dict(accuracy_kind='clean start-last CODI exact on canonical test L1-L7, this script', by_level=ex)),
    hyperparams=dict(script='codi_audit_artifacts/b29_ungated_resample.py, b29b_pooled_mean.py', swap='all six latent inputs at once',
                     means='same level: 2,000 train items per level (seed 21); pooled: 500 per level over L1-L7', dtype='float32'),
    seed=77, hardware='MacBook M4 Pro 24 GB, MPS', notes='Content reliance vs format damage: all-latent donors against same-level and pooled mean ablation.')
rec.save(predictions=rows); print(rec.run_id)
B = json.load(open(W + '/b30_gated_backup.json')); va = json.load(open(W + '/../synthetic_data_v2_l1to7w_startlast/depth2/valid.json'))
rows = []; ex = {}
C = ['clean', 'block_used', 'block_upto', 'block_idle', 'block_all']
for L, o in B.items():
    items = [r for r in va if int(r['level']) == int(L)]; ex[L] = {c: dict(hide=o[c]['hide'], acc=r3(o[c]['acc'])) for c in C}
    for i, r in enumerate(items):
        rows.append(dict(idx=len(rows), id=r['id'], level=int(L), gold_answer=str(o['gold'][i]), predicted_answer=str(o['clean']['ans'][i]),
                         correct=o['clean']['ans'][i] == o['gold'][i], **{c: o[c]['ans'][i] for c in C[1:]}))
rec = RunRecord(run_id=new_run_id('codi', 'v2-l1to7w-startlast-gated-idle-backup'), mechanism='codi', stage='full_run',
    model=ModelInfo(backbone='gpt2', checkpoint='rule-gated start-last CODI checkpoint-17820 (20261107-204814)', n_params=144499200),
    dataset=DatasetInfo(name='synthetic-depth2', split='validation', n_examples=len(rows), seed=20),
    metrics=dict(final_answer_accuracy=r3(np.mean([x['correct'] for x in rows])), unparseable_rate=0.0, compute_steps=6,
                 extra=dict(accuracy_kind='rule-gated CODI exact on valid L3-L5, clean', conditions=ex)),
    hyperparams=dict(script='codi_audit_artifacts/b30_gated_backup.py', change="answer tokens' attention mask over latent passes only", dtype='float32'),
    seed=20, hardware='MacBook M4 Pro 24 GB, MPS', notes='Can idle latents stand in when the pass the answer reads is hidden?')
rec.save(predictions=rows); print(rec.run_id)
