"""Log step 2 of HANDOFF_2026-11-07 (t2: SIM-CODI all-latent donors vs same-level mean ablation, test L1-L7). Usage: uv run python record10.py <audit dir>"""
import json, sys
import numpy as np
from latentreasoning.runlog.manifest import RunRecord, ModelInfo, DatasetInfo, new_run_id
W = sys.argv[1]; r3 = lambda x: round(float(x), 3)
R = json.load(open(W + '/t2_sim_resample.json')); B = json.load(open(W + '/b29_ungated_resample.json'))
rows = []; ex = {}
f = lambda d: {k: (r3(v) if isinstance(v, float) else v) for k, v in d.items()}
for L, o in R.items():
    ex[L] = dict(clean=r3(o['clean']), right=o['right'], diff_donor=f(o['diff']), same_value_donor=f(o['same']), same_level_mean=f(o['mean']),
                 plain_codi_b29=dict(clean=r3(B[L]['clean']), diff_follows=r3(B[L]['diff']['follows_donor']), same_keeps=r3(B[L]['same']['keeps_own']), mean_acc=r3(B[L]['mean']['acc_all'])))
    for i, iid in enumerate(o['ids']):
        rows.append(dict(idx=len(rows), id=iid, level=int(L), gold_answer=str(o['gold'][i]), predicted_answer=str(o['clean_ans'][i]), correct=o['clean_ans'][i] == o['gold'][i],
                         diff_donor=o['ids'][o['diff_donor'][i]], diff_target=o['diff_target'][i], diff_answer=o['diff_ans'][i],
                         same_donor=o['ids'][o['same_donor'][i]] if o['same_donor'][i] >= 0 else None, same_answer=o['same_ans'][i], mean_answer=o['mean_ans'][i]))
rec = RunRecord(run_id=new_run_id('codi', 'v2-l1to7w-startlast-sim-content-reliance'), mechanism='codi', stage='full_run',
    model=ModelInfo(backbone='gpt2', checkpoint='SIM-CODI start-last checkpoint-16335 (20261107-225603), decoder dropped', n_params=144499200),
    dataset=DatasetInfo(name='synthetic-depth2', split='test', n_examples=len(rows), seed=None),
    metrics=dict(final_answer_accuracy=r3(np.mean([x['correct'] for x in rows])), unparseable_rate=0.0, compute_steps=6,
                 extra=dict(accuracy_kind='clean SIM-CODI exact on canonical test L1-L7, this script', by_level=ex)),
    hyperparams=dict(script='codi_audit_artifacts/t2_sim_resample.py (copy of b29_ungated_resample.py, checkpoint changed)', swap='all six latent inputs at once',
                     means='same level: 2,000 train items per level (seed 21)', dtype='float32'),
    seed=77, hardware='MacBook M4 Pro 24 GB, MPS', notes='Content reliance in SIM-CODI: all-latent donors and same-value donors against same-level mean ablation, compared with plain CODI (b29).')
rec.save(predictions=rows); print(rec.run_id)
