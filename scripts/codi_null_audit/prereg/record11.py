"""Log step 3 of HANDOFF_2026-11-07 (t3: latent dynamics, three start-last models, valid L3-L6). Usage: uv run python record11.py <audit dir>"""
import json, sys
import numpy as np
from latentreasoning.runlog.manifest import RunRecord, ModelInfo, DatasetInfo, new_run_id
W = sys.argv[1]; r3 = lambda x: round(float(x), 3)
R = json.load(open(W + '/t3_dynamics.json')); va = json.load(open(W + '/../synthetic_data_v2_l1to7w_startlast/depth2/valid.json'))
U = {m: json.load(open(W + f'/t1_{m}.json')) for m in R}
rows = []; ex = {}
for m, RL in R.items():
    ex[m] = {L: dict(acc=r3(o['acc']), **{tr: {k: [r3(x) for x in np.median(np.array(v), 0)] for k, v in o[tr].items()} for tr in ('z', 'h')}) for L, o in RL.items()}
    for L, o in RL.items():
        u = U[m][L]; assert u['ids'] == o['ids']
        for i, iid in enumerate(o['ids']):
            rows.append(dict(idx=len(rows), id=iid, model=m, level=int(L), gold_answer=str(u['gold'][i]), predicted_answer=str(u['clean_ans'][i]), correct=u['clean_ans'][i] == u['gold'][i],
                             **{f'{tr}_{k}': o[tr][k][i] for tr in ('z', 'h') for k in ('step', 'rel_step', 'dir_cos', 'centred_cos', 'log_ratio')}))
rec = RunRecord(run_id=new_run_id('codi', 'v2-l1to7w-startlast-latent-dynamics'), mechanism='codi', stage='full_run',
    model=ModelInfo(backbone='gpt2', checkpoint='plain start-last CODI checkpoint-17820 (20261106-083843); gated checkpoint-17820 (20261107-204814), gate on; SIM-CODI checkpoint-16335 (20261107-225603)', n_params=144499200),
    dataset=DatasetInfo(name='synthetic-depth2', split='validation', n_examples=len(rows), seed=20),
    metrics=dict(final_answer_accuracy=r3(np.mean([x['correct'] for x in rows])), unparseable_rate=0.0, compute_steps=6,
                 extra=dict(accuracy_kind='clean exact pooled over three models x valid L3-L6 (not comparable to single-model records)', medians_by_model=ex)),
    hyperparams=dict(script='codi_audit_artifacts/t3_dynamics.py (collector, CODI venv), t3_plot.py (figure, repo env)', trajectories='z_t = prj(h_t), t = 0..6; h_t last-layer state before prj',
                     metrics='step ||D_t||, relative step ||D_t||/||x_t||, direction cos(D_t, D_{t-1}), centred consecutive cos (mean over level items and positions 0..5), log(||D_t||/||D_{t-1}||)', dtype='float32'),
    seed=20, hardware='MacBook M4 Pro 24 GB, MPS', notes='Latent dynamics in the metrics of 2607.09698: does plain CODI contract to a fixed point while gated and SIM step through the rules?')
rec.save(predictions=rows); print(rec.run_id)
