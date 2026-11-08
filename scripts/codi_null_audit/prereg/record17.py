"""Log t10 (routes: recurrent input vs look-back; start-last, gated, sim). Usage: uv run python record17.py <audit dir>"""
import json, sys
import numpy as np
from latentreasoning.runlog.manifest import RunRecord, ModelInfo, DatasetInfo, new_run_id
W = sys.argv[1]; r3 = lambda x: round(float(x), 3)
va = json.load(open(W + '/../synthetic_data_v2_l1to7w_startlast/depth2/valid.json')); rows = []; ex = {}
for tag, f in (('startlast', 't10_plain.json'), ('gated', 't10_gated.json'), ('sim', 't10_sim.json')):
    R = json.load(open(W + '/' + f)); ex[tag] = {}
    for L, o in R.items():
        ex[tag][L] = dict(clean=r3(o['clean']), noback_acc=r3(o['noback']['acc']), noback_kept_right=r3(o['noback']['kept_right']),
                          swaps={c: dict(ks=v['ks'], n=v['n'], follows=r3(v['follows_donor']), keeps=r3(v['keeps_own'])) for c, v in o['swap'].items()})
        items = [r for r in va if int(r['level']) == int(L)]; assert [r['id'] for r in items] == o['ids']
        for i, r in enumerate(items):
            rows.append(dict(idx=len(rows), id=r['id'], model=tag, level=int(L), gold_answer=str(o['gold'][i]), predicted_answer=str(o['clean_ans'][i]), correct=o['clean_ans'][i] == o['gold'][i],
                             noback_answer=o['noback']['ans'][i], **{c: v['ans'][i] for c, v in o['swap'].items()}))
rec = RunRecord(run_id=new_run_id('codi', 'v2-l1to7w-startlast-latent-routes'), mechanism='codi', stage='full_run',
    model=ModelInfo(backbone='gpt2', checkpoint='start-last CODI (20261106-083843); gated (20261107-204814), gate on; SIM-CODI (20261107-225603)', n_params=144499200),
    dataset=DatasetInfo(name='synthetic-depth2', split='validation', n_examples=len(rows), seed=20),
    metrics=dict(final_answer_accuracy=r3(np.mean([x['correct'] for x in rows])), unparseable_rate=0.0, compute_steps=6,
                 extra=dict(accuracy_kind='clean exact pooled over three models x valid L2-L5 (not comparable to single-model records)', by_model=ex)),
    hyperparams=dict(script='codi_audit_artifacts/t10_routes.py (copy of t1_single_use.py; MODEL=startlast|gated|sim)', noback='latent pass s >= 1 masked from earlier latent passes, that pass only; answer sees all',
                     donors='t1: same level, seeded Random(1000 L + j), differs at v_ks, predicted answer in 0..99 and != gold; recipients right', dtype='float32'),
    seed=20, hardware='MacBook M4 Pro 24 GB, MPS', notes='Which route carries the value between latent passes: recurrent input or look-back attention?')
rec.save(predictions=rows); print(rec.run_id)
