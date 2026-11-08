"""Log step 1 of HANDOFF_2026-11-07 (readable x used, three start-last models). Usage: uv run python record9.py <audit dir>
Needs t1_{plain,gated,sim}.json and t1_table.json (t1_table.py)."""
import json, sys
import numpy as np
from latentreasoning.runlog.manifest import RunRecord, ModelInfo, DatasetInfo, new_run_id
W = sys.argv[1]; r3 = lambda x: round(float(x), 3)
T = json.load(open(W + '/t1_table.json')); va = json.load(open(W + '/../synthetic_data_v2_l1to7w_startlast/depth2/valid.json'))
CKN = dict(plain='start-last CODI checkpoint-17820 (20261106-083843)', gated='rule-gated start-last CODI checkpoint-17820 (20261107-204814), gate on',
           sim='SIM-CODI start-last checkpoint-16335 (20261107-225603), decoder dropped')
rows = []; ex = {}
for m in ('plain', 'gated', 'sim'):
    U = json.load(open(W + f'/t1_{m}.json')); ex[m] = {L: dict(clean=T[m][L]['clean'], hide=T[m][L]['hide'],
        cells=[{k: v for k, v in c.items()} for c in T[m][L]['cells']]) for L in T[m]}
    for L, o in U.items():
        items = [r for r in va if int(r['level']) == int(L)]; assert [r['id'] for r in items] == o['ids']
        for i, r in enumerate(items):
            rows.append(dict(idx=len(rows), id=r['id'], model=m, level=int(L), gold_answer=str(o['gold'][i]), predicted_answer=str(o['clean_ans'][i]),
                             correct=o['clean_ans'][i] == o['gold'][i], values=o['V'][i],
                             swap={j: dict(donor=items[s['donor'][i]]['id'], target=s['target'][i], answer=s['ans'][i]) for j, s in o['swap'].items()},
                             mean_ablated={j: a['ans'][i] for j, a in o['mean_ablation'].items()}, hide={c: h['ans'][i] for c, h in o['hide'].items()}))
rec = RunRecord(run_id=new_run_id('codi', 'v2-l1to7w-startlast-readable-vs-used'), mechanism='codi', stage='full_run',
    model=ModelInfo(backbone='gpt2', checkpoint='; '.join(f'{k}: {v}' for k, v in CKN.items()), n_params=144499200),
    dataset=DatasetInfo(name='synthetic-depth2', split='validation', n_examples=len(rows), seed=20),
    metrics=dict(final_answer_accuracy=r3(np.mean([x['correct'] for x in rows])), unparseable_rate=0.0, compute_steps=6,
                 extra=dict(accuracy_kind='clean exact pooled over three models x valid L3-L6 (not comparable to single-model records)', by_model=ex, four_cell_counts=T['_counts'])),
    hyperparams=dict(script='codi_audit_artifacts/t1_single_use.py (MODEL=plain|gated|sim), t1_table.py; reads b27_gated_decoder.json, s2_sim_probe.json, b29_ungated_resample.json',
                     thresholds='readable: probe exact >= .5 (best of layers 0/2/4/8/12 at h{j+1}) with shuffled <= .06; used: swap follows >= .5 or <= .5 of the clean-right items stay right under same-level mean ablation (handoff said mean-ablated acc <= .5; changed before compiling, plain clean acc < .5)',
                     donors='same level, seeded shuffle random.Random(1000 L + j) as b28/s1, v_ks differs, predicted answer in 0..99 and != gold; recipients answered right',
                     mean_ablation='z_j replaced by its mean over 2,000 train items of the same level (seed 21)', dtype='float32'),
    seed=20, hardware='MacBook M4 Pro 24 GB, MPS', notes='Readable x used for every latent of three start-last models: does a decoder read values where the model does not use them?')
rec.save(predictions=rows); print(rec.run_id)
