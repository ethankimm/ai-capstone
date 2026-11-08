"""Log t15 (RuleChain route split, five models). Usage: uv run python record20.py <audit dir>"""
import json, sys
import numpy as np
from latentreasoning.runlog.manifest import RunRecord, ModelInfo, DatasetInfo, new_run_id
W = sys.argv[1]; r3 = lambda x: None if x is None else round(float(x), 3)
CKN = dict(gated='rule-gated start-last CODI (20261107-204814), gate on', sim='SIM-CODI start-last (20261107-225603)', startlast='start-last CODI (20261106-083843)',
           relay='lr-3e-3 L1-7w CODI, start first (20261103-193425)', stab='stabilised L1-7w CODI, start first (20261105-044444)')
SL = W + '/../synthetic_data_v2_l1to7w_startlast/depth2/valid.json'; SF = W + '/../synthetic_data_v2_l1to7w/depth2/valid.json'
rows = []; ex = {}
for m in CKN:
    R = json.load(open(W + f'/t15_{m}.json')); va = json.load(open(SL if m in ('gated', 'sim', 'startlast') else SF)); ex[m] = {}
    for L, o in R.items():
        ex[m][L] = dict(clean=r3(o['clean']), **{c: dict(n=v['n'], follows=r3(v['follows']), keeps=r3(v['keeps'])) for c, v in o['res'].items()})
        items = [r for r in va if int(r['level']) == int(L)]; assert [r['id'] for r in items] == o['ids']
        for i, r in enumerate(items):
            rows.append(dict(idx=len(rows), id=r['id'], model=m, level=int(L), gold_answer=str(o['gold'][i]), predicted_answer=str(o['clean_ans'][i]), correct=o['clean_ans'][i] == o['gold'][i],
                             **{c: v['ans'][i] for c, v in o['res'].items()}))
rec = RunRecord(run_id=new_run_id('codi', 'v2-l1to7w-route-split-five-models'), mechanism='codi', stage='full_run',
    model=ModelInfo(backbone='gpt2', checkpoint='; '.join(f'{k}: {v}' for k, v in CKN.items()), n_params=144499200),
    dataset=DatasetInfo(name='synthetic-depth2', split='validation', n_examples=len(rows), seed=20),
    metrics=dict(final_answer_accuracy=r3(np.mean([x['correct'] for x in rows])), unparseable_rate=0.0, compute_steps=6,
                 extra=dict(accuracy_kind='clean exact pooled over five models x valid L3-L5 (start-last data for gated/sim/startlast, start-first for relay/stab)', by_model=ex)),
    hyperparams=dict(script='codi_audit_artifacts/t15_rc_route_split.py (MODEL=gated|sim|startlast|relay|stab)',
                     conditions='both / input-only / entries-only / entries hidden from answer, per site and all sites; donors as t1 with equal question token length', dtype='float32'),
    seed=20, hardware='MacBook M4 Pro 24 GB, MPS', notes='The necessity check on five RuleChain models: which latents carry values through the recurrent input?')
rec.save(predictions=rows); print(rec.run_id)
