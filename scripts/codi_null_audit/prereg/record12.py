"""Log step 5 of HANDOFF_2026-11-07 (t5: SIM-CODI L7 with 7+ passes; L6 control). Usage: uv run python record12.py <audit dir>"""
import json, sys
import numpy as np
from importlib import util
from latentreasoning.runlog.manifest import RunRecord, ModelInfo, DatasetInfo, new_run_id
W = sys.argv[1]; r3 = lambda x: round(float(x), 3)
spec = util.spec_from_file_location('rc', 'scripts/codi_null_audit/rc.py'); rc = util.module_from_spec(spec); spec.loader.exec_module(rc)
R = json.load(open(W + '/t5_l7_passes.json')); va = {r['id']: r for r in json.load(open(W + '/../synthetic_data_v2_l1to7w_startlast/depth2/valid.json'))}
rows = []; ex = {}
for L, o in R.items():
    L_ = int(L); P = [rc.parse(va[i]['question']) for i in o['ids']]; V = np.array(o['V'])
    def rep(i, n):  # post hoc: rule L-1 applied n more times to v_{L-1}, then rule L
        x = V[i, L_ - 1]
        for _ in range(n): x = rc.apply(P[i][1][L_ - 2], x)
        return rc.apply(P[i][1][L_ - 1], x)
    ex[L] = {}
    for k, s in o['by_k'].items():
        A = np.array(s['ans']); n = int(k) - 6
        ex[L][k] = dict(share={c: r3(v) for c, v in s['share'].items()}, unique={c: r3(v) for c, v in s['unique'].items()}, none=r3(s['none']),
                        post_hoc_rule_Lminus1_repeated=r3(np.mean([A[i] == rep(i, n) for i in range(len(A))])) if n > 0 else None)
    for i, iid in enumerate(o['ids']):
        rows.append(dict(idx=len(rows), id=iid, level=L_, gold_answer=str(o['gold'][i]), predicted_answer=str(o['by_k']['6']['ans'][i]),
                         correct=o['by_k']['6']['ans'][i] == o['gold'][i], values=o['V'][i], ans_by_k={k: s['ans'][i] for k, s in o['by_k'].items()}))
rec = RunRecord(run_id=new_run_id('codi', 'v2-l1to7w-startlast-sim-extra-passes'), mechanism='codi', stage='full_run',
    model=ModelInfo(backbone='gpt2', checkpoint='SIM-CODI start-last checkpoint-16335 (20261107-225603), decoder dropped', n_params=144499200),
    dataset=DatasetInfo(name='synthetic-depth2', split='validation', n_examples=len(rows), seed=20),
    metrics=dict(final_answer_accuracy=r3(np.mean([x['correct'] for x in rows])), unparseable_rate=0.0, compute_steps=6,
                 extra=dict(accuracy_kind='SIM-CODI exact with the trained 6 passes, valid L7 + L6', by_level=ex)),
    hyperparams=dict(script='codi_audit_artifacts/t5_l7_passes.py', passes='k = 6..9 latent passes actually run (k > 6 outside training)', dtype='float32'),
    seed=20, hardware='MacBook M4 Pro 24 GB, MPS', notes='Why does SIM-CODI fail at L7 with 7+ passes? Which rule does the extra pass apply?')
rec.save(predictions=rows); print(rec.run_id)
