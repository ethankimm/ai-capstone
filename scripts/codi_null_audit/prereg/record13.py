"""Log t6 (start-last CODI: prefix/suffix donor swaps and answer-side pass hiding). Usage: uv run python record13.py <audit dir>"""
import json, sys
import numpy as np
from latentreasoning.runlog.manifest import RunRecord, ModelInfo, DatasetInfo, new_run_id
W = sys.argv[1]; r3 = lambda x: round(float(x), 3)
D = json.load(open(W + '/t6_prefix_hide.json')); P = D['P']; st = D['store']['startlast CODI']; R = st['res']
diff = [i for i in st['keep'] if P[i]['kind'] == 'diff']; d1 = [i for i in diff if P[i]['left'] >= 1]
sw = ['all'] + [f'prefix_{s}' for s in range(1, 6)] + [f'suffix_{s}' for s in range(1, 6)]
f = lambda c, ii: r3(np.mean([R[c][str(i)] == P[i]['dgold'] for i in ii]))
ex = dict(follows_by_left={str(l): {c: f(c, [i for i in diff if P[i]['left'] == l]) for c in sw} for l in range(4) if len([i for i in diff if P[i]['left'] == l]) >= 15},
          follows_edits_before_last={c: f(c, d1) for c in sw})
seen = {}
for i in diff: seen.setdefault(P[i]['rq'], i)
hd = [f'hide_ge_{s}' for s in range(1, 6)] + [f'only_{s}' for s in range(6)] + ['hide_all']
ex['hide_acc_by_level'] = {str(L): {c: r3(np.mean([R[c][str(i)] == P[i]['gold'] for i in seen.values() if P[i]['level'] == L])) for c in hd} for L in range(2, 6)}
rows = []
for i in diff:
    p = P[i]; rows.append(dict(idx=len(rows), id=p['item'], level=p['level'], edited_rule=p['rule'], rules_left=p['left'], gold_answer=str(p['gold']), donor_gold=p['dgold'],
                               predicted_answer=str(p['gold']), correct=True, **{c: R[c][str(i)] for c in R if str(i) in R[c]}))
rec = RunRecord(run_id=new_run_id('codi', 'v2-l1to7w-startlast-prefix-suffix-pooling'), mechanism='codi', stage='full_run',
    model=ModelInfo(backbone='gpt2', checkpoint='start-last CODI checkpoint-17820 (20261106-083843)', n_params=144499200),
    dataset=DatasetInfo(name='synthetic-depth2', split='test', n_examples=len(rows), seed=61),
    metrics=dict(final_answer_accuracy=1.0, unparseable_rate=0.0, compute_steps=6,
                 extra=dict(accuracy_kind='rows are diff pairs both answered right clean (accuracy 1.0 by construction)', **ex)),
    hyperparams=dict(script='codi_audit_artifacts/t6_prefix_hide.py (copy of scripts/codi_null_audit/b16_donor.py, same pairs), t6_table.py; offline b29 re-analysis in notes',
                     conditions='prefix_s: donor z_0..z_{s-1}, later recomputed; suffix_s: donor z_s..z_5; hide_ge_s / only_s: answer-token attention mask over latent passes', dtype='float32'),
    seed=16, hardware='MacBook M4 Pro 24 GB, MPS', notes='Do later passes build on a donor\'s early latents, and how does the answer combine the passes?')
rec.save(predictions=rows); print(rec.run_id)
