"""Log t7 (partial-information probes, start-last CODI vs CODI-0). Usage: uv run python record14.py <audit dir>"""
import json, sys
import numpy as np
from latentreasoning.runlog.manifest import RunRecord, ModelInfo, DatasetInfo, new_run_id
W = sys.argv[1]; r3 = lambda x: round(float(x), 3); R = json.load(open(W + '/t7_partial_probe.json'))
va = json.load(open(W + '/../synthetic_data_v2_l1to7w_startlast/depth2/valid.json')); LAY = [0, 1, 2, 3, 4, 8, 12]
def best(o, sites, t, key):
    return r3(max(o['cells'][f'{s}|{l}'][t][key] for s in sites for l in LAY))
ex = {}
for m, RL in R.items():
    ex[m] = {}
    for L, o in RL.items():
        j = int(L) - 1; grp = {'latents': [f'h{s}' for s in range(1, 7)], 'tail': ['t10', 't9', 't1'], 'colon': ['colon']} if m == 'codi' else {'tail': ['t10', 't9', 't1'], 'colon': ['colon']}
        ex[m][L] = dict(right=o['right'], n=o['n'], v_before_last={g: {t: dict(all=best(o, S, f'v{j}_{t}', 'all'), right=best(o, S, f'v{j}_{t}', 'right'), shuf=best(o, S, f'v{j}_{t}', 'shuf'))
                                                                          for t in ('exact', 'tens', 'ones', 'par', 'val')} for g, S in grp.items()},
                        majority={k: r3(v) for k, v in o['majority'].items() if k.startswith(f'v{j}_')})
rows = []
for L in R['codi']:
    items = [r for r in va if int(r['level']) == int(L)]
    for i, r in enumerate(items):
        g = R['codi'][L]['V'][i][-1]
        rows.append(dict(idx=len(rows), id=r['id'], level=int(L), gold_answer=str(g), predicted_answer=str(R['codi'][L]['ans'][i]), correct=R['codi'][L]['ans'][i] == g,
                         codi0_answer=R['codi0'][L]['ans'][i], values=R['codi'][L]['V'][i]))
rec = RunRecord(run_id=new_run_id('codi', 'v2-l1to7w-startlast-partial-probes'), mechanism='codi', stage='full_run',
    model=ModelInfo(backbone='gpt2', checkpoint='start-last CODI checkpoint-17820 (20261106-083843) and start-last CODI-0 checkpoint-17820 (20261106-052644)', n_params=144499200),
    dataset=DatasetInfo(name='synthetic-depth2', split='validation', n_examples=len(rows), seed=20),
    metrics=dict(final_answer_accuracy=r3(np.mean([x['correct'] for x in rows])), unparseable_rate=0.0, compute_steps=6,
                 extra=dict(accuracy_kind='start-last CODI exact on valid L3-L5 (this collector)', best_over_sites_and_layers=ex)),
    hyperparams=dict(script='codi_audit_artifacts/t7_partial_probe.py (collector of b23_right_wrong.py), t7_table.py', fit='8,000 train items per level, seed 20; ridge, penalty per target on held-out 15%; shuffled-label floor',
                     targets='per step: tens, ones, parity, exact, scalar within 2; per rule: branch', dtype='float32'),
    seed=20, hardware='MacBook M4 Pro 24 GB, MPS', notes='What do start-last latents hold if not the exact value: magnitude, digits, parity, branches?')
rec.save(predictions=rows); print(rec.run_id)
