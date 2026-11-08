"""Log t9 (attention flow by latent pass, four models). Usage: uv run python record16.py <audit dir>"""
import json, sys
import numpy as np
from latentreasoning.runlog.manifest import RunRecord, ModelInfo, DatasetInfo, new_run_id
W = sys.argv[1]; r3 = lambda x: round(float(x), 3); R = json.load(open(W + '/t9_attn_flow.json'))
ex = {}
for m, RL in R.items():
    ex[m] = {}
    for L, o in RL.items():
        ex[m][L] = {}
        for key in ('norm', 'raw'):
            S = np.array(o[f'{key}_share']).mean(1); ns = S[:, :5] / S[:, :5].sum(1, keepdims=True)
            ex[m][L][key] = dict(question=[r3(x) for x in ns[:, :3].sum(1)], earlier_latents=[r3(x) for x in ns[:, 3]], self=[r3(x) for x in ns[:, 4]], sink_total=[r3(x) for x in S[:, 5]])
        Dd = np.array(o['norm_by_distance']).mean(1); ex[m][L]['norm_by_distance_share'] = [[r3(x) for x in (Dd[s, :s] / Dd[s, :s].sum())] for s in range(1, 6)]
rows = [dict(idx=k, id=f'{m}-L{L}', model=m, level=int(L), gold_answer='', predicted_answer='', correct=False, n_items=R[m][L]['n']) for k, (m, L) in enumerate((m, L) for m in R for L in R[m])]
rec = RunRecord(run_id=new_run_id('codi', 'v2-l1to7w-latent-attention-flow'), mechanism='codi', stage='full_run',
    model=ModelInfo(backbone='gpt2', checkpoint='start-last CODI (20261106-083843); gated (20261107-204814), gate on; SIM-CODI (20261107-225603); lr-3e-3 relay (20261103-193425)', n_params=144499200),
    dataset=DatasetInfo(name='synthetic-depth2', split='validation', n_examples=sum(R[m][L]['n'] for m in R for L in R[m]), seed=20),
    metrics=dict(final_answer_accuracy=None, unparseable_rate=None, compute_steps=6, extra=dict(accuracy_kind='none: descriptive attention measures; rows are model x level aggregates', by_model=ex)),
    hyperparams=dict(script='codi_audit_artifacts/t9_attn_flow.py, t9_table.py', measures='raw attention (mean over heads) and Kobayashi norm ||sum_h a v W_O|| with W_O from c_proj (LoRA included); sets from b21 spans', dtype='float32'),
    seed=20, hardware='MacBook M4 Pro 24 GB, MPS', notes='Where does each latent pass read from: question or earlier latents? Start-last vs known chains (gated, SIM) and a known relay.')
rec.save(predictions=rows); print(rec.run_id)
