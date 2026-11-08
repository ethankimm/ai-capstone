"""Log b28 (per-latent swap and mean ablation, rule-gated CODI). Usage: uv run python record7.py <audit dir>"""
import json, sys
import numpy as np
from latentreasoning.runlog.manifest import RunRecord, ModelInfo, DatasetInfo, new_run_id
W = sys.argv[1]; R = json.load(open(W + '/b28_gated_swap.json'))
va = json.load(open(W + '/../synthetic_data_v2_l1to7w_startlast/depth2/valid.json'))
r3 = lambda x: round(float(x), 3); ex = {}; rows = []
for L, o in R.items():
    items = [r for r in va if int(r['level']) == int(L)]; gold = np.array(o['gold']); A0 = np.array(o['clean_ans'])
    assert len(items) == len(gold) and all(int(r['answer']) == g for r, g in zip(items, gold))
    ex[L] = dict(n=o['n'], clean=r3(o['clean']), by_latent={f'z{j}': dict(carries=f"v{s['k']}", n_pairs=s['n'], follows_donor=r3(s['follows_donor']), keeps_own=r3(s['keeps_own']),
                 donor_answer=r3(s['donor_answer']), mean_ablated_acc=r3(o['mean_ablation'][j]['acc'])) for j, s in o['swap'].items()})
    for i, r in enumerate(items):
        rows.append(dict(idx=len(rows), id=r['id'], level=int(L), gold_answer=str(gold[i]), predicted_answer=str(A0[i]), correct=bool(A0[i] == gold[i]), values=o['V'][i],
                         swap={j: dict(donor=items[s['donor'][i]]['id'], target=s['target'][i], answer=s['ans'][i]) for j, s in o['swap'].items()},
                         mean_ablated={j: a['ans'][i] for j, a in o['mean_ablation'].items()}))
rec = RunRecord(run_id=new_run_id('codi', 'v2-l1to7w-startlast-gated-latent-swap'), mechanism='codi', stage='full_run',
    model=ModelInfo(backbone='gpt2', checkpoint='rule-gated start-last CODI checkpoint-17820 (20261107-204814)', n_params=144499200),
    dataset=DatasetInfo(name='synthetic-depth2', split='validation', n_examples=len(rows), seed=20),
    metrics=dict(final_answer_accuracy=r3(np.mean([x['correct'] for x in rows])), unparseable_rate=0.0, compute_steps=6,
                 extra=dict(accuracy_kind='rule-gated CODI exact on valid L3-L7, clean run of this script, not test', swap=ex)),
    hyperparams=dict(script='codi_audit_artifacts/b28_gated_swap.py', donors='same level, seeded shuffle, v_k differs, predicted answer in 0..99 and != gold; recipients answered right',
                     mean_ablation='z_j replaced by its mean over 2,000 train items of the same level (seed 21)', dtype='float32'),
    seed=20, hardware='MacBook M4 Pro 24 GB, MPS', notes='Per-latent use in the gated model: does swapping one latent move the answer, and which latents are idle?')
rec.save(predictions=rows); print(rec.run_id)
