"""Log t17 (grid route split) as one RunRecord per task x method. Usage: uv run python scripts/necessity/record_t17.py <audit dir> <task> <method>"""
import json
import sys
from pathlib import Path

from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id

W, task, method = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
R = json.load(open(W / f't17_{task}_{method}.json')); P = R['pairs']; S = R['summary']
r3 = lambda x: None if x is None else round(float(x), 3)


def same(a, b):
    a, b = str(a or '').strip().rstrip('.').lower().replace(',', ''), str(b or '').strip().rstrip('.').lower().replace(',', '')
    if task == 'gsm8k':
        try: return float(a) == float(b)
        except ValueError: return False
    return a == b


right = [p for p in P if same(p['clean'], p['gold'])]
donor_ok = [p for p in P if same(p['donor_clean'], p['target'])]
sens = sum(same(p['donor_clean'], p['target']) for p in right) / max(len(right), 1)
rows = []
for k, p in enumerate(P):
    rows.append(dict(idx=k, item=p['i'], hop=p['hop'], n_hops=p['n_hops'], gold_answer=str(p['gold']), predicted_answer=str(p['clean']),
                     correct=same(p['clean'], p['gold']), target=str(p['target']), donor_answer=str(p['donor_clean']), qualified=p['qualified'],
                     **{c: v for c, v in p.items() if c.startswith(('both_', 'input_', 'kv_')) and c != 'input_all'}))
Rs = lambda d: {c: r3(v) for c, v in d.items() if c != 'input_all'}  # input_all is 0 by construction (every site's entries reset)
A = S['all']; best_in = max((A.get(f'input_{j}') or 0) for j in range(6)); best_b = max((A.get(f'both_{j}') or 0) for j in range(6))
ck = {('gsm8k', 'coconut'): 'hf:connordilgren/gpt2-gsm8k-coconut/checkpoint_33'}.get((task, method), f'hf:connordilgren/gpt2-{task}-{method}')
rec = RunRecord(run_id=new_run_id(method, f'{task}-grid-route-split'), mechanism=method, stage='full_run',
    model=ModelInfo(backbone='gpt2', checkpoint=ck, n_params=124442112 if method == 'coconut' else 144499200),
    dataset=DatasetInfo(name=task, split='test', n_examples=len(P), seed=0),
    metrics=dict(final_answer_accuracy=r3(len(right) / max(len(P), 1)), unparseable_rate=r3(sum(not p['clean'] for p in P) / max(len(P), 1)), compute_steps=6,
                 intervention_accuracy=r3(best_in),
                 extra=dict(accuracy_kind='recipient clean accuracy over candidate pairs (items repeat across hops)',
                            intervention_accuracy_definition='best single-site input-only follow rate (the necessity check)',
                            n_pairs=len(P), n_qualified=A['n'], donor_sensitivity=r3(sens),
                            donor_sensitivity_definition='among recipients answered right, share whose edited (donor) question the model answers as the edit implies',
                            best_full_swap_single_site=r3(best_b), follows=Rs(A), keeps=Rs(S['keeps_all']),
                            by_hop={h: Rs(v) for h, v in S['by_hop'].items()}, by_len={h: Rs(v) for h, v in S['by_len'].items()})),
    hyperparams=dict(script='scripts/necessity/grid_split.py (+ core.py, multi.py, adapters.py, pairs.py)', batch='27 conditions per pair in one batch (verified = single runs on the first 10 qualified pairs)',
                     pairs='ProsQA / ProntoQA: one gold-path relation rewired so the answer flips, equal token length (pairs.py), seed 0; GSM8K: 20260927-003331 twins, recipients matched by question text, equal length',
                     qualified='recipient answered right and donor answered its implied answer', dtype='float32', greedy=True, device='mps'),
    seed=0, hardware='MacBook M4 Pro 24 GB, MPS', notes=f'Necessity check on grid {method} ({task}): does any latent carry the donor value through its input only?')
rec.save(predictions=rows); print(rec.run_id)
