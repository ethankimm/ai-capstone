"""Log t16 (grid need test) as one RunRecord per task. Usage: uv run python scripts/necessity/record_t16.py <audit dir> <task>"""
import json
import sys
from pathlib import Path

from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id

W, task = Path(sys.argv[1]), sys.argv[2]
methods = [m for m in ('codi', 'coconut', 'no_cot', 'cot') if (W / f't16_{task}_{m}.json').exists()]
R = {m: json.load(open(W / f't16_{task}_{m}.json')) for m in methods}
r3 = lambda x: round(float(x), 3)
base = R[methods[0]]; n = base['n']; rows = []
for i in range(n):
    row = dict(idx=i, gold_answer=str(base['gold'][i]), n_steps=base['bins'][i])
    for m in methods:
        for k, a in R[m]['answers'].items(): row[f'{m}_{k}'] = a[i]
    rows.append(row)
main = 'coconut' if 'coconut' in R else methods[0]; mk = '6' if main in ('codi', 'coconut') else 'plain'
summ = {m: {k: dict(acc=r3(v['acc']), unparseable=r3(v['unparseable']), by_steps={b: dict(n=x['n'], acc=r3(x['acc'])) for b, x in v['by_bin'].items()})
            for k, v in R[m]['summary'].items()} for m in methods}
for row in rows:
    row['predicted_answer'] = row[f'{main}_{mk}']
    row['correct'] = None
ck = {'codi': f'hf:connordilgren/gpt2-{task}-codi' if task != 'gsm8k' else 'zen-E/CODI-gpt2', 'coconut': f'hf:connordilgren/gpt2-{task}-coconut',
      'no_cot': f'hf:connordilgren/gpt2-{task}-no-cot', 'cot': f'hf:connordilgren/gpt2-{task}-cot'}
rec = RunRecord(run_id=new_run_id(main, f'{task}-grid-need-by-path-length'), mechanism=main, stage='full_run',
    model=ModelInfo(backbone='gpt2', checkpoint='; '.join(f'{m}: {ck[m]}' for m in methods), n_params=124442112),
    dataset=DatasetInfo(name=task, split='test', n_examples=n, seed=None),
    metrics=dict(final_answer_accuracy=summ[main][mk]['acc'], unparseable_rate=summ[main][mk]['unparseable'], compute_steps=6,
                 early_termination_necessity=r3(summ[main]['6']['acc'] - summ[main]['0']['acc']) if main in ('codi', 'coconut') else None,
                 extra=dict(by_method=summ, need_definition='6-latent accuracy minus the same model at 0 latents, per gold step count',
                            accuracy_kind=f'{main} at 6 latents; other methods under extra.by_method')),
    hyperparams=dict(script='scripts/necessity/grid_accuracy.py', batch_size=1, dtype='float32', greedy=True, device='mps',
                     zero_latents='CODI: question + <bot> + <eot>; Coconut: question, <start-latent>, k passes, <end-latent> (k = 0..6)',
                     answer='CODI: text after "The answer is:"; Coconut / CoT / no-CoT: text after "###"; exact match after lower-casing'),
    seed=None, hardware='MacBook M4 Pro 24 GB, MPS',
    notes=f'Need test on {task}: do latent passes add accuracy, and does the gain grow with path length?')
for row in rows:
    g = row['gold_answer']; a = row['predicted_answer']
    if task != 'gsm8k':
        row['correct'] = str(a).strip().rstrip('.').lower() == g.strip().rstrip('.').lower()
    else:
        try: row['correct'] = float(str(a).replace(',', '')) == float(g)
        except ValueError: row['correct'] = False
rec.save(predictions=rows); print(rec.run_id)
