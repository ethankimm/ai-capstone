"""Log t14 (GSM8K look-back cut by chain length, full test). Usage: uv run python record19.py <audit dir>"""
import json, sys, collections
import numpy as np
from latentreasoning.runlog.manifest import RunRecord, ModelInfo, DatasetInfo, new_run_id
W = sys.argv[1]; r3 = lambda x: round(float(x), 3); R = json.load(open(W + '/t14_gsm8k_lookback_depth.json'))
by = collections.defaultdict(list)
for r in R: by[min(r['steps'], 6)].append(r)
ex = {}
for n in sorted(by):
    ok = [r for r in by[n] if r['clean_ok']]
    ex[str(n)] = dict(n=len(by[n]), clean_acc=r3(np.mean([r['clean_ok'] for r in by[n]])), n_right=len(ok),
                      **{f'kept_{c}': (r3(np.mean([r[c + '_ok'] for r in ok])) if ok else None) for c in ('nb_all', 'nb_even', 'nb_odd')})
acc = {c: r3(np.mean([r[c + '_ok'] for r in R])) for c in ('clean', 'nb_all', 'nb_even', 'nb_odd')}
rows = [dict(idx=r['idx'], id=f"gsm8k-test-{r['idx']}", steps=r['steps'], gold_answer=str(r['gold']), predicted_answer=str(r['clean']), correct=r['clean_ok'],
             nb_all=r['nb_all'], nb_even=r['nb_even'], nb_odd=r['nb_odd']) for r in R]
rec = RunRecord(run_id=new_run_id('codi', 'gsm8k-released-lookback-by-depth'), mechanism='codi', stage='full_run',
    model=ModelInfo(backbone='gpt2', checkpoint='hf:zen-E/CODI-gpt2@fd641b3 (released GSM8K CODI), fp32 MPS batch 1', n_params=144499200),
    dataset=DatasetInfo(name='gsm8k-aug', split='test', n_examples=len(rows), seed=None),
    metrics=dict(final_answer_accuracy=acc['clean'], unparseable_rate=None, compute_steps=6, extra=dict(accuracy_kind='full test, this collector', accuracy=acc, by_gold_steps=ex)),
    hyperparams=dict(script='codi_audit_artifacts/t14_gsm8k_lookback_depth.py (t12 model and run())', cuts='nb_all / nb_even / nb_odd: latent iterations masked from all / even / odd earlier latent sites', dtype='float32'),
    seed=11, hardware='MacBook M4 Pro 24 GB, MPS', notes='Released GSM8K CODI: does the look-back cut hurt more on longer chains, on the full test set?')
rec.save(predictions=rows); print(rec.run_id); print(acc)
