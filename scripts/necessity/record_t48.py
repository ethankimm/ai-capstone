"""Save the t48 run record (readability by refresh with the logit lens on pass outputs: released Llama-3.2-1B CODI and GPT-2 CODI).

  ../codi/.venv/bin/python scripts/necessity/record_t48.py
"""
import json
from pathlib import Path

from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id

A = Path(__file__).resolve().parents[3] / 'codi_audit_artifacts/t48'
R = {m: json.load(open(A / f'gsm8k_{m}_lens_h.json')) for m in ('llama', 'codi')}
preds = [dict(model=m, **it) for m, d in R.items() for it in d['items']]
extra = {m: dict(read=d['read'], skipped_multi_token=d['skipped'],
                 by_depth={str(k): dict(n=v['n'], own_readable=v['own_rate'], B_by_refresh=v['B'], stale_A=v['A'], chance=v['chance'])
                           for k, v in d['summary'].items()})
         for m, d in R.items()}
extra['t30_readZ_gpt2_codi_own_readable'] = {'1': 0.37, '2': 0.169, '3': 0.137}

rec = RunRecord(
    run_id=new_run_id('codi', 'gsm8k-readability-logit-lens-llama-vs-gpt2'),
    mechanism='codi',
    stage='full_run',
    model=ModelInfo(backbone='llama-3.2-1b-instruct (CODI, LoRA r128) and gpt2 (CODI, LoRA r128)',
                    checkpoint='hf:zen-E/CODI-llama3.2-1b-Instruct@b2c88ba; hf:zen-E/CODI-gpt2@fd641b3', n_params=None),
    dataset=DatasetInfo(name='gsm8k-aug', split='test', n_examples=len(preds), seed=0),
    metrics=dict(final_answer_accuracy=None, compute_steps=6, extra=extra),
    hyperparams=dict(readout='READ=h: top-10 tokens of lm_head(hidden_states[-1]) of each pass before the projection', topk=10,
                     pairs='llama: t47 qualified pairs answered right on both questions; gpt2: t30 pairs (t26 + long, QUAL=gold)'),
    seed=0,
    hardware='local Apple Silicon (MPS), 25.8 GB',
    notes='t48: readability by refresh with the standard logit lens, Llama-3.2-1B CODI vs GPT-2 CODI',
)
print('saved', rec.save(predictions=preds))
