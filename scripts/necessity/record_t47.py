"""Save the t47 run record (released Llama-3.2-1B CODI on GSM8K: accuracy gate + hop count + comparison with the GPT-2 checkpoints).

  ../codi/.venv/bin/python scripts/necessity/record_t47.py
"""
import json
from pathlib import Path

from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id

A = Path(__file__).resolve().parents[3] / 'codi_audit_artifacts/t47'
acc = json.load(open(A / 'gsm8k_llama_acc.json'))
cmp_ = json.load(open(A / 'gsm8k_llama_compare.json'))
spp = cmp_['llama']['hops']

preds = []
for f in ('gsm8k_llama.json', 'long_gsm8k_llama.json'):
    for p in json.load(open(A / f))['pairs']:
        if not p.get('qualified'): continue
        preds.append(dict(pair_set='t22' if f == 'gsm8k_llama.json' else 't26_long', recipient_idx=p['id'], edit=p['edit'], s_index=p['s'],
                          question_R=p['R']['question'], question_D=p['D']['question'], gold_R=p['R']['answer'], gold_D=p['D']['answer'],
                          answer_R=p['aR'], answer_D=p['aD'], R_right=p['R_right'], D_right=p['D_right'],
                          answers_by_refresh=p['A'], follows_D=p['follows'], hops=p['hops'], mk_matches_donor=p['mk_matches_donor']))
for x in acc['items']:
    preds.append(dict(pair_set='accuracy_gate', idx=x['idx'], question=x['question'], gold=x['gold'], pred=x['pred'], right=x['right']))

rec = RunRecord(
    run_id=new_run_id('codi', 'gsm8k-released-llama1b-hop-count'),
    mechanism='codi',
    stage='full_run',
    model=ModelInfo(backbone='llama-3.2-1b-instruct (LoRA r128/a32, projection 2048)',
                    checkpoint='hf:zen-E/CODI-llama3.2-1b-Instruct@b2c88ba; base config/tokenizer hf:unsloth/Llama-3.2-1B-Instruct with Meta tokenizer settings (adapters._llama_base)',
                    n_params=1_334_394_880),
    dataset=DatasetInfo(name='gsm8k-aug', split='test', n_examples=acc['n'], seed=0),
    metrics=dict(final_answer_accuracy=acc['acc'], compute_steps=6, extra=dict(
        published_accuracy=0.556, acc_by_depth_strict={b: v['mean'] for b, v in cmp_['llama']['acc_by_depth'].items()},
        hop_pairs_qualified=dict(t22=1315, t26_long=1142), mk_matches_donor='2457/2457',
        hops_by_serial_distance_all={s: spp['by_s'][s]['all'].get('hops') for s in ('2', '3', '4', '5')},
        hops_start_of_path_correct={D: v['hops'] for D, v in spp['start_of_path'].items()},
        steps_per_pass_start_of_path_correct={D: v['steps_per_pass'] for D, v in spp['start_of_path'].items()},
        within_item_slope_all=spp['within_item']['all']['slope'], within_item_slope_correct=spp['within_item']['correct']['slope'],
        gpt2_codi_reference=dict(hops_start_of_path_correct={D: v['hops'] for D, v in cmp_['codi']['hops']['start_of_path'].items()},
                                 within_item_slope_all=cmp_['codi']['hops']['within_item']['all']['slope']))),
    hyperparams=dict(num_latent=6, greedy=True, qual='own', first=0, dtype='fp32', pair_sets='t22 twins (1,471) + t26 long twins (1,305)'),
    seed=0,
    hardware='local Apple Silicon (MPS), 25.8 GB',
    notes='t47: hop count, steps per pass and accuracy of the released Llama-3.2-1B CODI on GSM8K, against the released GPT-2 checkpoints',
)
out = rec.save(predictions=preds)
print('saved', out)
