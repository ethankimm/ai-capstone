"""Save the t49 run record (released SIM-CODI Llama-3.2-1B on GSM8K: accuracy gate + hop count + readability READ=h, against Llama CODI
1B (t47) and the GPT-2 checkpoints). Run gsm8k_steps_per_pass.py and gsm8k_llama_compare.py first (outputs in t49/).

  ../codi/.venv/bin/python scripts/necessity/record_t49.py
"""
import json
from pathlib import Path

from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id

A = Path(__file__).resolve().parents[3] / 'codi_audit_artifacts/t49'
acc = json.load(open(A / 'gsm8k_simllama1b_acc.json'))
cmp_ = json.load(open(A / 'gsm8k_llama_compare.json'))
spp = cmp_['simllama1b']['hops']
lens = json.load(open(A / 'gsm8k_simllama1b_lens_h.json'))

preds, qual, mk = [], {}, [0, 0]
for f, tag in (('gsm8k_simllama1b.json', 't22'), ('long_gsm8k_simllama1b.json', 't26_long')):
    ps = [p for p in json.load(open(A / f))['pairs'] if p.get('qualified')]
    qual[tag] = len(ps)
    for p in ps:
        mk[0] += bool(p['mk_matches_donor']); mk[1] += 1
        preds.append(dict(pair_set=tag, recipient_idx=p['id'], edit=p['edit'], s_index=p['s'],
                          question_R=p['R']['question'], question_D=p['D']['question'], gold_R=p['R']['answer'], gold_D=p['D']['answer'],
                          answer_R=p['aR'], answer_D=p['aD'], R_right=p['R_right'], D_right=p['D_right'],
                          answers_by_refresh=p['A'], follows_D=p['follows'], hops=p['hops'], mk_matches_donor=p['mk_matches_donor']))
for x in acc['items']:
    preds.append(dict(pair_set='accuracy_gate', idx=x['idx'], question=x['question'], gold=x['gold'], pred=x['pred'], right=x['right']))
for x in lens['items']:
    preds.append(dict(pair_set='readability_READ_h', **x))

ref = lambda m: {D: v['hops'] for D, v in cmp_[m]['hops']['start_of_path'].items()} if 'hops' in cmp_.get(m, {}) else None
rec = RunRecord(
    run_id=new_run_id('codi', 'gsm8k-released-simcodi-llama1b-hop-count-readability'),
    mechanism='codi',
    stage='full_run',
    model=ModelInfo(backbone='llama-3.2-1b-instruct (SIM-CoT CODI, LoRA r128/a32, projection 2048)',
                    checkpoint='hf:internlm/SIM_COT-LLaMA3-CODI-1B (step decoder unused); base config/tokenizer hf:unsloth/Llama-3.2-1B-Instruct with Meta tokenizer settings (adapters._llama_base)',
                    n_params=None),
    dataset=DatasetInfo(name='gsm8k-aug', split='test', n_examples=acc['n'], seed=0),
    metrics=dict(final_answer_accuracy=acc['acc'], compute_steps=6, extra=dict(
        published_accuracy=0.561, missing_keys=acc['missing_keys'], unexpected_keys_step_decoder=acc['unexpected_keys'],
        acc_by_depth_strict={b: v['mean'] for b, v in cmp_['simllama1b']['acc_by_depth'].items()},
        hop_pairs_qualified=qual, mk_matches_donor=f'{mk[0]}/{mk[1]}',
        hops_by_serial_distance_all={s: spp['by_s'][s]['all'].get('hops') for s in ('2', '3', '4', '5') if s in spp['by_s']},
        hops_start_of_path_correct={D: v['hops'] for D, v in spp['start_of_path'].items()},
        steps_per_pass_start_of_path_correct={D: v['steps_per_pass'] for D, v in spp['start_of_path'].items()},
        within_item_slope_all=spp['within_item']['all']['slope'], within_item_slope_correct=spp['within_item']['correct']['slope'],
        readability_READ_h={str(k): dict(n=v['n'], own_readable=v['own_rate'], B_by_refresh=v['B'], stale_A=v['A'], chance=v['chance'])
                            for k, v in lens['summary'].items()},
        references=dict(llama_codi_1b_t47=ref('llama'), gpt2_codi=ref('codi'), gpt2_simcodi=ref('sim')))),
    hyperparams=dict(num_latent=6, greedy=True, qual='own', first=0, dtype='fp32', readout='READ=h (logit lens on pass outputs, t48)',
                     pair_sets='t22 twins (1,471) + t26 long twins (1,305)'),
    seed=0,
    hardware='local Apple Silicon (MPS), 25.8 GB',
    notes='t49: hop count, steps per pass, accuracy and READ=h readability of the released SIM-CODI Llama-3.2-1B on GSM8K',
)
print('saved', rec.save(predictions=preds))
