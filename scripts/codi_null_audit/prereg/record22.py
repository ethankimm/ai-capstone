"""Log t22 (released SIM-CODI GPT-2 vs released CODI GPT-2 on GSM8K: pass sweep + step decoder + lens; route split) and t24
(same-level mean ablation, round-1 / lr-3e-3 CODI). Usage (repo root):
  uv run python scripts/codi_null_audit/prereg/record22.py <audit dir> sweep_sim|sweep_codi|route_sim|route_codi|t24_r1|t24_lr3
Summaries come from scripts/necessity/sim_gsm8k_analyze.py (t22_summary.json)."""
import json, sys
from latentreasoning.runlog.manifest import RunRecord, ModelInfo, DatasetInfo, new_run_id
W, which = sys.argv[1], sys.argv[2]; r3 = lambda x: None if x is None else round(float(x), 3)
CK = dict(sim='hf:internlm/SIM_COT-GPT2-CODI (main, Sept 2025; step decoder present, not used at inference), fp32 MPS batch 1',
          codi='hf:zen-E/CODI-gpt2@fd641b3 (released GSM8K CODI, local HF cache), fp32 MPS batch 1')
PRE = 'scripts/codi_null_audit/prereg/t_predictions.md t22-t24 (commit 6a1e88b)'
HW = 'MacBook M4 Pro 24 GB, MPS'
if which.startswith('sweep_'):
    m = which.split('_')[1]; d = json.load(open(f'{W}/t22/t22_sweep_{m}.json')); S = json.load(open(f'{W}/t22/t22_summary.json'))[m]
    rows = [dict(idx=i, id=x['idx'], gold_answer=str(x['gold']), predicted_answer=str(x['ans']), correct=bool(x['correct']), n_steps=len(x['values']),
                 sweep=x['sweep'], sweep_correct=x['sweep_correct'], decoded=x['decoded'], lens_top1=x['lens']) for i, x in enumerate(d['items'])]
    rec = RunRecord(run_id=new_run_id('codi', f'gsm8k-released-{m}-pass-sweep-decoder'), mechanism='codi', stage='full_run',
        model=ModelInfo(backbone='gpt2', checkpoint=CK[m], n_params=144499200),
        dataset=DatasetInfo(name='gsm8k-aug', split='test', n_examples=len(rows), seed=None),
        metrics=dict(final_answer_accuracy=r3(S['acc']), unparseable_rate=r3(sum(x['predicted_answer'] == 'None' for x in rows) / len(rows)), compute_steps=6,
                     extra=dict(acc_by_k=[r3(sum(x['sweep_correct'][k] for x in rows) / len(rows)) for k in range(7)], sweep_by_n=S['sweep_by_n'],
                                step_decoder=S['decoder'], trace_off_right=S['trace_off_right'], trace_off_wrong=S['trace_off_wrong'],
                                own_error_any_decoded=S['own_error_any_decoded'], lens_n2_v1=S.get('lens_n2_v1'))),
        hyperparams=dict(script='scripts/necessity/sim_gsm8k.py sweep ' + m, analysis='scripts/necessity/sim_gsm8k_analyze.py',
                         sweep='answer after k = 0..6 latent passes (cache copied after each pass, CODI decode_answer path)',
                         decoder='SIM-CoT step decoder (decoder.* of internlm/SIM_COT-GPT2-CODI) applied greedily to z_0..z_6' + (' (control: weights from SIM, latents from CODI)' if m == 'codi' else ''),
                         lens='top-1 token, layers 4/6/8/10/12, question last position and each latent pass', prereg=PRE, dtype='float32'),
        seed=11, hardware=HW, notes=f't22a/c/d: pass sweep, step-decoder monitor and lens on released {m.upper()} GPT-2, GSM8K test')
elif which.startswith('route_'):
    m = which.split('_')[1]; d = json.load(open(f'{W}/t22/t22_route_{m}.json')); S = json.load(open(f'{W}/t22/t22_summary.json'))[m]
    rows = [dict(idx=i, **{k: v for k, v in p.items()}) for i, p in enumerate(d['pairs'])]
    rec = RunRecord(run_id=new_run_id('codi', f'gsm8k-released-{m}-route-split-all-sites'), mechanism='codi', stage='full_run',
        model=ModelInfo(backbone='gpt2', checkpoint=CK[m], n_params=144499200),
        dataset=DatasetInfo(name='gsm8k-aug', split='test', n_examples=len(rows), seed=0),
        metrics=dict(final_answer_accuracy=None, unparseable_rate=None, compute_steps=6,
                     extra=dict(route=S['route'], swap_faith=S.get('swap_faith'))),
        hyperparams=dict(script='scripts/necessity/sim_gsm8k.py route ' + m, analysis='scripts/necessity/sim_gsm8k_analyze.py',
                         pairs='1471 twin candidates of 20260927-004340 regenerated exactly (codi_audit_artifacts/t22/gsm8k_twin_candidates.json), qualified for this model: recipient and twin right, equal question token length',
                         conditions='per site s=0..5: both / input (pass s keys-values reset to the recipient) / kv (own z_s, twin keys-values) / kv_noans; both_all, kv_all; step decoder on downstream latents after both_s',
                         prereg=PRE, dtype='float32'),
        seed=11, hardware=HW, notes=f't22b: route split at all six sites, released {m.upper()} GPT-2, GSM8K')
else:
    m = which.split('_')[1]; R = json.load(open(f'{W}/t24_{m}.json')); rows = []; ex = {}
    for L, o in R.items():
        ex[L] = dict(n=o['n'], clean=r3(o['clean']), mean_acc=r3(o['mean']['acc_all']), mean_keeps_right=r3(o['mean']['keeps_own_right']),
                     diff_donor=dict(n=o['diff']['n'], follows=r3(o['diff']['follows_donor']), keeps=r3(o['diff']['keeps_own'])),
                     same_donor=dict(n=o['same']['n'], keeps=r3(o['same']['keeps_own'])))
        for i in range(o['n']):
            rows.append(dict(idx=len(rows), id=o['ids'][i], level=int(L), gold_answer=str(o['gold'][i]), predicted_answer=str(o['clean_ans'][i]),
                             correct=o['clean_ans'][i] == o['gold'][i], mean_ans=o['mean_ans'][i], diff_ans=o['diff_ans'][i], diff_target=o['diff_target'][i]))
    ckl = dict(r1='round-1 CODI rulechain-checkpoints/gpt2_depth2_codi_bs64/checkpoint-31280 (20261102-000204)',
               lr3='lr-3e-3 L1-7w CODI codi_runs_l1to7w_codi/checkpoint-17820 (20261103-193425)')
    rec = RunRecord(run_id=new_run_id('codi', f'v2-{m}-samelevel-mean-ablation'), mechanism='codi', stage='full_run',
        model=ModelInfo(backbone='gpt2', checkpoint=ckl[m], n_params=144499200),
        dataset=DatasetInfo(name='synthetic-depth2', split='test', n_examples=len(rows), seed=21),
        metrics=dict(final_answer_accuracy=r3(sum(x['correct'] for x in rows) / len(rows)), unparseable_rate=0.0, compute_steps=6, extra=dict(by_level=ex)),
        hyperparams=dict(script='codi_audit_artifacts/t24_samelevel_mean.py (b29_ungated_resample.py with MODEL switch), copy in scripts/codi_null_audit/prereg/',
                         mean='each z_s set to its mean over 2000 train items of the same level (seed 21)', donors='as b29: all six latents from a same-level donor with a different v_{L-1}',
                         levels='test L1-L5', prereg=PRE, dtype='float32'),
        seed=21, hardware=HW, notes=f't24: same-level mean ablation of all six latents, {m}, replacing the level-pooled means')
rec.save(predictions=rows); print(rec.run_id)
