"""Log t11 (GSM8K routes), t12 (GSM8K route split + odd/even look-back), t13 (GSM8K attention flow). Usage: uv run python record18.py <audit dir>"""
import json, sys
import numpy as np
from latentreasoning.runlog.manifest import RunRecord, ModelInfo, DatasetInfo, new_run_id
W = sys.argv[1]; r3 = lambda x: round(float(x), 3)
MI = ModelInfo(backbone='gpt2', checkpoint='hf:zen-E/CODI-gpt2@fd641b3 (released GSM8K CODI, local HF cache), fp32 MPS batch 1', n_params=144499200)
HW = 'MacBook M4 Pro 24 GB, MPS'
# t11
T = json.load(open(W + '/t11_gsm8k_routes.json')); P = T['pairs']
q = [r for r in P if str(r['base']) == str(r['gold']) and str(r['twin_pred']) == str(r['twin_answer'])]
C = ['all', 'single_0', 'single_2', 'single_4', 'even_02', 'even_04', 'even_24', 'even_all', 'prefix_2']
f = lambda c: r3(np.mean([str(r[c]) == str(r['twin_answer']) for r in q]))
ex = dict(n_qualified=len(q), full_test=T['full_test'], follows_twin={c: f(c) for c in C}, follows_twin_noback={c: f('nb_' + c) for c in C},
          recipient_right_noback=r3(np.mean([str(r['nb_clean']) == str(r['gold']) for r in q])),
          record_20260927={'all': .839, 'single_0': .199, 'single_2': .202, 'single_4': .360, 'prefix_2': .539})
rows = [dict(idx=i, id=f"gsm8k-test-{r['recipient_idx']}", step=r['step'], gold_answer=str(r['gold']), predicted_answer=str(r['base']), correct=str(r['base']) == str(r['gold']),
             twin_answer=r['twin_answer'], **{k: r[k] for k in r if k.startswith('nb_') or k in C}) for i, r in enumerate(P)]
rec = RunRecord(run_id=new_run_id('codi', 'gsm8k-released-latent-routes'), mechanism='codi', stage='full_run', model=MI,
    dataset=DatasetInfo(name='gsm8k-aug', split='test', n_examples=len(rows), seed=None),
    metrics=dict(final_answer_accuracy=r3(T['full_test']['clean']), unparseable_rate=None, compute_steps=6, extra=dict(accuracy_kind='full test (1319) clean, this collector', **ex)),
    hyperparams=dict(script='codi_audit_artifacts/t11_gsm8k_routes.py', pairs='317 aligned minimal pairs of 20260927-004340', noback='latent iteration masked from earlier latent iterations; answer sees all', dtype='float32'),
    seed=11, hardware=HW, notes='Released GSM8K CODI: site swaps (singles, even pairs, prefix) and the look-back cut, against the RuleChain route results.')
rec.save(predictions=rows); print(rec.run_id)
# t12
T = json.load(open(W + '/t12_gsm8k_route_split.json')); P = T['pairs']
q = [r for r in P if str(r['base']) == str(r['gold']) and str(r['twin_pred']) == str(r['twin_answer'])]; qs = [r for r in q if r['same_len']]
ex = dict(n_qualified=len(q), n_same_length=len(qs), full_test_acc=T['full_test'],
          recipients_right={c: r3(np.mean([str(r[c]) == str(r['gold']) for r in q])) for c in ('nb_all', 'nb_even', 'nb_odd')},
          route_split={s: {c: r3(np.mean([str(r[f'{c}_{s}']) == str(r['twin_answer']) for r in qs])) for c in ('both', 'input', 'kv', 'kv_noans')} for s in (0, 2, 4)},
          noans_keeps_recipient={s: r3(np.mean([str(r[f'noans_{s}']) == str(r['gold']) for r in qs])) for s in (0, 2, 4)})
rows = [dict(idx=i, id=f"gsm8k-test-{r['recipient_idx']}", step=r['step'], gold_answer=str(r['gold']), predicted_answer=str(r['base']), correct=str(r['base']) == str(r['gold']),
             **{k: v for k, v in r.items() if k not in ('recipient_idx', 'step', 'gold', 'base')}) for i, r in enumerate(P)]
rec = RunRecord(run_id=new_run_id('codi', 'gsm8k-released-route-split'), mechanism='codi', stage='full_run', model=MI,
    dataset=DatasetInfo(name='gsm8k-aug', split='test', n_examples=len(rows), seed=None),
    metrics=dict(final_answer_accuracy=None, unparseable_rate=None, compute_steps=6, extra=dict(accuracy_kind='pairs; full-test accuracy under partial look-back cuts in full_test_acc', **ex)),
    hyperparams=dict(script='codi_audit_artifacts/t12_gsm8k_route_split.py', input_s='twin z_s fed, that iteration keys/values reset to the recipient own', kv_s='recipient z_s fed, keys/values from the twin',
                     kv_noans_s='kv_s plus answer tokens masked from site s', dtype='float32'),
    seed=11, hardware=HW, notes='Released GSM8K CODI: does a latent value travel through the recurrent input or through its keys/values (attention)?')
rec.save(predictions=rows); print(rec.run_id)
# t13
T = json.load(open(W + '/t13_gsm8k_attn_flow.json')); ex = {}
for g, o in T.items():
    S = np.array(o['norm_share']).mean(1); ns = S[:, :4] / S[:, :4].sum(1, keepdims=True)
    ex[g] = dict(n=o['n'], earlier_latents=[r3(x) for x in ns[:, 1] + ns[:, 2]], even_prev=[r3(x) for x in ns[:, 1]], odd_prev=[r3(x) for x in ns[:, 2]], question=[r3(x) for x in ns[:, 0]])
rows = [dict(idx=i, id=f'gsm8k-test-{g}', group=g, gold_answer='', predicted_answer='', correct=g == 'right', n_items=o['n']) for i, (g, o) in enumerate(T.items())]
rec = RunRecord(run_id=new_run_id('codi', 'gsm8k-released-attention-flow'), mechanism='codi', stage='full_run', model=MI,
    dataset=DatasetInfo(name='gsm8k-aug', split='test', n_examples=sum(o['n'] for o in T.values()), seed=None),
    metrics=dict(final_answer_accuracy=r3(T['right']['n'] / (T['right']['n'] + T['wrong']['n'])), unparseable_rate=None, compute_steps=6,
                 extra=dict(accuracy_kind='share of test items answered right (grouping only); rows are aggregates', norm_share_nonsink=ex)),
    hyperparams=dict(script='codi_audit_artifacts/t13_gsm8k_attn_flow.py', measures='as t9: raw attention and Kobayashi norm, sets question / earlier even / earlier odd / self / sink', dtype='float32'),
    seed=11, hardware=HW, notes='Released GSM8K CODI: where each latent iteration reads from (t9 measures, the GSM8K row).')
rec.save(predictions=rows); print(rec.run_id)
