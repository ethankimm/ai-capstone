"""Log t21a (route split, SIM-CODI start-first), t21b (monitor test, same model) and t21c (prefill probe, four models).
Usage: uv run python record21.py <audit dir> t21a|t21b|t21c"""
import json, sys
import numpy as np
from latentreasoning.runlog.manifest import RunRecord, ModelInfo, DatasetInfo, new_run_id
W, which = sys.argv[1], sys.argv[2]; r3 = lambda x: None if x is None else round(float(x), 3)
SF = W + '/../synthetic_data_v2_l1to7w/depth2/valid.json'; SIMSF = 'SIM-CODI start-first checkpoint-17820 (20261108-130905)'
if which == 't21a':
    R = json.load(open(W + '/t15_simsf.json')); va = json.load(open(SF)); rows = []; ex = {}
    for L, o in R.items():
        ex[L] = dict(clean=r3(o['clean']), **{c: dict(n=v['n'], follows=r3(v['follows']), keeps=r3(v['keeps'])) for c, v in o['res'].items()})
        items = [r for r in va if int(r['level']) == int(L)]; assert [r['id'] for r in items] == o['ids']
        for i, r in enumerate(items):
            rows.append(dict(idx=len(rows), id=r['id'], level=int(L), gold_answer=str(o['gold'][i]), predicted_answer=str(o['clean_ans'][i]),
                             correct=o['clean_ans'][i] == o['gold'][i], **{c: dict(ans=v['ans'][i], donor=v['donor'][i], target=v['target'][i]) for c, v in o['res'].items()}))
    rec = RunRecord(run_id=new_run_id('codi', 'v2-l1to7w-sim-route-split'), mechanism='codi', stage='full_run',
        model=ModelInfo(backbone='gpt2', checkpoint=SIMSF + ', loaded without the step decoder', n_params=144499200),
        dataset=DatasetInfo(name='synthetic-depth2', split='validation', n_examples=len(rows), seed=20),
        metrics=dict(final_answer_accuracy=r3(np.mean([x['correct'] for x in rows])), unparseable_rate=0.0, compute_steps=6,
                     extra=dict(accuracy_kind='clean exact on valid L3-L7 (start-first)', by_level=ex)),
        hyperparams=dict(script='codi_audit_artifacts/t15_rc_route_split.py (MODEL=simsf LEVELS=3,4,5,6,7), copy in scripts/codi_null_audit/prereg/',
                         conditions='as t15 (20261108-073431): both / input-only / entries-only / entries hidden from answer per site, and all sites; donors as t1 with equal question token length',
                         prereg='scripts/codi_null_audit/prereg/t_predictions.md, t21 (commit 96a04d4)', dtype='float32'),
        seed=20, hardware='MacBook M4 Pro 24 GB, MPS', notes='The necessity check (t15 route split) on SIM-CODI start-first, L3-L7')
elif which == 't21b':
    R = json.load(open(W + '/t21b_simsf.json')); rows = []; ex = {}
    for L, o in R['res'].items():
        ex[L] = dict(clean=r3(o['clean']), onroute=[int(j) for j, v in o['onroute'].items() if v])
        for k, v in o['sites'].items(): rows.append(dict(idx=len(rows), level=int(L), cell=k, **{a: r3(b) if isinstance(b, float) else b for a, b in v.items()}))
    rec = RunRecord(run_id=new_run_id('codi', 'v2-l1to7w-sim-monitor-faithfulness'), mechanism='codi', stage='full_run',
        model=ModelInfo(backbone='gpt2', checkpoint=SIMSF + ', loaded without the step decoder', n_params=144499200),
        dataset=DatasetInfo(name='synthetic-depth2', split='validation', n_examples=len(rows), seed=None),
        metrics=dict(final_answer_accuracy=None, unparseable_rate=None, compute_steps=6, extra=dict(by_level=ex, n_train=R['n_train'],
            definitions=dict(dec_right='decoder = true v_ks on clean right items', mon_wrong='monitor answer (decoded value pushed through remaining rules) = model answer on clean wrong items',
                             swap='swapJ_LOC_JP: t21a both_J donor; dec_donor = decoder at JP reads the donor-implied value; mon_actual = monitor answer equals the actual answer'))),
        hyperparams=dict(script='scripts/necessity/rc_monitor.py simsf mps <out> 1200', n_train_per_level=1200, train_levels='2-7 (train split)',
                         decoders='linear 100-way per site x {input z_j, entries = keys+values at layers 3/6/9/12}', test='valid L3-L7, t21a items and donors',
                         prereg='t_predictions.md t21 (commit 96a04d4)'),
        seed=None, hardware='MacBook M4 Pro 24 GB, MPS', notes='Monitor test (t19 design) on SIM-CODI start-first, including own errors')
else:
    R = json.load(open(W + '/t21c_prefill.json')); rows = []; ex = {}
    for m, o in R.items():
        ex[m] = {}
        for L, x in o.items():
            c = x['cells']; best = lambda p, k: max(c[f'{p}|{l}|v{k}']['acc'] for l in (4, 8, 12))
            ex[m][L] = dict(n_fit=x['n_fit'], n_test=x['n_test'], diagonal=[r3(best(f'rule{k}', k)) for k in range(1, int(L))],
                            at_or_after=[r3(max(best(f'rule{a}', k) for a in range(k, int(L) + 1))) for k in range(1, int(L))],
                            bot=[r3(best('bot', k)) for k in range(1, int(L))])
            for k, v in c.items(): rows.append(dict(idx=len(rows), model=m, level=int(L), cell=k, **{a: r3(b) for a, b in v.items()}))
    CK = dict(simsf=SIMSF, stab='start-first CODI checkpoint-17820 (20261105-044444)', sim='SIM-CODI start-last checkpoint-16335 (20261107-225603)',
              startlast='start-last CODI checkpoint-17820 (20261106-083843)')
    rec = RunRecord(run_id=new_run_id('codi', 'v2-l1to7w-prefill-chain-probe'), mechanism='codi', stage='full_run',
        model=ModelInfo(backbone='gpt2', checkpoint='; '.join(f'{k}: {v}' for k, v in CK.items()), n_params=144499200),
        dataset=DatasetInfo(name='synthetic-depth2', split='validation', n_examples=len(rows), seed=30),
        metrics=dict(final_answer_accuracy=None, unparseable_rate=None, compute_steps=0,
                     extra=dict(cells='best of layers 4/8/12, exact; diagonal = v_k at the end of rule k; at_or_after = best over rule ends m >= k; bot = <bot>', by_model=ex)),
        hyperparams=dict(script='scripts/necessity/prefill_probe.py', probes='ridge to 100-way one-hot, standardised, penalty from a held-out 15%; shuffled control on the diagonal and at <bot>',
                         n_fit_per_level=2000, test='all valid items of L3-L6', positions='period ending each rule, and <bot>', prereg='t_predictions.md t21 (commit 96a04d4)'),
        seed=30, hardware='MacBook M4 Pro 24 GB, MPS', notes='Does the question pass run the chain by itself, next to the latents?')
rec.save(predictions=rows); print(rec.run_id)
