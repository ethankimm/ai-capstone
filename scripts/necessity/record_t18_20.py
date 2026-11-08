"""Log t18 (ProsQA probes), t19 (RuleChain monitor), t20 (teacher dependence). Usage: uv run python scripts/necessity/record_t18_20.py <audit dir> t18|t19|t20"""
import json
import sys
from pathlib import Path

from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id

W, which = Path(sys.argv[1]), sys.argv[2]
r3 = lambda x: None if x is None else round(float(x), 3)
HW = 'MacBook M4 Pro 24 GB, MPS'
RC = dict(gated='rule-gated start-last CODI (20261107-204814)', sim='SIM-CODI start-last (20261107-225603)', startlast='start-last CODI (20261106-083843)',
          relay='lr-3e-3 L1-7w CODI (20261103-193425)', stab='stabilised L1-7w CODI (20261105-044444)')

if which == 't18':
    for m in ('coconut', 'codi'):
        R = json.load(open(W / f't18_{m}.json')); res = {k: {a: r3(b) if not isinstance(b, int) else b for a, b in v.items()} for k, v in R['res'].items()}
        rows = [dict(idx=i, probe=k, **v) for i, (k, v) in enumerate(sorted(res.items()))]
        rec = RunRecord(run_id=new_run_id(m, 'prosqa-grid-path-frontier-probes'), mechanism=m, stage='full_run',
            model=ModelInfo(backbone='gpt2', checkpoint=f'hf:connordilgren/gpt2-prosqa-{m}', n_params=124442112 if m == 'coconut' else 144499200),
            dataset=DatasetInfo(name='prosqa', split='test', n_examples=500, seed=0),
            metrics=dict(final_answer_accuracy=None, compute_steps=6, decoding_accuracy=res['lat0|L12|path_1']['acc'],
                         extra=dict(decoding_accuracy_definition='path_1 probe accuracy at latent 0, layer 12 (argmax among the item graph nodes)', probes=res)),
            hyperparams=dict(script='scripts/necessity/grid_probe.py', n_train=R['n_train'], train_split='prosqa train (seed 0 shuffle)', layers=[6, 9, 12],
                             probe='linear, standardised, Adam lr 1e-2, wd 1e-3, 300 full-batch epochs; shuffled-label control per probe'),
            seed=0, hardware=HW, notes=f'ProsQA {m}: are path nodes / BFS frontiers readable at the question tail vs the latents (and the paper LM-head readout)?')
        rec.save(predictions=rows); print(rec.run_id)

elif which == 't19':
    rows = []; ex = {}
    for m in ('gated', 'sim', 'startlast', 'relay'):
        R = json.load(open(W / f't19_{m}.json'))['res']; ex[m] = {}
        for L, o in R.items():
            ex[m][L] = dict(clean=r3(o['clean']), onroute=[int(j) for j, v in o['onroute'].items() if v],
                            sites={k: {a: r3(b) if isinstance(b, float) else b for a, b in v.items()} for k, v in o['sites'].items()})
            for k, v in o['sites'].items(): rows.append(dict(idx=len(rows), model=m, level=int(L), cell=k, **{a: r3(b) if isinstance(b, float) else b for a, b in v.items()}))
    rec = RunRecord(run_id=new_run_id('codi', 'v2-l1to7w-rulechain-monitor-faithfulness'), mechanism='codi', stage='full_run',
        model=ModelInfo(backbone='gpt2', checkpoint='; '.join(f'{k}: {RC[k]}' for k in ('gated', 'sim', 'startlast', 'relay')), n_params=144499200),
        dataset=DatasetInfo(name='synthetic-depth2', split='validation', n_examples=len(rows), seed=None),
        metrics=dict(final_answer_accuracy=None, compute_steps=6, extra=dict(by_model=ex,
            definitions=dict(dec_right='decoder = true v_ks on clean right items', mon_wrong='monitor answer (decoded value pushed through remaining rules) = model answer on clean wrong items',
                             swap='swapJ_LOC_JP: t15 both_J donor; dec_donor = decoder at JP reads the donor-implied value; mon_actual = monitor answer equals the actual answer',
                             onroute='t15 input-only follow >= .5'))),
        hyperparams=dict(script='scripts/necessity/rc_monitor.py', n_train_per_level=1200, train_levels='2-6 (train split)', decoders='linear 100-way per site x {input z_j, entries = keys+values at layers 3/6/9/12}',
                         epochs=200, lr=3e-3, wd=1e-3, test='valid L3-L5, t15 items and donors'),
        seed=None, hardware=HW, notes='Monitor steps 2-3 on RuleChain: are decoders at on-route latents faithful and accurate decoders at off-route latents decorative?')
    rec.save(predictions=rows); print(rec.run_id)

elif which == 't20':
    rows = []; ex = {}
    for tag in ('rulechain_startlast', 'rulechain_relay', 'rulechain_stab', 'rulechain_sim', 'gsm8k_zen', 'prosqa_grid'):
        R = json.load(open(W / f't20_{tag}.json')); ex[tag] = dict(scale=r3(R['scale']), by_k={k: {a: r3(b) if isinstance(b, float) else b for a, b in v.items()} for k, v in R['summary'].items()})
        for it in R['items']:
            for k, v in it['per_k'].items(): rows.append(dict(idx=len(rows), run=tag, item=it['i'], k_from_end=int(k), **v))
    rec = RunRecord(run_id=new_run_id('codi', 'teacher-dependence-rulechain-gsm8k-prosqa'), mechanism='codi', stage='full_run',
        model=ModelInfo(backbone='gpt2', checkpoint='RuleChain: ' + ', '.join(RC[k] for k in ('startlast', 'relay', 'stab', 'sim')) + '; GSM8K: zen-E/CODI-gpt2; ProsQA: hf:connordilgren/gpt2-prosqa-codi', n_params=144499200),
        dataset=DatasetInfo(name='synthetic-depth2 (valid L3-L6) + gsm8k-aug test (>= 3 steps) + prosqa test (>= 4 steps); split field = test, RuleChain items are from valid', split='test', n_examples=len(rows), seed=0),
        metrics=dict(final_answer_accuracy=None, compute_steps=6, extra=dict(by_run=ex,
            definitions=dict(flip="P(teacher's first answer token changes; ProsQA: preferred option, teacher-forced)",
                             move='median ||h12 change|| at the ":" of "The answer is:" / median distance between two items', k='counted from the end of the shown CoT, 1 = last shown value'))),
        hyperparams=dict(script='scripts/necessity/teacher_dependence.py', n=300, corruption='one shown CoT value replaced positionally (its result and next mention), same token length'),
        seed=0, hardware=HW, notes='Which CoT values does the CODI teacher\'s answer state (the distillation target) depend on?')
    rec.save(predictions=rows); print(rec.run_id)
