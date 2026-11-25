"""Save the t58 run records (third-task pilot; pre-registered and scored in scripts/codi_null_audit/prereg/t_predictions.md).
The pilot script keeps only per-k aggregates (no per-item outputs), so `predictions` holds one row per (epoch, split, k).

  ../codi/.venv/bin/python scripts/necessity/record_t58.py
"""
import json, re, sys
from pathlib import Path

from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id

sys.path.insert(0, str(Path(__file__).parent)); from third_task_score import score
A = Path(__file__).resolve().parents[3] / 'codi_audit_artifacts' / 't58'
HW = 'local Apple Silicon M4 Pro (MPS), 24 GB'
CMD = '../codi/.venv/bin/python scripts/necessity/third_task_pilot.py {t} mps ../codi_audit_artifacts/t58/{t}.json 4000 {e}'
HP = dict(lr=5e-5, batch_size=16, weight_decay=.01, warmup_steps=200, schedule='linear decay', n_per_k=4000, ks='1-8',
          test_per_k=300, loss='cross-entropy on the answer token(s) at the last prompt position(s)', script='scripts/necessity/third_task_pilot.py')
DESC = dict(
    sn='S5 word problem over 6 fixed generators (Liu et al. 2023; Merrill et al. 2024), 120 single-token labels, no CoT',
    brew='in-context permutations (brew surface, own generator: 10 colours, 3 ingredients as rule table), no CoT',
    brewcot='brew items with chain of thought (colour after every stir), greedy decode restricted to colours',
    sn24='S5 word problem over 24 random generators, test sequences held out for k >= 3, no CoT',
    sn24cot='sn24 items with chain of thought (running-product label after each generator)',
    sn24pause='sn24 items with 6 pause tokens after the prompt, no CoT (matched no-loop control)',
    sn24s2='sn24 with a second data seed, no CoT')
MECH = dict(brewcot='explicit_cot', sn24cot='explicit_cot', sn24pause='filler_tokens')
VERDICT = dict(
    sn='FAILS the gate: reach 3.87 > 3.5, inflated by memorisation (216 sequences at k = 3; train .62 vs test .43 at k = 4). P1 held.',
    brew='INCONCLUSIVE, stopped after epoch 2 by deviation (2026-11-25 00:30:55): at chance at every k, train included; dropped after brewcot.',
    brewcot='P4 MISSED: k = 1 plateaus at ~.40 from epoch 0; brew dropped without the 10-epoch rerun.',
    sn24='PASSES the gate (P1, P2 held; P3 missed): cliff between k = 2 (all test pairs seen in training) and k = 3 (held out); reach is an upper bound on composition.',
    sn24cot='P5 held: the CoT teacher solves every k (>= .997): sn24 is learnable step by step.',
    sn24pause='P6 held: 6 pause tokens add no single-pass reach (2.50 vs 2.49).',
    sn24s2='P7 held: second seed reach 2.50, same verdict.')


def rows(by_epoch):
    return [dict(epoch=int(e), split=sp, k=int(k), accuracy=v) for e, d in by_epoch.items() for sp in ('test', 'test_path', 'train')
            for k, v in d.get(sp, {}).items()]


def brew_from_log():
    by = {}
    for line in open(A / 'brew.log'):
        m = re.match(r"brew ep (\d+) \d+s test (\{.*?\}) train (\{.*?\})", line)
        if m: by[m[1]] = dict(test=eval(m[2]), train=eval(m[3]))
    return dict(task='brew', chance=.1, epochs=5, by_epoch=by)


for t in ('sn', 'brew', 'brewcot', 'sn24', 'sn24cot', 'sn24pause', 'sn24s2'):
    d = brew_from_log() if t == 'brew' else json.load(open(A / f'{t}.json'))
    last = d['by_epoch'][max(d['by_epoch'], key=int)]
    s = None if t == 'brew' else score(A / f'{t}.json')
    ep = 3 if t == 'brewcot' else 5
    rec = RunRecord(
        run_id=new_run_id(MECH.get(t, 'baseline_no_cot'), f't58-third-task-pilot-{t}'), mechanism=MECH.get(t, 'baseline_no_cot'), stage='pilot',
        model=ModelInfo(backbone='gpt2', checkpoint='not saved (pilot)', n_params=124_439_808),
        dataset=DatasetInfo(name=f't58-{t}', split='test', n_examples=300 * 8, seed=0),
        metrics=dict(final_answer_accuracy=round(sum(last['test'].values()) / len(last['test']), 4), compute_steps=6 if t == 'sn24pause' else 0,
                     extra=dict(test_by_k=last['test'], train_by_k=last['train'], test_path_by_k=last.get('test_path'),
                                chance=d['chance'], reach=None if s is None else s['reach'], gate_passed=None if s is None else s['passed'],
                                epochs_run=len(d['by_epoch']))),
        hyperparams=dict(HP, epochs=ep, task=t), seed=0, hardware=HW,
        notes=f't58 pilot {t}: single-pass reach of GPT-2 fine-tuned on a candidate third task')
    out = Path(rec.save(predictions=rows(d['by_epoch']))); out = out.parent if out.is_file() else out
    test = ' / '.join(f'{v:.3f}' for v in last['test'].values())
    (out / 'notes.md').write_text(f"""## 2026-11-24/25 — t58 third-task pilot, {t}: {VERDICT[t]} ({MECH.get(t, 'baseline_no_cot')}, t58)

**Goal:** advisor asked for a task orthogonal in content with the same serial structure. t58 measures single-pass reach of pretrained GPT-2 (proxy for CODI-0) on candidate tasks before any CODI training. Pre-registered in scripts/codi_null_audit/prereg/t_predictions.md (t58, 2026-11-24 22:17:14 +07; follow-ups 2026-11-25 00:14:34 and 00:36:08; deviation 00:30:55).
**Task:** {DESC[t]}.
**Command:** `{CMD.format(t=t, e=ep)}` (from ai-capstone/); scorer `scripts/necessity/third_task_score.py`.
**Gate:** pass if accuracy at k = 1 >= .90 and 1.5 <= reach <= 3.5; reach = interpolated k where final-epoch test accuracy crosses the midpoint between chance and accuracy at k = 1.
**Results (final epoch, test, k = 1-8):** {test}; chance {d['chance']:.3f}{'' if s is None else f"; reach {s['reach']}"}.
**Verdict:** {VERDICT[t]}
**Caveats:** single pilot run per setting (sn24 has a second seed); no per-item outputs or checkpoints saved; `predictions.jsonl` holds per-(epoch, split, k) accuracies. Reference: RuleChain start-last CODI-0 reach ~2.6, start-first ~5.
""")
    print('saved', out)
