"""Save the t53-t57 run records (pre-registered and scored in scripts/codi_null_audit/prereg/t_predictions.md).
Run the scorers first: prefix_sub_score.py (t53/score.json), cot_route_score.py (t54/score.json), epoch_hops_score.py (t55/score.json),
validity_score.py (t56/score_t56_t57.json).

  ../codi/.venv/bin/python scripts/necessity/record_t53_t57.py
"""
import json
from pathlib import Path

from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id

A = Path(__file__).resolve().parents[3] / 'codi_audit_artifacts'
load = lambda p: json.load(open(p))
RC_CKPT = 'BASE/codi_runs_l1to7w_startlast{,_gated,_sim}/codi/run/checkpoint-* (selected checkpoints, adapters.RuleChainCODI)'
GSM_CKPT = 'hf:zen-E/CODI-gpt2; BASE/grid_checkpoints/simcot_gpt2_codi (released SIM-CODI)'
HW = 'local Apple Silicon (MPS), 25.8 GB'


def save(rec, preds, notes):
    d = Path(rec.save(predictions=preds)); d = d.parent if d.is_file() else d
    (d / 'notes.md').write_text(notes)
    print('saved', d, len(preds), 'predictions')


# ---------------------------------------------------------------- t53 prefix substitution
s53 = load(A / 't53/score.json')
P = []
for m in ('gated', 'startlast', 'gsm8k_codi'):
    for p in load(A / f't53/{m}.json')['pairs']:
        P.append(dict(model=m, **{k: v for k, v in p.items() if k not in ('R', 'D')},
                      question_R=p.get('R', {}).get('question'), question_D=p.get('D', {}).get('question')))
save(RunRecord(
    run_id=new_run_id('codi', 'prefix-substitution-entry-point'), mechanism='codi', stage='full_run',
    model=ModelInfo(backbone='gpt2 (rule-gated CODI, start-last CODI; released GSM8K CODI)', checkpoint=f'{RC_CKPT}; {GSM_CKPT}', n_params=None),
    dataset=DatasetInfo(name='rulechain-l1to7w-startlast + gsm8k-aug', split='validation', n_examples=len(P), seed=53),
    metrics=dict(compute_steps=6, extra=dict(score_file='codi_audit_artifacts/t53/score.json',
                 exploratory_gated_slope_e_on_pos=.96, exploratory_gated_e_eq_rule_plus_1='375/375', exploratory_startlast_slope=-.09,
                 gsm8k_codi_entry_slope_anyuse=-2.24, gsm8k_codi_hops_slope_anyuse=.86)),
    hyperparams=dict(cap_per_cell=25, pairs='t31 pairs, both answered right', script='scripts/necessity/prefix_sub.py'),
    seed=53, hardware=HW, notes='t53: install B\'s first c latents and count saved refreshes; where does the edit enter the loop?'),
    P, """## 2026-11-23 — Prefix substitution (t53): in rule-gated CODI the edit enters the loop exactly one pass after the edited rule; in start-last CODI it enters at a fixed point (~latent 4) wherever the rule sits; in released GSM8K CODI later edits enter later (codi, t53)

**Goal:** t53 (pre-registered 2026-11-23 19:01 +07, P2 corrected 19:02 before any output): are A's and B's latents before the edited step really the same, and where does the edit enter the loop? Tests whether the flat GSM8K within-item slope came from early latents already carrying the edit.
**Method:** for each pair answered right on both sides, d_j = ||z_j(A) - z_j(B)|| normalised by the distance to an unrelated item; for c = 0..K a run with B's first c natural latents and A's for the rest, refreshed as in hops.py; S_c = h_0 - h_c refreshes saved; entry e = smallest c with S_c > 0.
**Data:** t31 pairs: rule-gated and start-last CODI (RuleChain valid L2-L6, CAP 25 per cell), released GSM8K CODI (414 pairs). The GSM8K SIM-CODI run was stopped at pair 241 by team decision (no output).
**Command:** `../codi/.venv/bin/python scripts/necessity/prefix_sub.py <model> mps ../codi_audit_artifacts/t53/<model>.json`; scoring `scripts/necessity/prefix_sub_score.py ../codi_audit_artifacts/t53/score.json`.
**Results:** checks pass on every pair (h_0 = t31, own answers, c = K run = B). Rule-gated: latents before the carrier identical to A's (.00), carrier .75 (P2 held); on pairs with h_0 >= 1 the entry is exactly rule + 1 on 375 / 375 rule edits, within-level slope .96. Start-last CODI: late-rule edits mostly h_0 = 0 (latents ~ A's, .01-.08); pairs with h_0 >= 1 enter at latent 3.9-4.3 at every position (slope -.09). GSM8K CODI: entry slope +.56 per gold step index; on the any-use distance -2.2 passes per step and hops +.86 per step (11-13 items); latents before a late edit at .05 of the reference; edit-specific information only in z_0, z_2, z_4.
**Predictions:** P0 held; P1, P3 missed as scored (coding flaw: h_0 = 0 pairs coded e = K + 1); P2 held; P4-P6 missed in the direction of chaining.
**Interpretation:** the method recovers a moving entry point when the chain is forced (rule gate); the standard RuleChain model gathers the edit at one point in the loop; the released GSM8K model's entry moves with the edit, consistent with short genuine chains once distance is measured from every step that uses the edited number (t54).
**Caveats:** exploratory restriction to h_0 >= 1 chosen after seeing the coding flaw; GSM8K any-use cells are small; one released checkpoint.
""")

# ---------------------------------------------------------------- t54 the model's own CoT route
s54 = load(A / 't54/score.json'); P = []
for m in ('codi', 'sim'):
    for r in load(A / f't54/{m}.json')['rows']: P.append(dict(model=m, **r))
save(RunRecord(
    run_id=new_run_id('codi', 'gsm8k-model-cot-route-depth'), mechanism='codi', stage='full_run',
    model=ModelInfo(backbone='gpt2 (released CODI and SIM-CODI, teacher mode)', checkpoint=GSM_CKPT, n_params=None),
    dataset=DatasetInfo(name='gsm8k-aug', split='test', n_examples=len(P), seed=0),
    metrics=dict(compute_steps=0, extra=dict(score=s54, teacher_accuracy=dict(codi=.462, sim=.470),
                 anyuse_slope_correct=dict(codi=.84, sim=1.00), anyuse_slope_all_codi=.36, labelled_slope_all_codi=.16)),
    hyperparams=dict(decoding='greedy, CoT written by the shared teacher weights', script='scripts/necessity/cot_route.py'),
    seed=0, hardware=HW, notes='t54: depth and edit distance from the CoT the released checkpoints write themselves; any-use gold distance'),
    P, """## 2026-11-23 — The model's own CoT route (t54): within-item hop slope +.88 on the model's route; the flat +.16 slope reported since t42 is mostly a twin-pair labelling artefact (codi, t54)

**Goal:** t54 (pre-registered 2026-11-23 19:53 +07): measure serial distance on the route the model itself writes (the released CODI checkpoints share weights with their CoT teacher), to test whether the flat within-item slope comes from gold-route mismatch.
**Command:** `../codi/.venv/bin/python scripts/necessity/cot_route.py <codi|sim> cpu ../codi_audit_artifacts/t54/<m>.json`; scoring `scripts/necessity/cot_route_score.py ../codi_audit_artifacts/t54/score.json`; post hoc `scripts/necessity/gsm8k_anyuse_distance.py`.
**Results:** parse 1.00; teacher accuracy .462 (CODI) / .470 (SIM-CODI); model-route depth equals gold depth on .60 / .61 of items. Within-item slope on model-route distance +.88 [+.63, +1.18] (CODI, 15 items) and +.87 (SIM-CODI, 11 items) against +.16 / +.03 on the labelled distance for the same pairs. Start-of-path crossings by model-route depth 2 / 3 / 4: CODI 1.08 / 1.95 / 2.43.
**Post hoc (labelled as such in the prereg):** the t22 twin pairs carry the distance of one gold step, but the edited number often feeds several gold steps (about 20% of correct pairs). Measured from any gold step using the edited number, the within-item slope on correct pairs is +.84 [+.59, +1.26] (CODI, 13 items) and +1.00 (SIM-CODI, 8 items); on all qualified CODI pairs +.36 against +.16. Coconut +.30 and SIM-Coconut +.40 on all pairs (computed 2026-11-24).
**Predictions:** P0, P1, P3 held; P2 missed (slope +.88, not flat); P4 mixed (tiny cells).
**Interpretation:** on GSM8K problems it solves, released CODI chains about one crossing per serial step through depth 3; the earlier "count follows item depth, not edit position" was a labelling error. The paper's GSM8K text was rewritten on 2026-11-24 accordingly.
**Caveats:** few items with edits at genuinely different distances; the written route is a proxy for the latent route.
""")

# ---------------------------------------------------------------- t55 epoch sweep
s55 = load(A / 't55/score.json'); P = []
for tag in ('startlast_8910', 'startlast_11880', 'startlast_14850', 'startlast_17820', 'r1'):
    for p in load(A / f't55/{tag}.json')['pairs']:
        P.append(dict(checkpoint=tag, **{k: p.get(k) for k in ('level', 'edit', 's', 'id', 'qualified', 'R_right', 'D_right', 'aR', 'aD', 'hops', 'A')}))
save(RunRecord(
    run_id=new_run_id('codi', 'rulechain-epoch-sweep-hop-count'), mechanism='codi', stage='full_run',
    model=ModelInfo(backbone='gpt2 (start-last CODI at epochs 6/8/10/12; round-1 40-epoch start-first CODI)',
                    checkpoint='BASE/codi_runs_l1to7w_startlast/codi/run/checkpoint-{8910,11880,14850,17820}; rulechain-checkpoints gpt2_depth2_codi_bs64/checkpoint-31280',
                    n_params=None),
    dataset=DatasetInfo(name='rulechain-l1to7w (start-last; start-first for round 1)', split='validation', n_examples=len(P), seed=0),
    metrics=dict(compute_steps=6, extra=dict(score=s55)),
    hyperparams=dict(qual='own', cap=80, levels='2-6', env='CODI_CKPT', script='scripts/necessity/hops.py'),
    seed=0, hardware=HW, notes='t55: does chaining grow with training? hop count over start-last CODI epochs and on the 40-epoch round-1 CODI'),
    P, """## 2026-11-24 — Epoch sweep (t55): start-last CODI's accuracy doubles from epoch 8 to 12 while crossings per pair stay ~1.5; the 40-epoch round-1 CODI uses ~1 crossing at every distance (codi, t55)

**Goal:** t55 (pre-registered 2026-11-23 20:52 +07): is the standard RuleChain null an artefact of the 12-epoch budget (the recipe is 40)?
**Command:** `CODI_CKPT=<checkpoint> QUAL=own ../codi/.venv/bin/python scripts/necessity/hops.py startlast mps ../codi_audit_artifacts/t55/<tag>.json` (round 1: model key stab); scoring `scripts/necessity/epoch_hops_score.py ../codi_audit_artifacts/t55/score.json`.
**Results:** epoch-12 run reproduces t31 on 2000 / 2000 pairs. Correct pairs, mean hops at s = 3-4: 1.14 (n 7) / 1.60 (10) / 1.52 (31) / 1.47 (51) at epochs 6 / 8 / 10 / 12, while valid accuracy at L3 rises .14 / .24 / .41 / .52. Round-1 40-epoch start-first CODI: 1.05 / 1.12 / 1.14 / 1.31 / 1.25 at s = 2-6, flat.
**Predictions:** P0 held; P1 missed (cell size at epoch 6; no rise from epoch 8 on); P2 missed (slope not comparable across epochs); P3 missed in level (~1 crossing, not ~0), direction held.
**Interpretation:** within the epochs we have, more training buys accuracy without longer chains; the 40-epoch start-first run behaves like a copy (near-identical latents). A 40-epoch start-last run remains the clean test.
**Caveats:** start-last only to epoch 12; round 1 is start-first at lr 3e-3; the queue paused overnight (Mac asleep), results unaffected.
""")

# ---------------------------------------------------------------- t56 / t57 validity
S = load(A / 't56/score_t56_t57.json'); V = S['verdicts']
keep = ('level', 'pos', 's', 'edit', 'delta', 'idR', 'idD', 'aR', 'aD', 'goldR', 'goldD', 'qualified', 'hops', 'mk_matches_donor')
P = []
for m in ('gated', 'sim', 'startlast', 'startlast0'):
    for mode in ('sets', 'wide'):
        f = A / f't56/{m}_{mode}.json'
        if not f.exists(): continue
        for tag, v in load(f).items():
            if not isinstance(v, dict) or 'acc' not in v: continue
            P.append(dict(model=m, set=tag, kind='accuracy', acc=v['acc']))
            for p in v.get('pairs', []): P.append(dict(model=m, set=tag, kind='pair', **{k: p.get(k) for k in keep}))
save(RunRecord(
    run_id=new_run_id('codi', 'rulechain-distribution-shift-validity'), mechanism='codi', stage='full_run',
    model=ModelInfo(backbone='gpt2 (rule-gated CODI, SIM-CODI, start-last CODI, start-last CODI-0)', checkpoint=RC_CKPT, n_params=None),
    dataset=DatasetInfo(name='rulechain-l1to7w-startlast shifted sets (orig, rw, hc, hc1, sv, l8, wide)', split='test', n_examples=len(P), seed=56),
    metrics=dict(compute_steps=6, extra=dict(t56=V['t56'], wide=V['wide'])),
    hyperparams=dict(sets='scripts/necessity/validity_sets.py', script='scripts/necessity/validity.py (modes sets, wide)', cap=40, qual='own'),
    seed=56, hardware=HW, notes='t56: inference-only distribution shift: reworded rules, held-out constants, shifted start values, values to 115, L8 with more passes'),
    P, """## 2026-11-24 — Distribution shift (t56): every model's accuracy collapses under reworded rules, held-out constants and unseen numbers, but the crossing pattern survives every shift; rule-gated CODI solves L8 with 8 passes, SIM-CODI does not (codi, t56)

**Goal:** t56 (pre-registered 2026-11-24 13:3x +07; scope change and wide-set addendum before the affected outputs): advisor feedback asked for validity of RuleChain and the hop count under inference-time shift.
**Sets (validity_sets.py, 300 items per level L2-L7, filters applied, none in train):** orig (canonical test), rw (reworded rules), hc (all constants outside training ranges), hc1 (one held-out rule), sv (start 100-115), l8 (L = 8 at K = 6 / 7 / 8), wide (values 0-115, readout extended to 0-115).
**Command:** `../codi/.venv/bin/python scripts/necessity/validity.py <model> mps <sets|wide> ../codi_audit_artifacts/t56/<model>_<mode>.json`; scoring `scripts/necessity/validity_score.py ../codi_audit_artifacts/t56/score_t56_t57.json`.
**Results:** rule-gated accuracy rw .26-.04, hc .44-.12, sv .00-.06, wide ~.04; SIM-CODI rw .23-.02, hc .33-.05; hc1 flat in L for the chaining models (.68-.76 / .60-.66), falling for start-last (.66 -> .06). Start-edit hops of the chaining models >= L - 1 in every L2-L6 cell of every shifted set; start-last <= 2.07 everywhere. Rule-gated rule edits exact at L2-L6 on 800/800, 551/562, 695/696, 767/768, 578/578; at L7 every edit gives L - i - 1 (6 latents, 7 rules). L8: rule-gated .00 / .28 / .76 at K = 6 / 7 / 8; SIM-CODI .00 / .06 / .02 although its count reaches 7.89.
**Predictions:** P0, P1, P3, P4 held; P2 missed as scored (L7 formula); P5 held for rule-gated and start-last, missed for SIM-CODI; W0-W3 held.
**Interpretation:** the serial structure measured by the hop count is robust to input shift; per-step accuracy is template- and range-bound. Errors under one unfamiliar rule stay at that rule.
**Caveats:** single seeds; shifted sets test generalisation the models were never trained for.
""")

keep57 = keep + ('t31_hops', 'aX')
P = []
for m in ('gated', 'sim', 'startlast'):
    for part, rows in load(A / f't56/{m}_robust.json').items():
        if not isinstance(rows, list): continue
        for p in rows:
            P.append(dict(model=m, part=part, **{k: p.get(k) for k in keep57}, margin=[round(x, 2) for x in p.get('margin', [])]))
save(RunRecord(
    run_id=new_run_id('codi', 'hop-count-robustness-swap-margin-size-donor'), mechanism='codi', stage='full_run',
    model=ModelInfo(backbone='gpt2 (rule-gated CODI, SIM-CODI, start-last CODI)', checkpoint=RC_CKPT, n_params=None),
    dataset=DatasetInfo(name='rulechain-l1to7w-startlast valid (t31 pairs + new size pairs)', split='validation', n_examples=len(P), seed=57),
    metrics=dict(compute_steps=6, extra=dict(t57=V['t57'])),
    hyperparams=dict(script='scripts/necessity/validity.py (mode robust)', size_pairs_cap=25),
    seed=57, hardware=HW, notes='t57: robustness of the hop count: swapped pairs, margin-based counts, edit size, unrelated donor'),
    P, """## 2026-11-24 — Hop-count robustness (t57): the count survives swapping A and B, a margin threshold, edit size and an unrelated donor for the chaining models; for start-last CODI it is noisier but every disagreement is downward (codi, t57)

**Goal:** t57 (pre-registered with t56): is the count itself robust to how it is measured?
**Command:** `../codi/.venv/bin/python scripts/necessity/validity.py <model> mps robust ../codi_audit_artifacts/t56/<model>_robust.json`; scoring as t56. The rule-gated run was paused 14:38-19:03 (SIGSTOP, battery), results unaffected.
**Results:** repro equals t31 on 1.00 of pairs. Swap within 1: 1.00 / 1.00 / .857; means identical (2.16 / 2.77 / 1.34). Margin hop equals argmax hop on .984 / .975 / .657, and where it differs it is always lower. Onset earlier, never later, for s > 1 (111 / 576 / 584 pairs). Start-last half hop 1.04-1.19 at every s. Large minus small edits +.01 / +.10 / +.28. Unrelated donor within .10 of the start-edit count (chaining models); start-last <= 1.96.
**Predictions:** P0 held; P1 missed for start-last and on the strict equality bar for rule-gated (.938); P2 held for the chaining models' margin agreement, missed for start-last and for SIM-CODI's onset; P3 held for the chaining models, missed for start-last; P4 held.
**Interpretation:** the argmax count is an upper bound, as the paper states; by the margin measure start-last CODI uses about one crossing. SIM-CODI's onset one refresh before the flip fits its count lying one above the rule gate's.
**Caveats:** single seeds; post hoc reading of the disagreement direction.
""")
