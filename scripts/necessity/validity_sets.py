"""t56: inference-only RuleChain evaluation sets for validity tests (pre-registered in prereg/t_predictions.md, t56).

  python3 scripts/necessity/validity_sets.py            (from ai-capstone/; writes ../codi_audit_artifacts/t56/sets/<set>_<order>.json)

Sets (levels L2-L7, N items per level, every item passes the generator's identity / attractor / order filters):
  orig  the first N test items per level of the canonical split (reference)
  hc    held-out constants: every constant of every rule is drawn from outside its training range; all values stay in 0..99
  hc1   one held-out rule: one rule (uniform position) has held-out constants, the others training ranges; values in 0..99
  sv    shifted start: start value in 100..115 (never seen in training); rule 1 must bring it into 0..99, later values stay in 0..99
  rw    reworded: the orig items with a new wording of each rule type (same content, same 'Rule i:' / 'Start with x.' frame)
  l8    L = 8 (one rule longer than any CODI training item), training ranges
Each record keeps the structured item (x0, rules) so pairs are built without parsing text.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from latentreasoning.tasks import synthetic_v2 as S  # noqa: E402

BASE = REPO.parent
OUT = BASE / 'codi_audit_artifacts/t56/sets'
N = 300
LEVELS = [2, 3, 4, 5, 6, 7]

# training ranges (synthetic_v2.sample_rule) and held-out ranges (disjoint from training for every constant)
TRAIN = dict(P=dict(a=(1, 25)), T=dict(t=(20, 80), a=(1, 20), b=(1, 20)), D=dict(t=(10, 50), b=(1, 50)))
HELD = dict(P=dict(a=(26, 49)), T=dict(t=(81, 95), a=(21, 35), b=(21, 35)), D=dict(t=(51, 70), b=(51, 70)))
SV_RANGE = (100, 115)


def rule_ok(kind, t, a, b, rng_kind):
    """Constants inside the given range family (TRAIN keeps the generator's coupled bounds)."""
    R = rng_kind[kind]
    ok = all(R[k][0] <= v <= R[k][1] for k, v in dict(t=t, a=a, b=b).items() if k in R)
    if rng_kind is TRAIN and kind == 'T': ok = ok and a <= min(20, 100 - t) and b <= min(20, t)
    if rng_kind is TRAIN and kind == 'D': ok = ok and b <= t
    if rng_kind is HELD and kind == 'D': ok = ok and b <= t
    return ok


def sample_rule(rng, fam):
    if fam is TRAIN: return S.sample_rule(rng)
    kind = rng.choice('PTD'); R = fam[kind]; d = {k: rng.randint(*R[k]) for k in R}
    if kind == 'D': d['b'] = rng.randint(R['b'][0], d['t'])
    return S.Rule(kind, **d)


def walk(x0, rules, x0_range=(0, 99), vmax=99):
    """Values [x0, v1, .., vL] or None if x0 is outside x0_range or a later value leaves 0..vmax (vmax 115: the wide set only)."""
    if not x0_range[0] <= x0 <= x0_range[1]: return None
    s, out = x0, [x0]
    for r in rules:
        st = (r.step(s) if s <= 99 else _step_big(r, s)) if vmax == 99 else _step_big(r, s, vmax)
        if st is None: return None
        s = st[3]; out.append(s)
    return out


def _step_big(r, s, vmax=99):
    """Rule.step for a start value above 99 (only rule 1 under sv; the result must land in 0..99), or any value under vmax."""
    if r.kind == 'P': v = s // 2 if s % 2 == 0 else s + r.a
    elif r.kind == 'T': v = s - r.b if s >= r.t else s + r.a
    else: v = 2 * s if s < r.t else s - r.b
    return (s, '?', 0, v) if 0 <= v <= vmax else None


def accepted(x0, rules, rng, x0_range, vmax=99):
    v = walk(x0, rules, x0_range, vmax)
    if v is None: return False
    vals = v[1:]; n = len(vals)
    if vals[-1] == x0: return False
    if n >= S.ATTRACTOR_MIN_STEPS and max(vals.count(x) for x in vals) / n > S.ATTRACTOR_FRAC: return False
    if n >= 2:
        for _ in range(S.ORDER_BINDING_PERMS):
            perm = list(rules); rng.shuffle(perm)
            if tuple(perm) != tuple(rules):
                w = walk(x0, perm, x0_range, vmax)
                if w is not None and w[-1] == vals[-1]: return False
    return True


# ---------------------------------------------------------------- wording
def text(r, reworded=False):
    if not reworded: return r.text()
    if r.kind == 'P': return f"when the number is even, cut it in half; when it is odd, increase it by {r.a}"
    if r.kind == 'T': return f"when the number is {r.t} or more, decrease it by {r.b}; when it is less, increase it by {r.a}"
    return f"when the number is less than {r.t}, multiply it by two; when it is not, decrease it by {r.b}"


def render(x0, rules, order, reworded=False):
    body = ' '.join(f'Rule {i + 1}: {text(r, reworded)}.' for i, r in enumerate(rules))
    if order == 'first': return f'Start with {x0}. {body} What is the final number?'
    return f'{body} Start with {x0}. What is the final number?'


def rule_json(r): return dict(kind=r.kind, t=r.t, a=r.a, b=r.b)
def rule_of(d): return S.Rule(d['kind'], t=d['t'], a=d['a'], b=d['b'])


def record(sid, i, L, x0, rules, order, reworded=False, vmax=99):
    v = walk(x0, rules, (0, 115), vmax)
    return dict(id=f'{sid}-L{L}-{i:04d}', set=sid, level=L, x0=x0, rules=[rule_json(r) for r in rules], values=v,
                answer=str(v[-1]), reworded=reworded, question=render(x0, rules, order, reworded))


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    canon = {o: json.load(open(BASE / d / 'depth2/test.json')) for o, d in
             (('first', 'synthetic_data_v2_l1to7w'), ('last', 'synthetic_data_v2_l1to7w_startlast'))}
    train_keys = set()
    for o in ('first',):
        for r in json.load(open(BASE / 'synthetic_data_v2_l1to7w/depth2/train.json')):
            c = S.parse_chain_question(r['question']); train_keys.add(c.key())
    stats = {}
    for order in ('first', 'last'):
        sets = {k: [] for k in ('orig', 'rw', 'hc', 'hc1', 'sv', 'l8')}
        for L in LEVELS:
            # orig / rw: the same canonical test items (start-first file; the start-last file holds the same content reordered)
            items = [r for r in canon['first'] if int(r['level']) == L][:N]
            for i, r in enumerate(items):
                c = S.parse_chain_question(r['question'])
                sets['orig'].append(record('orig', i, L, c.x0, c.rules, order))
                sets['rw'].append(record('rw', i, L, c.x0, c.rules, order, reworded=True))
            for sid, fam, xr in (('hc', HELD, (1, 99)), ('sv', TRAIN, SV_RANGE)):
                rng = random.Random(f't56-{sid}-L{L}'); got = []
                while len(got) < N:
                    rules = tuple(sample_rule(rng, fam) for _ in range(L)); x0 = rng.randint(*xr)
                    if accepted(x0, rules, rng, xr) and (x0, rules) not in got: got.append((x0, rules))
                sets[sid] += [record(sid, i, L, x0, rules, order) for i, (x0, rules) in enumerate(got)]
            rng = random.Random(f't56-hc1-L{L}'); got = []
            while len(got) < N:
                j = rng.randrange(L); rules = tuple(sample_rule(rng, HELD if k == j else TRAIN) for k in range(L)); x0 = rng.randint(1, 99)
                if accepted(x0, rules, rng, (1, 99)) and (x0, rules) not in got: got.append((x0, rules))
            sets['hc1'] += [dict(record('hc1', i, L, x0, rules, order), held_rule=next(k + 1 for k, r in enumerate(rules)
                            if not rule_ok(r.kind, r.t, r.a, r.b, TRAIN))) for i, (x0, rules) in enumerate(got)]
        rng = random.Random('t56-l8'); got = []
        while len(got) < N:
            rules = tuple(S.sample_rule(rng) for _ in range(8)); x0 = rng.randint(1, 99)
            if accepted(x0, rules, rng, (1, 99)) and (x0, rules) not in got: got.append((x0, rules))
        sets['l8'] = [record('l8', i, 8, x0, rules, order) for i, (x0, rules) in enumerate(got)]
        for sid, recs in sets.items():
            json.dump(recs, open(OUT / f'{sid}_{order}.json', 'w'))
            stats[f'{sid}_{order}'] = dict(n=len(recs), in_train=sum(('depth2', r['x0'], tuple(rule_of(d) for d in r['rules'])) in train_keys
                                                                      for r in recs if sid != 'orig'))
    # sanity: orig start-last questions equal the canonical start-last test file
    sl = {r['id']: r for r in canon['last']}
    o = json.load(open(OUT / 'orig_last.json')); c = [r for L in LEVELS for r in canon['last'] if int(r['level']) == L]
    lv = {L: [r for r in canon['last'] if int(r['level']) == L][:N] for L in LEVELS}
    same = sum(a['question'] == b['question'] for a, b in zip(o, [r for L in LEVELS for r in lv[L]]))
    stats['orig_last_matches_canonical'] = f'{same}/{len(o)}'
    # held-out check: no hc constant inside its training range; sv start values all >= 100
    hc = json.load(open(OUT / 'hc_first.json'))
    stats['hc_constants_in_train_range'] = sum(rule_ok(d['kind'], d['t'], d['a'], d['b'], TRAIN) for r in hc for d in r['rules'])
    stats['sv_min_start'] = min(r['x0'] for r in json.load(open(OUT / 'sv_first.json')))
    json.dump(stats, open(OUT / 'stats.json', 'w'), indent=1)
    print(json.dumps(stats, indent=1))


WIDE_MAX = 115


def build_wide():
    """wide (t56 addendum): every value may lie in 0..115 (start 1..115); training-range rules; at least one value >= 100 per item.
    Written to wide_<order>.json only; the other sets are not touched."""
    stats = {}
    for order in ('first', 'last'):
        recs = []
        for L in LEVELS:
            rng = random.Random(f't56-wide-L{L}'); got = []
            while len(got) < N:
                rules = tuple(S.sample_rule(rng) for _ in range(L)); x0 = rng.randint(1, WIDE_MAX)
                if not accepted(x0, rules, rng, (1, WIDE_MAX), WIDE_MAX) or (x0, rules) in got: continue
                if max(walk(x0, rules, (1, WIDE_MAX), WIDE_MAX)) >= 100: got.append((x0, rules))
            recs += [record('wide', i, L, x0, rules, order, vmax=WIDE_MAX) for i, (x0, rules) in enumerate(got)]
        json.dump(recs, open(OUT / f'wide_{order}.json', 'w'))
        stats[order] = dict(n=len(recs), answer_ge_100=sum(int(r['answer']) >= 100 for r in recs),
                            by_L={L: sum(int(r['answer']) >= 100 for r in recs if r['level'] == L) for L in LEVELS})
    json.dump(stats, open(OUT / 'stats_wide.json', 'w'), indent=1); print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    build_wide() if sys.argv[1:] == ['wide'] else build()
