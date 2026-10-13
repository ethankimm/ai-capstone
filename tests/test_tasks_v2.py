"""Synthetic tasks v2 (rule chain, paircount, distinct): validity, parsing, splits, determinism."""
import random
from collections import Counter

import pytest

from latentreasoning.tasks import synthetic_v2 as V

SMALL = dict(seed=7, train_total=600, valid_total=120, test_per_level=15, monitor_per_level=4)
TASKS = ("depth2", "paircount", "distinct", "modal")


@pytest.fixture(scope="module")
def built():
    return {t: V.build_task(t, **SMALL) for t in TASKS}


def _check(task, rec):
    p = V.parse_question(task, rec["question"])
    r = V.render(task, p)
    assert r["question"] == rec["question"] and r["steps"] == rec["steps"] and r["answer"] == rec["answer"]
    assert isinstance(rec["answer"], str)
    prev_v = None
    for st in rec["steps"]:
        left, op, right, v = V.parse_step(st)
        assert V.apply_op(op, left, right) == v
        assert 0 <= v <= 99
        if task == "depth2" and prev_v is not None:
            assert left == prev_v
        prev_v = v
    assert prev_v == int(rec["answer"])  # last step value == answer (CODI drops the last step)
    return p


def test_examples_valid(built):
    for task, (splits, _) in built.items():
        for recs in splits.values():
            for rec in recs:
                assert rec["task"] == task
                p = _check(task, rec)
                if task == "depth2":
                    assert len(rec["steps"]) == rec["level"] == p.n_steps
                    assert 0 <= rec["nudge_depth"] <= rec["level"]
                else:  # modal has one extra final answer step
                    assert len(rec["steps"]) == rec["level"] + (task == "modal")


def test_rule_constants_and_branches():
    rng = random.Random(0)
    for _ in range(2000):
        r = V.sample_rule(rng)
        if r.kind == "P":
            assert 1 <= r.a <= 25
        elif r.kind == "T":
            assert 20 <= r.t <= 80 and 1 <= r.a <= 20 and 1 <= r.b <= min(20, r.t) and r.t - 1 + r.a <= 99
        else:
            assert 10 <= r.t <= 50 and 1 <= r.b <= r.t
    assert V.Rule("P", a=5).step(8) == (8, "/", 2, 4)
    assert V.Rule("P", a=5).step(7) == (7, "+", 5, 12)
    assert V.Rule("T", t=40, a=7, b=12).step(40) == (40, "-", 12, 28)
    assert V.Rule("T", t=40, a=7, b=12).step(39) == (39, "+", 7, 46)
    assert V.Rule("D", t=30, b=9).step(29) == (29, "*", 2, 58)
    assert V.Rule("D", t=30, b=9).step(30) == (30, "-", 9, 21)
    assert V.Rule("P", a=25).step(99) is None  # leaves 0..99


def test_example_format():
    c = V.RuleChain(57, (V.Rule("P", a=5), V.Rule("T", t=40, a=7, b=12), V.Rule("D", t=30, b=9)))
    r = V.render_chain(c)
    assert r["question"] == (
        "Start with 57. Rule 1: if it is even, halve it; otherwise add 5. "
        "Rule 2: if it is at least 40, subtract 12; otherwise add 7. "
        "Rule 3: if it is below 30, double it; otherwise subtract 9. What is the final number?")
    assert r["steps"] == ["<<57+5=62>>", "<<62-12=50>>", "<<50-9=41>>"] and r["answer"] == "41"


def test_filters_applied():
    rng = random.Random(1)
    # final == start is rejected: 20 -> halve -> 10 -> (D t=30 b=5: below 30, double) -> 20
    back = V.RuleChain(20, (V.Rule("P", a=1), V.Rule("D", t=30, b=5)))
    assert V._final(20, back.rules) == 20 and not V.chain_accepted(back, rng)
    # order-binding: two rules that commute on this input are rejected
    same = V.RuleChain(60, (V.Rule("T", t=50, a=3, b=4), V.Rule("T", t=20, a=3, b=5)))
    assert V._final(60, same.rules) == V._final(60, same.rules[::-1]) == 51
    assert not V.chain_accepted(same, rng)
    for n in (2, 5, 12):
        for _ in range(30):
            c = V.sample_chain(rng, n)
            walk = V.run_chain(c.x0, c.rules)
            assert walk is not None and walk[-1][3] != c.x0
            if n >= V.ATTRACTOR_MIN_STEPS:
                vals = [w[3] for w in walk]
                assert max(Counter(vals).values()) / n <= V.ATTRACTOR_FRAC


def test_nudge_depth_bounds():
    rng = random.Random(2)
    ds = [V.nudge_depth(V.sample_chain(rng, 8)) for _ in range(100)]
    assert all(0 <= d <= 8 for d in ds) and sum(d == 8 for d in ds) > 50  # saturated by design


def test_width_semantics():
    p = V.PairCount((1, 49, 10, 40, 25), 0)  # (1,49), (10,40) sum to 0 mod 50; 25 alone
    assert p.answer() == 2 and p.partners() == [0, 1, 0, 1, 0]
    r = V.render_paircount(p)
    assert r["steps"] == ["<<0+0=0>>", "<<0+1=1>>", "<<1+0=1>>", "<<1+1=2>>", "<<2+0=2>>"] and r["answer"] == "2"
    d = V.Distinct((5, 5, 7, 5, 9))
    assert d.answer() == 3
    assert V.render_distinct(d)["steps"] == ["<<0+1=1>>", "<<1+0=1>>", "<<1+1=2>>", "<<2+0=2>>", "<<2+1=3>>"]


def test_width_answer_balance():
    rng = random.Random(3)
    for n in (6, 16):
        ans = Counter(V.sample_paircount(rng, n).answer() for _ in range(400))
        assert set(ans) <= set(range(0, min(4, n // 2) + 1)) and max(ans.values()) / 400 < 0.4
    for n in (8, 24):
        ans = Counter(V.sample_distinct(rng, n).answer() for _ in range(400))
        assert max(ans.values()) / 400 < 0.4


def test_split_disjoint_balanced_deterministic(built):
    again = {t: V.build_task(t, **SMALL) for t in TASKS}
    for task, (splits, stats) in built.items():
        assert splits == again[task][0] and stats == again[task][1]
        keys = {s: [V.parse_question(task, r["question"]).key() for r in splits[s]] for s in ("train", "valid", "test")}
        assert len(set(keys["test"])) == len(keys["test"])
        assert not set(keys["test"]) & set(keys["valid"]) and not set(keys["test"]) & set(keys["train"])
        assert not set(keys["valid"]) & set(keys["train"])
        assert Counter(r["level"] for r in splits["test"]) == {lv: 15 for lv in V.TASK_LEVELS[task]}
        ids = [r["id"] for r in splits["test"]]
        assert len(set(ids)) == len(ids)
        assert stats["generator_version"] == V.GENERATOR_VERSION


def test_single_token_numbers():
    tok = pytest.importorskip("transformers").AutoTokenizer
    try:
        t = tok.from_pretrained("gpt2", local_files_only=True)
    except Exception:
        pytest.skip("gpt2 tokenizer not cached")
    for v in range(100):
        assert len(t.encode(f" {v}")) == 1 and len(t.encode(str(v))) == 1
    rng = random.Random(4)
    longest = max(len(t.encode(V.render_chain(V.sample_chain(rng, 12))["question"])) for _ in range(50))
    assert longest <= 245  # CODI truncates question at 256 tokens


def test_modal_semantics():
    p = V.Modal(((3, "*", 4), (20, "-", 8), (5, "+", 7), (1, "+", 2), (0, "+", 9)))
    assert p.values() == [12, 12, 12, 3, 9] and p.answer() == 12
    r = V.render_modal(p)
    assert r["question"] == ("Expressions: 3*4, 20-8, 5+7, 1+2, 0+9. "
                             "Which value do the most expressions evaluate to?")
    assert r["steps"] == ["<<3*4=12>>", "<<20-8=12>>", "<<5+7=12>>", "<<1+2=3>>", "<<0+9=9>>", "<<12+0=12>>"]
    assert r["answer"] == "12"
    tie = V.Modal(((1, "+", 1), (4, "-", 2), (1, "+", 2), (2, "+", 1)))
    with pytest.raises(AssertionError):
        tie.answer()


def test_modal_sampler_properties():
    rng = random.Random(5)
    best_guess = Counter()
    for n in (6, 10, 16):
        for _ in range(300):
            p = V.sample_modal(rng, n)
            assert len(p.exprs) == n and len(set(p.exprs)) == n  # no repeated expression string
            counts = p.counts()
            (v, m), *rest = counts.most_common()
            assert all(c < m for _, c in rest) and m >= 2
            assert all(0 <= x <= 99 for x in p.values())
            assert all(0 <= a <= 99 and 0 <= b <= 99 for a, _, b in p.exprs)
            best_guess[v] += 1
    assert max(best_guess.values()) / sum(best_guess.values()) < 0.05  # answers spread over 0..99
