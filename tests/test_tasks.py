"""Synthetic depth/width tasks: correctness, twins/decoys, splits, determinism, tokenization."""
import random
import re
from collections import Counter

import pytest

from latentreasoning.tasks import synthetic as S
from latentreasoning.tasks.generate import _split_quota, build_task, dumps_records

SMALL = dict(seed=123, train_total=1500, valid_total=300, test_per_level=20, monitor_per_level=5)


@pytest.fixture(scope="module")
def built():
    return {task: build_task(task, **SMALL) for task in ("depth", "width")}


def _check_example(task, rec):
    """Values in range, steps recompute from the question, answer correct. Returns values."""
    p = S.parse_question(task, rec["question"])
    assert S.render(p) == {k: rec[k] for k in ("question", "steps", "answer")}
    vals = []
    for k, st in enumerate(rec["steps"]):
        left, op, right, v = S.parse_step(st)
        assert S.apply_op(op, left, right) == v
        assert 0 <= v <= 99
        if op == "*":  # no `*1` literal (a previous *value* of 1 is fine)
            literals = (left, right) if task == "width" or k == 0 else (p.operands[k],)
            assert 1 not in literals
        if task == "depth" and k > 0:
            assert vals[-1] in (left, right)
            if p.swapped[k]:
                assert left == p.operands[k] and right == vals[-1]
            else:
                assert left == vals[-1] and right == p.operands[k]
        vals.append(v)
    if task == "depth":
        assert vals == p.values()
        assert rec["answer"] == str(vals[-1])
        assert len(vals) == rec.get("level", len(vals))
    else:
        assert all(2 <= x <= 9 for pair in p.pairs for x in pair)
        assert rec["answer"] == str(max(vals))
        assert vals.count(max(vals)) == 1
    return vals


def test_examples_valid(built):
    for task, (splits, _) in built.items():
        for split, recs in splits.items():
            for rec in recs:
                assert rec["task"] == task
                _check_example(task, rec)


def test_example_format():
    p = S.DepthProblem(7, ("*", "-", "*", "+"), (3, 5, 2, 9), (False,) * 4)
    assert S.render(p) == {
        "question": "a = 7*3; b = a-5; c = b*2; d = c+9. What is d?",
        "steps": ["<<7*3=21>>", "<<21-5=16>>", "<<16*2=32>>", "<<32+9=41>>"],
        "answer": "41",
    }
    w = S.WidthProblem(((7, 3), (4, 6), (9, 2), (5, 5), (8, 4)))
    assert S.render(w) == {
        "question": "7*3, 4*6, 9*2, 5*5, 8*4. Which product is largest?",
        "steps": ["<<7*3=21>>", "<<4*6=24>>", "<<9*2=18>>", "<<5*5=25>>", "<<8*4=32>>"],
        "answer": "32",
    }
    # commuted later step round-trips through the parser
    q = S.DepthProblem(7, ("*", "-", "*"), (3, 5, 2), (False, False, True))
    r = S.render(q)
    assert r["question"] == "a = 7*3; b = a-5; c = 2*b. What is c?"
    assert r["steps"][-1] == "<<2*16=32>>"
    assert S.parse_depth_question(r["question"]) == q


def test_grid_interior_not_implemented():
    with pytest.raises(NotImplementedError, match="Phase B two-knob grid"):
        S.sample_problem(random.Random(0), n_chains=2, n_steps=2)
    assert isinstance(S.sample_problem(random.Random(0), 1, 5), S.DepthProblem)
    assert isinstance(S.sample_problem(random.Random(0), 4, 1), S.WidthProblem)


def test_space_sizes_match_enumeration():
    for n in (1, 2):
        probs = S.enumerate_depth(n)
        assert len(probs) == S.depth_space_size(n) == len({p.key() for p in probs})
        assert all(S.depth_valid(p) for p in probs)
    probs = S.enumerate_width(2)
    assert len(probs) == S.width_space_size(2)
    assert S.depth_space_size(1) == 190


def test_sampler_valid_deep():
    rng = random.Random(0)
    for _ in range(2000):
        p = S.sample_depth(rng, 12)
        assert S.depth_valid(p)
        assert all(0 <= v <= 99 for v in p.values())


def _differs_in_one_operand(base, q):
    if isinstance(base, S.DepthProblem):
        a = (base.x0, *base.operands)
        b = (q.x0, *q.operands)
        assert base.ops == q.ops and base.swapped == q.swapped
    else:
        a = tuple(x for pair in base.pairs for x in pair)
        b = tuple(x for pair in q.pairs for x in pair)
    return sum(x != y for x, y in zip(a, b)) == 1


def test_twins(built):
    for task, (splits, _) in built.items():
        n_twins = 0
        for ex in splits["test_full"]:
            base = S.parse_question(task, ex["question"])
            base_vals = _check_example(task, ex)
            assert len(ex["twins"]) in (0, len(S.LADDER_TARGETS))
            for tw in ex["twins"]:
                n_twins += 1
                vals = _check_example(task, tw)
                q = S.parse_question(task, tw["question"])
                assert _differs_in_one_operand(base, q)
                assert tw["answer"] != ex["answer"]
                assert tw["abs_change_answer"] == abs(int(tw["answer"]) - int(ex["answer"]))
                k = tw["step_changed"]
                assert vals[k] != base_vals[k]
                assert tw["signed_change_step"] == vals[k] - base_vals[k]
                assert tw["rel_change_step"] == pytest.approx(abs(vals[k] - base_vals[k]) / base_vals[k], abs=1e-6)
                if task == "depth":
                    # upstream untouched; injectivity: every downstream value changes
                    assert vals[:k] == base_vals[:k]
                    assert all(v != b for v, b in zip(vals[k:], base_vals[k:]))
                    pos = tw["operand_position"]
                    old = base.x0 if (k == 0 and pos == 0) else base.operands[k]
                    assert (tw["old_operand"], pos in (0, 1)) == (old, True)
                    if k > 0:
                        assert pos == 1
                else:
                    assert k == base.argmax()
                    assert tw["old_operand"] == base.pairs[k][tw["operand_position"]]
                    assert tw["new_operand"] == q.pairs[k][tw["operand_position"]]
                t = tw["target_rel_change"]
                rel = tw["rel_change_step"] if task == "depth" else tw["rel_change_answer"]
                assert tw["within_tolerance"] == (t / S.LADDER_TOLERANCE <= rel + 1e-6 and rel - 1e-6 <= t * S.LADDER_TOLERANCE)
        assert n_twins > 0


def test_decoys(built):
    for task, (splits, _) in built.items():
        n = Counter()
        for ex in splits["test_full"]:
            base = S.parse_question(task, ex["question"])
            base_vals = _check_example(task, ex)
            for d in ex["decoys"]:
                n[d["decoy_type"]] += 1
                vals = _check_example(task, d)
                assert d["answer"] == ex["answer"]
                assert d["question"] != ex["question"]
                if task == "depth":
                    assert vals == base_vals  # every step value unchanged
                    if d["decoy_type"] == "same_value":
                        assert d["step_changed"] == 0
                else:
                    q = S.parse_question(task, d["question"])
                    assert _differs_in_one_operand(base, q)
                    assert d["step_changed"] != base.argmax()
                    assert q.argmax() == base.argmax()
        if task == "depth":
            assert n["commute"] > 0 and n["same_value"] > 0
        else:
            assert n["smaller_product"] > 0


def test_splits_disjoint(built):
    for task, (splits, _) in built.items():
        keys = {}
        for split in ("train", "valid", "test"):
            ks = {S.parse_question(task, r["question"]).key() for r in splits[split]}
            keys[split] = ks
            if split != "train":
                assert len(ks) == len(splits[split])  # no duplicates outside train
        assert not keys["train"] & keys["valid"]
        assert not keys["train"] & keys["test"]
        assert not keys["valid"] & keys["test"]
        mon = {r["question"] for r in splits["valid_monitor"]}
        assert mon <= {r["question"] for r in splits["valid"]}
        ids = [r["id"] for s in ("train", "valid", "test") for r in splits[s]]
        assert len(ids) == len(set(ids))


def test_level_balance(built):
    for task, (splits, stats) in built.items():
        tr = Counter(r["level"] for r in splits["train"])
        assert sum(tr.values()) == SMALL["train_total"]
        assert max(tr.values()) - min(tr.values()) <= 1
        va = Counter(r["level"] for r in splits["valid"])
        quota = _split_quota(SMALL["valid_total"], sorted(va))
        assert max(quota.values()) - min(quota.values()) <= 1
        for level, c in va.items():
            # balanced except where the level's space is too small (10% cap)
            cap = int(0.1 * stats["levels"][str(level)]["space_size"])
            assert c == min(cap, quota[level])
        mon = Counter(r["level"] for r in splits["valid_monitor"])
        assert all(mon[lv] == min(SMALL["monitor_per_level"], va[lv]) for lv in va)


def test_deterministic():
    a, sa = build_task("depth", levels=[1, 2, 5], **SMALL)
    b, sb = build_task("depth", levels=[1, 2, 5], **SMALL)
    for split in a:
        assert dumps_records(a[split]) == dumps_records(b[split])
    assert sa == sb
    c, _ = build_task("depth", levels=[1, 2, 5], **{**SMALL, "seed": 124})
    assert dumps_records(a["train"]) != dumps_records(c["train"])
    w1, _ = build_task("width", levels=[2, 7], **SMALL)
    w2, _ = build_task("width", levels=[2, 7], **SMALL)
    assert dumps_records(w1["test_full"]) == dumps_records(w2["test_full"])


# --------------------------------------------------------------------------- tokenizer


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:  # no network and no cache
        pytest.skip(f"gpt2 tokenizer unavailable: {e}")


def test_numbers_single_token(tok):
    for v in range(100):
        assert len(tok.encode(str(v))) == 1, v
        assert len(tok.encode(" " + str(v))) == 1, v


def _digit_tokens(tok, text):
    pieces = [tok.decode([i]) for i in tok.encode(text)]
    for p in pieces:
        if any(ch.isdigit() for ch in p):
            assert p.strip().isdigit(), (p, text)  # no digit merged with other chars
    return [p.strip() for p in pieces if p.strip().isdigit()]


def _check_in_context(tok, rec):
    """Every integer literal must be exactly one token, in each training-repo format."""
    q, steps, ans = rec["question"], rec["steps"], rec["answer"]
    # Coconut: question + "\n", each step + "\n", "### " + answer, tokenized separately
    coconut_segments = [q + "\n"] + [s + "\n" for s in steps] + ["### " + ans]
    for seg in coconut_segments:
        assert _digit_tokens(tok, seg) == re.findall(r"\d+", seg), seg
    # Same content as one concatenated string, and a CODI-style rendering
    full = q + "\n" + "\n".join(steps) + "\n### " + ans
    assert _digit_tokens(tok, full) == re.findall(r"\d+", full)
    codi = q + " " + " ".join(steps) + " The answer is: " + ans
    assert _digit_tokens(tok, codi) == re.findall(r"\d+", codi)


def test_single_token_in_context(tok, built):
    # every result value 0..99 inside a step string, for each op and operand shape
    for v in range(100):
        for left in (0, 7, 42, 99):
            for op in S.OPS:
                for right in (2, 9):
                    s = f"<<{left}{op}{right}={v}>>"
                    for seg in (s + "\n", " " + s, s):
                        assert _digit_tokens(tok, seg) == [str(left), str(right), str(v)]
    for v in range(100):
        assert _digit_tokens(tok, f"### {v}") == [str(v)]
    # real rendered examples, including twins and decoys
    for task, (splits, _) in built.items():
        for ex in splits["test_full"][::3]:
            _check_in_context(tok, ex)
            for sub in ex["twins"] + ex["decoys"]:
                _check_in_context(tok, sub)
        for ex in splits["train"][:300]:
            _check_in_context(tok, ex)


def test_width_answer_balanced():
    """v1.1 sampler: answer not dominated by 72/81 at the top width level."""
    rng = random.Random(0)
    n = 6000
    c = Counter(max(S.sample_width(rng, 16).products()) for _ in range(n))
    assert set(c) == set(S.WIDTH_MAX_VALUES)
    assert max(c.values()) / n < 0.10
    # probability helper agrees with the sampler's support and normalizes at L2
    assert sum(S.width_problem_prob(p) for p in S.enumerate_width(2)) == pytest.approx(1.0)


def test_width_answer_balanced_in_splits(built):
    splits, _ = built["width"]
    top = [r for r in splits["train"] + splits["test"] if r["level"] == 16]
    c = Counter(r["answer"] for r in top)
    assert max(c.values()) / len(top) < 0.15  # small fixture; full data is ~4%


def test_no_multiply_zero(built):
    """v1.2: no step of the form 0*c or c*0, in base examples, twins or decoys."""
    splits, _ = built["depth"]
    recs = splits["train"] + splits["valid"] + splits["test"]
    for ex in splits["test_full"]:
        recs += ex["twins"] + ex["decoys"]
    for r in recs:
        for st in r["steps"]:
            left, op, right, _ = S.parse_step(st)
            assert not (op == "*" and 0 in (left, right)), (st, r["question"])
    rng = random.Random(1)
    for _ in range(2000):
        p = S.sample_depth(rng, 12)
        vals = [p.x0] + p.values()
        assert all(not (op == "*" and vals[k] == 0) for k, op in enumerate(p.ops))
