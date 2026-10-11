import random

import pytest

from latentreasoning.data.gsm8k_aug import Example, load_local_sample
from latentreasoning.data.minimal_pairs import (
    DEFAULT_DELTAS, WIDE_CHANGE_SET, Change, generate_change_set_pairs, generate_minimal_pair,
    generate_minimal_pair_for_change, generate_minimal_pairs, hidden_operands, parse_change_set,
    passes_hidden_operand_filter,
)
from latentreasoning.eval.counterfactual import num_equal, parse_steps, qualifying_steps
from latentreasoning.eval.twin_magnitude import bin_summary, magnitude_fields, rel_bin

SAMPLE = load_local_sample()
JANET = SAMPLE[0]  # "<<16-3-4=9>> <<9*2=18>>", answer 18


def test_default_deltas_unchanged():
    assert DEFAULT_DELTAS == (1, -1, 2, -2, 3, -3)
    assert generate_minimal_pairs(SAMPLE) == generate_minimal_pairs(SAMPLE, deltas=DEFAULT_DELTAS)


def test_parse_change_set():
    cs = parse_change_set(WIDE_CHANGE_SET)
    assert Change("add", 20.0) in cs and Change("mul", 0.5) in cs and Change("mul", 3.0) in cs
    assert [c.label for c in parse_change_set("5,-5,x1.25")] == ["+5", "-5", "x1.25"]
    with pytest.raises(ValueError):
        parse_change_set("0")


def test_multiplicative_twin_rounds_half_up_and_reexecutes():
    pair = generate_minimal_pair_for_change(JANET, 0, Change("mul", 1.5))
    assert pair is not None and pair.change == "x1.5"
    assert "24" in pair.twin.question  # 16 * 1.5
    steps = parse_steps(pair.twin.rationale)
    assert steps[0]["val"] == "17" and pair.twin.answer == "34"  # 24-3-4, *2
    assert pair.delta == 8
    # x1.25 of 3 = 3.75 -> 4 (half-up rounding, never Python's banker's rounding)
    ex = Example(question="She has 3 bags with 4 apples each.", rationale="<<3*4=12>>", answer="12", idx=0)
    p = generate_minimal_pair(ex, 0, 0, factor=1.25)
    assert p is not None and p.twin.answer == "16"


def test_multiplicative_twin_rejected_when_rounding_is_a_noop():
    ex = Example(question="She has 1 bag with 4 apples.", rationale="<<1*4=4>>", answer="4", idx=0)
    assert generate_minimal_pair(ex, 0, 0, factor=1.25) is None  # round(1.25) == 1


def test_change_set_pairs_are_deduplicated_and_consistent():
    rng = random.Random(0)
    pairs = generate_change_set_pairs(SAMPLE, lambda e: qualifying_steps(parse_steps(e.rationale)),
                                      parse_change_set(WIDE_CHANGE_SET), rng=rng)
    keys = [(p.original.idx, p.twin.question) for p in pairs]
    assert len(keys) == len(set(keys))
    for p in pairs:
        assert p.change is not None
        assert any(num_equal(st["val"], p.twin.answer) for st in parse_steps(p.twin.rationale))
        assert not num_equal(p.twin.answer, p.original.answer) or p.step < len(parse_steps(p.original.rationale)) - 1


def test_hidden_operands():
    ex = Example(question="Tom runs 3 miles a day. How far in a week?", rationale="<<3*7=21>>", answer="21", idx=0)
    assert hidden_operands(ex.question, ex.rationale) == ["7"]
    ok = Example(question="Tom runs 3 miles a day for 7 days.", rationale="<<3*7=21>> <<21+3=24>>", answer="24", idx=0)
    assert hidden_operands(ok.question, ok.rationale) == []


def test_hidden_operand_filter_rejects_collisions():
    # perturbed number 9 collides with the step result 9: literal substitution would rewrite both
    ex = Example(question="Ann has 9 pens and buys 5 and 4 more.", rationale="<<5+4=9>> <<9+9=18>>", answer="18", idx=0)
    pair = generate_minimal_pair(ex, 0, 1)
    assert pair is not None
    if pair.perturbed_number == "9":
        assert not passes_hidden_operand_filter(pair)


def test_magnitude_fields_and_bins():
    f = magnitude_fields("100", "110", "105")
    assert f["abs_answer_change"] == 10 and f["rel_answer_change"] == pytest.approx(0.10)
    assert f["fraction_moved"] == pytest.approx(0.5)
    assert magnitude_fields("0", "5", "5")["rel_answer_change"] is None
    assert magnitude_fields("10", "20", None)["fraction_moved"] is None
    assert rel_bin(0.0) == "<=5%" and rel_bin(0.05) == "<=5%" and rel_bin(0.0501) == "5-15%"
    assert rel_bin(1.0) == "40-100%" and rel_bin(2.0) == ">100%" and rel_bin(None) is None
    rows = [{"rel_answer_change": 0.1, "fraction_moved": 1.0, "matches_twin": True},
            {"rel_answer_change": 0.1, "fraction_moved": 0.0, "matches_twin": False}]
    s = bin_summary(rows)
    assert s["5-15%"]["n"] == 2 and s["5-15%"]["steer_rate"] == 0.5
    assert s["5-15%"]["mean_fraction_moved"] == 0.5 and s["<=5%"]["n"] == 0
