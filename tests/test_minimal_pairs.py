from latentreasoning.data.gsm8k_aug import load_local_sample
from latentreasoning.data.minimal_pairs import generate_minimal_pair, generate_minimal_pairs
from latentreasoning.eval.counterfactual import num_equal, parse_steps

SAMPLE = load_local_sample()
JANET = SAMPLE[0]  # "<<16-3-4=9>> <<9*2=18>>", answer 18


def test_generate_minimal_pair_perturbs_question_and_chain_consistently():
    pair = generate_minimal_pair(JANET, step=0, delta=1)
    assert pair is not None
    assert "17" in pair.twin.question and "16" not in pair.twin.question.replace("16-3-4", "")
    twin_steps = parse_steps(pair.twin.rationale)
    assert twin_steps[0]["val"] == "10"  # 17-3-4
    assert twin_steps[1]["val"] == "20"  # 10*2
    assert pair.twin.answer == "20"


def test_twin_answer_is_self_consistent_with_its_own_chain():
    """The twin isn't scored against `counterfactual_answer`'s single-value chain
    propagation (that model assumes a number flows through exactly one downstream path;
    a raw question number reused as a literal operand in several steps independently --
    common in GSM8K-Aug -- breaks that assumption). Its answer only needs to be what its
    OWN re-executed chain says, which is what makes it usable as ground truth directly."""
    for pair in generate_minimal_pairs(SAMPLE):
        twin_steps = parse_steps(pair.twin.rationale)
        assert any(num_equal(st["val"], pair.twin.answer) for st in twin_steps)


def test_generate_minimal_pair_returns_none_for_out_of_range_step():
    assert generate_minimal_pair(JANET, step=99, delta=1) is None


def test_generate_minimal_pairs_only_yields_pairs_that_change_the_target_step():
    for pair in generate_minimal_pairs(SAMPLE):
        orig_steps = parse_steps(pair.original.rationale)
        twin_steps = parse_steps(pair.twin.rationale)
        assert not num_equal(orig_steps[pair.step]["val"], twin_steps[pair.step]["val"])
