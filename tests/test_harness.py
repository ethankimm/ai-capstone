from latentreasoning.data.gsm8k_aug import Example
from latentreasoning.eval.harness import run_eval, score_outputs

EXAMPLES = [Example(question=f"q{i}", rationale="", answer=a, idx=i) for i, a in enumerate(["18", "3", "7"])]


def test_score_outputs_scores_and_records_in_order():
    r = score_outputs(EXAMPLES, ["so the answer is 18", "#### 3.0", "no digits here"])
    assert r.n == 3 and r.final_answer_accuracy == 2 / 3 and r.unparseable_rate == 1 / 3
    assert [rec["idx"] for rec in r.records] == [0, 1, 2]
    assert [rec["correct"] for rec in r.records] == [True, True, False]
    assert r.records[2]["predicted_answer"] is None
    assert r.sec_per_example is None


def test_score_outputs_rejects_length_mismatch():
    import pytest

    with pytest.raises(ValueError):
        score_outputs(EXAMPLES, ["18"])


def test_run_eval_wraps_score_outputs():
    r = run_eval(lambda q: "18", EXAMPLES)
    assert r.final_answer_accuracy == 1 / 3 and r.sec_per_example is not None
