from latentreasoning.data.gsm8k_aug import Example, load_local_sample, parse_intermediate_values


def test_parse_intermediate_values():
    assert parse_intermediate_values("<<16-3-4=9>> <<9*2=18>>") == ["9", "18"]
    assert parse_intermediate_values("plain natural language rationale") == []


def test_example_intermediate_values_ends_with_answer():
    ex = load_local_sample(n=1)[0]
    assert isinstance(ex, Example)
    assert ex.intermediate_values[-1] == ex.answer
