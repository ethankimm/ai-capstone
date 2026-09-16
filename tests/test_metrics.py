from latentreasoning.eval.metrics import accuracy, extract_final_number, is_correct


def test_extract_final_number_hash_marker():
    assert extract_final_number("blah blah #### 42") == "42"


def test_extract_final_number_last_number_fallback():
    assert extract_final_number("She had 3 apples then bought 4 more, giving 7.") == "7"


def test_extract_final_number_thousands_separator():
    assert extract_final_number("total: 1,234") == "1234"


def test_extract_final_number_none_when_no_number():
    assert extract_final_number("no numbers here") is None


def test_is_correct_numeric_equivalence():
    assert is_correct("18", "18")
    assert is_correct("18.0", "18")
    assert not is_correct("19", "18")
    assert not is_correct(None, "18")


def test_accuracy():
    preds = ["18", "3", None, "999"]
    golds = ["18", "3", "70000", "540"]
    assert accuracy(preds, golds) == 0.5
