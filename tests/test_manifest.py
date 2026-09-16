import json

import pytest

from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, index_row, new_run_id


def _record(**kw):
    base = dict(
        run_id="test-run",
        mechanism="filler_tokens",
        stage="pilot",
        model=ModelInfo(backbone="gpt2", checkpoint="gpt2"),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=5, seed=0),
        metrics={"final_answer_accuracy": 0.4, "compute_steps": 8},
        seed=42,
    )
    return RunRecord(**{**base, **kw})


def test_new_run_id_contains_mechanism_and_slug():
    run_id = new_run_id("filler_tokens", "gsm8k-aug-sample")
    assert "filler_tokens" in run_id and "gsm8k-aug-sample" in run_id


def test_save_writes_manifest_predictions_and_notes(tmp_path):
    preds = [{"idx": 0, "raw_output": "7", "correct": True}]
    path = _record().save(out_dir=tmp_path, predictions=preds)
    assert json.loads(path.read_text())["seed"] == 42
    assert [json.loads(l) for l in (tmp_path / "predictions.jsonl").read_text().splitlines()] == preds
    notes = (tmp_path / "notes.md").read_text()
    assert "test-run" in notes and "TODO" in notes


def test_save_does_not_overwrite_existing_notes(tmp_path):
    (tmp_path / "notes.md").write_text("my notes")
    _record().save(out_dir=tmp_path)
    assert (tmp_path / "notes.md").read_text() == "my notes"


@pytest.mark.parametrize("bad", [dict(mechanism="coconut"), dict(stage="final"),
                                 dict(dataset=DatasetInfo(split="dev"))])
def test_validation_rejects_unknown_values(bad):
    with pytest.raises(ValueError):
        _record(**bad)


def test_baseline_mechanism_names_allowed():
    _record(mechanism="baseline_last_number_in_question")


def test_index_row_accepts_record_or_dict():
    rec = _record()
    row = index_row(rec)
    assert row == index_row(rec.to_dict())
    assert "filler_tokens" in row and "0.400" in row and "| 42 |" in row
