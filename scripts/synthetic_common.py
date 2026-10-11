"""Shared helpers for the synthetic depth / width tasks, used by `train_synthetic.py`
(our filler-token / explicit-CoT trainer, main venv) and `eval_synthetic_coconut.py`
(Coconut-repo checkpoints, Coconut venv). Stdlib + `latentreasoning.data.gsm8k_aug.Example`
only, so it imports in either venv.

Dataset files (written by `latentreasoning/tasks/`, owned elsewhere):
  ~/Documents/Penn/CIS5980/synthetic_data/{depth,width}/{train,valid,valid_monitor,test,test_full}.json
each a JSON list of {"question", "steps": ["<<7*3=21>>", ...], "answer", "level", "task", "id"}.

Mapping to this repo's `Example`: `rationale = " ".join(steps)` -- the same
space-separated `<<a*b=c>>` calculator format as GSM8K-Aug's `cot` field, so
`explicit_cot.build_example_ids` / the faithful filler mixture train on exactly the
format they already train on for GSM8K. (Coconut's own format joins steps with "\\n"
instead -- that's `coconut_cot_text` below, used only on the Coconut side.)
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from latentreasoning.data.gsm8k_aug import Example  # noqa: E402

DEFAULT_DATA_ROOT = Path.home() / "Documents/Penn/CIS5980/synthetic_data"


@dataclass
class SyntheticSet:
    path: str
    rows: list[dict]
    examples: list[Example]  # Example.idx = position in the file

    @property
    def task(self) -> str:
        tasks = {r.get("task") for r in self.rows}
        return tasks.pop() if len(tasks) == 1 else "mixed"

    @property
    def level_range(self) -> list[int]:
        levels = [int(r["level"]) for r in self.rows]
        return [min(levels), max(levels)] if levels else []

    def subset(self, n: int | None) -> "SyntheticSet":
        if n is None or n <= 0:
            return self
        return SyntheticSet(self.path, self.rows[:n], self.examples[:n])


def load_synthetic(path: str | Path) -> SyntheticSet:
    rows = json.loads(Path(path).read_text())
    examples = [
        Example(question=r["question"], rationale=" ".join(r["steps"]), answer=str(r["answer"]).strip(), idx=i)
        for i, r in enumerate(rows)
    ]
    return SyntheticSet(str(path), rows, examples)


def annotate_records(records: list[dict], dataset: SyntheticSet) -> list[dict]:
    """Add `id` / `level` / `task` / `n_steps` to `score_outputs` records (same order as
    `dataset.examples`), so per-level accuracy and id-paired McNemar work downstream."""
    assert len(records) == len(dataset.rows)
    for rec, row in zip(records, dataset.rows):
        assert rec["idx"] is not None
        rec["id"] = row.get("id", rec["idx"])
        rec["level"] = int(row["level"])
        rec["task"] = row.get("task")
        rec["n_steps"] = len(row["steps"])
    return records


def coconut_cot_text(row: dict) -> str:
    """Coconut `dataset.py` training target for cot: steps joined by "\\n", then "### ans"."""
    return "\n".join(row["steps"]) + "\n### " + str(row["answer"])


# ---------------------------------------------------------------------------------------
# Coconut run.py stdout/stderr log parsing (run.py itself is used verbatim, so per-epoch
# training time and monitor accuracy come from its tqdm bar / prints)
# ---------------------------------------------------------------------------------------
_COCONUT_TRAIN_BAR = re.compile(
    r"Training Epoch: (\d+)/\d+[^|\n]*\|[^|\n]*\|\s*(\d+)/(\d+) \[([0-9:]+)<"
)
_COCONUT_VAL_ACC = re.compile(r"Accuracy on validation set: (\d+) / (\d+) = ([0-9.eE+-]+)")


def _hms_to_seconds(s: str) -> int:
    secs = 0
    for part in s.split(":"):
        secs = secs * 60 + int(part)
    return secs


def parse_coconut_log(text: str) -> dict:
    """From a captured `torchrun ... run.py cfg.yaml 2>&1 | tee train.log`:
    `seconds_train_epoch` = {epoch: elapsed of the training tqdm bar when it reached
    total} -- that bar is opened right before the train loop and closed right after,
    so it excludes checkpoint save, val loss and val generation (1 s resolution);
    `val_accuracy` = the per-epoch "Accuracy on validation set" values, in order."""
    text = text.replace("\r", "\n")
    epochs: dict[int, int] = {}
    for m in _COCONUT_TRAIN_BAR.finditer(text):
        ep, n, total, elapsed = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        if n >= total:  # run.py also steps on the last partial accumulation -> n can be total+1
            epochs[ep] = _hms_to_seconds(elapsed)
    vals = [{"correct": int(a), "total": int(b), "accuracy": float(c)} for a, b, c in _COCONUT_VAL_ACC.findall(text)]
    return {"seconds_train_epoch": dict(sorted(epochs.items())), "val_accuracy": vals}
