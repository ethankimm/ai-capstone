"""Zero-model sanity-floor baseline: predict the last number that appears in the
question text itself. No model, no deps beyond stdlib. Every real mechanism should
clear this floor by a wide margin -- if a trained checkpoint doesn't, something in the
pipeline (prompting, answer extraction, data) is broken before the mechanism is even a
suspect. `scripts/run_baseline.py` runs it end to end without torch or a GPU.
"""
from __future__ import annotations

from latentreasoning.eval.metrics import NUMBER_RE

name = "baseline_last_number_in_question"


def predict(question: str) -> str:
    matches = NUMBER_RE.findall(question)
    return matches[-1] if matches else "0"
