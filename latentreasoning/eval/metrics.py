"""Answer extraction + scoring, shared across all four mechanisms so accuracy numbers
mean the same thing regardless of who generated them or how.
"""
from __future__ import annotations

import re

NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_final_number(text: str) -> str | None:
    """Pull the final numeric answer out of free-form model output.

    Checks, in order: an explicit "#### X" marker (GSM8K convention), then the last
    number appearing anywhere in the text. Strips thousands separators. Returns None if
    no number is found -- callers should treat that as a wrong/unparseable answer, not
    silently skip it (an unparseable output is a real failure mode worth counting).
    """
    if "####" in text:
        tail = text.split("####")[-1]
        m = NUMBER_RE.search(tail)
        if m:
            return m.group(0).replace(",", "")
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def is_correct(predicted: str | None, gold: str) -> bool:
    """Numeric-equal comparison (18 == 18.0 == "18"), not string-equal."""
    if predicted is None:
        return False
    try:
        return float(predicted) == float(gold)
    except ValueError:
        return predicted.strip() == gold.strip()


def accuracy(predictions: list[str | None], golds: list[str]) -> float:
    if not predictions:
        return 0.0
    correct = sum(is_correct(p, g) for p, g in zip(predictions, golds))
    return correct / len(predictions)
