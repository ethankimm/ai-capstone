"""Shared scoring so `final_answer_accuracy` means the same thing for every mechanism
and collaborator, however the outputs were produced.

The contract is `score_outputs(examples, raw_outputs)`: run your own eval loop (the
CODI repo's, a batched HF `generate`, a custom looped-model decoder, ...), collect one
raw output string per example, and hand both lists here. `run_eval` is a convenience
wrapper for the simplest case -- a `generate_fn(question) -> str` closure called one
example at a time -- not something you have to fit your mechanism into.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from latentreasoning.data.gsm8k_aug import Example
from latentreasoning.eval.metrics import accuracy, extract_final_number, is_correct


@dataclass
class EvalResult:
    n: int
    final_answer_accuracy: float
    unparseable_rate: float
    sec_per_example: float | None = None  # only known if the loop was timed here
    records: list[dict] = field(default_factory=list)  # per-example detail, for error analysis


def score_outputs(examples: Iterable[Example], raw_outputs: Sequence[str]) -> EvalResult:
    """Score `raw_outputs[i]` (free-form model output text) against `examples[i]`.

    `records` are plain dicts, one per example, in order -- add mechanism-specific keys
    to them afterwards (loop iterations used, filler length, decoded intermediate
    guesses, ...) and pass them to `RunRecord.save(predictions=...)`.
    """
    examples = list(examples)
    if len(raw_outputs) != len(examples):
        raise ValueError(f"{len(raw_outputs)} outputs for {len(examples)} examples")
    records = []
    for ex, raw in zip(examples, raw_outputs):
        pred = extract_final_number(raw)
        records.append({
            "idx": ex.idx,
            "question": ex.question,
            "gold_answer": ex.answer,
            "raw_output": raw,
            "predicted_answer": pred,
            "correct": is_correct(pred, ex.answer),
        })
    n = len(examples)
    return EvalResult(
        n=n,
        final_answer_accuracy=accuracy([r["predicted_answer"] for r in records], [ex.answer for ex in examples]),
        unparseable_rate=sum(r["predicted_answer"] is None for r in records) / n if n else 0.0,
        records=records,
    )


def run_eval(generate_fn: Callable[[str], str], examples: Iterable[Example]) -> EvalResult:
    """Convenience: call `generate_fn(question)` per example, time it, score it."""
    examples = list(examples)
    start = time.perf_counter()
    raw_outputs = [generate_fn(ex.question) for ex in examples]
    elapsed = time.perf_counter() - start
    result = score_outputs(examples, raw_outputs)
    result.sec_per_example = elapsed / len(examples) if examples else 0.0
    return result
