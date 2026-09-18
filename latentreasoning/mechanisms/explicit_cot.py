"""Explicit CoT baseline: ordinary autoregressive generation, the model reasons in
visible output tokens before stating the final answer. Every latent mechanism is
compared against this (and against the zero-model floor in `latentreasoning/baselines/`).

Owner's notes go here. Unbudgeted -- report the actual CoT token count post-hoc under
`metrics["extra"]["cot_tokens"]` instead of `compute_steps`.

Gotchas for whoever implements it on GPT-2:
- Use exactly the same prompt template at train and eval time; the scorer only needs
  a number in the output (`#### N` or last number, see `latentreasoning.eval.metrics`).
- GPT-2's BPE encodes a trailing space as its own token, so a template ending in
  "Answer: " shifts the first generated token's distribution. End on a non-space.

Training target is the full visible rationale (the `cot` field's calculator
annotations, e.g. "<<16-3-4=9>> <<9*2=18>>") followed by `#### {answer}` -- ordinary
next-token LM fine-tuning, loss masked over the question/prompt span only.
`compute_steps` doesn't apply (unbudgeted); report the actual generated CoT token
count post-hoc under `metrics["extra"]["cot_tokens"]`.

`scripts/train_explicit_cot.py` is the actual training/eval loop that uses these.
"""
from __future__ import annotations

from typing import Any

name = "explicit_cot"

PROMPT_TEMPLATE = "Question: {question}\nAnswer:"
TARGET_TEMPLATE = " {rationale}\n#### {answer}"


def build_example_ids(tokenizer: Any, question: str, rationale: str, answer: str) -> tuple[list[int], list[int]]:
    """`(input_ids, labels)` for one training example: prompt + rationale + answer +
    EOS. `labels` masks (`-100`) the prompt span so loss is only over the completion."""
    prompt_ids = tokenizer(PROMPT_TEMPLATE.format(question=question), add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(
        TARGET_TEMPLATE.format(rationale=rationale, answer=answer), add_special_tokens=False
    )["input_ids"]
    eos = tokenizer.eos_token_id
    input_ids = prompt_ids + target_ids + [eos]
    labels = [-100] * len(prompt_ids) + target_ids + [eos]
    return input_ids, labels


def build_eval_prompt_ids(tokenizer: Any, question: str) -> list[int]:
    return tokenizer(PROMPT_TEMPLATE.format(question=question), add_special_tokens=False)["input_ids"]
