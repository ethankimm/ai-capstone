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
"""
from __future__ import annotations

name = "explicit_cot"
