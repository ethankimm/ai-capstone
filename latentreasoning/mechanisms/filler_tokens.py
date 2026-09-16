"""Filler tokens (Pfau, Merrill & Bowman 2024, arXiv:2404.15758): `compute_steps`
content-free tokens (e.g. repeated ".") between the question and the answer instead
of a genuine step-by-step rationale. Tests whether hidden computation happens across
those extra forward passes even though the tokens carry no task information.

Owner's notes go here. `compute_steps` = number of filler tokens; `compute_steps=0` is
the no-filler control -- always log it alongside any nonzero sweep point.

Gotchas for whoever implements it on GPT-2:
- GPT-2's BPE merges runs of "." into single tokens ("..." is one token), so a string
  of N dots does *not* tokenize to N filler tokens. Splice the filler token id N times
  into `input_ids` directly, and do it the same way at train and eval time.
- Pfau et al. found filler tokens only help with dense/parallel supervision on
  synthetic tasks; a null result on GSM8K-Aug is plausible and still a result.
"""
from __future__ import annotations

name = "filler_tokens"
