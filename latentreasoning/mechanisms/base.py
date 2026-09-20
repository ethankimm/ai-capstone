"""Mechanism names, and the (deliberately thin) convention around them.

One module per mechanism under `latentreasoning/mechanisms/` -- it belongs to whoever owns
that mechanism and holds their citation, reference repo, `compute_steps` definition,
and gotchas. There is no shared `generate_fn` / model-loading / training contract on
purpose: for the Sep 25 reproductions everyone builds their mechanism their own way
(own training loop, own prompt format, own generation), and we compare methodologies
and gaps afterwards. What *is* shared, so numbers stay comparable:

  1. data   -- `latentreasoning.data.gsm8k_aug.load_gsm8k_aug` with the eval settings in
               `configs/mechanisms/*.yaml` (`eval_n`, `eval_seed`, split)
  2. scoring -- `latentreasoning.eval.harness.score_outputs(examples, raw_outputs)`
  3. logging -- `latentreasoning.runlog.manifest.RunRecord` (see the `record-run` skill)

`compute_steps` is the one cross-mechanism knob: filler tokens emitted / loop
iterations r / continuous-thought tokens. Budgeted mechanisms report it in
`metrics["compute_steps"]`; `explicit_cot` is unbudgeted and reports its actual CoT
length under `metrics["extra"]["cot_tokens"]`.

Looking ahead (a note, not a contract): the Oct 9 milestone decodes intermediate states
with linear/non-linear probes, J-lens, and logit lens, which needs *per-layer hidden
states at the scratchpad positions* (the filler span, the continuous-thought span, each
loop iteration's output). Keep those positions cheap to identify in whatever you build
so exposing them is a small addition, not a rewrite.
"""
from __future__ import annotations

MECHANISM_NAMES = ("explicit_cot", "filler_tokens", "codi", "recurrent_depth", "coconut")
