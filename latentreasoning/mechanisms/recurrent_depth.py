"""Recurrent depth (vertical): loop a recurrent Transformer block r times per token
instead of stacking r layers. Geiping et al. arXiv:2502.05171; reference implementation
github.com/seal-rg/recurrent-pretraining (their huginn-0125 checkpoint is 3.5B -- too
big to reuse; this needs its own small looped GPT-2). Highest-risk reproduction, so it
goes first (see PROPOSAL.md).

Before deciding how to decode intermediate states, see "Looped Transformers under the
Jacobian Lens" (PROPOSAL.md references): loop iterations largely overwrite rather than
accumulate state.

---

**What the paper does (Sec. 3, verified against arXiv:2502.05171v2):** prelude P
(l_p layers) embeds tokens to `e`; a recurrent core R (l_r layers) is iterated
`s_i = R(A[s_{i-1}; e])` for r steps, where `A: R^{2h} -> R^h` is a linear adapter on
the *concatenation* of the previous state and the prelude output (so `e` is re-injected
every iteration); a coda C (l_c layers) + LM head reads out `s_r`. Their large model is
(l_p, l_r, l_c) = (2, 4, 2). `s_0 ~ N(0, 2/5 I)` is fresh random noise per forward pass.
Training samples r per micro-batch from a log-normal Poisson,
`tau ~ N(log(r_bar) - sigma^2/2, sigma), r ~ Poisson(e^tau) + 1` with `r_bar = 32,
sigma = 1/2`, and backpropagates only through the last `k = 8` iterations (earlier ones
run without grad; the prelude still gets gradient through the injection). At test time
r is a free knob: accuracy improves with r and saturates -- that is the claim to verify.

**How this maps onto pretrained `gpt2` here** (`recurrent_depth_model.LoopedGPT2`):
- Layer split is (4, 4, 4): prelude = blocks 0-3, core = blocks 4-7 (looped), coda =
  blocks 8-11, so every pretrained weight is kept and at r=1 the unrolled network is
  plain 12-layer GPT-2. The paper's 2:4:2 ratio would drop 4 pretrained blocks.
- Adapter is initialised to `A[e; s] = e` (identity on the `e` half, zero on the state
  half), so at init the model *is* GPT-2 for any r (every iteration returns R(e)) and
  training has to learn to use the loop -- a null result (flat accuracy in r) is
  therefore a real finding about fine-tuning a pretrained net into a looped one, not
  an init artefact.
- The state is LayerNorm'd before it enters the adapter (`A[e; LN(s)]`). The paper
  RMSNorms the adapter *output* and uses sandwich-norm blocks; GPT-2 blocks are pre-LN
  only and can't be retrofitted without changing the pretrained function, so the norm
  is moved to the state path. This also bounds the state across iterations (the state
  is overwritten, not accumulated, so no residual-norm blow-up).
- No learned embedding scale gamma (pretrained embeddings), no sandwich norms.

**Prompt / target format:** direct answer, no visible rationale --
`"Question: {q}\nAnswer:"` -> `" #### {a}"` + EOS, loss only on the answer span.
Identical to `filler_tokens` at `compute_steps=0`, so that run is the no-scratchpad
control for this mechanism (both are "GPT-2 fine-tuned to answer directly").

**`compute_steps` = r**, the number of core iterations at *eval* time (each = 4
transformer blocks over every position). One trained checkpoint is evaluated at a
sweep of r; the run's manifest reports the eval at r = r_bar (the training mean) as
`compute_steps` / `final_answer_accuracy`, and the whole sweep under
`metrics["extra"]["sweep_by_r"]` with per-r predictions in `predictions_r{r}.jsonl`.
Compute-matching against filler tokens is not automatic: r=32 here is 32 extra passes
of 4 blocks over *all* positions, not 32 extra positions through 12 blocks.

Gotchas:
- No KV cache for generation: every new token re-runs the looped forward on the full
  prefix. Fine for the ~6-token `#### N` answers here; the noise `s_0` is drawn once
  per example for the full max length and sliced, so earlier positions' states are
  stable across decoding steps (as a cache would give).
- `create_causal_mask` from transformers is used for the left-padded eval batches --
  verify padded == unpadded outputs in a smoke test whenever transformers is bumped.

`scripts/train_recurrent_depth.py` is the training/eval loop that uses these.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

name = "recurrent_depth"

PROMPT_TEMPLATE = "Question: {question}\nAnswer:"
ANSWER_TEMPLATE = " #### {answer}"

# Paper defaults (Sec. 3.3): r_bar = 32, sigma = 1/2, backprop through last k = 8.
DEFAULT_MEAN_RECURRENCE = 32
DEFAULT_LOGNORMAL_SIGMA = 0.5
DEFAULT_BACKPROP_LAST_K = 8
DEFAULT_INIT_STATE_STD = math.sqrt(2 / 5)


def build_example_ids(tokenizer: Any, question: str, answer: str) -> tuple[list[int], list[int]]:
    """`(input_ids, labels)` for one training example: prompt + answer + EOS, `labels`
    masks (`-100`) the prompt span. Same format as `filler_tokens` at compute_steps=0."""
    prompt_ids = tokenizer(PROMPT_TEMPLATE.format(question=question), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(ANSWER_TEMPLATE.format(answer=answer), add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    input_ids = prompt_ids + answer_ids + [eos]
    labels = [-100] * len(prompt_ids) + answer_ids + [eos]
    return input_ids, labels


def build_eval_prompt_ids(tokenizer: Any, question: str) -> list[int]:
    return tokenizer(PROMPT_TEMPLATE.format(question=question), add_special_tokens=False)["input_ids"]


def sample_num_recurrences(
    rng: np.random.Generator,
    mean_recurrence: float = DEFAULT_MEAN_RECURRENCE,
    sigma: float = DEFAULT_LOGNORMAL_SIGMA,
) -> int:
    """One draw of r from the paper's log-normal Poisson (Eq. 1-2):
    `tau ~ N(log(r_bar) - sigma^2/2, sigma)`, `r = Poisson(exp(tau)) + 1`.
    E[exp(tau)] = r_bar, so E[r] = r_bar + 1. Call once per training batch."""
    tau = rng.normal(math.log(mean_recurrence) - 0.5 * sigma**2, sigma)
    return int(rng.poisson(math.exp(tau))) + 1
