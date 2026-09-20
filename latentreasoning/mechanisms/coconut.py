"""Coconut (horizontal continuous-thought): a multi-stage curriculum that replaces
explicit CoT reasoning steps, one at a time, with `<|latent|>` tokens whose embedding
is the model's own last-layer hidden state from the position before them -- so the
"thought" is spliced back in as an input, in-sequence, rather than generated via a
separate loop the way CODI's is. arXiv:2412.06769 (Hao et al.); reference
implementation github.com/facebookresearch/coconut.

Owner's notes (Henning, 2026-09-19). Added to extend the causal-patching /
decodability sweep already run on CODI to the other width-based LRM this project
studies, specifically to compare against Dilgren & Wiegreffe (COLM 2026, "Are Latent
Reasoning Models Easily Interpretable?", arXiv:2604.04902) -- their Finding 2 claims
Coconut+GPT2 encodes gold reasoning traces in vocabulary-projected latents 54-93% of
the time when correct, a much higher decodability rate than CODI's near-zero odd
positions in our own repo. Like CODI, nothing is re-implemented here: uses the
released checkpoint `connordilgren/gpt2-gsm8k-coconut` (checkpoint_33) from Dilgren &
Wiegreffe's own HF collection (their fork only adds attention-ablation hooks on top of
`facebookresearch/coconut`'s `Coconut` module -- the state dict loads unchanged into
the vanilla class, see `scripts/coconut_common.py`). No training run in this repo --
their own paper's stage-0-CoT-then-curriculum recipe (`~/Projects/coconut`) is
available if a from-scratch reproduction is ever needed, but the released checkpoint
is the paper-comparable one and is what all logged `coconut` runs so far use.

`compute_steps` = number of `<|latent|>` tokens in the sequence between
`<|start-latent|>` and `<|end-latent|>` = 6 at this checkpoint's final curriculum
stage (`c_thought=2` continuous thoughts per reasoning step x `max_latent_stage=3`
stages, matching CODI's 6 and the paper's own Table 1 for GSM8k-Aug). Each one is a
chunked forward pass over the growing sequence (not a fresh single-token step like
CODI's loop) -- KV cache is trimmed and reused between passes exactly as
`facebookresearch/coconut`'s `Coconut.forward` does; see `scripts/coconut_common.py`
for the adapted version with donor-interchange override support.

Gotchas:
- Checkpoint state dict keys are `base_causallm.*` (verified: `missing=0 unexpected=0`
  loading into a plain `GPT2LMHeadModel` after `resize_token_embeddings`) plus a
  redundant `embedding.weight` (same tensor as `wte`, since `self.embedding` is just a
  Python reference to an existing submodule, not new params) -- only load the
  `base_causallm.*` keys.
- Needs `torch==2.5.1` / `transformers==4.46.2` (Coconut's own pin) in a dedicated
  venv, NOT this repo's `--extra train` env: newer `transformers` (verified against
  5.17.0 locally) returns a `Cache` object from `past_key_values` with no
  `to_legacy_cache()`/`__getitem__`, but ALSO refuses legacy tuples when passed back
  in (`create_causal_mask` requires a real `Cache`) -- neither direction works outside
  the pinned version. `scripts/coconut_common._to_legacy_cache` handles the object ->
  tuple direction for forward-compatibility, but was only verified end-to-end
  (checkpoint load + 6-latent generate, 2/5 correct on a real slice, consistent with
  the paper's 33.1%) inside a pinned venv.
- Answer delimiter is `###`, not CODI's "The answer is:" -- `extract_answer_after_delimiter`
  in `coconut_common.py`, matching the Dilgren & Wiegreffe repo's own `extract_answer`.
- Gold step values come from `<<a op b=c>>`-style `steps` entries in
  `gsm_valid-gold-reasoning-trace_test.json` (Dilgren & Wiegreffe's data prep, same
  underlying GSM8K-Aug corpus/questions as our own `gsm8k_aug.py` loader, different
  distribution channel -- da03/Internalize_CoT_Step_by_Step text files rather than the
  `zen-E` HF parquet) -- generate locally via
  `~/Projects/are-lrms-easily-interpretable/preprocessing/prepare_gsm8k.py` (the
  Llama-tokenizer step in that script fails without gated Llama access; irrelevant to
  the GPT-2-only files we use, they're written before that step runs).
"""
from __future__ import annotations

name = "coconut"
