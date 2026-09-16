"""Recurrent depth (vertical): loop a recurrent Transformer block r times per token
instead of stacking r layers. Geiping et al. arXiv:2502.05171; reference implementation
github.com/seal-rg/recurrent-pretraining (their huginn-0125 checkpoint is 3.5B -- too
big to reuse; this needs its own small looped GPT-2). Highest-risk reproduction, so it
goes first (see PROPOSAL.md).

Owner's notes go here. Which block(s) loop, how r (`compute_steps`) is chosen, and
weight-tying across iterations are the owner's decisions.

Before deciding how to decode intermediate states, see "Looped Transformers under the
Jacobian Lens" (PROPOSAL.md references): loop iterations largely overwrite rather than
accumulate state.
"""
from __future__ import annotations

name = "recurrent_depth"
