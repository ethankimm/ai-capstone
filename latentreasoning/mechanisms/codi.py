"""CODI (horizontal continuous-thought): self-distillation between an explicit-CoT
teacher pass and a continuous-thought student pass sharing one model, aligning hidden
activations on the final-answer token. arXiv:2502.21074; reference implementation
github.com/zhenyi4/codi (GPT-2 on GSM8K-Aug is the paper's own setting -- reproduce
that number first).

Owner's notes go here. How many continuous-thought tokens count as `compute_steps`,
and how CODI's training adapts to the shared `gpt2` backbone, are the owner's
decisions.
"""
from __future__ import annotations

name = "codi"
