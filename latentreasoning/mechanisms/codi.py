"""CODI (horizontal continuous-thought): self-distillation between an explicit-CoT
teacher pass and a continuous-thought student pass sharing one model, aligning hidden
activations on the final-answer token. arXiv:2502.21074; reference implementation
github.com/zhenyi4/codi (GPT-2 on GSM8K-Aug is the paper's own setting -- reproduce
that number first).

Owner's notes (Henning, 2026-09-18). Nothing is re-implemented here on purpose: the
reference repo *is* the implementation, run verbatim from an isolated venv --
`scripts/codi_setup.sh` (pinned commit + their requirements.txt + one data-loading
patch), `scripts/train_codi.sh` (their `train_gpt2_gsm8k-aug.sh`, deviations listed in
the header), `scripts/eval_codi.py` (their `test.py` model loading + greedy decode,
scored through `score_outputs` on the shared slice).

`compute_steps` = number of continuous-thought ("latent") tokens autoregressively
generated between `<bot>` and `<eot>` at inference = `--inf_latent_iterations` (6 in
the paper, matching `--num_latent 6` at training). Each one is a full forward pass
whose last-layer hidden state (through a 768-d projection + LayerNorm) is fed back as
the next input embedding -- so 6 latents ~ 6 extra single-token forward passes with KV
cache, comparable in FLOPs to 6 filler tokens, not to 6 recurrent-depth loops (see the
compute-budget rule in the record-run skill before quoting one against the other).

Paper <-> repo cross-reference (paper Appendix A vs `scripts/train_gpt2_gsm8k-aug.sh`
at commit 2c23146, checked 2026-09-18): identical. lr 3e-3, 40 epochs, effective batch
128 (64 x 2 accum), LoRA r=128 alpha=32 on c_attn/c_proj/c_fc (dropout 0.1 hardcoded
in train.py), bf16, AdamW + cosine + 3% warmup, weight decay 0.1, grad clip 2.0,
6 latents, projection 768 + LN, smooth-L1 distill loss / teacher std, alpha=beta=gamma=1,
seed 11. Reported: 43.7% GSM8K test (greedy, 6 latents); Table A5: 38.4% at 20 epochs.
~36h on one A100 80GB. Released weights: HF `zen-E/CODI-gpt2` (a single
`pytorch_model.bin` in exactly the format their `test.py` loads).

Gotchas:
- Their `load_dataset("zen-E/GSM8k-Aug")` hits the same datasets/pyarrow false-positive
  ArrowInvalid as our loader did; `scripts/codi_streaming.patch` adds `streaming=True`
  (semantics unchanged). Their eval loads `gsm8k/main` test instead -- same 1319
  problems as GSM8k-Aug's test file; `eval_codi.py` records any text mismatch.
- The paper script only saves weights at the very end (`--save_strategy no`); ours
  checkpoints every 3000 steps so a pod restart doesn't cost 36h.
- Known upstream attention-mask quirk (`--fix_attn_mask False` by default, and in the
  paper): after the question is encoded with a mask, the latent / answer steps pass
  `attention_mask=None`, so left-padding IS attended in batched inference. Outputs
  therefore depend on batch composition -- `eval_codi.py` uses their exact protocol
  (file order, batch 128) so the number is theirs; don't re-batch and expect identical
  strings.
- `<bot>`/`<eot>`/`[PAD]` are appended to the vocab and, with LoRA freezing the base,
  their embeddings are never trained -- random-init and frozen. That's the paper's
  setup; don't "fix" it in a reproduction.
- Paper-default per-device batch 64 OOMs on 16GB (three full-sequence forwards per
  step + LoRA); the A100 80GB it was tuned on is the safe choice for a faithful run.
- The reference `test.py` samples by default; `test_gpt2.sh` passes `--greedy True`,
  which is the reported protocol. `extract_answer_number` = last number in the output,
  same convention as `latentreasoning.eval.metrics`.
"""
from __future__ import annotations

name = "codi"
