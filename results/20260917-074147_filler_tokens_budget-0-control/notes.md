## 2026-09-17 — Pilot: filler_tokens no-filler control (filler_tokens, run_id: 20260917-074147_filler_tokens_budget-0-control)

**Goal:** First reproduction pass for the filler_tokens mechanism (Pfau, Merrill &
Bowman 2024) on GPT-2 / GSM8K-Aug — verify the train/eval pipeline works end to end on
RunPod and get an initial read on whether filler tokens help over no scratchpad at
all, before committing to a full-scale run.
**Mechanism / model:** `filler_tokens`, gpt2 / results/20260917-074147_filler_tokens_budget-0-control/ckpt, compute_steps=0.
**Data:** gsm8k-aug test n=200 seed=0; run seed=42.
**Hyperparams:** {'lr': 5e-05, 'epochs': 3, 'batch_size': 16, 'filler_token': '.', 'train_n': 20000}
**Command:** `uv run python scripts/train_filler_tokens.py --compute-steps 0 --train-n 20000 --stage pilot --hardware "RunPod RTX A4000 (community)" --slug budget-0-control`
**Headline results:** `final_answer_accuracy=0.010`, `unparseable_rate=0.000`, `compute_steps=0`
**Interpretation:** No-filler control at pilot scale — 2/200 correct. Compare against
`20260917-074839_filler_tokens_budget-32` (same scale, compute_steps=32: 2.5%) — at
this pilot scale filler tokens looked like they helped. **But see the full-scale
runs** (`*-full`): that ordering reverses once trained on the full dataset, so this
pilot gap should be read as noisy, not a confirmed effect.
**Gotchas hit:** `load_gsm8k_aug` had two real bugs discovered and fixed during this
run (not filler_tokens-specific — affects every mechanism using the shared loader):
(1) `datasets`' pyarrow batch JSON reader throws a false-positive `ArrowInvalid` on
the train file; fixed with `streaming=True`. (2) the HF repo's *test* split ships as a
differently-shaped file (one JSON object of column arrays, not JSON-lines) and needed
a dedicated loader path (`_load_test_split`). See `latentreasoning/data/gsm8k_aug.py`.
**Caveats:** Pilot scale only — 20,000 of 384,620 available train examples (~5%), 3
epochs. Not representative of the mechanism's real ceiling.
**Theory (added 2026-09-18):** the full pattern across all filler_tokens runs this
session (this pair, the full-scale pair, and the paper-faithful pilot pair) is
explained by a structural mismatch between GSM8K-Aug and the task class the filler-
tokens paper actually tested — see
`20260917-190640_filler_tokens_budget-32-full`'s notes for the full writeup, or
`HANDOFF.md`'s "Paper-faithful methodology audit" section for the citation trail.
**Next:** Full-scale run done, see `20260917-163049_filler_tokens_budget-0-control-full`.
