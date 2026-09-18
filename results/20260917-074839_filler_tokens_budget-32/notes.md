## 2026-09-17 — Pilot: filler_tokens compute_steps=32 (filler_tokens, run_id: 20260917-074839_filler_tokens_budget-32)

**Goal:** Same pilot pass as the compute_steps=0 control, at the config's default
nonzero sweep point (32 forced filler tokens) — initial read on whether filler tokens
help over no scratchpad at all, before committing to a full-scale run.
**Mechanism / model:** `filler_tokens`, gpt2 / results/20260917-074839_filler_tokens_budget-32/ckpt, compute_steps=32.
**Data:** gsm8k-aug test n=200 seed=0; run seed=42.
**Hyperparams:** {'lr': 5e-05, 'epochs': 3, 'batch_size': 16, 'filler_token': '.', 'train_n': 20000}
**Command:** `uv run python scripts/train_filler_tokens.py --compute-steps 32 --train-n 20000 --stage pilot --hardware "RunPod RTX A4000 (community)" --slug budget-32`
**Headline results:** `final_answer_accuracy=0.025`, `unparseable_rate=0.000`, `compute_steps=32`
**Interpretation:** 5/200 correct vs. the control's 2/200 at this pilot scale — looked
like filler tokens more than doubled accuracy. **This did not hold up at full scale**
(see `20260917-190640_filler_tokens_budget-32-full`, 12.0% vs. the full-scale
control's 13.0% — the ordering flips). At n=200 with single-digit correct counts,
this pilot gap (2 vs 5) is not a reliable signal on its own.
**Gotchas hit:** Same `load_gsm8k_aug` bugs as the compute_steps=0 pilot run — see that
run's notes.md for detail.
**Caveats:** Pilot scale only — 20,000/384,620 train examples (~5%), 3 epochs.
**Theory (added 2026-09-18):** see `20260917-190640_filler_tokens_budget-32-full`'s
notes for the full explanation of why the null holds up across pilot/full/faithful —
short version: GSM8K-Aug arithmetic chains are a sequential/instance-adaptive task
(each step needs the previous step's numeric result), and the paper's own reference
code tests exactly this task shape (`dot_filler_serial`/`serial_cot` in their
`src/match3.py`) and reports filler tokens fail on it too, staying at baseline. Our
result replicates that, in a new domain (natural language) and model (pretrained
GPT-2 vs. their from-scratch ~30M-param model).
**Next:** Full-scale run done, see `20260917-190640_filler_tokens_budget-32-full` — that's
the number that actually matters for comparing against the paper.
