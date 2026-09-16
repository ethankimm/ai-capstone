## 2026-09-16 — Repo bring-up: sanity-floor baseline (baseline_last_number_in_question, run_id: 20260916-045350_baseline_last_number_in_question_gsm8k-aug-sample)

**Goal:** validate the data-loading → eval-harness → run-manifest pipeline end to end
before spending any compute, and establish a floor every real mechanism should clear.
**Mechanism / model:** `baseline_last_number_in_question` (predict the last number in
the question text) — no model, no training, `compute_steps=0`.
**Data:** GSM8K-Aug test, bundled offline sample (`latentreasoning/data/gsm8k_aug_sample.jsonl`,
30 real examples), seed=0 (unused — sample is fixed).
**Hyperparams:** none.
**Command:** `uv run python scripts/run_baseline.py`
**Headline results:** `final_answer_accuracy=0.033` (1/30), `unparseable_rate=0.000`.
**Interpretation:** floor is ~0 as expected — a word problem's last mentioned number is
essentially never its answer. Confirms answer extraction and scoring aren't silently
miscounting, and gives later runs a concrete floor.
**Gotchas hit:** initial `NUMBER_RE` (`-?\d[\d,]*\.?\d*`) matched a trailing bare "."
(pulled "7." out of "...giving 7."); fixed by requiring a digit after the decimal point.
**Caveats:** n=30; a sanity check, not a reportable result. First run on Sep 13
against the pre-squash scaffold; re-logged here on the initial commit (same data,
same deterministic result).
**Next:** `explicit_cot` or `filler_tokens` on GPT-2 + real GSM8K-Aug on RunPod, logged
against this floor.
