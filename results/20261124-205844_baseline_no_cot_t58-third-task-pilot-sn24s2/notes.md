## 2026-11-24/25 — t58 third-task pilot, sn24s2: P7 held: second seed reach 2.50, same verdict. (baseline_no_cot, t58)

**Goal:** advisor asked for a task orthogonal in content with the same serial structure. t58 measures single-pass reach of pretrained GPT-2 (proxy for CODI-0) on candidate tasks before any CODI training. Pre-registered in scripts/codi_null_audit/prereg/t_predictions.md (t58, 2026-11-24 22:17:14 +07; follow-ups 2026-11-25 00:14:34 and 00:36:08; deviation 00:30:55).
**Task:** sn24 with a second data seed, no CoT.
**Command:** `../codi/.venv/bin/python scripts/necessity/third_task_pilot.py sn24s2 mps ../codi_audit_artifacts/t58/sn24s2.json 4000 5` (from ai-capstone/); scorer `scripts/necessity/third_task_score.py`.
**Gate:** pass if accuracy at k = 1 >= .90 and 1.5 <= reach <= 3.5; reach = interpolated k where final-epoch test accuracy crosses the midpoint between chance and accuracy at k = 1.
**Results (final epoch, test, k = 1-8):** 1.000 / 1.000 / 0.007 / 0.013 / 0.007 / 0.013 / 0.007 / 0.023; chance 0.008; reach 2.5.
**Verdict:** P7 held: second seed reach 2.50, same verdict.
**Caveats:** single pilot run per setting (sn24 has a second seed); no per-item outputs or checkpoints saved; `predictions.jsonl` holds per-(epoch, split, k) accuracies. Reference: RuleChain start-last CODI-0 reach ~2.6, start-first ~5.
