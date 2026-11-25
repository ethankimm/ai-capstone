## 2026-11-24/25 — t58 third-task pilot, sn: FAILS the gate: reach 3.87 > 3.5, inflated by memorisation (216 sequences at k = 3; train .62 vs test .43 at k = 4). P1 held. (baseline_no_cot, t58)

**Goal:** advisor asked for a task orthogonal in content with the same serial structure. t58 measures single-pass reach of pretrained GPT-2 (proxy for CODI-0) on candidate tasks before any CODI training. Pre-registered in scripts/codi_null_audit/prereg/t_predictions.md (t58, 2026-11-24 22:17:14 +07; follow-ups 2026-11-25 00:14:34 and 00:36:08; deviation 00:30:55).
**Task:** S5 word problem over 6 fixed generators (Liu et al. 2023; Merrill et al. 2024), 120 single-token labels, no CoT.
**Command:** `../codi/.venv/bin/python scripts/necessity/third_task_pilot.py sn mps ../codi_audit_artifacts/t58/sn.json 4000 5` (from ai-capstone/); scorer `scripts/necessity/third_task_score.py`.
**Gate:** pass if accuracy at k = 1 >= .90 and 1.5 <= reach <= 3.5; reach = interpolated k where final-epoch test accuracy crosses the midpoint between chance and accuracy at k = 1.
**Results (final epoch, test, k = 1-8):** 1.000 / 1.000 / 1.000 / 0.433 / 0.083 / 0.023 / 0.010 / 0.003; chance 0.008; reach 3.87.
**Verdict:** FAILS the gate: reach 3.87 > 3.5, inflated by memorisation (216 sequences at k = 3; train .62 vs test .43 at k = 4). P1 held.
**Caveats:** single pilot run per setting (sn24 has a second seed); no per-item outputs or checkpoints saved; `predictions.jsonl` holds per-(epoch, split, k) accuracies. Reference: RuleChain start-last CODI-0 reach ~2.6, start-first ~5.
