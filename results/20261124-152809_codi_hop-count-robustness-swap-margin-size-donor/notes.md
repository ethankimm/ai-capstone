## 2026-11-24 — Hop-count robustness (t57): the count survives swapping A and B, a margin threshold, edit size and an unrelated donor for the chaining models; for start-last CODI it is noisier but every disagreement is downward (codi, t57)

**Goal:** t57 (pre-registered with t56): is the count itself robust to how it is measured?
**Command:** `../codi/.venv/bin/python scripts/necessity/validity.py <model> mps robust ../codi_audit_artifacts/t56/<model>_robust.json`; scoring as t56. The rule-gated run was paused 14:38-19:03 (SIGSTOP, battery), results unaffected.
**Results:** repro equals t31 on 1.00 of pairs. Swap within 1: 1.00 / 1.00 / .857; means identical (2.16 / 2.77 / 1.34). Margin hop equals argmax hop on .984 / .975 / .657, and where it differs it is always lower. Onset earlier, never later, for s > 1 (111 / 576 / 584 pairs). Start-last half hop 1.04-1.19 at every s. Large minus small edits +.01 / +.10 / +.28. Unrelated donor within .10 of the start-edit count (chaining models); start-last <= 1.96.
**Predictions:** P0 held; P1 missed for start-last and on the strict equality bar for rule-gated (.938); P2 held for the chaining models' margin agreement, missed for start-last and for SIM-CODI's onset; P3 held for the chaining models, missed for start-last; P4 held.
**Interpretation:** the argmax count is an upper bound, as the paper states; by the margin measure start-last CODI uses about one crossing. SIM-CODI's onset one refresh before the flip fits its count lying one above the rule gate's.
**Caveats:** single seeds; post hoc reading of the disagreement direction.
