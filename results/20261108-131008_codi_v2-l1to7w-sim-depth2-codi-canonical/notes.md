## 2026-11-08 — SIM-CODI start-first on the canonical L1-L12 test: L1-L7 as the main record, L8 .006; out of range it mostly runs rules 1-6 plus the last rule (43% at L8) or stops after rule 7 (37%) (codi, run_id: 20261108-131008_codi_v2-l1to7w-sim-depth2-codi-canonical)

**Goal:** Secondary check on L8-L12, levels this model never trained on (TRAINING_PROTOCOL sections 10.3 and 14).
**Mechanism / model:** As 20261108-130905 (SIM-CODI, start-first, checkpoint-17820).
**Data:** `synthetic_data_v2`: canonical L1-L12 test, 500/level, start value first. L1-L7 items are the main record's, item for item.
**Command:** `scripts/pod/codi_stab_arm.sh`; `eval_command.txt`. Out-of-range readings on the laptop from `predictions.jsonl`: `codi_audit_artifacts/sim_ood.py` (does the prediction equal an intermediate value v_k) and `sim_subsets.py` (parse each chain, `parse_chain_question`, 0 gold mismatches; for every split "first k rules + last m rules", does the prediction equal that subset's final value).
**Headline results:**
- **Test, exact:** L1-L7 as the main record (.946-1.000), L8 .006, L9 .078, L10 .012, L11 .010, L12 .018; overall .586, d50 7.19, AUC .592. Start-first CODI on L8-L12: .078, .038, .046, .014, .016. The CoT teacher in the same model: L8 .472, L9 .358, L10 .184, L11 .120, L12 .072 (start-first CODI's teacher .988, .684, .164, .052, .036).
- **What it outputs out of range:** rules 1-6 plus the last rule on 43% of L8 items, 32% at L9, 22% at L10, 14% at L11, 9% at L12; rules 1-7 only on 37%, 20%, 21%, 16%, 9%. Prediction equals v7 on 37% at L8; matches no intermediate on 55-77% at L8-L12. Start-last SIM-CODI (20261107-225709): rules 1-6 plus last on 65% at L8.

**Interpretation:** As on start-last, the drop from .946 to .006 sits at the edge of training, so d50 7.19 marks the training range, not reach. The out-of-range pattern is the start-last SIM-CODI pattern, weaker: mostly six rules then the last rule, or stop after rule 7, which fits one rule per latent plus one in the answer pass. Behavioural only; the latent battery on the main record tests it directly.
**Gotchas hit:** None.
**Caveats:** One seed. Subsets overlap, so shares need not sum to 1. The CoT teacher also fails past L7, and here worse than start-first CODI's teacher at L8-L9 (.472 vs .988), so length generalisation is weak in every arm.
**Next:** None for this record; see the main record.
