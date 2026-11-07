## 2026-11-07 — SIM-CODI start-last on the canonical L1-L12 test: L1-L7 as the main record, L8 0.006; out of range it mostly applies rules 1-6 plus the last rule (65% at L8, falling to 16% at L12) (codi, run_id: 20261107-225709_codi_v2-l1to7w-startlast-sim-depth2-codi-canonical)

**Goal:** Secondary check on L8-L12, levels this model never trained on (TRAINING_PROTOCOL sections 11.2 and 13).
**Mechanism / model:** As 20261107-225603 (SIM-CODI, start-last, checkpoint-16335).
**Data:** `synthetic_data_v2_startlast`: canonical L1-L12 test, 500/level, start value after the rules. L1-L7 items are the main record's, item for item.
**Command:** `scripts/pod/codi_stab_arm.sh`; `eval_command.txt`. The rule-subset check ran on the laptop from `predictions.jsonl`: parse each chain from the question (`parse_chain_question`, 0 gold mismatches), then for every split "first k rules + last m rules" ask whether the prediction equals that subset's final value.
**Headline results:**
- **Test, exact:** L1-L6 1.000, L7 0.998, L8 0.006, L9 0.070, L10 0.014, L11 0.014, L12 0.030; overall 0.594, d50 7.22, AUC 0.602. Start-last CODI on L8-L12: .054, .038, .036, .010, .018. The CoT teacher in the same model: L8 .622, L9 .356, L10 .186, L11 .082, L12 .086.
- **What it outputs out of range:** the final value of rules 1-6 plus the last rule on 65% of L8 items, 42% at L9, 26% at L10, 19% at L11, 16% at L12. Rules 1-7 only: 32%, 20%, 22%, 20%, 22%. A random subset matches about 10% by coincidence. Rule-gated start-last CODI (20261107-204918) gives rules 1-6 plus the last rule on 97-99% at every level; start-last CODI shows no such pattern.

**Interpretation:** The drop from 0.998 to 0.006 sits at the edge of training, not at a reach limit, so d50 7.22 marks the training range and should not be quoted as reach. The output pattern says how this model splits the work: at L8 it mostly runs six rules and then applies the last rule in the answer pass, which is the split the gate forces and fits one rule per latent. Unlike the gated model it does this less cleanly and less often as chains grow, and a third of L8 answers instead stop after rule 7. Behavioural evidence only; the latent ablation, donor and probe checks on the main record test it directly.
**Gotchas hit:** None.
**Caveats:** One seed. Subsets overlap (two subsets can give the same value), so the shares need not sum to 1. The CoT teacher also fails past L7, so length generalisation is weak for every arm here.
**Next:** None for this record; see the main record.
