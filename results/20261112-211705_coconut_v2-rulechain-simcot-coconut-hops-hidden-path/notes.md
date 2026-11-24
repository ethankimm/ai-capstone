## 2026-11-12 — Kevin's RuleChain SIM-Coconut (t37): one crossing per step and then some (F_0 = 0, hops s + .3-.6, capped at 6 latents), latents reveal middle-rule branches (+.36 over question + answer) (coconut, run_id: 20261112-211705_coconut_v2-rulechain-simcot-coconut-hops-hidden-path)

**Goal:** t37 (pre-registered): hop count and hidden-path detection for the one Coconut-stack RuleChain model whose latents beat single-pass reach.
**Mechanism / model:** pilot_k8_simcot_c1m6/checkpoint_26 (Coconut c = 1, M = 6, SIM-CoT step loss; record 20261104-144235). Loader check: 360 / 360 logged predictions reproduced.
**Data:** synthetic-depth2 test L2-L6 (start value first), 1,937 qualified pairs (own answers differ; 786 with a wrong answer). Hidden path: fit on depth2 train (2,000 per level), scored on test L4-L6.
**Command:** `QUAL=own ... hops.py simcoconut_rc mps <out>`; `... hidden_path.py simcoconut_rc mps <out>` (queue `codi_audit_artifacts/t37/queue.sh`).
**Headline results:**
- Mean hops s = 1-6: 1.61 / 2.56 / 3.46 / 4.32 / 5.32 / 5.99; F_0 = 0 in every bin; rule-1 edits need L + 1 crossings (L4 4.54); s = 6 at the 6-latent cap.
- Wrong pairs: same staircase (3.71 vs 3.28 both right).
- Hidden path: question + answer .50-.70; best latent set .92-.98; mean gain +.36; shuffled <= .55.
**Predictions:** all held (QA <= .70 at the edge, .700).
**Interpretation:** The clearest staircase of any model: no answer changes with fewer crossings than steps. With t36 this extends "per-pass structure sets the hop count" to the Coconut stack. The cap at 6 latents matches the accuracy collapse at k = 7.
**Gotchas hit:** checkpoint arrived as checkpoint_26.zip (a torch zip file, loads directly).
**Caveats:** one seed; under-trained flag in the original record.
**Next:** per-position probes (t38c, done).
