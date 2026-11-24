## 2026-11-12 — Hop-guided reading (t38): the item's own hop count predicts when a value becomes readable in rule-gated (.99 vs .76) and relay CODI (.90 vs .39) but not in SIM-CODI; branch information sits at pass k - 1 in all four staircase models (codi, run_id: 20261112-211705_codi_v2-rulechain-hop-guided-timing-position)

**Goal:** t38 (pre-registered): does the hop count tell a monitor when and where to read, beyond reading all latents?
**Mechanism / model:** the t26 RuleChain CODI models (t38a/b, offline); gated, sim, simsf and Kevin's SIM-Coconut (t38c).
**Data:** t38a/b: t29 readability records (t26/rcread_*.json) joined one to one to their t26 pairs. t38c: RuleChain L4-L6, as t34 / t37.
**Command:** `python3 scripts/necessity/hop_guided.py <out>`; `PER_POS=1 ... hidden_path.py <model> mps <out>` (queue `codi_audit_artifacts/t38/queue.sh`).
**Headline results:**
- t38a, first-read refresh = structural d vs = hop-aligned h_k: gated .76 vs .99 (disagreements, n 707: .01 vs .99; within-cell r .96); relay .39 vs .90; sim .67 vs .32 (.66 with one more crossing, post hoc); simsf .68 vs .29 (.67).
- t38b: the newly refreshed latent z_(n_r - 1) reads the value, where legible: gated 1.00, sim .89, simsf .90, relay .77 (unconditional .24-.61).
- t38c: the best single position for rule k's branch is pass k - 1 in 35 of 36 cells (Spearman 1.0 at every level, all four models); gated H_(k-1) = 1.00; single position within .05 of all latents except three SIM cells.
**Predictions:** t38a held for gated, missed for sim / simsf; t38b held; t38c held except the within-.05 check in three cells.
**Interpretation:** Where to look: in staircase models, rule k is decided in pass k - 1, the position a one-crossing-per-step count implies, and one position carries nearly all of it. When: the item's own hop count beats chain structure where the two differ (gated start edits, relay), but in SIM models it adds nothing beyond structure.
**Gotchas hit:** rc_readability.py stores no pair ids; the join rebuilds its pair selection and checks every (L, k, A, B).
**Caveats:** t38b is partly built in (earlier sites were already refreshed); linear probes; one seed per model.
**Next:** paper 5.3 / 5.6.
