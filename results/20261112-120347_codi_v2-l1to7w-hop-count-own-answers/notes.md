## 2026-11-12 — RuleChain hop count with own-answer qualification (t31): wrong answers keep the staircase in gated / SIM-CODI; start-last CODI and the relay model cross more on wrong items, not less; stabilised CODI still ignores its latents (codi, run_id: 20261112-120347_codi_v2-l1to7w-hop-count-own-answers)

**Goal:** t31 (pre-registered 2026-11-11 23:30): remove t26's selection on correct items. A pair counts if the model's own answers to recipient and donor differ, right or wrong; hops = first refresh from which the answer stays the model's own donor answer.
**Mechanism / model:** the six RuleChain CODI models of t26 (20261112-060438_codi_v2-l1to7w-hop-count-six-models), same checkpoints.
**Data:** valid L2-L6, the t26 minimal pairs (2,000 candidates per model, 80 recipients per level x edit cell). Qualified (own answers differ) / any-wrong: startlast 1735 / 1399, relay 1926 / 938, stab 1912 / 1078, gated 1978 / 273, sim 1966 / 311, simsf 1980 / 383.
**Command:** `QUAL=own ../codi/.venv/bin/python scripts/necessity/hops.py <model> mps <out>` (queue `codi_audit_artifacts/t31/queue.sh`).
**Headline results** (mean hops: all / both right / any wrong):
- gated 2.16 / 2.23 / 1.69; sim 2.77 / 2.84 / 2.41; simsf 2.95 / 2.97 / 2.86. Wrong pairs keep the staircase (gated s - 1, sim / simsf about s in every bin).
- startlast 1.34 / .79 / 1.47; wrong pairs at s = 4 / 5 / 6: 1.80 / 1.86 / 1.91 (n 254 / 180 / 121).
- relay 1.15 / .88 / 1.43; wrong answers use .3-.6 more hops than right ones at s = 2-3.
- stab .20 / .11 / .26; F_0 on wrong pairs .74 / .75 at s = 5 / 6.
**Predictions:** both-right reproduces t26 within .10: held for gated / sim / simsf / startlast, missed for relay (.11) and stab (.20, s = 6, n = 22; the both-right bins are a smaller subsample than t26's). startlast <= 1.6 missed; its low-confidence alternative (wrong uses more hops) held. relay within +-.30 missed at s = 2-3. stab F_0 >= .80 missed at s = 5-6. gated / sim / simsf "too few wrong pairs" missed (46-104 per bin at s <= 4); all-pairs within .15 of t26 held.
**Interpretation:** Selection on correctness did not create the staircase: gated and SIM-CODI cross once per rule on items they get wrong too. Start-last CODI and the relay model do not fail deep items by skipping crossings; their wrong answers use slightly more crossings than their right ones but stay near two, far below s.
**Gotchas hit:** none; M_6 = donor run on every pair.
**Caveats:** one seed per model; hops is an upper bound on serial steps; any-wrong pairs mix "recipient wrong" and "donor wrong".
**Next:** t34 hidden-path detection and t35 layer-wise reach on the same models.
