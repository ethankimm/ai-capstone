## 2026-11-14 — Local consistency of successive latents on wrong items (t43b): crossed latents are not more consistent than later ones (codi, run_id: 20261114-081823_codi_rulechain-latent-local-consistency-wrong-items)

**Goal:** t43b: fix t43's confound. On wrong items, does the value read at z_(j+1) follow from the value read at z_j (rules in between applied), and more so inside the item's crossings?
**Mechanism / model:** same seven RuleChain models and probes as t43.
**Data:** wrong items of valid + test L2-L7; links where at least one side differs from gold.
**Command:** `LOCAL=1 ../codi/.venv/bin/python -W ignore scripts/necessity/faithful.py <model> mps codi_audit_artifacts/t43b/<model>.json`.
**Headline results:** compute links (next latent reads a later step) inside vs outside C: simsf .13 vs .18 (n 238 / 77), simcoconut_rc .04 vs .10, startlast .03 vs .03, relay .03 vs .01; chance .01-.02. Copy links beyond C .47-.71 (persistence). gated 4 and sim 10 links: not scorable.
**Interpretation:** No sign that the crossed latents carry the model's own (wrong) chain; t43c shows this is not just readout error for simsf and simcoconut_rc.
**Gotchas hit:** copy links (same step read again) dominate the registered all-links score; split by kind added before the full run (prereg addendum).
**Caveats:** value probes trained on correct runs; level-mean C.
**Next:** t43c ceiling; t45 branches.
