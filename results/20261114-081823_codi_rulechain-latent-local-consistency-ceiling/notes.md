## 2026-11-14 — Readout ceiling for the local-consistency test (t43c): the t43b null is real for SIM models, readout-limited for low crossers (codi, run_id: 20261114-081823_codi_rulechain-latent-local-consistency-ceiling)

**Goal:** t43c: the same link test on correctly answered items, where the true values are known, gives the consistency a faithful chain reaches given readout error.
**Mechanism / model:** simsf, simcoconut_rc, startlast, relay.
**Data:** correct items of valid + test L2-L7.
**Command:** `CEIL=1 LOCAL=1 ../codi/.venv/bin/python -W ignore scripts/necessity/faithful.py <model> mps codi_audit_artifacts/t43c/<model>.json` (parallel runner `codi_audit_artifacts/t44/run_one.sh`).
**Headline results:** compute links inside C: simsf .63, simcoconut_rc .57 (chance .01); startlast .05, relay .06. simsf by link .72 / .61 / .59 / .49 / .43 / .34 (z1->z2 .. z6->z7).
**Interpretation:** Pre-registered rule: for simsf and simcoconut_rc the wrong-item rates (.13, .04) are far below a reachable ceiling, so their crossed latents' value readouts do not track the model's own errors. For startlast and relay the ceiling is ~.06, so per-item faithfulness at their crossings is untested.
**Gotchas hit:** none.
**Caveats:** value readouts only; branch readouts are tested in t45.
**Next:** t45.
