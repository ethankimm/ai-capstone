## 2026-11-12 — GSM8K hop count with own answers and first-token scoring (t31 + t32), released Coconut: hops 1.5, falling with depth; answer tokens add no crossings (coconut, run_id: 20261112-120347_coconut_gsm8k-grid-hop-count-own-answers-first-token)

**Goal:** as the CODI / SIM record of the same run (t31 own-answer qualification, t32 first answer token), for the released GSM8K Coconut.
**Mechanism / model:** connordilgren grid Coconut GPT-2 GSM8K (grid_checkpoints), fp32 MPS, batch 1.
**Data:** t22 twins (1,471) and t27a long twins (1,305). Qualified (own answers differ): 1238 + 1109 (t26: 220 + 42).
**Command:** `QUAL=own FIRST=1 ../codi/.venv/bin/python scripts/necessity/hops.py gsm8k_coconut mps <out>`; long twins with `CANDS=t26/gsm8k_long_twins.json`.
**Headline results:**
- Both-right reproduces t26 (1.46 vs 1.45; long twins identical).
- Mean hops by s: 1.38 / 1.56 / 1.69 at s = 2 / 3 / 4; long twins s = 4-8 1.58 / 1.44 / 1.29 / 1.34 / .90. F_0 about .2 on long twins.
- Long twins s = 4, hop shares 0-4: any-wrong .16 / .36 / .29 / .12 / .05.
- t32: agreement 1.00; first-token hops = full-answer hops on .960-.970 of pairs; first-token hops slightly lower.
**Predictions:** t31 within +-.30: held at s = 2 (+.03), missed at s = 3 (-.34). t32 held except t22 s = 6 (-.21, n 23).
**Interpretation:** Coconut uses fewer crossings than CODI / SIM and fewer as items get longer; about one long-twin pair in five needs none.
**Gotchas hit:** none.
**Caveats:** one released checkpoint; its vocabulary-projection readout fails chance (t30), so readability needs a probe.
**Next:** none beyond the paper figure.
