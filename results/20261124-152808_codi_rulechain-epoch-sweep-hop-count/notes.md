## 2026-11-24 — Epoch sweep (t55): start-last CODI's accuracy doubles from epoch 8 to 12 while crossings per pair stay ~1.5; the 40-epoch round-1 CODI uses ~1 crossing at every distance (codi, t55)

**Goal:** t55 (pre-registered 2026-11-23 20:52 +07): is the standard RuleChain null an artefact of the 12-epoch budget (the recipe is 40)?
**Command:** `CODI_CKPT=<checkpoint> QUAL=own ../codi/.venv/bin/python scripts/necessity/hops.py startlast mps ../codi_audit_artifacts/t55/<tag>.json` (round 1: model key stab); scoring `scripts/necessity/epoch_hops_score.py ../codi_audit_artifacts/t55/score.json`.
**Results:** epoch-12 run reproduces t31 on 2000 / 2000 pairs. Correct pairs, mean hops at s = 3-4: 1.14 (n 7) / 1.60 (10) / 1.52 (31) / 1.47 (51) at epochs 6 / 8 / 10 / 12, while valid accuracy at L3 rises .14 / .24 / .41 / .52. Round-1 40-epoch start-first CODI: 1.05 / 1.12 / 1.14 / 1.31 / 1.25 at s = 2-6, flat.
**Predictions:** P0 held; P1 missed (cell size at epoch 6; no rise from epoch 8 on); P2 missed (slope not comparable across epochs); P3 missed in level (~1 crossing, not ~0), direction held.
**Interpretation:** within the epochs we have, more training buys accuracy without longer chains; the 40-epoch start-first run behaves like a copy (near-identical latents). A 40-epoch start-last run remains the clean test.
**Caveats:** start-last only to epoch 12; round 1 is start-first at lr 3e-3; the queue paused overnight (Mac asleep), results unaffected.
