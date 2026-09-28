## 2026-09-28 — Experiment A1, Coconut pass0 only → CODI: pre-reasoning latent alone carries only 15-39% of the unrestricted map's cf_joint (codi, run_id: 20260927-225629_codi_xmech-ctrl-site0-coconut-to-codi)

**Goal:** Reverse direction of `20260927-224224` (CODI z0 → Coconut). Does Coconut's pass0 — the
vector computed from the question alone, before any of Coconut's own splice-and-recompute passes —
carry most of the unrestricted CODI↔Coconut map's transfer power into CODI?
**Design** (`xmech_common.run_transplant`'s new `feat`/`feat_sites` params; `xmech_codi.py --mode
ctrl --ctrl_kind site0`): identical transplant to the unrestricted P4 run
(`20260927-092012_codi_xmech-coconut-to-codi`) — same 259 eval-split ladder recipients/donors, same
fit split, same carriers (z0,z2,z4). Only the map's input changes: Coconut pass0 alone (768 dims)
instead of all 6 passes (4608 dims).
**Mechanism / model:** CODI `hf:zen-E/CODI-gpt2@fd641b3` (target, aligned sites), Coconut
`hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33` (source, feature only).
**Command:** `xmech_codi.py $CODI_FLAGS --mode ctrl --ctrl_kind site0 --questions
xmech_questions.jsonl --codi_latents codi_latents.pt --coconut_latents coconut_latents.pt --slug
xmech-ctrl-site0-coconut-to-codi --stage full_run` (full line in `eval_command.txt`). Pod
`jbzj5cd4diacxb` (RTX A40 secure, $0.49/hr, EU-SE-1); ran alongside 7 other jobs on one GPU.
**Headline results** (259 recipients; siblings `20260927-225810` (passes 1-5) and
`20260927-225751` (plain GPT-2); reference unrestricted run `20260927-092012`, mapped_all cf_joint
L2/L3/L4 = 0.26/0.25/0.21):

| cf_joint (null ≤0.006) | L2 | L3 | L4 |
|---|---|---|---|
| own_carriers (this run) | 0.359 | 0.386 | 0.193 |
| **mapped_carriers (pass0 only)** | **0.039** | **0.058** | **0.081** |
| shuffled_carriers | 0.015 | 0.023 | 0.004 |
| % of unrestricted mapped_all (0.26/0.25/0.21) | 15% | 23% | 39% |

Map fit (held-out R², Coconut pass0 → CODI site): z0 0.262, z1 0.471, z2 0.122, z3 0.404, z4 0.127,
z5 0.387. self_mapped_carriers (recipient's own pass0 mapped into itself) unchanged 32.8%.

**Interpretation:**
- **Unlike the reverse direction, Coconut's pass0 alone is NOT sufficient.** 15-39% of the
  unrestricted map — clearly on the "reasoning latents add transferable content beyond the
  question" side of the decision rule at L2/L3 (well below 40%), borderline at L4. Coconut needs
  its LATER passes to supply what the unrestricted map uses.
- This is the sharpest asymmetry in Experiment A: CODI's z0 carries 74-110% of ITS OWN direction's
  unrestricted transfer (`20260927-224224`), but Coconut's pass0 carries only 15-39% of its
  direction. Consistent with the two mechanisms' architectures: CODI's z0 is the output of a full
  autoregressive pass over the question BEFORE any loop iteration begins (already "digested"),
  whereas Coconut's pass0 fills the first of six fixed `<|latent|>` slots inside one forward pass,
  with less computation behind it at that point.
- See the sibling complement run (`20260927-225810`, passes 1-5): it alone reaches ~100-129% of the
  unrestricted rate, confirming the missing content is squarely in Coconut's later passes.
**Caveats:** n=259, one donor draw per level. own_carriers here (0.359/0.386/0.193) is close to but
not identical to the "0.39/0.39/0.20" quoted in round-3 notes for the same nominal condition — a
few points off, plausibly rounding in that summary; doesn't affect this run's ratio, computed
against the precisely-read unrestricted mapped_all.
**Next:** `20260927-225751` (A2 plain-GPT2 control, same direction) and `20260927-225810` (the
complement) complete this direction's picture.
