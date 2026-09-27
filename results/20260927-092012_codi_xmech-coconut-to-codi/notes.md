## 2026-09-27 — P4 cross-mechanism transplant, Coconut → CODI: mapped Coconut latents carry a donor's intermediate values into CODI (cf_joint 26/25/21% vs own 39/39/20%, shuffled ≤1.5%) (codi, run_id: 20260927-092012_codi_xmech-coconut-to-codi)

**Goal / design:** reverse direction of `20260927-085211_coconut_xmech-codi-to-coconut` (read that record
for the full design): ridge maps from all six Coconut passes to each CODI site z_s, fit on the same 8000
fit questions; transplant into CODI recipients at z0/z2/z4 with the same ladder donors and conditions.
**Mechanism / model:** CODI `hf:zen-E/CODI-gpt2@fd641b3` (target, aligned sites), Coconut checkpoint_33 (source).
**Command:** `xmech_codi.py --mode transplant --stage full_run` after the Coconut stage; pod
`b7i2qndzf0w1dh` (L4 secure). See `eval_command.txt`.
**Headline results:** 259 recipients (base-correct in both models).

Map fit (held-out R², shuffled): z0 0.54 (−0.07), z1 0.59, **z2 0.36**, z3 0.55, **z4 0.31** (−0.15), z5 0.52.
The value-carrying sites z2/z4 are the hardest to predict from Coconut.

| level | own cf_joint | **mapped** cf_joint | shuffled | mapped_all |
|---|---|---|---|---|
| L2 | 0.39 | **0.26** | 0.004 | 0.27 |
| L3 | 0.39 | **0.25** | 0.015 | 0.21 |
| L4 | 0.20 | **0.21** | 0.000 | 0.16 |

(null ≤0.6% throughout; donor_final ≤0.10 for every condition.) self_mapped: answer unchanged **74.1%**.

**Interpretation:**
- Transfer works in this direction too: mapped Coconut latents reach 64–105% of CODI's own donor latents,
  with the shuffled map at zero. At L4 mapped equals own (0.21 vs 0.20).
- Unlike CODI → Coconut, mapping into all six CODI sites still transfers at L3/L4 (0.21/0.16): CODI's
  non-carrier sites are inert (aligned E3), so overwriting them costs little.
- Weaker reconstruction (74% vs 90%) matches the lower R² at z2/z4.
**Caveats:** as in the forward record — the map uses all source information, not only values.
**Next:** see the forward record.
