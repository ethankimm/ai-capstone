## 2026-09-28 — Experiment A1 complement, Coconut pass1-5 → CODI: post-pass0 latents alone reach 100-129% of the unrestricted map's cf_joint (codi, run_id: 20260927-225810_codi_xmech-ctrl-sites1to5-coconut-to-codi)

**Goal:** Complement view of `20260927-225629` (Coconut pass0 only): with pass0 held out, do
Coconut's remaining five passes (1-5, where the splice-and-recompute reasoning actually accumulates)
carry the unrestricted map's full transfer power on their own?
**Design:** identical to `20260927-225629` except the ridge map reads Coconut passes 1,2,3,4,5
concatenated (5x768=3840 dims) instead of pass0 alone (`xmech_codi.py --mode ctrl --ctrl_kind
sites1to5`). Same 259 recipients, same fit split, same carriers (z0,z2,z4).
**Mechanism / model:** CODI `hf:zen-E/CODI-gpt2@fd641b3` (target, aligned sites), Coconut
`hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33` (source, feature only, pass0 excluded).
**Command:** `xmech_codi.py $CODI_FLAGS --mode ctrl --ctrl_kind sites1to5 --slug
xmech-ctrl-sites1to5-coconut-to-codi --stage full_run` (full line in `eval_command.txt`). Pod
`jbzj5cd4diacxb` (RTX A40 secure, $0.49/hr, EU-SE-1); ran alongside 7 other jobs on one GPU.
**Headline results** (259 recipients; siblings `20260927-225629` (pass0 only) and
`20260927-225751` (plain GPT-2); reference unrestricted run `20260927-092012`, mapped_all cf_joint
L2/L3/L4 = 0.26/0.25/0.21):

| cf_joint (null ≤0.006) | L2 | L3 | L4 |
|---|---|---|---|
| own_carriers (this run) | 0.359 | 0.386 | 0.193 |
| **mapped_carriers (passes 1-5)** | **0.259** | **0.286** | **0.270** |
| shuffled_carriers | 0.019 | 0.019 | 0.008 |
| % of unrestricted mapped_all (0.26/0.25/0.21) | 100% | 114% | 129% |

Map fit (held-out R², Coconut passes1-5 → CODI site): z0 0.540, z1 0.581, z2 0.360, z3 0.536, z4
0.312, z5 0.519 — the highest R² of any source-feature set in either direction, roughly double
pass0-only's (`20260927-225629`). self_mapped_carriers unchanged 75.3% — this condition also
reconstructs the recipient's own content best of the three A1/A2 conditions in this direction.

**Interpretation:**
- **Coconut's passes 1-5 alone fully account for (and slightly exceed, within noise) the
  unrestricted 6-pass map's transfer into CODI** — confirming that pass0's near-total exclusion
  from the transfer (`20260927-225629`, 15-39%) is not a dimensionality artifact (both conditions
  are 5x vs 1x768) but a real statement about WHERE in Coconut's pipeline the transferable value
  lives: after the question-encode pass, not in it.
- Together with the CODI-side result (`20260927-224224`/`20260927-224922`, where EITHER half
  reaches most of the ceiling), this is the round's cleanest asymmetry: CODI spreads transferable
  value redundantly across its whole trajectory including step 0; Coconut concentrates it in the
  passes after step 0. Both are consistent with the DAS probe results from round 4
  (`20260927-185312` CODI: no dims are special; `20260927-185326` Coconut: its DAS dims at the
  carrier pass do hold the value) — the same underlying pattern seen from a different angle.
**Caveats:** n=259, one donor draw per level. Values above 100% (114%, 129%) are within plausible
noise range at this n but also consistent with pass0 (question framing) being mildly UNHELPFUL for
a cross-problem donor swap, since it carries donor-problem-specific framing the recipient doesn't
need.
**Next:** This completes Experiment A (both directions logged: `20260927-224224/-224642/-224922`
into Coconut, this run + `20260927-225629/-225751` into CODI). See `MEMORY`/results index for the
combined report; Experiment B (`20260927-234206`, `20260928-005033`) tackles the DAS cross-problem
basis question in parallel.
