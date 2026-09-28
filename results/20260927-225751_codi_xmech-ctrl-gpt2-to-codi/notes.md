## 2026-09-28 — Experiment A2, plain GPT-2 (no latent training) → CODI: question text alone carries only 7-9% of the unrestricted map's cf_joint (codi, run_id: 20260927-225751_codi_xmech-ctrl-gpt2-to-codi)

**Goal:** Reverse direction of `20260927-224642` (plain GPT-2 → Coconut) — the cleanest
question-only control, this time mapping into CODI.
**Design** (`gpt2_plain_dump.py` + `xmech_common.run_transplant`'s `feat`/`feat_sites` params;
`xmech_codi.py --mode ctrl --ctrl_kind gpt2`): same plain-GPT2 feature dump as the Coconut-direction
run (`[layer6 last-token, layer12 last-token, layer12 mean-over-question-tokens]`, 2304 dims), same
259 eval-split ladder recipients/donors as the unrestricted P4 run, pinned by the TRUE
CODI/Coconut correctness intersection. Same fit split, same carriers (z0,z2,z4).
**Mechanism / model:** CODI `hf:zen-E/CODI-gpt2@fd641b3` (target, aligned sites); plain `gpt2`
(124M, no fine-tuning) supplies the source features.
**Command:** `xmech_codi.py $CODI_FLAGS --mode ctrl --ctrl_kind gpt2 --gpt2_latents
gpt2_latents.pt --slug xmech-ctrl-gpt2-to-codi --stage full_run` (full line in `eval_command.txt`).
Pod `jbzj5cd4diacxb` (RTX A40 secure, $0.49/hr, EU-SE-1); ran alongside 7 other jobs on one GPU.
**Headline results** (259 recipients; siblings `20260927-225629` (pass0 only) and
`20260927-225810` (passes 1-5); reference unrestricted run `20260927-092012`, mapped_all cf_joint
L2/L3/L4 = 0.26/0.25/0.21):

| cf_joint (null ≤0.006) | L2 | L3 | L4 |
|---|---|---|---|
| own_carriers (this run) | 0.359 | 0.386 | 0.193 |
| **mapped_carriers (plain GPT-2)** | **0.019** | **0.023** | **0.015** |
| shuffled_carriers | 0.042 | 0.019 | 0.012 |
| % of unrestricted mapped_all (0.26/0.25/0.21) | 7% | 9% | 7% |

Map fit (held-out R², plain-GPT2 features → CODI site): z0 0.029, z1 0.140, z2 -0.004, z3 0.157, z4
0.007, z5 0.149 — the lowest R² of any feature set tested in either direction, and the mapped
condition here is actually BELOW its own shuffled-pair control at L2 (0.019 vs 0.042) — both are
near zero, i.e. this map has essentially no signal to exploit either way.
self_mapped_carriers unchanged only 23.6%.

**Interpretation:**
- **Confirms the question-confound control in both directions.** 7-9% of the unrestricted map's
  transfer, consistent with the Coconut-direction result (`20260927-224642`, 5-8%) and well inside
  the "<40%" branch of the decision rule. Neither direction's unrestricted P4 result is explained
  by re-encoding the question text alone.
- The R² near zero (and mapped ≤ shuffled at L2) means this condition isn't just "weak signal
  diluted by noise" — plain GPT-2's representation of these questions carries close to nothing a
  linear map can turn into CODI's site-specific value information, at least at these three probe
  points.
**Caveats:** n=259. As with the Coconut-direction gpt2 control, a richer GPT-2 feature set might
move this number somewhat, but the size of the gap to either mechanism's own pass0/z0
(`20260927-224224`, `20260927-225629`) makes it unlikely to change the qualitative conclusion.
**Next:** This closes out Experiment A (both directions x all three conditions logged). Decision:
plain-GPT2 control rules out the question confound cleanly in both directions; the source's own
pre-reasoning encoding (z0/pass0) is informative for CODI→Coconut but not for Coconut→CODI — a
genuine mechanism asymmetry, not an artifact of the map's input dimensionality (z0-only and
pass0-only are both 768 dims).
