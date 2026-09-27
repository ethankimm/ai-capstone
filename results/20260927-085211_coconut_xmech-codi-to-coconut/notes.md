## 2026-09-27 — P4 cross-mechanism transplant, CODI → Coconut: mapped CODI latents carry a donor's intermediate values into Coconut about as well as Coconut's own latents (cf_joint 32/28/24% vs own 30/30/30%, shuffled map ≤2%) (coconut, run_id: 20260927-085211_coconut_xmech-codi-to-coconut)

**Goal:** RESEARCH_PLAN §4 P4: do the two mechanisms encode intermediate values in a shared (linearly
mappable) format? Map CODI's latents into Coconut's latent slots and transplant them across problems.
**Design** (`scripts/xmech_common.py`, `scripts/xmech_codi.py`, `scripts/xmech_coconut.py`):
- One question set fed to both models (`xmech_common.py` builder): **fit** = 8000 GSM8K-Aug train
  questions, **eval** = all 1319 test questions.
- Dumps: CODI z0..z5 (aligned, z_s feeds iteration s+1); Coconut passes 0..5.
- Maps: ridge, all 6 CODI sites concatenated (4608 dims) → each Coconut pass (768), standardized inputs,
  λ ∈ {0.1 … 1e4} picked on a held-out 20% of the fit split (1e3 for every pass).
- Transplant on the eval split into Coconut recipients at passes 1,4 (carriers), with the P1 ladder donors
  (`ladder_common.py`; L2 same ops, L3 same length, L4 any), recipients and donors base-correct in BOTH
  models: **own** (Coconut's own donor latents — ceiling), **mapped** (map(CODI donor latents)),
  **shuffled** (map fit on permuted pairs — control), **mapped_all** (all 6 passes), and
  **self_mapped** (map(CODI RECIPIENT latents) — reconstruction check). Buckets + permutation null as in P1.
**Mechanism / model:** Coconut `hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33` (target) and CODI
`hf:zen-E/CODI-gpt2@fd641b3` (source). Base accuracy on eval: Coconut 0.331, CODI 0.419.
**Command:** three stages on pod `b7i2qndzf0w1dh` (L4 secure): `xmech_codi.py --mode dump` (27 min) →
`xmech_coconut.py --stage full_run` (dump 15 min + fit + transplant) → reverse run. See `eval_command.txt`.
**Headline results:** 259 recipients.

Map fit (held-out eval R², shuffled in brackets): pass 0 0.35 (−0.06), **pass 1 0.31** (−0.03), 2 0.37,
3 0.32, **pass 4 0.18** (−0.04), 5 0.28. Linear CKA CODI site × Coconut pass: 0.06–0.27 (max z0↔pass 1
0.26, z1/z3/z5↔pass 3/5 0.18–0.27).

| level | condition | unchanged | donor_final | **cf_joint** (null) | other |
|---|---|---|---|---|---|
| L2 | own | 0.08 | 0.14 | 0.30 (0.004) | 0.44 |
| L2 | **mapped** | 0.08 | 0.09 | **0.32** (0.004) | 0.49 |
| L2 | shuffled | 0.20 | 0.00 | 0.01 | 0.68 |
| L2 | mapped_all | 0.06 | 0.10 | 0.26 | 0.52 |
| L3 | own / **mapped** / shuffled | 0.05 / 0.05 / 0.19 | 0.04 / 0.02 / 0.02 | 0.30 / **0.28** / 0.02 | |
| L4 | own / **mapped** / shuffled | 0.08 / 0.09 / 0.20 | 0.03 / 0.03 / 0.01 | 0.30 / **0.24** / 0.01 | |
| L4 | mapped_all | 0.09 | 0.04 | 0.07 | 0.67 |

self_mapped (CODI recipient's own latents mapped into Coconut): answer unchanged **89.6%**.

**Interpretation:**
- **CODI's intermediate values are linearly translatable into Coconut's format.** A map fit on 8000 other
  problems turns a CODI donor's latents into Coconut pass-1/4 vectors that make Coconut finish its own
  program on the donor's values 24–32% of the time — 80–105% of what Coconut's own donor latents do, against
  ≤2% for a map that saw the right marginals but no pairing. The reconstruction check (90% unchanged) says
  the map preserves the recipient's own content.
- Mapping into ALL six passes works at L2 (0.26) but collapses at L3/L4 (0.10/0.07): the non-carrier
  passes encode problem-specific structure that does not transfer, consistent with P1 (Coconut's full
  latent set carries the donor's answer/program).
- Low CKA with a working linear map: the shared content occupies a small subspace (cf. DAS: ~16 dims
  carry the value in each mechanism — `20260927-094215`, `20260927-084449`).
**Caveats:** The map sees all six CODI sites, so it can use any information in them (including
question-derived features), not only the computed values; cf_joint scoring targets computed step values,
which the recipient's question does not contain, but this is not a dimension-level test. One donor draw
per level; n=259.
**Next:** Restrict the map's input to the value subspace found by DAS and test whether it alone transfers;
per-step analysis (CODI z0 → Coconut pass 1 for step 0).
