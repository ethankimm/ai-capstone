## 2026-09-27 — E2 rerun with aligned sites, CODI: single-site cross-problem patching moves z0's step-0 value ~8% of the time, for any donor; no other site carries its assigned step (codi, run_id: 20260927-062443_codi_qualified-patch-aligned)

**Goal:** Rerun E2 (`20260920-085206_codi_qualified-patch`) with the site-indexing fix: site s = z_s, the
latent fed into iteration s+1 (z_0 = latent-0), donor z_s → `override_input_at={s+1: …}`. Same design,
seed, slice and site→step map (`site_to_step(s, 6, max_step)`) as before, so the only change is the fix.
**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3`, 6 latents, eval mode, greedy.
**Data:** gsm8k-aug test n=600 seed=0 (the E2/E3 slice); base accuracy 0.435, 261 base-correct, 370 pairs.
**Command:** `scripts/patch_qualified_codi.py … --slug qualified-patch-aligned --stage full_run --eval_n 600
--n_pairs_per_site 200 --output_dir /tmp/o` (full flags in `eval_command.txt`). Pod `5wb7p2e1kr568o`
(RTX A6000 secure, $0.53/hr — A5000 out of stock), in parallel with E4 and the two ladder runs; ~10 min.
**Headline results:** `intervention_accuracy` (matches_cf, real donor, pooled) = 0.041, random donor 0.041.

| site (step) | n | changed real / random / mean | matches_cf real | random | old E2 (shifted) real cf |
|---|---|---|---|---|---|
| z0 (0) | 190 | 0.43 / 0.43 / 0.31 | **0.075** | **0.081** | 0.005 |
| z1 (1) | 121 | 0.19 / 0.18 / 0.10 | 0.008 | 0.000 | 0.025 |
| z2 (2) | 52 | 0.62 / 0.64 / 0.42 | 0.000 | 0.000 | 0.000 |
| z3–z5 (4–6) | 2–3 each | — | 0 | 0 | 0 |

(`mean` has no counterfactual by construction — matches_cf is None; see `outcome_taxonomy`.)

**Interpretation:**
- The fix moves z0 from 0.5% to 7.5–8.1% counterfactual match: the donor's latent-0 does carry its
  step-0 value into the recipient's own chain in a minority of cases.
- Real ≈ random is NOT a null here: E2's "random donor" is a second real donor scored against its OWN
  value, so both transferring their value at the same rate is exactly what value transfer predicts.
  The proper control is a permutation null (score against an unrelated donor's value), which the
  ladder run has: `20260927-063622_codi_ladder-patch` gives z0 cf_single 8–12% against a null of 0.1%.
- z1 carries nothing (inert in aligned E3 too). z2 is tested against step 2 by the positional map, but
  aligned E3 and the ladder say z2/z4 carry later values for chains of ≥3 steps — and only 52 recipients
  have a step-2 qualifying step, so this cell is thin. The positional map (max_step=8 over 6 sites) is a
  poor fit for CODI's even-site layout; the ladder's cf_joint/cf_single scoring supersedes it.
**Gotchas hit:** The E2 design's random-donor "control" is not a control for value transfer (above).
**Caveats:** Single slice, greedy; z3–z5 cells have n ≤ 3.
**Next:** Use the ladder run for cross-problem claims; E2 stays as the like-for-like correction of the old run.
