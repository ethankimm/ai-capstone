## 2026-09-27 — Necessity under zero / mean / norm-matched noise, CODI: thoughts are necessary under every replacement, but which site looks necessary depends on the replacement, and sampling recovers most of what truncation loses (codi, run_id: 20260927-005051_codi_necessity-all-replacements)

**Goal:** RESEARCH_PLAN round 1, item 2: fill the Q2 "necessary?" cells with every replacement type
(Jin et al.'s point that zero / mean ablation are off-distribution in different ways), with
preserved-when-correct (PWC), the correct→wrong / wrong→correct flip split, and cheap pass@k.
Extends `20260919-192228_codi_early-termination-ablate-all` (truncation, ablate-all zero/mean,
single mean on the 200-slice) to norm-matched noise, single-site zero/noise, the full test set for
every condition, and sampling.
**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3`, 6 latents, eval mode, batch 1.
**Data:** gsm8k-aug test, all 1319, every condition.
**Design:** `scripts/necessity_codi.py` + `scripts/necessity_common.py`. Sites z0..z5, z_s = latent fed
into iteration s+1 (z0 = latent-0), aligned with `override_input_at` (same numbering as the aligned E3
rerun and Coconut passes 0..5). trunc_k = k thoughts then eot (KV-snapshot path, verified equivalent in
the prior run). Replacements: zero; mean = population mean of z_s over 300 seeded test examples (norms
62–68); noise = Gaussian direction scaled to this example's clean ||z_s||, seed idx·1000+s.
PWC = P(correct | baseline correct). pass@k: 10 answers sampled at T=0.7 from the (deterministic)
latent state, unbiased estimator; conditions trunc_6, trunc_0, ablate_all_{zero,mean,noise}.
**Command:** 4 shards in parallel then merge, see `scripts/necessity_codi.py` docstring; flags as in
`eval_command.txt` (`--output_dir /tmp/o` added, required by the CODI TrainingArguments parser).
Pod `k030n8z4jrafxm` (A5000 secure, $0.27/hr), ~40 min wall shared with 4 other jobs (7.4 s/ex/shard).
**Headline results:** `final_answer_accuracy=0.419` (553/1319, identical to the prior run),
**`early_termination_necessity=0.181`** (0.419 − 0.238, identical to the prior run).

| condition | acc | PWC | c→w | w→c | pass@1 | pass@10 | pass@10 \| base correct | pass@10 \| base wrong |
|---|---|---|---|---|---|---|---|---|
| baseline trunc_6 | 0.419 | 1 | 0 | 0 | 0.407 | 0.519 | 0.998 | 0.174 |
| trunc_0 | 0.238 | 0.454 | 302 | 63 | 0.218 | 0.509 | 0.832 | 0.277 |
| ablate_all_zero | 0.231 | 0.441 | 309 | 61 | 0.214 | 0.494 | 0.770 | 0.294 |
| ablate_all_mean | **0.105** | **0.201** | 442 | 27 | 0.089 | 0.307 | 0.450 | 0.204 |
| ablate_all_noise | 0.141 | 0.259 | 410 | 43 | 0.128 | 0.397 | 0.644 | 0.218 |

Truncation curve k=0..6: 0.238, 0.243, 0.231, 0.370, 0.370, 0.411, 0.419 (unchanged from the prior run).

Single-site accuracy (PWC in parentheses), baseline 0.419:

| site | zero | mean | noise |
|---|---|---|---|
| z0 (latent-0) | 0.408 (0.95) | **0.298 (0.65)** | **0.305 (0.66)** |
| z1 | 0.353 (0.80) | 0.399 (0.90) | 0.368 (0.81) |
| z2 | 0.349 (0.79) | 0.375 (0.84) | 0.353 (0.77) |
| z3 | 0.315 (0.71) | 0.325 (0.73) | 0.374 (0.83) |
| z4 | **0.305 (0.68)** | 0.354 (0.78) | 0.371 (0.82) |
| z5 | 0.415 (0.97) | 0.413 (0.96) | 0.409 (0.95) |

All ablate-all McNemar p < 1e-38; flips are overwhelmingly correct→wrong.
**Interpretation:**
- **Necessary under every replacement.** No content-free replacement preserves more than 44% of
  correct answers. Mean replacement is the most damaging (PWC 0.20, below the no-thoughts 0.45) and zero
  is nearly identical to truncation, so the "worse than nothing" effect of the prior run is specific to
  plausible-norm, wrong-content inputs (mean, noise), consistent with Jin et al.'s concern.
- **pass@k:** removing the scratchpad (trunc_0) keeps aggregate pass@10 at 0.509 vs 0.519, because
  no-thought sampling is more diverse on problems the full model misses (0.28 vs 0.17). On problems the
  full model solves, thoughts make the right answer near-certain (pass@10 0.998 vs 0.832 without them;
  0.45 under mean). So thoughts mostly concentrate probability on the right answer where the model can
  solve the problem; they are not the only route to it. Mean/noise inputs remove capability outright.
- **Single-site necessity depends on the replacement.** z0 is harmless under zero but the most necessary
  site under mean or noise; z4 is the most necessary under zero. z5 is unnecessary under all three
  (it only feeds the last iteration before eot). The odd sites z1/z3 carry no transplantable value in
  the aligned E3 (`20260927-004340`: ≤0.3% steering) yet removing them costs 5–10 pp: load-bearing but
  content-invariant across twins, plausibly structural placeholders. Single-site "necessity" claims must
  name the replacement.
**Gotchas hit:** CODI's HfArgumentParser needs `--output_dir` (TrainingArguments); any path works.
**Caveats:** pass@k samples only the answer tokens (latents are deterministic by design), T=0.7 only;
numeric answers have a small guessing floor, not estimated. Noise is one draw per (example, site).
**Next:** Coconut counterpart `necessity_coconut.py` (same design); cross-reference with the aligned E3
site pattern (steering on z0/z2/z4, necessity on z0/z3/z4 depending on replacement).
