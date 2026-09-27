## 2026-09-27 — Does Coconut's learned 16-dim DAS subspace decode the step values? Yes, at its carrier pass: pass 1 → step 0 and pass 4 → step 1, well above random 16 dims and close to the full vector (coconut, run_id: 20260927-185326_coconut_das-subspace-probe)

**Goal:** Round-3 follow-up to E4. A learned 16-dim rotation of passes 1+4 steers 61–63% of minimal pairs
(`20260927-084449`, rerun `20260927-182915`). Does it also decode the step values, better than random dims?
**Design:** identical to `20260927-185312_codi_das-subspace-probe` (same script and invocation); Coconut
dump from `xmech_coconut.py --mode dump`, rotation `rot_1+4_k16.pt` from
`20260927-182915_coconut_das-minimal-pair-saverot`. Model-correct examples only (fit n=7378 / eval n=432 for
step 0). Majority last-digit baseline 0.42/0.40/0.39.
**Mechanism / model:** `coconut`, gpt2 / `hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33`, 6 passes.
**Command:** see the CODI record.
**Headline results** (eval split):

| feature (dims) | step 0 knn / digit | step 1 knn / digit | step 0 ridge R² |
|---|---|---|---|
| DAS passes 1+4 (32) | 0.48 / 0.72 | 0.39 / 0.69 | 0.09 |
| random 2×16 (32) | 0.37 / 0.53 | 0.26 / 0.48 | 0.17 |
| complement (1504) | 0.48 / 0.74 | 0.31 / 0.69 | 0.73 |
| **DAS@pass1 (16)** | **0.60 / 0.72** | 0.08 / 0.42 | 0.08 |
| random@pass1 (16) | 0.50 / 0.52 | 0.07 / 0.39 | |
| full pass1 (768) | 0.65 / 0.75 | 0.09 / 0.33 | |
| **DAS@pass4 (16)** | 0.24 / 0.43 | **0.51 / 0.69** | |
| random@pass4 (16) | 0.23 / 0.42 | 0.34 / 0.47 | |
| full pass4 (768) | 0.33 / 0.59 | 0.56 / 0.70 | |

k=32 rotation (unlogged, `--no_log`): same ordering, DAS@pass1 knn 0.63 / digit 0.77, DAS@pass4 0.52 / 0.73; 3-site DAS vs random 0.51 vs 0.45 (step 0), 0.39 vs 0.30 (step 1). Step 2: nothing decodes (knn ≤ 0.07, digit ≈ majority). Ridge tol5 ≤ 0.09 everywhere.
**Interpretation:**
- **In Coconut the DAS subspace is where the value is concentrated.** At its carrier pass, 16 learned dims
  recover almost everything the full 768-dim vector gives (pass 1/step 0: 0.60 vs 0.65 knn, 0.72 vs 0.75
  digit; pass 4/step 1: 0.51 vs 0.56, 0.69 vs 0.70), and clearly beat 16 random dims (0.50 / 0.52; 0.34 / 0.47).
- The same 16-dim basis serves both passes, and decoding confirms the causal site ↔ step map from P1
  (pass 1 = step 0, pass 4 = step 1); each pass holds only its own step.
- **Contrast with CODI** (`20260927-185312`): there random dims decode as well as the DAS dims. Both
  mechanisms have a small causally sufficient subspace, but Coconut's value is localized in it while
  CODI's is spread redundantly across the vector.
- Linear log-magnitude probes fail in the DAS dims (R² 0.09 < random 0.17): the code is token-like, not
  a number line — consistent with Coconut's latents being near the answer-token embeddings (logit lens).
**Caveats:** as in the CODI record (knn benefits from common values; one random draw per single site;
empty predictions.jsonl).
**Next:** P4 restricted to these subspaces (`xmech-subspace-*` runs).
