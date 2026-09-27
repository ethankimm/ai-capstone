## 2026-09-27 — Does CODI's learned 16-dim DAS subspace decode the step values? No better than a random 16-dim subspace: CODI stores the value redundantly across the vector (codi, run_id: 20260927-185312_codi_das-subspace-probe)

**Goal:** Round-3 follow-up to E4. A learned 16-dim rotation of z0/z2/z4 steers 63–70% of minimal pairs
(`20260927-094215`, rerun `20260927-184022`). Does that subspace also linearly or otherwise *decode* the
step values, and does it do so better than a random 16-dim subspace?
**Design** (`scripts/das_subspace_probe.py`, model-free, local CPU): features from the P4 latent dumps
(`xmech_codi.py --mode dump`, 9319 questions): fit = 8000 GSM8K-Aug train questions, eval = the test set,
model-correct examples only (fit n=6301 / eval n=548 for step 0). Rotation = `rot_0+2+4_k16.pt` from
`20260927-184022_codi_das-minimal-pair-saverot`. Readouts per rationale step j: ridge on signed-log1p value
(R², tol5, exact), **knn_exact** (nearest fit example's step value is the same; standardized Euclidean),
**last_digit** (linear softmax on the last integer digit; majority baseline 0.43/0.42/0.37).
Feature sets: DAS coordinates (all 3 sites, 48 dims; per site 16), random 16-dim subspaces (mean of 5
draws for the 3-site set; one draw per single site), the 752-dim complement, full vectors.
**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3`, 6 latents, aligned sites.
**Command:** `uv run python scripts/das_subspace_probe.py --questions xmech_questions.jsonl --codi_latents
codi_latents.pt --coconut_latents coconut_latents.pt --codi_rotation rotations/codi/rot_0+2+4_k16.pt
--coconut_rotation rotations/coconut/rot_1+4_k16.pt --stage full_run` (inputs kept outside the repo at
`~/Documents/Penn/CIS5980/xmech_artifacts/`, ~350 MB). One invocation writes this and the Coconut record.
**Headline results** (eval split):

| feature (dims) | step 0 knn / digit | step 1 knn / digit | step 0 ridge R² |
|---|---|---|---|
| DAS z0+z2+z4 (48) | 0.54 / 0.72 | 0.20 / 0.54 | 0.10 |
| random 3×16 (48) | 0.54 / 0.68 | 0.22 / 0.54 | 0.44 |
| complement (2256) | 0.57 / 0.76 | 0.23 / 0.53 | 0.68 |
| **DAS@z0 (16)** | **0.61** / 0.72 | 0.05 / 0.42 | 0.07 |
| random@z0 (16) | 0.61 / 0.74 | 0.07 / 0.43 | |
| full z0 (768) | 0.63 / 0.80 | 0.07 / 0.45 | |
| DAS@z2 (16) | 0.28 / 0.53 | **0.23** / 0.56 | |
| random@z2 (16) | 0.29 / 0.51 | 0.23 / 0.58 | |

(k=32 rotation, run unlogged with `--no_log`: DAS ≈ random again, knn 0.54 vs 0.55 at step 0; the ridge gap narrows, R² 0.50 vs 0.62.) Ridge tol5 ≤ 0.09 and exact ≤ 0.06 for every feature set.
**Interpretation:**
- **The causally privileged subspace is not decode-privileged in CODI.** Any 16 random dims at a site
  decode that site's step (z0 → step 0, z2 → step 1) as well as the DAS dims, and nearly as well as the
  whole 768-dim vector. The value is spread redundantly across CODI's latent; DAS found *a* direction set
  the model reads out, not the only place the value lives.
- The site ↔ step mapping from decoding matches the causal one (z0 carries step 0, z2 step 1 — aligned E3).
- A linear read of log-magnitude is a poor probe here (DAS R² 0.10, *below* random 0.44): the value is not
  on a linear number line in these dims. Token-like readouts (nearest neighbour, last digit) work.
**Gotchas hit:** First version of the script had only the ridge probe; it made the DAS subspace look
uninformative. Added knn_exact / last_digit and the single-site random/full controls before logging.
**Caveats:** knn_exact benefits from repeated values across GSM8K problems (small integers are common) —
compare rows against each other, not to 0. One random draw per single site. `predictions.jsonl` is empty
(aggregate probe run; per-example predictions not saved).
**Next:** Compare with Coconut (`20260927-185326`), where the DAS dims *are* privileged; the subspace
transplant (P4 restricted) tests whether the DAS coordinates alone carry the value across mechanisms.
