## 2026-11-12 — Released SIM-CoT Coconut on GSM8K (t36): 2.6-3.2 hops at every length, the most of any GSM8K model, but no staircase (coconut, run_id: 20261112-211705_coconut_gsm8k-released-simcot-coconut-hop-count)

**Goal:** t36 (pre-registered): does step supervision give Coconut deeper latent use on GSM8K? Comparison to plain Coconut (t31).
**Mechanism / model:** hf:internlm/SIM_COT-GPT2-Coconut @676c8a8, checkpoint_28 (base_causallm only; the step decoder is not used). 10 latents (SIM-CoT gsm_simcot.yaml: c_thought 2 x max_latent_stage 5).
**Data:** GSM8K-Aug test twins: t22 (1,471 candidates) and long twins (1,305). Qualified (own answers differ): 1,287 + 1,111.
**Command:** `QUAL=own FIRST=1 ../codi/.venv/bin/python scripts/necessity/hops.py gsm8k_simcoconut mps <out>` (queue `codi_audit_artifacts/t36/queue.sh`).
**Headline results:**
- Validity: on 200 recipients, accuracy .295 at 6 latents, .39 at 10 (plateau .38-.40 from 8 to 16); plain Coconut .325.
- Mean hops by listed steps: t22 s = 2-5 2.05 / 2.49 / 3.10 / 2.96; long twins s = 4-8 2.61 / 2.95 / 2.57 / 3.20 / 2.00. Plain Coconut long twins 1.58 / 1.44 / 1.29 / 1.34.
- F_0 .07 / .06 (plain Coconut ~.2). s = 4 long twins with >= 3 hops: .50.
- First token: agreement 1.00; first-token hops = full-answer hops on .969 / .939 of pairs.
**Predictions:** all held (above plain Coconut in every bin; no staircase; not falling with s, pooled s = 6-7 2.80 vs 2.61; F_0 <= .15; t32 checks; M_K = donor run on every pair).
**Interpretation:** Step supervision deepens Coconut's latent use on GSM8K (about one more crossing than CODI / SIM-CODI, 1.2-1.9 more than plain Coconut) but, as with SIM-CODI, the count does not grow with length past ~3 steps.
**Gotchas hit:** the released checkpoint needs 10 latents; at the adapter's default 6 it loses 10 points.
**Caveats:** 10 latents vs 6 for the other GSM8K models (no model's counts approach its cap); one released checkpoint.
**Next:** readability with SIM-CoT's own decoder (in the checkpoint, unused so far).
