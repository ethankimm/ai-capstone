## 2026-11-08 — Attention shares do not separate GSM8K CODI from start-last CODI: GSM8K latents put .08-.36 of their attention output on earlier latents (start-last .09-.27), with value sites reading mostly the punctuation sites before them. The regimes differ in what the attended entries carry (route split), not in how much attention flows (codi, run_id: 20261108-061510_codi_gsm8k-released-attention-flow)

**Goal:** The missing GSM8K row of the attention-flow comparison (`20261108-040037_..._latent-attention-flow`). Is GSM8K CODI's look-back visibly different from start-last CODI's, enough to tell a horizontal chain from width by attention alone?
**Mechanism / model:** `zen-E/CODI-gpt2` @ fd641b3, fp32 MPS, batch 1, as t11 / t12.
**Data:** all 1,319 GSM8K-Aug test items (560 answered right, 759 wrong).
**Command:** `~/Documents/Penn/CIS5980/codi_audit_artifacts/t13_gsm8k_attn_flow.py` (CODI venv, `run_guarded.sh`, `caffeinate`), record `record18.py`; 20-item smoke first.
**Measures:** t9's measures for the query of latent iteration s (input z_s), every layer:
- raw attention;
- norm share ||sum_h a v W_O|| (W_O from c_proj, LoRA included).

Key sets: question (sink excluded), earlier even sites, earlier odd sites, self, sink. Shares non-sink, mean over layers. The sink takes .45-.49 of raw attention.

**Headline results** (items answered right; wrong items within .04):

| Site | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| question | .90 | .81 | .65 | .72 | .56 | .68 |
| earlier latents | .00 | .08 | .26 | .20 | .36 | .26 |
| of which even (value) sites | .00 | .08 | .05 | .12 | .10 | .15 |
| of which odd (punctuation) sites | .00 | .00 | .21 | .07 | .27 | .12 |

- **About as much look-back as start-last.** Earlier latents get .08-.36 here and .09-.27 in start-last CODI (L3-L5), with the same rise across passes. Raw attention agrees (GSM8K .09-.41).
- **Alternating, not uniform.** Value sites z2 and z4 look back more (.26, .36) than the punctuation sites after them (.20, .26), and they read mostly the punctuation site just before them: z4's look-back is .48 at distance 1 (z3).
- **Right and wrong items look the same** (within .04).

**Predictions (`t_predictions.md`):**
- *More look-back than start-last:* only slightly, and only at z4.
- *Concentrated on even (value) sites:* **wrong**. Most of it goes to odd sites.

**Interpretation:** On attention shares alone, GSM8K CODI and start-last CODI cannot be told apart: both spend a quarter to a third of a late latent's attention on earlier latents. What separates them is what those entries carry, and that comes from interventions:
- In GSM8K CODI a single site's keys/values move the answer, and the first value is relayed by later latents (`20261108-061509_..._route-split`).
- In start-last CODI single sites never move the answer and the passes act as interchangeable votes (`20261108-032531`).

The heavy attention to odd sites fits the odd/even look-back control (cutting odd sites costs as much as cutting even ones). Odd sites hold no answer-relevant value, but later latents use them, perhaps as structure. This is not tested further here.
**Gotchas hit:** None.
**Caveats:**
- Descriptive. Shares mix every layer and head.
- One released checkpoint.
- The norm measure ignores MLPs and later layers.
**Next:** full-test depth split (t14). Then the regime table across both tasks.
