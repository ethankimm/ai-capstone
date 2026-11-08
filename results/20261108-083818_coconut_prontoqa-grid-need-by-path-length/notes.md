## 2026-11-08 — ProntoQA grid need test: Coconut .991 with 0 latent passes vs .990 with 6; CODI .946 vs .953. No need for latent passes; no-CoT .879, CoT .993 (coconut, run_id: 20261108-083818_coconut_prontoqa-grid-need-by-path-length)

**Goal:** As the ProsQA record (same session): do grid latent models need their latent passes on ProntoQA? Pre-registered (t16, commit 3574431).
**Mechanism / model:** `connordilgren/gpt2-prontoqa-{coconut,codi,no-cot,cot}`; Coconut k = 0..6, CODI 6 and 0 latents.
**Data:** ProntoQA test, all 800. Every item has exactly 6 gold steps, so there is no path-length split (one bin).
**Hyperparams / command:** as the ProsQA record; `grid_accuracy.py prontoqa <method> mps <out>`; raw JSON `codi_audit_artifacts/t16_prontoqa_*.json`.
**Results:** Coconut k = 0..6: .991 .991 .991 .991 .991 .990 .990. CODI 6 latents .953, 0 latents .946 (need +.007). no-CoT .879, CoT .993.
**Interpretation:** Matches the prediction (within .03). Coconut and CODI both beat no-CoT by .07-.11, but neither needs its latent passes; the gap is a training effect. Binary answers.
**Caveats:** no depth variation in this test set; one checkpoint per method; 0 latents off-distribution.
**Next:** t17 route split.
