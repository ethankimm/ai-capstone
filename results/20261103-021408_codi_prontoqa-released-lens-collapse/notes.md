## 2026-11-03 — Released ProntoQA CODI: latents are one constant vector for every item and position (99.3-99.9% of energy shared, raw cross-item cos 1.000); 0 latents = 6 latents (0.946 vs 0.953) (codi, run_id: 20261103-021408_codi_prontoqa-released-lens-collapse)

**Goal:** ProntoQA half of the logic-task collapse check; full story in `20261103-021023_codi_prosqa-released-lens-collapse`.
**Mechanism / model:** `codi`, `connordilgren/gpt2-prontoqa-codi`, grid `CODIInference`, 6 latents.
**Data:** D&W `prontoqa_test.json`, all 800.
**Hyperparams:** identical to the ProsQA run.
**Command:** `scripts/logic_codi_lens_collapse.py --dataset prontoqa`, see `eval_command.txt`.
**Headline results:** accuracy 0.953 (D&W 94.8; our Oct 11 run 0.953); **0 latents 0.946**. Raw within-item cos 0.998-0.999, **raw cross-item cos 1.000** at every layer. Energy share: constant 0.9925-0.9989 at every layer (layer 12: 0.997, item 0.002, position 0.0005). Logit lens: 'ito' (layer 0) / ',' (layer 12) for every item; step-word hits 0.000.
**Interpretation:** Same as ProsQA: a global constant latent, inert (0 latents and random swap change nothing). Strongest collapse of the four settings checked.
**Gotchas hit:** Centred "share of variance between positions" is 0.91-0.95 here, which would look un-collapsed on its own; it is a fraction of a residual that is ~0.1-0.7% of the energy. Use the decomposition.
**Caveats:** As ProsQA. ProntoQA answers are binary, so accuracy differences are coarse.
**Next:** see the ProsQA run.
