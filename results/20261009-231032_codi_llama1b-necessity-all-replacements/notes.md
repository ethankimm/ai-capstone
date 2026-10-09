## 2026-10-09 — CODI Llama-3.2-1B: necessity of the latents under every replacement (codi, run_id: 20261009-231032_codi_llama1b-necessity-all-replacements)

**Goal:** CODI counterpart to `20261009-231035_coconut_llama1b-necessity-all-replacements`. Same design as GPT-2
`20260927-005051_codi_necessity-all-replacements`. How much does Llama-1B CODI need its latents, and do the
necessary slots match the transplant carriers z1/z4 (`20261009-102724_codi_llama1b-minimal-pair-patch`)?
**Mechanism / model:** `codi`, `unsloth/Llama-3.2-1B-Instruct@5a8abab` + `connordilgren/llama32-1b-gsm8k-codi@5f670c7`,
bf16, LoRA r=128 on q/k/v/o/up/down/gate, prj_dim 2048, 6 latents.
**Data:** gsm8k-aug test, all 1319; run seed 11.
**Hyperparams:** greedy; mean over 300 test examples; pass@10 at T=0.7; max_new_tokens 64.
**Command:** `eval_command.txt`; 3 shards + merge. Code change: `necessity_codi.py` records the real backbone.
**Headline results:** baseline 0.566 (746/1319; paper 0.560). `early_termination_necessity=0.090`.

| condition | Llama acc (PWC) | GPT-2 acc (PWC) | Llama pass@10 | GPT-2 pass@10 |
|---|---|---|---|---|
| baseline | 0.566 (1) | 0.419 (1) | 0.639 | 0.519 |
| trunc_0 | 0.475 (0.73) | 0.238 (0.45) | **0.711** | 0.509 |
| ablate_all_zero | 0.444 (0.69) | 0.231 (0.44) | 0.622 | 0.494 |
| ablate_all_mean | **0.315 (0.52)** | 0.105 (0.20) | 0.456 | 0.307 |
| ablate_all_noise | 0.478 (0.74) | 0.141 (0.26) | 0.656 | 0.397 |

Truncation curve k=0..6: 0.475, 0.428, 0.500, 0.493, 0.475, 0.566, 0.566. It is not monotone: one latent is
worse than none, and the jump comes at k=5.
Single-site PWC (zero / mean / noise): z0 0.91/0.90/0.86, **z1 0.82/0.76/0.83**, z2 0.85/0.96/0.83,
z3 0.87/0.93/0.85, **z4 0.88/0.83/0.87**, z5 0.98/0.98/0.98 (n.s. under all three).

**Interpretation:**
- **Necessary, but less than on GPT-2.** Every ablate-all condition is significant (McNemar p ≤ 4e-12), and mean
  is again the most damaging replacement (PWC 0.52, worse than no latents, the same pattern as GPT-2). But
  no latents still keeps 73% of correct answers (GPT-2 45%). Without latents, pass@10 is *higher* than with them (0.711 vs
  0.639): as on GPT-2, the latents concentrate probability on one answer rather than being the only route to it.
- **Necessity matches the transplant carriers.** Under mean replacement, the two most necessary sites are z1 (0.76) and
  z4 (0.83), exactly the slots that carry transplantable value (single-slot steering 0.38 / 0.22). z5 is
  unnecessary, as on GPT-2. Two independent tests (necessity, sufficiency for steering) pick the same
  slots on Llama, and on GPT-2 they picked different ones (z0/z2/z4 for steering, with z0/z4 most necessary).
- Across both methods, the 1B model leans on its latents less than GPT-2 small does. CODI keeps most of its
  latent-content story (transplantable, necessary at the carrier slots), and Coconut keeps almost none of it beyond pass 0.
**Gotchas hit:** a second, byte-identical merge of the same shards (`20261009-220539_codi_llama1b-necessity-all-replacements`)
appeared on the pod at 22:05 UTC, probably from a parallel session. Not copied back; this run is the scripted merge.
**Caveats:** single seed. The non-monotone truncation curve suggests that truncated CODI is off-distribution (it was
trained with exactly 6 latents), so truncation is a weaker probe of CODI than ablation.
**Next:** large-delta twin rerun with the hidden-operand filter; within-method baseline.
