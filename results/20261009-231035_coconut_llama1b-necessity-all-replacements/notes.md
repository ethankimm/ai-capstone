## 2026-10-09 — Coconut Llama-3.2-1B: necessity of the latents under every replacement (coconut, run_id: 20261009-231035_coconut_llama1b-necessity-all-replacements)

**Goal:** Follow-up to the Llama twin transplant (`20261009-094526_coconut_llama1b-minimal-pair-patch`, all-slot
steering 0.216 vs GPT-2 0.827). Does Llama-1B Coconut use its continuous thoughts at all? Two readings to separate:
(a) the latents are mostly unused, or (b) they are needed, but their content isn't transplantable.
Same design as the GPT-2 run `20260927-005834_coconut_necessity-all-replacements`.
**Mechanism / model:** `coconut`, `unsloth/Llama-3.2-1B-Instruct@5a8abab` (tokenizer patched to Meta's defaults) +
`connordilgren/llama32-1b-gsm8k-coconut@49dc864/checkpoint_10`, 6 latent passes, fp32 (as `load_coconut` loads).
**Data:** `gsm_original_test.json`, all 1319 GSM8K test questions.
**Hyperparams:** greedy; mean over 300 test examples; pass@10 at T=0.7 (answer tokens only); max_new_tokens 48.
**Command:** `eval_command.txt`; 3 shards + merge (`run_necessity.sh` pattern from `necessity_coconut.py`'s docstring).
Code change: the sampled decode now stops on `<|eot_id|>` like `finish_and_decode`.
**Headline results:** baseline 0.359 (473/1319; paper 0.357). `early_termination_necessity=0.151`.

| condition | Llama acc (PWC) | GPT-2 acc (PWC) | Llama pass@10 | GPT-2 pass@10 |
|---|---|---|---|---|
| baseline (6 latents) | 0.359 (1) | 0.331 (1) | 0.491 | 0.415 |
| trunc_0 (no latents) | **0.208 (0.37)** | 0.081 (0.18) | 0.348 | 0.168 |
| ablate_all_zero | 0.272 (0.67) | 0.033 (0.08) | 0.431 | 0.091 |
| ablate_all_mean | 0.297 (0.75) | 0.090 (0.22) | 0.440 | 0.222 |
| ablate_all_noise | 0.259 (0.64) | 0.066 (0.15) | 0.415 | 0.152 |

Truncation curve k=0..6: 0.208, **0.320**, 0.330, 0.344, 0.352, 0.356, 0.359. On GPT-2 it is graded:
0.081, 0.083, 0.152, 0.190, 0.234, 0.272, 0.331.
Single-pass PWC (zero / mean / noise): p0 0.74/0.77/0.64, p1 0.84/0.83/0.76, p2 0.87/0.85/0.80,
p3 0.91/0.89/0.87, p4 0.94/0.92/0.91 (n.s. under zero), p5 0.97/0.94/0.95 (n.s.).

**Interpretation:**
- **Neither pure reading holds; the latents are needed but shallow.** Removing all latents loses 63% of
  correct answers (PWC 0.37, McNemar p=2e-24), so the latent block is needed. But a single latent recovers
  almost everything (0.208 → 0.320 of 0.359), and passes 2–5 add 4 pp in total. With all six present,
  content-free replacements (zero/mean/noise) keep 64–75% of correct answers; on GPT-2 they keep 8–22%.
  So most of Llama Coconut's work is the latent block existing, plus pass 0's content. Passes 1–5 contribute little.
- **This matches the transplant exactly.** There, pass 0 alone steered 0.19 of the 0.216 all-slot rate, and passes 1–5
  added ~2.5 pp. Necessity falls monotonically with pass index here (p0 most necessary, p4/p5 n.s.), the
  same ordering as transplant steering. On GPT-2 the carrier was pass 1 (PWC ~0.3, steering 0.56).
  Pass 0 is the hidden state at `<|start-latent|>`, computed from the question alone, so on Llama Coconut the
  "reasoning" that can be moved or removed is mostly a re-encoding of the question, not multi-step latent computation.
- **Scale shifts reliance off the latents.** Ablating all latents costs 33–36 pp on GPT-2 but only 6–10 pp on
  Llama, and Llama without latents (0.208) is 2.6× GPT-2 without latents (0.081). The 1B model answers more
  from the question directly. For the "do findings port" question this means the Coconut transplant null on
  Llama is real, but it is about where the computation lives, not a broken pipeline.
**Gotchas hit:** with 6 shards sharing one A40, Coconut ran at ~38 s/question per shard. The answer decode
recomputes the whole sequence per token (no KV cache, as in the reference generate), so the run took 4.7 h.
A KV-cached batched decode is worth adding for future Llama Coconut runs (validate against the gate first).
**Caveats:** single seed; greedy except the pass@10 columns. GPT-2 numbers come from `20260927-005834`
(same script and data).
**Next:** large-delta twin rerun with the hidden-operand filter (see the twin notes); within-method baseline.
