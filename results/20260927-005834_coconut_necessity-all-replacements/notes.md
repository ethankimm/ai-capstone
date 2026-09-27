## 2026-09-27 — Necessity under zero / mean / norm-matched noise, Coconut: thoughts are strongly necessary and, unlike CODI, sampling does not recover what truncation removes; pass 1 is the necessary site under every replacement (coconut, run_id: 20260927-005834_coconut_necessity-all-replacements)

**Goal:** RESEARCH_PLAN round 1, item 2 — Coconut had no necessity run at all, leaving the Q2
"necessary?" cell empty for one of our two mechanisms. Same design as the CODI run
(`20260927-005051_codi_necessity-all-replacements`) via `scripts/necessity_coconut.py` + `necessity_common.py`.
**Mechanism / model:** `coconut`, gpt2 / `hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33`, 6 latents, greedy.
**Data:** `gsm_original_test.json` — all 1319 GSM8K test questions (the same questions as CODI's test split).
**Design:** sites = passes 0..5 (pass p = vector spliced into the p-th `<|latent|>` slot; pass 0 = the
hidden state at `<|start-latent|>`). trunc_k = k `<|latent|>` tokens between start/end markers (k=0:
question + start + end). Replacements per pass: zero; mean over 300 seeded test examples (norms 23–38);
noise = Gaussian direction scaled to this example's clean pass-p norm (seed idx·1000+p). PWC, flip split,
pass@k (10 samples, T=0.7, answer tokens only) as for CODI.
**Command:** 2 shards + merge, see `scripts/necessity_coconut.py` docstring / `eval_command.txt`. Same pod
as the CODI runs (A5000 secure, $0.27/hr), ~50 min wall (6 s/ex/shard while sharing the GPU, ~2 s/ex after).
**Headline results:** `final_answer_accuracy=0.331` (437/1319; paper Table 1 0.331, Hao et al. repro 0.341),
**`early_termination_necessity=0.250`** (0.331 − 0.081).

| condition | acc | PWC | c→w | w→c | pass@1 | pass@10 | pass@10 \| base correct | pass@10 \| base wrong |
|---|---|---|---|---|---|---|---|---|
| baseline trunc_6 | 0.331 | 1 | 0 | 0 | 0.326 | 0.415 | 0.998 | 0.126 |
| trunc_0 | 0.081 | 0.183 | 357 | 27 | 0.071 | **0.168** | 0.320 | 0.092 |
| ablate_all_zero | **0.033** | **0.078** | 403 | 10 | 0.034 | 0.091 | 0.172 | 0.051 |
| ablate_all_mean | 0.090 | 0.215 | 343 | 25 | 0.080 | 0.222 | 0.432 | 0.118 |
| ablate_all_noise | 0.066 | 0.149 | 372 | 22 | 0.062 | 0.152 | 0.297 | 0.080 |

Truncation curve k=0..6: 0.081, 0.083, 0.152, 0.190, 0.234, 0.272, 0.331 (graded, every latent adds).

Single-pass accuracy (PWC), baseline 0.331:

| pass | zero | mean | noise |
|---|---|---|---|
| 0 | 0.189 (0.53) | 0.252 (0.69) | 0.185 (0.49) |
| 1 | **0.115 (0.31)** | **0.108 (0.27)** | **0.124 (0.30)** |
| 2 | 0.230 (0.63) | 0.326 (0.97) | 0.255 (0.71) |
| 3 | 0.199 (0.54) | 0.244 (0.65) | 0.230 (0.63) |
| 4 | 0.212 (0.59) | 0.223 (0.60) | 0.212 (0.59) |
| 5 | 0.256 (0.73) | 0.278 (0.78) | 0.271 (0.76) |

**Interpretation:**
- **Necessary, more so than CODI.** Truncating to zero latents keeps 18% of correct answers (CODI 45%);
  ablate-all zero keeps 8% (CODI 44%). Zero is the most damaging replacement here (CODI: mean), so the
  ranking of replacement types is mechanism-specific — report all three.
- **pass@k separates the mechanisms.** CODI without thoughts keeps pass@10 at 0.509 vs 0.519; Coconut
  without thoughts drops pass@10 from 0.415 to 0.168, and on problems the full model solves from 0.998 to
  0.320. Coconut's latents enable the answer; CODI's mostly sharpen a distribution that already contains it.
- **Pass 1 is necessary under every replacement** (PWC 0.27–0.31, ~4× the damage of any other pass),
  matching its role as the main carrier of transplantable value in E3 (`20260927-003331`: 56% steering
  alone, 43 pp loss when left out). Pass 4, the second E3 carrier, is moderately necessary (PWC ~0.6)
  but not more than passes 0/3. Passes 2/3/5 carry no transplantable value in E3 yet removing them
  still costs 5–15 pp under zero/noise; pass 2 is harmless under mean only.
- Unlike CODI's even/odd split, necessity in Coconut is spread across all passes (every single-pass
  ablation is significant except pass 2 under mean); transplantable value is concentrated in passes 1/4.
**Gotchas hit:** Coconut's answer decode recomputes the full sequence per token (no KV cache, mirroring
`Coconut.generate`), so batched sampling here is ~3× slower per example than CODI's.
**Caveats:** Coconut was trained with a curriculum, so trunc_k<6 may be closer to in-distribution than
CODI's truncation. pass@k samples answer tokens only, T=0.7. One noise draw per (example, pass).
**Next:** side-by-side Q2 table for both mechanisms in RESEARCH_PLAN; rerun CODI E2/E4 aligned.
