## 2026-10-09 — CODI Llama-3.2-1B: reproduction gate + E3 twin transplant (codi, run_id: 20261009-102724_codi_llama1b-minimal-pair-patch)

**Goal:** First cross-family check. (1) Reproduction gate: does the released Llama-3.2-1B CODI checkpoint
reproduce the paper's GSM8K-Aug accuracy under our eval code? (2) Does the GPT-2 E3 minimal-pair (twin)
transplant result (aligned all-slot 0.839, value on z0/z2/z4; `20260927-004340_codi_minimal-pair-patch-aligned`)
hold on a different, 8x larger model family? The design is identical to the GPT-2 run: same script, same caps
and same aligned site convention (patch site s as `{s+1: twin z_s}`).
**Mechanism / model:** `codi`, backbone `unsloth/Llama-3.2-1B-Instruct@5a8abab` (ungated mirror of
`meta-llama/Llama-3.2-1B-Instruct`; tokenizer/config patched to Meta's defaults: no pad token,
PreTrainedTokenizerFast, no pad_token_id) + `connordilgren/llama32-1b-gsm8k-codi@5f670c7` (Dilgren & Wiegreffe
release), compute_steps=6.
**Data:** gsm8k-aug test, full n=1319; run seed=11.
**Hyperparams:** bf16, LoRA r=128 alpha=32 on q/k/v/o/up/down/gate_proj, prj_dim=2048, num_latent=6, greedy;
pair caps 1000 / non-responsive cap 300; deltas ±1..3.
**Command:** see `eval_command.txt` (run from `/workspace/codi` with the CODI env; `--model_name_or_path
/workspace/llama32-1b-instruct --ckpt_dir /workspace/codi_llama --prj_dim 2048 --eval_n 0 --n_pairs 1000
--n_nonresponsive 300`). Code changes needed: `eval_codi.build_model` picks LoRA targets by family;
`patch_minimal_pair_codi.py` records the real backbone.
**Headline results:**
- **Gate passes:** base accuracy **0.566** (746/1319) vs 0.560 in Dilgren & Wiegreffe Table 1 (CODI paper 0.556).
  Checkpoint loaded with missing=0 unexpected=0.
- Twin transplant, n=471 qualified pairs:

| condition | Llama-1B matches_twin | changed | GPT-2 (n=317) |
|---|---|---|---|
| **ALL-SLOT aligned** | **0.692** [0.649, 0.732] | 0.828 | 0.839 [0.795, 0.875] |
| ALL-SLOT legacy | 0.594 | 0.773 | 0.713 |

| site | single | leave-one-out | prefix z0..s | GPT-2 single / LOO |
|---|---|---|---|---|
| z0 | 0.000 | 0.699 | 0.000 | 0.199 / 0.729 |
| z1 | **0.384** | **0.251** | 0.384 | 0.003 / 0.839 |
| z2 | 0.028 | 0.686 | 0.467 | 0.202 / 0.659 |
| z3 | 0.000 | 0.696 | 0.469 | 0.000 / 0.836 |
| z4 | **0.217** | **0.482** | 0.667 | 0.360 / 0.546 |
| z5 | 0.004 | 0.667 | 0.692 | 0.000 / 0.842 |

- Non-responsive pairs (n=272, twin answered wrong): all-slot moves the recipient to the twin's wrong answer
  0.474 [0.416, 0.534] (GPT-2 0.533).
- CFR: P(twin correct | original correct) = 471/743 = 0.634 [0.599, 0.668] (GPT-2 0.581);
  P(twin correct | original wrong) = 0.085.

**Interpretation:** The core Q1 finding replicates across families: transplanting the twin's latents moves
the answer to the twin's answer in most pairs (0.69), and also carries the twin's wrong answers. That means the
latents carry the problem-specific value, not just a "be correct" signal. The effect is weaker than on GPT-2
(0.69 vs 0.84, non-overlapping CIs). Localization is sparse in both models: a few slots carry the value and
the rest are inert (single ≈ 0, LOO ≈ all-slot). **The slots differ, though.** GPT-2 uses z0/z2/z4 (even
slots), while Llama uses z1 (strongest) and z4, and z0 is inert. So the qualitative claim ports ("value lives
in a sparse subset of slots"), but a slot-index-level claim does not. Any cross-model mapping needs to be
learned per model, not assumed by position.
**Gotchas hit:** the gated meta-llama repo needs a token, so we used the unsloth mirror with the tokenizer
config patched to Meta's. LoRA target modules had been hard-coded to GPT-2 names (c_attn etc.), which would
have silently loaded nothing for Llama. Fixed, and the load is confirmed by missing=0 unexpected=0.
**Caveats:** single seed, greedy. Qualified pairs differ between models (each model's own correct originals),
so the GPT-2 vs Llama comparison is on different pair sets. The legacy row is kept only for continuity.
**Next:** Q2 decode control on Llama (does a probe read the value at z1/z4 where the transplant says it lives?);
a within-method baseline (CODI GPT-2 seed vs seed) to answer the advisor's pretraining-artifact concern.
