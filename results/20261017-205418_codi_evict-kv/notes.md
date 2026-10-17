## 2026-10-17 — Answer-phase KV eviction: CODI's answer reads slots 1 and 5 (and a little of 3), mostly at layers 8-11; partial eviction is worse than evicting everything; mean-merging the 6 slots is worth nothing (codi, run_id: 20261017-205418_codi_evict-kv)

**Goal:** Idea #4 of the KV/latent-efficiency brainstorm: use KV-cache eviction as a causal
test. The 6 continuous thoughts are computed honestly (no patching, no ablation of the latents
themselves); only the **answer phase** (eot + every generated answer token) is blocked from
attending to chosen thought positions at chosen layers. This isolates the answer → latent
attention read, which the necessity runs (`20260927-005051_codi_necessity-all-replacements`)
can't separate from the latents' effect on each other. Three questions: which slots does the
answer need, which slots suffice, and at which layers is the read? Also one compression
variant: can the 6 slots' K/V be mean-pooled into 1?
**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3` (released checkpoint), 6
thoughts, paper inference protocol (LoRA r=128/α=32, projection 768+LN, greedy), **fp32 on
CPU** (the earlier CODI runs were bf16 on CUDA).
**Data:** gsm8k-aug test, all 1319 examples in file order; seed 11.
**Design:** A forward pre-hook on each of the 12 `GPT2Attention` modules adds `finfo.min` to the
chosen key columns while the answer phase runs. Slot i = the sequence position of iteration i's
input (absolute `L0 + i - 1`), the same convention as `override_input_at` and
`ablate_attention_codi.py`. So slot 1 holds the projected bot-position state (the "latent 0"),
and slot i holds iteration i−1's output. The 6th iteration's output is never written to the
cache. Checked on one example with `output_attentions`: masking slots {1,3} at layers 4-7 gives
exactly 0 mass there and leaves the other 8 layers untouched. Conditions: full;
evict_all; keep_only_i / evict_only_i; keep_135 (the slots the patching runs found carry the
value) vs keep_246; evict one layer band (0-3 / 4-7 / 8-11) for all slots, or keep only one band;
merge_mean (per layer, the 6 slots' K and V averaged into slot 1, slots 2-6 evicted). Paired
McNemar against full.
**Command:** (local CODI checkout at `codi@2c23146` + `codi_streaming.patch`, venv from its
`requirements.txt` minus the `nvidia-*`/`triton` lines)
```
HF_HUB_OFFLINE=1 .venv/bin/python <repo>/scripts/evict_kv_codi.py \
  --ckpt_dir ~/.cache/huggingface/hub/models--zen-E--CODI-gpt2/snapshots/fd641b3d3edc59e4f534b55588e906588c9e36bb/ \
  --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" --model_name_or_path gpt2 --seed 11 --model_max_length 512 \
  --lora_r 128 --lora_alpha 32 --lora_init --greedy True --num_latent 6 --use_prj True --prj_dim 768 \
  --prj_no_ln False --prj_dropout 0.0 --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True \
  --use_lora True --output_dir /tmp/x --device cpu --n 0
```
2916 s on an Apple Silicon CPU (2.2 s/example for all 23 conditions), $0.
**Headline results** (n=1319; "lost/gained" = flips vs full, paired):

| condition | acc | 95% CI | lost | gained |
|---|---|---|---|---|
| full | 0.425 | [0.398, 0.451] | – | – |
| evict_all | 0.229 | [0.207, 0.252] | 316 | 58 |
| keep_only_1 / 2 / 3 / 4 / 5 / 6 | 0.262 / 0.220 / 0.221 / 0.221 / 0.227 / 0.216 | | | |
| evict_only_1 | 0.362 | [0.336, 0.388] | 99 | 16 |
| evict_only_2 | 0.425 | | 8 | 9 |
| evict_only_3 | 0.407 | | 41 | 18 (p=0.004) |
| evict_only_4 | 0.422 | | 14 | 10 |
| evict_only_5 | 0.369 | [0.344, 0.396] | 103 | 30 |
| evict_only_6 | 0.418 | | 18 | 9 |
| keep_135 | 0.369 | [0.344, 0.396] | 95 | 22 |
| keep_246 | **0.109** | [0.093, 0.127] | 445 | 29 |
| evict_band L0-3 / L4-7 / L8-11 | 0.306 / 0.369 / **0.127** | | | |
| keep_band L0-3 / L4-7 / L8-11 | 0.113 / 0.181 / 0.262 | | | |
| merge_mean | 0.230 | [0.208, 0.253] | 309 | 52 |

**Interpretation:**
- **The attention read is worth ~20 points.** Evicting all 6 slots drops accuracy 42.5 → 22.9.
  The latents still exist, but the answer can't see them.
- **Necessity is concentrated in slots 1 and 5, with slot 3 a distant third.** Evicting any one
  of 2, 4 or 6 alone costs nothing. This is an independent confirmation of the patching result
  (value concentrated in slots 1, 3, 5). It uses a different intervention: eviction at read time,
  not content replacement at write time.
- **No single slot is sufficient.** keep_only_i ≈ evict_all for every i. Slot 1 is slightly
  above, at 26%.
- **Slots 1, 3 and 5 together recover ~71% of the latent contribution:** (36.9 − 22.9) /
  (42.5 − 22.9). Slots 2, 4 and 6 are each individually free to drop, but dropping them jointly
  costs 5.6 points, so they are partly redundant with one another.
- **Partial eviction can be worse than total eviction.** keep_246 (10.9%), evict L8-11 (12.7%),
  keep L0-3 (11.3%) and keep L4-7 (18.1%) are all *below* evict_all (22.9%). Reading some of
  the latent channel without the rest misleads the answer more than reading none of it. So the
  read is coordinated across slots and layers, not additive.
- **The read happens mostly late (layers 8-11), but no band is sufficient on its own.**
  keep_band L8-11 is only 26.2%.
- **Naive compression fails.** Mean-pooling the 6 slots' K/V into one entry is no better than
  evicting all of them. The slots are not interchangeable. For the efficiency framing: you can
  drop single slots 2/4/6 for free and half the latent KV (keep_135) at −5.6 points, but there
  is no free lunch here at 6 slots.

**Gotchas hit:** under transformers 4.52.4, CODI's GPT-2 returns the legacy tuple KV cache (not a
`DynamicCache`). Inside the attention call it *is* a `DynamicCache`, which is what the mask hook
uses for the key length. `eval_codi.build_model` hard-coded `cuda` + bf16; it now takes `device`
/ `dtype` kwargs, with defaults unchanged.
**Caveats:**
- fp32/CPU full accuracy is 42.5% on the full test set, vs 43.7% for bf16/CUDA in
  `20260918-021217_codi_released-weights-6lat`. Compare conditions only within this run.
- An additive mask renormalises the softmax: the evicted mass goes to the remaining keys (mostly
  question tokens and the attention sink). Some of the "worse than evict_all" effect may be this
  redistribution putting more weight on the surviving latent slots, rather than "misleading
  content". A value-zeroing variant (keep the key, zero V) would separate the two.
- The mask applies to every answer-phase token, so later answer tokens also lose access through
  earlier answer tokens' own reads.
- One checkpoint, one task, greedy decoding only.
**Next:** (1) a value-zero variant to separate renormalisation from content; (2) the same
harness on Coconut (slots 2 and 5 per patching) for the cross-method comparison; (3) Ouro
per-loop KV (idea #5) needs a pod.
