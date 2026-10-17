## 2026-10-17 — Answer-phase KV eviction, Coconut: the answer reads the later slots (5, 6, then 2) almost only at layers 8-11, and no partial eviction is worse than evicting everything, unlike CODI (coconut, run_id: 20261017-212529_coconut_evict-kv)

**Goal:** The Coconut mirror of `20261017-205418_codi_evict-kv`. The 6 thoughts are computed
honestly; only the answer phase is blocked from attending to chosen thought slots at chosen
layers. Which slots does the answer read, at which layers, and does Coconut show CODI's
"partial eviction is worse than none" pattern?
**Mechanism / model:** `coconut`, gpt2 / `hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33`
(released, loaded into the vanilla module via `coconut_common.load_coconut`), 6 latents, greedy,
max_new_tokens 48, fp32 on CPU (transformers 4.52.4).
**Data:** `gsm_original_test.json` (Dilgren & Wiegreffe prep, the same 1319 GSM8K test questions
as CODI's split), all 1319 in file order; answer after `###`.
**Design:** Same mask and condition grid as the CODI run, from the new shared module
`scripts/evict_kv_common.py` (the CODI script was refactored onto it, and reproduces its
committed outputs on the first 8 examples). Coconut differences:
- The latent passes are `coconut_common.run_passes`, unmasked.
- The answer phase starts at `<|end-latent|>`. The last pass (slot 6 + `<|end-latent|>`) and the
  decode run with a KV cache, and the mask hits only queries at or after `<|end-latent|>`.
  `--check_reference` compared the unmasked cached decode with `finish_and_decode` (the reference
  no-cache `Coconut.generate` tail) on every example: **0 mismatches / 1319**.
- Slot i = the i-th `<|latent|>` = pass i−1. So the necessity/patching runs' passes 1 and 4 are
  slots 2 and 5. Extra conditions: keep_25 (the patching carriers), keep_1346 (their complement),
  and keep_135 / keep_246 for comparison with CODI.
**Command:**
```
HF_HUB_OFFLINE=1 <codi-venv>/bin/python scripts/evict_kv_coconut.py \
  --checkpoint_path ~/.cache/huggingface/hub/models--connordilgren--gpt2-gsm8k-coconut/snapshots/24f1422fcf6d557fd359fc818a4d1d0fde4aac7f/checkpoint_33 \
  --data_file ~/Projects/are-lrms-easily-interpretable/data/gsm_original_test.json --n 0 --check_reference True
```
1256 s on an Apple Silicon CPU (25 conditions + the reference decode), $0.
**Headline results** (n=1319, paired vs full):

| condition | acc | 95% CI | lost | gained |
|---|---|---|---|---|
| full | 0.331 | [0.306, 0.357] | – | – |
| evict_all | 0.082 | [0.068, 0.098] | 356 | 27 |
| keep_only_1 / 2 / 3 / 4 / 5 / 6 | 0.089 / 0.138 / 0.094 / 0.149 / 0.187 / 0.221 | | | |
| evict_only_1 / 3 / 4 | 0.327 / 0.328 / 0.324 | | ≤24 | (all p > 0.1) |
| evict_only_2 | 0.296 | | 64 | 18 |
| evict_only_5 | 0.272 | | 104 | 26 |
| evict_only_6 | 0.271 | | 101 | 21 |
| keep_25 / keep_1346 | 0.232 / 0.241 | | | |
| keep_135 / keep_246 | 0.196 / 0.262 | | | |
| evict_band L0-3 / L4-7 / L8-11 | 0.324 / 0.312 / **0.178** | | | |
| keep_band L0-3 / L4-7 / L8-11 | 0.081 / 0.149 / **0.310** | | | |
| merge_mean | 0.141 | [0.123, 0.161] | 276 | 25 |

Full accuracy 0.331 matches the earlier Coconut baseline (0.331 in
`20260927-005834_coconut_necessity-all-replacements`).

**Interpretation:**
- **The answer's attention read is worth ~25 points out of 33** (evict_all 0.082). That is
  proportionally much more than CODI (20 of 42.5). It is consistent with the necessity runs:
  Coconut depends on its latents more than CODI does.
- **Slots 2, 5 and 6 are read; slots 1, 3 and 4 are each free.** Single-slot sufficiency rises with
  position: slot 6 alone recovers 0.221, slot 1 alone nothing (0.089).
- **Read ≠ necessity.** Replacing pass 1 (slot 2) *content* was the most damaging intervention in
  the necessity run (~4× any other pass). Blocking the answer's direct *read* of slot 2 costs
  only 3.5 points. So slot 2 matters mostly by feeding the later thoughts, not by being read at
  answer time. Slot 6, the last thought, was outside the patching analysis and is read directly.
- **The read is almost entirely in layers 8-11.** Keeping only the top band recovers 0.310 of
  0.331, and evicting layers 0-3 or 4-7 is nearly free. Efficiency-wise, about 2/3 of the latent
  KV (layers 0-7) can be dropped for −2 points. CODI can't do this (keep_band L8-11 = 0.262 vs
  0.425).
- **No partial condition is worse than evict_all**, unlike CODI (where keep_246, evict L8-11,
  keep L0-3 and keep L4-7 all fell below evict_all). Coconut's read degrades gracefully and
  roughly additively; CODI's is coordinated.
- **Mean-merging keeps a little** (0.141 vs 0.082). Unlike CODI, the averaged entry isn't worthless.
  But see the caveat: only slots 1-5 are merged here.

**Gotchas hit:** the reference `finish_and_decode` runs without a cache, so a mask that only applies
to answer-phase queries needs query-position awareness. `AttnMasker` now masks only rows with
absolute position >= `ans_start`. For CODI that is every row, so behaviour there is unchanged.
**Caveats:**
- merge_mean merges slots 1-5 only. Slot 6's K/V is produced inside the answer-phase pass, so it
  isn't in the cache yet at merge time; it is evicted with 2-5. So this isn't identical to CODI's
  6-slot merge.
- The additive mask renormalises the softmax onto the remaining keys (same caveat as CODI).
- Greedy only, one checkpoint; fp32/CPU.
**Next:** a value-zero variant for both methods; depth2 needs trained Coconut/CODI checkpoints
(none exist yet).
