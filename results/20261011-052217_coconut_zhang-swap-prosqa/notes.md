## 2026-10-11 — Zhang-style random swap, coconut on prosqa (coconut, run_id: 20261011-052217_coconut_zhang-swap-prosqa)

**Goal:** Pilot from PILOT_ZHANG_SWAP.md: replicate Zhang et al. 2025's random-permutation latent/CoT swap (arXiv 2512.21711, Table 3) on the Dilgren & Wiegreffe GPT-2 grid, next to our E3 twin swap on the same recipients (GSM8K only). Does the 'latents are placeholders' verdict depend on the intervention, the task, or the method?
**Mechanism / model:** `coconut`, openai-community/gpt2 / hf:connordilgren/gpt2-prosqa-coconut/checkpoint_40, compute_steps=6.
**Data:** prosqa test n=500 seed=0; run seed=0.
**Hyperparams:** {'method': 'coconut', 'num_latents': 6, 'perm_seed': 0, 'n_twin_cap': 400, 'max_new_tokens': 48, 'deltas': [1, -1, 2, -2, 3, -3]}
**Command:** `scripts/swap_random_coconut.py --method coconut --dataset prosqa --checkpoint_path <grid>/gpt2-prosqa-coconut/checkpoint_40 --checkpoint_label hf:connordilgren/gpt2-prosqa-coconut/checkpoint_40 --data_dir ~/Projects/are-lrms-easily-interpretable/data --device cpu --slug zhang-swap-prosqa --stage pilot --hardware 'local Apple Silicon CPU'  (venv pinned to Coconut's requirements: torch 2.5.1, transformers 4.46.2)`
**Headline results:** `final_answer_accuracy=0.980`, `compute_steps=6`
**Results:** Random swap: accuracy 0.980 -> 0.980, IR 0.0% [0, 0.76], steer 0/499 (n=500).
**Interpretation:** Same as ProntoQA: placeholder verdict, latents unused. Consistent with Dilgren & Wiegreffe (latent = no-CoT on ProsQA) and with Jin et al.'s late-checkpoint ProsQA nulls.
**Gotchas hit:** The local transformers 5.x breaks coconut_common's legacy KV-cache slicing, so I used a scratch venv pinned to Coconut's requirements. Grid CoT/no-CoT checkpoints keep the base vocab (50257), while the Coconut ones have +3 latent tokens; the loader now resizes to the checkpoint's vocab.
**Caveats:** Single seed and permutation. GPT-2 only. CODI is not in this pilot (its scripts hard-code CUDA): CODI GSM8K/ProntoQA/ProsQA is the next step. Steer is defined as P(swapped answer == donor's own base answer AND != recipient's base answer | donor answer differs). On ProntoQA, binary answers make ~50% steer the chance level. Swaps are full-chain (all 6 latent passes / the whole reasoning text). The CoT splice re-tokenizes the donor's decoded text.
**Next:** CODI rows (GSM8K zen-E + grid ProntoQA/ProsQA, on RunPod); Llama-1B grid; no-CoT parity check on our own pool; decide the paper angle (PILOT_ZHANG_SWAP.md).
