## 2026-10-11 — Zhang-style random swap, codi on prontoqa (codi, run_id: 20261011-061753_codi_zhang-swap-prontoqa)

**Goal:** CODI rows of the Zhang-swap pilot (PILOT_ZHANG_SWAP.md): does the pattern from Coconut (random swap = placeholder on logic tasks, load-bearing but content-blind on GSM8K; twin swap reads the content) hold for CODI?
**Mechanism / model:** `codi`, gpt2 / hf:connordilgren/gpt2-prontoqa-codi/pytorch_model.bin, compute_steps=6.
**Data:** prontoqa test n=800 seed=0; run seed=0.
**Hyperparams:** {'method': 'codi', 'num_latents': 6, 'perm_seed': 0, 'n_twin_cap': 400, 'max_new_tokens': 64, 'deltas': [1, -1, 2, -2, 3, -3], 'dtype': 'float32', 'greedy': True}
**Command:** `scripts/swap_random_codi.py --dataset prontoqa --checkpoint_path <grid>/gpt2-prontoqa-codi/pytorch_model.bin --grid_repo ~/Projects/are-lrms-easily-interpretable --data_dir ~/Projects/are-lrms-easily-interpretable/data --device cpu --fp32 True --slug zhang-swap-prontoqa --stage pilot --hardware 'local Apple Silicon CPU'  (venv: torch 2.5.1, transformers 4.46.2, peft 0.15.2)`
**Headline results:** `final_answer_accuracy=0.953`, `compute_steps=6`
**Results:** Base accuracy 0.953 (Dilgren & Wiegreffe report 94.8). Random swap: accuracy 0.953 -> 0.953, IR 0.0% [0, 0.48], steer 0 (n=800).
**Interpretation:** Placeholder verdict, same as Coconut on ProntoQA: CODI ignores its latents on this task, consistent with Dilgren's latent = no-CoT parity.
**Gotchas hit:** `eval_codi.build_model` hard-codes CUDA, so this loads with the grid repo's device-agnostic `CODIInference` and reimplements its latent loop with an override hook. bf16 on CPU took ~3.6 s/example, so it is cast to fp32 (~35x faster). On the n=30 smoke test, 1/30 base answers differed from bf16; full-set base accuracy matches the references.
**Caveats:** fp32, not the reference bf16 protocol. Single permutation seed. GPT-2 only. Full-chain swap (z_0..z_5, aligned). On ProntoQA, binary answers put chance-level steer at ~50%.
**Next:** Llama-1B grid; write up the use-vs-content table across methods and tasks. Multi-dataset direction: see MULTI_DATASET.md.
