## 2026-09-23 — CODI decode+patch, top-10 added: closes some of the ANY-iteration gap, not the reproduction gap (codi, run_id: 20260923-044632_codi_decode-patch-full-eval-top10)

**Goal:** Extend the ANY-iteration decodability addendum on `20260919-184323_codi_decode-patch-full-eval`
(top1/top5 only, rescored locally from cached predictions) with top-10, which needed a fresh
model pass since only top-1/top-5 token strings were cached at decode time -- top-10 candidates
were never saved. Exact re-run of `20260919-184323`'s command (same checkpoint, same held-out
split, same patch-pair selection seed) with `scripts/decode_patch_codi.py` extended to compute
and cache `top10` alongside `top1`/`top5`, and both the matched-iteration and ANY-iteration
metrics extended to report top-10.

**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3` (released checkpoint),
compute_steps=6, paper inference protocol (LoRA r=128/α=32, projection 768+LN, greedy,
`model.eval()` fix in place).

**Data:** gsm8k-aug test, all 1319 examples; half A (idx even, n=660) fits `best_iter_for_step`,
half B (idx odd, n=659) is where every decoding number and patch pair comes from. Train-corpus
baseline from 20,000 train examples. 350 patch pairs (227 focus iter 2, 123 other live iters),
`random.Random(0)` -- identical pair selection to `20260919-184323`.

**Command:**
```
cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/decode_patch_codi.py \
  --ckpt_dir <hf cache path for zen-E/CODI-gpt2@fd641b3> --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \
  --slug decode-patch-full-eval-top10 --stage full_run --hardware "RunPod RTX A5000 (secure)" \
  --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \
  --lora_r 128 --lora_alpha 32 --lora_init --greedy True \
  --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
  --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \
  --full_test True --mapping_split True --n_patch_pairs 350
```
Fresh RunPod RTX A5000 (secure, CA-MTL-1, $0.27/hr). CODI env via `codi_setup.sh` (~2 min).
Decode pass 264s (n=1319) + patch sweep 350 pairs (~3 min). Total pod uptime including
provisioning, both mechanisms' env setup (Coconut ran on the same pod, see its own run),
checkpoint downloads, and result transfer back: ~14 min ≈ **$0.06**.

**Headline results:** `final_answer_accuracy=0.419` (553/1319, matches `20260919-184323`
exactly -- same checkpoint/protocol/seed, confirms this is a faithful re-run, not a new
measurement).

| | top1 | top5 | top10 |
|---|---|---|---|
| matched-iteration (held out, n=2126) | 0.1834 | 0.2658 | **0.3189** |
| ANY-iteration (n=2126) | 0.2300 | 0.3358 | **0.4097** |

Paper-style metric (Table 3, correct-only, by step count):

| steps | matched-iter top5 | ANY-iter top5 | **ANY-iter top10** | paper (Shen et al.) |
|---|---|---|---|---|
| 1 (n=9) | 0.667 | 0.667 | **0.778** | 0.971 |
| 2 (n=119) | 0.067 | 0.134 | **0.168** | 0.839 |
| 3 (n=81) | 0.000 | 0.000 | **0.049** | 0.750 |

Patch sweep numbers (answer-changed, McNemar, read-out tracking) match `20260919-184323`
within sampling noise -- see that run's notes for the full causal-patching writeup; this run
doesn't change any faithfulness conclusion, only decodability.

**Interpretation:** Widening to top-10 continues the pattern from the top1/top5 ANY-iteration
addendum: real additional signal, still nowhere near closing the reproduction gap. Top-10
ANY-iteration roughly doubles top1 matched-iteration (18.3%→41.0% overall; individual buckets
up to 0.778 at 1-step) but every step-count bucket stays well below Shen et al.'s reported
97.1%/83.9%/75.0%, and the 3-step bucket -- completely flat at 0.000 under both top5 variants
-- only breaks off the floor at top10 (0.049), still far below 0.750. Widening k recovers
some of what a single-position, single-candidate read misses, but the shape of the gap
(collapses fastest at higher step counts) is unchanged -- consistent with `next_experiments.md`'s
existing read that the gap is more about population size (paper's full 1319-example test set
vs. this run's ~9-81-example step-count buckets) and/or a different reading rule than about a
top-k cutoff specifically.

**Gotchas hit:**
- None new in the run itself. Setup-side: the local repo had uncommitted changes to
  `decode_patch_codi.py`/`decode_patch_coconut.py` (the top-10 addition) at the time this run
  was launched -- the pod ran the *working-tree* version of the script, not a committed one.
  `manifest.json`'s `git.commit` is recorded as `3b1e1ee2...-dirty` (the HEAD the working tree
  was based on, with `dirty: true`) rather than a commit that actually contains these changes,
  since that commit didn't exist yet at run time. The script diff that produced this run is the
  one committed alongside this notes.md.
- `manifest.json`'s `author` field was empty on save (pod's `git config user.name` unset, same
  gap noted in earlier runs) -- filled by hand.

**Caveats:** Same as `20260919-184323` (paper-style buckets small-n at n=9/119/81; single seed).
The ANY-iteration/ANY-pass metrics specifically only test "does the value appear anywhere in
top-k across all 6 positions" -- they say nothing about causal use (see that run's patch-sweep
results for the faithfulness side, unchanged here).

**Next:** This closes out the top-k side of the decodability reproduction-gap question at the
current pair-selection/mapping design. If still worth chasing: the "correct-only, all-steps-in-top-k"
metric is still bottlenecked by population size at 2-3+ steps (n=81-119) -- a fresh pull from
the paper's own full 1319-example test set structure (not the held-out half-B slice) would be
the next lever, separate from k.
