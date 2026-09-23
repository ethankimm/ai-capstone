## 2026-09-23 — CODI decode+patch, latent-0 + checked-steps fix: reproduction gap to Shen et al. Table 3 closes to within ~6-12pp (codi, run_id: 20260923-052604_codi_decode-patch-full-eval-latent0)

**Goal:** Combine two methodology fixes identified by re-reading the reference repo
(`probe_latent_token.py`) against a real decoded-output dump, plus a paper text re-check
(Shen et al., Sec 3.5), both flagged as likely explanations for the large gap between our
CODI decodability numbers and the paper's reported Table 3 (97.1%/83.9%/75.0% for
1/2/3-step problems):

1. **Never-scored position ("latent 0").** The reference repo's own probe script scores 7
   candidate positions per example: the pre-loop encode-pass hidden state ("latent 0",
   `lm_head(latent_embd)` before the first loop iteration) plus the 6 loop outputs
   ("latent 1"-"latent 6"). `decode_patch_codi.py`'s `run_thoughts()` computes this exact
   pre-loop hidden state (it's what feeds iteration 1) but never captured its logits or
   scored it -- only iterations 1-6 were ever candidates. Confirmed directly in code
   (`run_thoughts`, line ~110-116 before this run's fix) and against a real decode dump
   from the reference repo, where latent-0 is very often the first, cleanest hit for step 1.
2. **Wrong step-count bucketing.** CODI's own paper excludes the final CoT step from
   distillation supervision ("this behavior would undermine the quality of the target
   hidden activations", Sec 3.5). Checking the reference dump: the final step's numeric
   value essentially never appears among any of the 7 decoded positions, while non-final
   steps often do. Strong evidence Table 3's "N-step" buckets count only *checked*
   (non-final) steps, i.e. `total_steps - 1`, not the raw chain length -- and that scoring
   the final step (which the model was never trained to represent there) was inflating our
   denominator with an unwinnable case.

Both were independently verified for free against already-cached predictions before
spending any GPU time: the checked-steps rebucketing alone (no latent-0, rescored locally
from `20260923-044632_codi_decode-patch-full-eval-top10`'s predictions.jsonl) took the
"1 checked step" bucket from ~15-17% to **86.6%/87.4%** (top5/top10) -- right in the
paper's range -- while "2 checked steps" stayed low (~8.6%/19.8%), a real but incomplete
fix. This run adds latent-0 (which needed a fresh decode pass -- its logits were never
cached) on top of that fix to get the combined number.

**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3` (released checkpoint),
compute_steps=6, paper inference protocol (LoRA r=128/α=32, projection 768+LN, greedy,
`model.eval()` fix in place). `run_thoughts()` called with the new `include_latent0=True`
flag (default False, every other caller/script unaffected -- see the flag's docstring for
the positional-indexing hazard this guards against).

**Data:** gsm8k-aug test, all 1319 examples; half A (idx even, n=660) fits
`best_iter_for_step` (unaffected by latent-0 -- `build_matrix` explicitly skips iter-0
entries so the matched-iteration mapping and `patch_iter_focus` selection stay defined
over iterations 1-6 only, same as every prior run); half B (idx odd, n=659) is where every
decoding number and patch pair comes from.

**Command:**
```
cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/decode_patch_codi.py \
  --ckpt_dir <hf cache path for zen-E/CODI-gpt2@fd641b3> --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \
  --slug decode-patch-full-eval-latent0 --stage full_run --hardware "RunPod RTX A40 (secure)" \
  --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \
  --lora_r 128 --lora_alpha 32 --lora_init --greedy True \
  --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
  --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \
  --full_test True --mapping_split True --n_patch_pairs 350
```
Fresh RunPod RTX A40 (secure, EU-SE-1, $0.49/hr -- A5000 had zero secure stock at
provisioning time). `codi_setup.sh` (~2 min) + checkpoint download (~3s). Decode pass
283s (n=1319, includes the one extra logits capture per example for latent-0 --
negligible added cost since it reuses the already-computed encode-pass forward output, no
extra forward pass) + patch sweep ~5 min (350 pairs). Total pod uptime including
provisioning, env setup, checkpoint download, and result transfer: ~14 min ≈ **$0.11**.

**Headline results:** `final_answer_accuracy=0.415` (548/1319 -- within noise of
`20260923-044632`'s 0.419 and the original `20260919-184323`'s 0.419; small run-to-run
greedy-decode variation across different pods/GPUs, same as noted in that run's
"sampling noise" caveat, not a regression).

| | top1 | top5 | top10 |
|---|---|---|---|
| matched-iteration (iters 1-6 only, unaffected by latent-0) | 0.1849 | 0.2643 | 0.3184 |
| **ANY-iteration incl. latent-0** (n=2126) | **0.3617** | **0.4694** | **0.5400** |

ANY-iteration jumped sharply from the top10-only-1-6 run's 0.2300/0.3358/0.4097 (all
iterations 1-6) to 0.3617/0.4694/0.5400 once latent-0 is included -- confirming it's a
real, frequently-hit position, not a marginal one.

**The combined-fix result** (checked steps = `n_steps - 1`, drop the final step's gold
value, ANY-iteration including latent-0):

| checked steps (→ total CoT length) | our top5 | our top10 | n | paper (Shen et al.) | gap (top10) |
|---|---|---|---|---|---|
| 1 (2-step CoT) | 0.891 | **0.908** | 119 | 0.971 | 6.3pp |
| 2 (3-step CoT) | 0.750 | **0.8125** | 80 | 0.839 | 2.7pp |
| 3 (4-step CoT) | 0.628 | **0.674** | 43 | 0.750 | 7.6pp |

For comparison, the two unfixed/partially-fixed numbers on the exact same population:

| checked steps | original (total-chain, matched-iter, top5) | ANY-iter incl. latent-0, raw total-chain bucketing | **combined fix (checked-steps + latent-0)** |
|---|---|---|---|
| "2-step" (raw) / 1 checked | 0.067 | 0.160 | **0.891 (top5) / 0.908 (top10)** |
| "3-step" (raw) / 2 checked | 0.000 | 0.113 | **0.750 (top5) / 0.8125 (top10)** |

**Interpretation:** The combined fix takes the reproduction gap from 55-84 percentage
points (original matched-iteration, total-chain-length bucketing) down to **3-8 points**
at top10, across all three checked-step buckets with usable n (43-119). This is
substantially better than either fix alone -- checked-steps bucketing alone closed most of
the 1-checked-step gap but left 2-checked-steps at ~20%; adding latent-0 on top brings
2-checked-steps to 81% (vs. paper's 83.9%) and even gives 3-checked-steps a real number
(67.4% vs. 75.0%) where before it was near-zero. **This substantially reverses the
project's earlier "CODI decodability doesn't reproduce the paper" framing** -- the
original claim was a real measurement, but of a different, harder, and not-quite-matching
quantity (matched single-iteration read, full-chain-conjunction, final step included) than
what Table 3 most likely reports. Once measured comparably, CODI's continuous thoughts are
about as decodable as the paper claims.

**What this does NOT change:** the causal-patching / faithfulness results elsewhere in the
repo (`20260919-184323`, `20260920-190420` minimal-pair, `20260920-085206` qualified-patch,
etc.) are untouched -- those test whether a *specific* injected value propagates, which
latent-0 and the checked-steps bucketing don't bear on. This run's own patch-sweep numbers
(focus-iter answer-changed 29.1% patch / 28.6% control, McNemar p=0.09 on the pooled
n=350) are consistent with the prior full-eval run's null within sampling noise -- the
faithfulness gap (decodable but not causally used) stands. **Decodability and
faithfulness are now on much firmer, more paper-comparable footing simultaneously**: CODI
is decodable close to the paper's own numbers, and still not shown to be causally faithful
under patching.

**Gotchas hit:**
- `run_thoughts()`'s `records` list gets a `{"iter": 0, ...}` entry prepended when
  `include_latent0=True`, which shifts every other entry's *position* in the list by one.
  One pre-existing block in this same file (`paper_metric`, the original un-fixed
  matched-iteration metric) indexed `per_iter` *positionally*
  (`per_iter[best_iter_for_step[s] - 1]`) rather than by the `"iter"` key -- this would have
  silently grabbed the wrong iteration's candidates once latent-0 shifted the list. Fixed
  to look up by `"iter"` key before this run (the bug was dormant in every prior run, since
  `include_latent0` defaults to `False` and was never set anywhere before this file's
  change -- no previously-logged numbers are affected).
- `build_matrix()` now explicitly skips any `"iter": 0` entry so the matched-iteration
  mapping/`patch_iter_focus` selection stay restricted to loop iterations 1-6, unaffected
  by latent-0 -- confirmed by `answer_changed_patch_focus_iter`/`patch_iter_focus=2`
  matching the pattern of prior runs.
- Working-tree/commit caveat, same as the top-10 runs: `manifest.json`'s `git.commit` is
  `3b1e1ee2...-dirty` (HEAD the working tree was based on) since the latent-0/checked-steps
  script changes weren't committed yet at run time.
- `manifest.json`'s `author` field was empty on save (pod git config unset) -- filled by
  hand.

**Caveats:**
- n=43 at "3 checked steps" (4-step CoT) is the smallest well-powered bucket here --
  67.4%/75.0% is a real result but with wider uncertainty than the n=80/119 buckets.
- The checked-steps hypothesis (final step excluded from Table 3's buckets) is inferred
  from the paper's own distillation-supervision text plus the reference dump's pattern, not
  from reading Shen et al.'s actual scoring code (not available) -- it fits the data
  extremely well but isn't a first-party confirmation.
- Single seed, greedy decode; `final_answer_accuracy` varies by ~1pp run-to-run across
  different pods for reasons not fully pinned down (same GPU-nondeterminism caveat noted in
  `20260919-184323`'s correction).
- `decoding_matrix_*_FULL_POPULATION` in the manifest still only covers iterations 1-6 (not
  latent-0) -- the full structural odd/even-iteration finding from earlier runs is
  unaffected/unchanged by this addition.

**Next:**
- This is likely the number to cite for CODI decodability in the Oct 9 writeup, replacing
  the flatter "far below the paper" framing from `20260919-184323`/`20260923-044632` --
  cite both: the original/partial numbers as the record of what was measured and why it
  looked worse, this run as the corrected comparison.
- Worth checking whether Coconut's own D&W paper has an analogous excluded-final-step or
  missing-candidate-position convention before assuming any of this generalizes --
  `coconut_common.py`'s `run_passes` already scores a pass-0-equivalent position (Coconut's
  splicing works differently, no missing pre-loop position there), and the final-step
  exclusion rationale is specific to Shen et al.'s distillation training, not established
  for D&W's Coconut training regime.
- If the residual 3-8pp gaps are worth chasing further: try scoring against the paper's
  presumably much larger n (their full 1319-example test set, not this run's ~660-example
  held-out half) before reading the remaining gap as a real discrepancy.
