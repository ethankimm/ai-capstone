## 2026-09-23 — Coconut decode+patch, top-10 added: ANY-pass top10 lands near D&W's Finding 2 range at low step counts (coconut, run_id: 20260923-044803_coconut_decode-patch-pilot-top10)

**Goal:** Coconut counterpart to `20260923-044632_codi_decode-patch-full-eval-top10`. Extends
the ANY-pass decodability addendum on `20260920-031246_coconut_decode-patch-pilot` (top1/top5
only, rescored locally) with top-10, which needed a fresh model pass (top-10 candidates were
never cached at decode time). Exact re-run of the original pilot's command (same checkpoint,
same eval_n/seed, same pair design) with `scripts/decode_patch_coconut.py` extended to cache
`top10` and report matched-pass/ANY-pass top-10 alongside top-1/top-5.

**Mechanism / model:** `coconut`, backbone `openai-community/gpt2`, checkpoint
`hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33`, compute_steps=6.

**Data:** `gsm_valid-gold-reasoning-trace_test.json` (Dilgren & Wiegreffe's data prep), n=200
(shuffled, seed=0) -- identical slice to the original pilot.

**Command:**
```
.venv_coconut/bin/python scripts/decode_patch_coconut.py \
  --checkpoint_path <hf cache path for connordilgren/gpt2-gsm8k-coconut checkpoint_33> \
  --data_dir /workspace/coconut_data \
  --slug decode-patch-pilot-top10 --stage pilot --hardware "RunPod RTX A5000 (secure)" \
  --num_latents 6 --full_test False --eval_n 200 --eval_seed 0 --n_patch_pairs 60 --n_grouped_pairs 100
```
Same pod as the CODI top-10 run above (RunPod RTX A5000, secure, CA-MTL-1, $0.27/hr), separate
pinned venv (`torch==2.5.1`/`transformers==4.46.2`). Decode+patch pass: 10.6s decode + ~15s
patch/grouped sweep -- Coconut's single-shared-forward-pass extraction is far cheaper than
CODI's per-iteration loop, as in every prior run. Combined pod time for both mechanisms
(provisioning, both env setups, checkpoint downloads, both runs, result transfer): ~14 min ≈
**$0.06** total (not separately billed per mechanism).

**Headline results:** `final_answer_accuracy=0.360` (matches the original pilot exactly --
same checkpoint/data/seed, confirms a faithful re-run).

| | top1 | top5 | top10 |
|---|---|---|---|
| matched-pass (held out, n=368) | 0.288 | 0.391 | **0.462** |
| ANY-pass (n=368) | 0.315 | 0.535 | **0.630** |

Paper-style breakdown (correct-only, by step count), ANY-pass top10 (new -- no matched-pass or
top5 equivalent was computed at decode time for the original pilot, so only the top10 column
is directly new; top5 values are from the local rescore addendum on the original pilot's notes):

| steps | ANY-pass top5 (this run) | **ANY-pass top10** | n |
|---|---|---|---|
| 1 | 1.000 | **1.000** | 2 |
| 2 | 0.778 | **0.889** | 18 |
| 3 | 0.286 | **0.429** | 7 |
| 4 | 0.167 | **0.500** | 6 |
| 5 | 0.000 | **0.000** | 1 |

(Recomputed locally from this run's cached `predictions.jsonl`, same method as
`scripts/rescore_any_iter_top5.py`, extended to k=10 -- `decode_patch_coconut.py` doesn't print
a by-step-count breakdown itself, only the overall ANY-pass top1/top5/top10 numbers above; those
overall numbers were cross-checked against this same recomputation and match exactly, 0.315/0.535/0.630
at n=368.)

Single-slot / grouped patch numbers match the original pilot's `steered_to_donor` results
exactly (same seed, same pairs) -- no new causal-patching finding here, see the original
pilot's notes and its Metric B addendum (`steered_to_donor_audit.md`) for that side.

**Interpretation:** Top-10 pushes ANY-pass decodability further into D&W's reported Finding-2
range (54-93%) at low-to-mid step counts: 2-step 88.9% (inside the range), 4-step 50.0% (just
below it), 3-step 42.9% (below it) -- though at small n (6-18). This is a
bigger jump than CODI's top10 ANY-iteration widening (which stayed far below Shen et al.'s
numbers even at top10) -- consistent with the qualitative story already established across
these two mechanisms: Coconut's live per-pass read is closer to D&W's own search-based
decodability than CODI's per-iteration read is to Shen et al.'s. Still not the same measurement
as D&W's actual backtracking search (this is "in top-10 of a live vocabulary projection at any
of 6 fixed passes", not "found anywhere in a searched subsequence"), and still small-n outside
the 1-2 step buckets.

**Gotchas hit:**
- Same working-tree/commit caveat as the CODI top-10 run: `manifest.json`'s `git.commit` is
  `3b1e1ee2...-dirty` (HEAD at run time, uncommitted top-10 script changes on top) rather than a
  commit containing this run's actual code.
- `manifest.json`'s `author` field was empty on save (pod git config unset) -- filled by hand.
- None on the decode/patch side -- identical environment and checkpoint to the original pilot,
  no new pinning issues.

**Caveats:** Same as the original pilot (n=200 slice, single seed, step-count buckets small
beyond 1-2 steps: n=2/18/7/6/1). ANY-pass says nothing about causal use -- see the original
pilot's patch-sweep and its Metric B addendum for the faithfulness side, unchanged here.

**Next:** Closes out the top-k side of the decodability question for Coconut at this data
scale. A larger n (the full 1194-example gold-trace file, per `next_experiments.md`'s note on
`decode_patch_coconut.py`'s ceiling) would tighten the 2-5 step buckets before reading too much
into the 33-83% range above.
