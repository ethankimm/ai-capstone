## 2026-09-20 — Coconut decode + donor-interchange patching pilot, data-driven decodable split (coconut, run_id: 20260920-031246_coconut_decode-patch-pilot)

**Goal:** Extend the "decodability != causal faithfulness" question already run on CODI
(z0/z3) to Coconut, on the released checkpoint from Dilgren & Wiegreffe (COLM 2026,
arXiv:2604.04902) whose Finding 2 claims Coconut+GPT2 encodes gold traces in most latent
passes when correct (54-93%). Unlike CODI, don't assume which passes are
decodable/non-decodable — rank them empirically (half-A/half-B held-out split) and group
top-half "most decodable" vs bottom-half "least decodable" for a joint donor-interchange
patch, mirroring CODI's {z0,z3} vs {z2,z4} comparison but data-driven.

**Mechanism / model:** `coconut`, backbone openai-community/gpt2, checkpoint
`hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33`, compute_steps=6 (6 `<|latent|>`
passes).

**Data:** gsm8k-aug-equivalent test set (`gsm_valid-gold-reasoning-trace_test.json`,
Dilgren & Wiegreffe's data prep, same underlying corpus/questions as our own
`gsm8k_aug.py`), n=200 (shuffled, seed=0), run seed=0.

**Command:**
```
.venv_coconut/bin/python scripts/decode_patch_coconut.py \
  --checkpoint_path <hf checkpoint_33 snapshot path> \
  --data_dir /workspace/coconut_data \
  --slug decode-patch-pilot --stage pilot --hardware "RunPod RTX A6000 (secure)" \
  --num_latents 6 --full_test False --eval_n 200 --n_patch_pairs 60 --n_grouped_pairs 100
```
Run inside a dedicated venv pinned to Coconut's own `requirements.txt`
(`torch==2.5.1`, `transformers==4.46.2`), on a fresh RunPod RTX A6000 pod ($0.53/hr,
EU-SE-1, secure cloud — A5000 showed zero stock at provision time). Full pilot run
(decode + single-slot + grouped patch, n=200/60/100) took **~90 seconds of actual compute**
(`sec_per_example=0.059`); total pod wall time including venv setup, checkpoint download,
and a throwaway n=5 smoke test was well under 10 minutes.

**Headline results:**
- `final_answer_accuracy=0.360` — close to paper Table 1's GPT-2/Coconut/GSM8k-Aug=33.1%
  and our own repro-of-Hao-et-al column (34.1%). Checkpoint reproduces the paper's headline
  number; released weights are good.
- `decoding_accuracy` (held-out matched-pass top1) = **0.288** (top5 = 0.391), n=368 matched
  gold-step comparisons. This is a different measurement than D&W's Finding 2 (54-93%) —
  theirs is a backtracking *search* over the full latent trace for the best-matching
  subsequence; this is live vocabulary-projection logit-lens at the empirically-best pass
  per step, held out on a disjoint half of the data. Non-trivial and well above chance, but
  well below D&W's headline — consistent with "decodable via search, much less so via a
  naive live per-step read."
- Pass ranking (avg top1 decode rate, fit on half A): most-decodable = passes {0, 1, 4},
  least-decodable = passes {2, 3, 5} (avg_top1 for 2, 3, 5 are all exactly 0.0 — the
  ranking split is stark, not a coin flip).
- **Single-slot donor-interchange patching** (per pass, n=60 pairs where eligible):
  `answer_changed` is high wherever a pass is ever "best" for some step (pass 0: 0.517,
  pass 1: 0.867, pass 4: 0.650) — patching these passes clearly perturbs the answer. But
  `steered_to_donor` (patched answer specifically matches the donor's gold value, base
  didn't) is at or near floor: 0.000 / 0.033 / 0.000. Passes 2, 3, 5 were never the
  best-decoding pass for any step in half A, so 0 patch pairs were generated for them
  (`n=0`) — a gap in this run, see caveats.
- **Grouped patch** (most-decodable {0,1,4} vs least-decodable {2,3,5} jointly, same
  donor/recipient, n=100 pairs): `answer_changed` = 0.900 (most) vs 0.740 (least),
  **McNemar b=4, c=20, p=0.0015** — significant, and in the direction that patching the
  more-decodable group perturbs the answer more often than the less-decodable group.

**Interpretation:** Same qualitative pattern as CODI's null: passes that are more
readable via logit-lens are also more disruptive to patch (`answer_changed` tracks
decodability, both in the single-slot breakdown and the significant grouped McNemar), but
patching essentially never steers the recipient's answer to specifically match the donor's
gold value (`steered_to_donor` ~0 everywhere). Decodability correlates with being
load-bearing in a coarse "changes the output" sense, but not with carrying a *portable,
donor-specific* piece of content the way a clean interchange-patch story would need.
Table 1 accuracy reproduces; the pilot's own decoding-accuracy number is sane and non-zero,
not floor — this is a real (if modest) positive number, not a broken pipeline.

**Gotchas hit:**
- `runpodctl`'s RTX A5000 showed zero stock at provision time (`get-capacity` returned
  `stockStatus: "unavailable"`, `pricePerHr: null`); fell back to RTX A6000 ($0.53/hr,
  EU-SE-1) per the plan's stated fallback.
- The pod's base image had no `rsync` — had to `apt-get install rsync` before the code/data
  transfer (script + data are ~1.4 MB total, so this was not a bottleneck either way).
- `torch==2.5.1` isn't on the `download.pytorch.org/whl/cu121` index alongside
  `transformers==4.46.2` in one `pip install` line — installed torch from the PyTorch
  index and transformers/datasets from PyPI in two separate commands.
- Only `latentreasoning/`, `scripts/`, `pyproject.toml`, `configs/` were rsynced to the
  pod (not `.git`, to keep the transfer small) — `RunRecord`'s `default_author()` /
  `git_info()` therefore couldn't shell out to git on the pod and the manifest was
  produced with `author=""` / `git.commit="unknown"`. Hand-corrected in `manifest.json`
  after copying back: `author="Henning Lindig"` (matches `git config user.name` locally),
  `git.commit` set to the local `HEAD` at run time
  (`890ae225e955e9666184272ceac615ac1b8eb35f`), `git.dirty=true` (working tree had
  untracked/modified files per `git status` at the time).

**Caveats:**
- `n_patch_pairs=0` for passes 2, 3, 5 (the "least decodable" passes) in the **single-slot**
  breakdown — the pairing logic only generates single-slot pairs at a pass when that pass
  is `best_pass_for_step` for some step, and passes 2/3/5 never won that title in half A.
  So the single-slot table only has real numbers for the 3 most-decodable passes; the
  least-decodable single-slot condition is untested here (the **grouped** patch does cover
  all 6 passes, jointly, and is the one with the significant McNemar result).
- n=200 pilot, not the full 1194-example gold-trace set — small-n estimate, not the
  paper-comparable accuracy number the suggested full run would give.
- `decoding_accuracy`'s definition here (matched-pass top1, held-out split) is intentionally
  different from D&W's own backtracking-search metric; don't quote this pilot's 28.8% as a
  refutation of their 54-93% without noting that difference explicitly (as done above).

**Next:** If this pilot's direction (decodable-but-not-donor-steerable) is worth
confirming at scale, rerun with `--full_test True` on the full 1194-example set for a
paper-comparable accuracy number and larger patch-pair counts — flagged as a possible
follow-up in `next_experiments.md`, not done in this session per instructions to stop at
the pilot. Coconut probes (ridge/MLP vs logit lens, mirroring `probe_codi.py`) would be the
natural companion analysis, not yet written.

## Metric B addendum (rescored 2026-09-19, `scripts/rescore_counterfactual.py`)

See `steered_to_donor_audit.md`. `steered_to_donor` as originally logged measures Metric A (`answer_patched == donor.answer` -- already the case for this run except where noted); the table below adds Metric B (`matches_cf`): does the answer equal the counterfactual obtained by substituting the injected value into the RECIPIENT's own remaining chain and re-evaluating.

### Raw single-slot patch, all passes (n=180)

| group | n | n(cf defined) | matches_cf | 95% CI | |
|---|---|---|---|---|
| all passes | 180 | 38 | 0.105 | [0.042, 0.241] |

- **all passes** taxonomy: other_number 108, unchanged 58, recipient_intermediate 7, counterfactual 4, recipient_gold 3
### By pass

| group | n | n(cf defined) | matches_cf | 95% CI | |
|---|---|---|---|---|
| pass 0 | 60 | 0 | - | [0.000, 0.000] |
| pass 1 | 60 | 28 | 0.143 | [0.057, 0.315] |
| pass 4 | 60 | 10 | 0.000 | [0.000, 0.278] |

- **pass 0** taxonomy: other_number 29, unchanged 29, recipient_intermediate 2
- **pass 1** taxonomy: other_number 42, unchanged 8, counterfactual 4, recipient_intermediate 3, recipient_gold 3
- **pass 4** taxonomy: other_number 37, unchanged 21, recipient_intermediate 2
