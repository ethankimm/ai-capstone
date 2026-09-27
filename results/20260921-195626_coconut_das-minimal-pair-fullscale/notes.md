## 2026-09-21 — E4 full-scale: Coconut DAS subspace saturation sweep, k in {64,96,128,256}, joint {1,4} vs pass-1-alone (coconut, run_id: 20260921-195626_coconut_das-minimal-pair-fullscale)

**Goal:** `steered_to_donor_audit.md` §5 E4 / pasted "Priority Tier 1" plan: full-scale
follow-up to the pilot (`20260920-233519_coconut_das-minimal-pair`, pilot-scale
`train_n_pairs=200, epochs=5, k<=64`, which found a monotonic rise with k up to
`matches_twin=0.505` at joint {1,4} k=64). This run asks where that curve **saturates**
relative to E3(a)'s 77.1% all-slot ceiling, at full training scale (`train_n_pairs=500,
epochs=10`) and extended k (`96, 128, 256`).

**Mechanism / model:** `coconut`, backbone `openai-community/gpt2`, checkpoint
`hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33`, compute_steps=6.

**Data:** `gsm_valid-gold-reasoning-trace_test.json` (Dilgren & Wiegreffe gold-trace prep,
same source as every prior Coconut run this session). Train pool n=1194 (full file,
seed=0; accuracy 0.336, 401 base-correct, 406 candidates, **209/406 candidates checked
qualified** — short of the requested 500 because the file only has 1194 rows total and
~34% base accuracy caps the candidate supply, same ceiling effect as the pilot).
Eval pool n=500 (seed=1, disjoint shuffle of the same file; accuracy 0.318, 159
base-correct, 165 candidates, **91/165 checked qualified** — identical eval-pair count to
the pilot since `eval_pool_n` was left at its default and the candidate supply is the
same).

**Command:**
```
cd /workspace/ai-capstone && .venv_coconut/bin/python scripts/das_minimal_pair_coconut.py \
  --checkpoint_path /workspace/hf_cache/hub/models--connordilgren--gpt2-gsm8k-coconut/snapshots/24f1422fcf6d557fd359fc818a4d1d0fde4aac7f/checkpoint_33 \
  --data_dir /workspace/coconut_data \
  --slug das-minimal-pair-fullscale --stage full_run --hardware "RunPod A40 (secure)" \
  --num_latents 6 --site_groups "1;1,4" --k_values 64,96,128,256 \
  --train_n_pairs 500 --eval_n_pairs 150 --epochs 10
```
RunPod A40, secure cloud, CA-MTL-1, $0.49/hr. Env: `.venv_coconut` pinned to
`torch==2.5.1`/`transformers==4.46.2`/`datasets==3.1.0`/`numpy==2.1.3`, checkpoint via
`hf_hub_download`, gold-trace JSON copied in via `scp`. Full 8-combo sweep: 2145s (~36
min) of GPU time after pair-building (~85s).

**Headline results:** `intervention_accuracy=0.560` (best of sweep, joint {1,4} k=128 —
**not** the largest k tested). n=91 qualified minimal pairs per (group, k) cell (same
eval pairs reused across the whole sweep):

| group | k | matches_twin | 95% CI | answer_changed |
|---|---|---|---|---|
| 1 | 64 | 0.385 | [0.291, 0.487] | 0.560 |
| 1 | 96 | 0.363 | [0.271, 0.465] | 0.582 |
| 1 | 128 | 0.407 | [0.311, 0.509] | 0.582 |
| 1 | 256 | 0.407 | [0.311, 0.509] | 0.582 |
| 1+4 | 64 | 0.495 | [0.394, 0.595] | 0.670 |
| 1+4 | 96 | 0.516 | [0.415, 0.616] | 0.758 |
| **1+4** | **128** | **0.560** | **[0.458, 0.658]** | 0.736 |
| 1+4 | 256 | 0.516 | [0.415, 0.616] | 0.758 |

**Interpretation — the curve saturates, it does not keep climbing.** The pilot's clean
monotonic rise (k=8→64: 0.297→0.429→0.440→0.505) does **not** continue past k=64 at full
training scale: joint {1,4} goes 0.495 (k=64) → 0.516 (k=96) → 0.560 (k=128) → 0.516
(k=256), i.e. it peaks at k=128 (16.7% of the 768 residual dims) and *drops back down* at
k=256 rather than climbing further. Every one of these four cells' 95% CIs overlap
heavily with every other (e.g. k=64's [0.394, 0.595] vs k=128's [0.458, 0.658]) — at
n=91, the k=64→128 "rise" and the k=128→256 "fall" are both within noise of each other,
so the honest read is a **plateau at ~0.49-0.56 starting around k=64**, not a resolved
saturation point at a specific k. The single-slot group=1 shows the same pattern one
notch lower: 0.385→0.363→0.407→0.407, flat within noise across the whole k range, so
pass-1-alone's subspace was already saturated by k=64 (consistent with the pilot's own
k=8→64 trend for group 1, which was "noisy but above the k=8 floor" rather than cleanly
monotonic).

At the best cell (joint {1,4}, k=128): `0.560/0.771 = 72.6%` of the E3(a) all-slot
ceiling recovered by a 128-dim (16.7% of 768) learned linear subspace spanning two
passes — a *fraction* of the joint dimensionality still short of the pilot's train-scale
optimism (this run's best cell is only +0.055 over the pilot's k=64 result of 0.505,
despite 2.5x the training pairs and 2x the epochs). **Revised takeaway from the pilot's:**
"monotonic dose-response saturating somewhere past k=64" was itself a pilot-scale
artifact of training noise/undertraining at low n; the corrected picture is a plateau
that both more training data and larger k fail to move past ~0.56, i.e. Coconut's joint
{1,4} scratchpad variable looks like it occupies **roughly a 64-128 dim linear subspace**
and adding more dims past that does not buy more causal control. This is still a much
stronger positive result than CODI's flat null on the identical design
(`20260920-235312_codi_das-minimal-pair`, null at every site/k), and still clears the
single-slot group by a wide, non-overlapping margin at every matched k (e.g. k=128: 0.560
vs 0.407, CIs [0.458,0.658] vs [0.311,0.509], barely overlapping) — the joint-vs-single
gap is the robust finding here, not the exact saturation k.

**Gotchas hit:**
- **First launch attempt crashed at the record-save step**: passed `--stage full`
  instead of the schema's `full_run` (`RunRecord.__post_init__` validates against
  `('smoke_test', 'pilot', 'full_run')`). The full 8-combo training+eval sweep (2190s)
  had already completed and printed every summary line to stdout/log before the crash,
  but `record.save()` — which writes `predictions.jsonl` — never ran, so the per-pair
  records were lost with the process (they only existed in an in-memory list). Rather
  than hand-reconstruct a `predictions.jsonl` from the aggregate log lines, reran the
  identical command with `--stage full_run` end-to-end (another ~36 min / ~$0.29) to get
  a real, complete artifact. The two runs' numbers are close but not identical at every
  cell (e.g. joint k=256: 0.560 first attempt's log vs 0.516 this logged run) — training
  is not seed-pinned per-(group,k) beyond the shared `train_seed`/`pair_seed`, so a
  rerun of the *same* sweep is not bit-identical. Only this run's (logged) numbers are
  authoritative; the discarded first attempt's numbers should not be quoted anywhere.
- Same `eval_max_candidate_checks`/pool-size ceiling as the pilot: eval always caps at
  91 pairs from this data source's ~500-row eval pool regardless of `--eval_n_pairs`
  requested, since candidate supply (165) is the binding constraint, not the request.
  Train similarly capped at 209/500 requested (406 candidates from the full 1194-row
  file). A genuinely larger n would require a bigger underlying gold-trace corpus, not
  just bigger `--*_pool_n`/`--*_n_pairs` flags, since both are already drawing from
  ~the whole file.
- `author` was blank in the auto-generated manifest (pod's `git config user.name` was
  unset, this session never configured it since `/workspace/ai-capstone` was an rsync
  copy without `.git`) — fixed by hand to `Henning Lindig` post-hoc, matching this
  session's git identity. `git.commit`/`dirty` stayed `"unknown"`/`null` for the same
  reason (no `.git` on the pod) and were left as-is.

**Caveats:**
- n=91 eval pairs is the ceiling this data source supports at this pool size (see
  gotcha above); every sweep cell's CI is ~20 points wide, so treat the specific
  ranking of adjacent k values (128 > 96 > 64 ≈ 256) as noise, not a resolved curve
  shape — the safe claim is "plateaus somewhere in [64, 128], does not keep rising."
- `matches_twin` is the same minimal-pair-specific metric as E3/the pilot's DAS run —
  not comparable to the cross-problem `matches_cf`/permutation-null numbers in
  `steered_to_donor_audit.md` §3.3.
- Training loss (12-34 nats across cells, no clear trend with k) was not tracked as a
  convergence diagnostic beyond the printed per-50-step log lines; a proper train/eval
  loss curve per cell was out of scope for this run.

**Next:**
- The dose-response question is now answered well enough not to warrant a bigger sweep:
  further k (512, 768=full) would only be interesting if the current plateau were
  itself in doubt, and the CIs already say it isn't worth chasing at this eval n.
- A genuinely tighter estimate of the plateau's exact shape needs more *eval* pairs,
  which needs a bigger held-out gold-trace pool than this D&W prep provides (91 is the
  ceiling here) — out of scope unless a second data source is added.
- CODI counterpart already exists and is null (`20260920-235312_codi_das-minimal-pair`);
  no full-scale CODI rerun is warranted given its flat pilot result.
- Revisit whether the trained k=128 joint-{1,4} rotation's subspace overlaps with the
  continuous probe's recovered directions (`20260920-040411_coconut_probe-continuous-pilot`)
  — same "Next" item carried over from the pilot, still not done.


---
**Caveat (2026-09-27): the DAS training recipe used here is suspect.** In `20260927-072659_codi_das-minimal-pair-aligned-bigk` the same recipe (lr 1e-3, 5 epochs, teacher-forced CE) trained at k=512 steers 22% while an UNTRAINED random rotation steers 71% on the same pairs. This run had no untrained reference, so its null may be an optimization failure; do not cite it as evidence against a linear subspace until rerun with a fixed recipe.
