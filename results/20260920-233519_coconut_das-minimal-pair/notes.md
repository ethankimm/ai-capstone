## 2026-09-20 — E4: DAS on same-problem minimal-pair donors, joint multi-site subspace, Coconut -- monotonic dose-response with k, best 50.5% at joint {1,4} k=64 (coconut, run_id: 20260920-233519_coconut_das-minimal-pair)

**Goal:** `steered_to_donor_audit.md` §5 E4 / `next_experiments.md`'s pasted E4 plan,
Coconut counterpart to `20260920-235312_codi_das-minimal-pair`, conditional on E3(a)
(`20260920-195725_coconut_minimal-pair-patch`): ALL-SLOT minimal-pair patching steers to
the twin's answer 77.1% of the time, with pass 1 ALONE already carrying 48.6% and pass 4
alone an independent 28.6% (passes 0/2/3/5 near-inert). E4 asks whether a learned k-dim
linear subspace (Distributed Alignment Search) explains pass 1's signal alone, and
whether a JOINT subspace spanning passes {1,4} explains more of the ALL-SLOT upper bound
than either pass alone -- tested at k in {8,16,32,64}.

**Mechanism / model:** `coconut`, backbone `openai-community/gpt2`, checkpoint
`hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33`, compute_steps=6 (6 `<|latent|>`
passes).

**Data:** `gsm_valid-gold-reasoning-trace_test.json` (Dilgren & Wiegreffe gold-trace
prep, same source as every prior Coconut run this session), n=1194 for the training pool
(seed=0; accuracy 0.336, 401 base-correct, 406 candidates, 200/392 checked to reach 200
qualified training pairs) and n=500 for eval (seed=1, disjoint shuffle from train per
`das_coconut.py`'s convention; accuracy 0.318, 159 base-correct, 165 candidates, 91/165
checked -- only 91 qualified pairs found within the pool, short of the 100 requested,
capped by `--eval_max_candidate_checks 900` exhausting the eval pool's candidate supply
at this pool size).

**Command:**
```
cd /workspace/ai-capstone && .venv_coconut/bin/python scripts/das_minimal_pair_coconut.py \
  --checkpoint_path /workspace/hf_cache/models--connordilgren--gpt2-gsm8k-coconut/snapshots/24f1422fcf6d557fd359fc818a4d1d0fde4aac7f/checkpoint_33 \
  --data_dir /workspace/coconut_data \
  --slug das-minimal-pair --stage pilot --hardware "RunPod A40 (secure)" \
  --num_latents 6 --site_groups "1;1,4" --k_values 8,16,32,64 \
  --train_n_pairs 200 --eval_n_pairs 100 --epochs 5
```
Same RunPod A40 pod as the CODI run above ($0.49/hr, CA-MTL-1), run in parallel with it
(46GB GPU, plenty of headroom for two small models at once). Env: fresh `.venv_coconut`
pinned to `torch==2.5.1`/`transformers==4.46.2`/`datasets==3.1.0`/`numpy==2.1.3`
(`uv` unavailable on `$PATH` in the non-login SSH shell, fell back to `python3 -m venv` +
pip, no functional difference). Checkpoint via `hf_hub_download`, gold-trace JSON copied
in directly via `scp` (no network dependency needed on the pod for that file). Pair-
building: ~1.5 min. Full 8-combo training+eval sweep: ~24 min (1430s cumulative).

**Headline results:** `intervention_accuracy=0.505` (best of sweep, joint {1,4} k=64).
n=91 qualified minimal pairs per (group, k) cell:

| group | k | matches_twin | 95% CI | answer_changed |
|---|---|---|---|---|
| 1 | 8 | 0.242 | [0.165, 0.339] | 0.429 |
| 1 | 16 | 0.341 | [0.252, 0.443] | 0.604 |
| 1 | 32 | 0.275 | [0.194, 0.374] | 0.516 |
| 1 | 64 | 0.308 | [0.222, 0.409] | 0.571 |
| 1+4 | 8 | 0.297 | [0.213, 0.397] | 0.538 |
| 1+4 | 16 | 0.429 | [0.332, 0.531] | 0.637 |
| 1+4 | 32 | 0.440 | [0.342, 0.542] | 0.670 |
| 1+4 | 64 | **0.505** | [0.405, 0.606] | 0.681 |

**Interpretation:** Sharp contrast with CODI's flat null on the identical experimental
design. Two clean trends: (1) at fixed group, `matches_twin` roughly rises with k for
both the single-slot pass-1 group (0.242→0.341→0.275→0.308, noisy but above the k=8
floor at every larger k) and the joint {1,4} group, which is MONOTONIC in k
(0.297→0.429→0.440→0.505, each step's CI barely overlapping the k=8 cell); (2) at every
matched k, the joint group beats the single-slot group (e.g. k=64: 0.505 vs 0.308) --
patching pass 4's subspace on top of pass 1's captures more of the twin's answer than
either alone, consistent with E3(b)'s finding that pass 1 and pass 4 are independently
load-bearing (48.6% and 28.6% raw-vector single-slot). At k=64 (8.3% of the 768 dims),
the joint subspace recovers 0.505/0.771 = 65.5% of the ALL-SLOT raw-vector upper bound.
Note the subspace constraint does cost some signal relative to the unconstrained
full-vector patch -- pass 1 alone's best subspace result (0.341 at k=16) does not reach
pass 1's own raw single-slot number from E3(b) (0.486) -- but a FRACTION of the
dimensions (up to 8.3% at k=64) still recovers a MAJORITY of the joint effect, which is
the DAS hypothesis's actual claim (a low-dimensional linear code, not literally "as good
as the raw vector at low k"). Net: Coconut's same-problem minimal-pair signal looks like
a genuine,
moderate-dimensional linear "scratchpad variable" -- especially once pass 4's subspace is
added to pass 1's -- while CODI's structurally identical test (previous run this
session) found nothing at any k. This is the clearest confirmation yet of the
CODI/Coconut asymmetry E3 first surfaced: same overall raw-vector effect size (E3(a):
70.1% vs 77.1%), but only Coconut's decomposes into anything resembling an addressable,
low-dimensional linear code.

**Gotchas hit:**
- Eval pool (n=500, seed=1) only yielded 91/100 requested qualified pairs before
  `eval_max_candidate_checks` (900, well above the 165 candidates actually generated)
  exhausted the pool's own candidate supply -- not a script bug, just this data source's
  base accuracy (31.8-33.6%) yielding fewer base-correct-both-sides pairs per unit pool
  size than CODI's ~79% accuracy. A larger `--eval_pool_n` would close the gap to 100 if
  an exact n is needed for a future rerun.
- Same as the CODI run: no other new gotchas this session (venv, checkpoint, and data
  transfer all reused verbatim from earlier sessions' setup).

**Caveats:**
- Pilot-scale reduction from the plan's `n=500` spec: `train_n_pairs=200`, `epochs=5`
  (1430s for the full 8-combo sweep). Given the CLEAR upward trend with k here (unlike
  CODI's flat null), a full-scale rerun (`--train_n_pairs 500`, more epochs, and
  possibly k>64) is well-motivated -- see Next.
- `matches_twin` is the same minimal-pair-specific metric as E3's patch runs.
- n=91 per cell is modest; the k=8→64 trend for the joint group is outside adjacent CIs'
  overlap at the extremes (k=8 CI upper 0.397 vs k=64 CI lower 0.405, non-overlapping)
  but the middle steps (16→32) are not individually significant pairwise -- treat the
  trend as real but its exact shape (linear? saturating?) as underdetermined at this n.

**Next:**
- CODI counterpart is `20260920-235312_codi_das-minimal-pair` (same session, same pod) --
  flat null at every (site, k), the opposite of this result.
- Full-scale rerun for Coconut specifically (`--train_n_pairs 500 --epochs 10`,
  `--eval_pool_n` raised to actually reach `eval_n_pairs=200`, and extending
  `--k_values` past 64 e.g. 96,128 to see where the curve saturates relative to the
  77.1% ALL-SLOT ceiling) is the natural next step given the clean positive trend here --
  CODI does not warrant the same investment given its flat null across the full k range
  already tested.
- Worth checking whether the trained k=64 joint-{1,4} rotation's target subspace overlaps
  with anything a linear/MLP probe recovers (`20260920-040411_coconut_probe-continuous-pilot`)
  -- if the SAME dimensions matter for both readout and causal control, that would be a
  much stronger "this is the scratchpad variable's actual encoding" claim than either
  result alone.
