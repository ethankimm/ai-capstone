## 2026-09-20 — E4: DAS on same-problem minimal-pair donors, joint multi-site subspace, CODI -- flat null across every (site, k) (codi, run_id: 20260920-235312_codi_das-minimal-pair)

**Goal:** `steered_to_donor_audit.md` §5 E4 / `next_experiments.md`'s pasted E4 plan,
conditional on E3(a) (`20260920-190420_codi_minimal-pair-patch`): ALL-SLOT minimal-pair
patching steers to the twin's answer 70.1% of the time, but no single iteration alone
carries more than 10.9%, with a sharp jump from cumulative prefix 1..3 (8.0%) to prefix
1..4 (46.7%). E4 asks whether that jump is explained by a k-dimensional LINEAR SUBSPACE
(a learned orthogonal rotation R, Distributed Alignment Search) rather than the full
768-dim vector -- tested at two site configurations: iteration 4 alone, and the joint
prefix {1,2,3,4} sharing one rotation, swept over k in {8,16,32,64}.

**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3` (released checkpoint),
compute_steps=6, same LoRA r=128/α=32, projection 768+LN, greedy decode as every prior
CODI run this session.

**Data:** gsm8k-aug train n=2500 (seed=0) for the minimal-pair training pool (accuracy
0.790, 1975 base-correct, 1755 candidates, 200/296 candidates checked to reach 200
qualified training pairs); validation n=800 (seed=0) for eval (accuracy 0.790, 632
base-correct, 575 candidates, 100/165 checked to reach 100 qualified eval pairs).

**Command:**
```
cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/das_minimal_pair_codi.py \
  --ckpt_dir /workspace/hf_cache/models--zen-E--CODI-gpt2/snapshots/fd641b3d3edc59e4f534b55588e906588c9e36bb \
  --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \
  --slug das-minimal-pair --stage pilot --hardware "RunPod A40 (secure)" \
  --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \
  --lora_r 128 --lora_alpha 32 --lora_init --greedy True \
  --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
  --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \
  --site_groups "4;1,2,3,4" --k_values 8,16,32,64 \
  --train_n_pairs 200 --eval_n_pairs 100 --epochs 5
```
Fresh RunPod A40 (secure, CA-MTL-1, $0.49/hr -- A5000/A6000 had no stock this session).
Ran in parallel with the Coconut counterpart below on the same GPU (46GB, plenty of
headroom for two small models). Pair-building (decode 2500+800 examples, filter
candidates): ~7.5 min. Full 8-combo training+eval sweep: ~35 min (2079s cumulative per
the log's elapsed counter). Total pod time this session for both mechanisms combined:
~45 min GPU, ≈$0.37.

**Headline results:** `intervention_accuracy=0.04` (best of sweep). n=100 qualified
minimal pairs per (group, k) cell (Wilson 95% CIs from `predictions.jsonl`/manifest):

| group | k | matches_twin | 95% CI | answer_changed |
|---|---|---|---|---|
| 4 | 8 | 0.020 | [0.006, 0.070] | 0.890 |
| 4 | 16 | 0.000 | [0.000, 0.037] | 0.950 |
| 4 | 32 | 0.000 | [0.000, 0.037] | 0.870 |
| 4 | 64 | 0.010 | [0.002, 0.054] | 0.500 |
| 1+2+3+4 | 8 | 0.010 | [0.002, 0.054] | 0.910 |
| 1+2+3+4 | 16 | 0.020 | [0.006, 0.070] | 0.860 |
| 1+2+3+4 | 32 | **0.040** | [0.016, 0.098] | 0.890 |
| 1+2+3+4 | 64 | 0.020 | [0.006, 0.070] | 0.820 |

**Interpretation:** A clean, flat null across every cell -- no (site, k) combination gets
above 4%, all eight 95% CIs overlap and sit far below E3(a)'s ALL-SLOT raw-vector upper
bound (70.1%) and even below E3(b)'s single-slot iter-4 raw patch (3.6%). Doubling k from
8 to 64 (a 1%→8.3% share of the 768 dims) produces no monotonic trend in either site
config -- the best cell (joint prefix, k=32) is not distinguishable from noise relative to
its neighbors. `answer_changed` stays high (82-95%) throughout, confirming the
intervention is doing *something* to the forward pass (consistent with z0/z3-style
positions being load-bearing under ablation), it just never lands on the twin's specific
answer. This rules out the "prefix-1..4 jump is a low-dimensional linear subspace"
hypothesis for CODI: whatever raw-vector interchange patching captures at prefix 1..4
(E3(b)'s 46.7%) is NOT recoverable by any rotation-constrained subspace up to k=64/768
dims tested here, even though R has 20M target degrees of freedom to work with (a full
768x768 orthogonal matrix) and even though the *training* target itself is the easiest
possible case (twin's own answer, no cross-problem confound). Combined with the earlier
cross-problem DAS null (`20260920-050602_codi_das-pilot`, 17/18 null), this is now three
independent nulls for CODI under three different training-target regimes (donor's own
gold, minimal-pair single-site, minimal-pair joint-site) -- the raw-vector prefix effect
looks like it depends on something a linear change-of-basis structurally can't express
(a genuinely nonlinear, entangled function of the 768 dims, or dependence on exact values
outside any fixed k-dim subspace).

**Gotchas hit:**
- None new. Same RunPod A40 pod as the Coconut run (see its notes for the venv/data-
  transfer details); `codi_setup.sh` ran clean, checkpoint `snapshot_download` cached
  from earlier session, `uv` not on `$PATH` in the non-login SSH shell so `codi_setup.sh`
  fell back to it via its own PATH-export logic without issue.
- `build_minimal_pairs` decodes the ENTIRE requested pool unconditionally before
  filtering (no early exit once `n_pairs` is reached) -- at `train_pool_n=2500` this cost
  ~5.7 min of decode time that a tighter default pool size (candidate yield here was
  200/296=68%, far above the ~23% seen in E3's original 600-example slice) would have
  avoided. Not wrong, just conservative; worth tuning `--train_pool_n` down for a rerun.

**Caveats:**
- Pilot-scale reduction from the plan's `n=500` training-tuple spec: `train_n_pairs=200`,
  `epochs=5` (2079s for the full 8-combo sweep; the full n=500 spec would run
  proportionally longer, roughly 2.5x, ~85 min for CODI alone). Given the result is a
  flat null with no upward trend in k or n hinting at an emerging signal, a full-scale
  rerun is not obviously worth the added compute -- see Next.
- `matches_twin` here is the same minimal-pair-specific metric as E3's patch runs (twin's
  own re-executed answer, valid without the cross-problem confound by construction).
- Eval n=100 per cell (Wilson CIs above) is small enough that a true effect below ~4-5%
  could be present but undetected; this null is "no large or moderate effect", not proof
  of exactly zero.

**Next:** Coconut counterpart is `20260920-233519_coconut_das-minimal-pair` (same
session, same pod) -- shows the OPPOSITE result: a clear monotonic dose-response with k
(0.242→0.505, best at joint {1,4} k=64), a materially stronger and cleaner positive
finding. The CODI/Coconut asymmetry from E3 (diffuse vs. localized raw-vector effect)
now extends to linear-subspace recoverability too: Coconut's localized effect IS explained
by a moderate-k linear subspace; CODI's diffuse effect is NOT, at any k tested. This is
itself the E4 "do different latent scratchpads think alike" answer for this angle: no --
Coconut's intermediate state looks close to a genuine low-dimensional linear "scratchpad
variable"; CODI's does not, even though both show a raw-vector same-problem effect of
similar overall magnitude (E3(a): 70.1% vs 77.1%).
