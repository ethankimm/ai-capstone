## 2026-09-20 — Distributed Alignment Search on Coconut: learned subspace still doesn't steer (coconut, run_id: 20260920-053935_coconut_das-pilot)

**Goal:** Coconut counterpart to `20260920-050602_codi_das-pilot`, run in the same
session for the same reason: test whether a learned k-dim subspace of the patched
latent hidden state can steer a recipient's answer toward a donor's, where raw
full-vector donor-interchange patching (`20260920-031246_coconut_decode-patch-pilot`)
was a clean null (`steered_to_donor` at or near floor across every pass, despite
`answer_changed` tracking decodability). Same DAS mechanism as `das_codi.py`
(orthogonal rotation R, k-dim subspace split, trained to maximize the donor's own
gold-answer likelihood, transformer frozen) but hooked at Coconut's structurally
different intervention point: the continuous input-embedding stack at the
`<|latent|>` token position for pass i (vs. CODI's per-iteration loop input).

**Mechanism / model:** `coconut`, backbone openai-community/gpt2, checkpoint
`hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33`, compute_steps=6 (6 `<|latent|>`
passes). New script `scripts/das_coconut.py`, reusing `load_coconut`/
`encode_question`/`finish_and_decode`/`extract_answer_after_delimiter`/`wilson_ci`/
`_to_legacy_cache` from `coconut_common.py`. `coconut_common.run_passes` is hardwired
`@torch.no_grad()`, so this script reimplements a resumable, optionally-grad-enabled
version of the same loop (`run_pass_range`) rather than editing that shared module.

**Data:** `gsm_valid-gold-reasoning-trace_test.json` (same corpus/questions as our own
`gsm8k_aug.py`, Dilgren & Wiegreffe's data prep), shuffled with a seed disjoint from
`decode_patch_coconut.py`'s eval slice (train pool seed=0, eval pool seed=1) so
training and evaluation pairs don't overlap the prior pilot's own eval set.

**Hyperparams:** all 6 latent passes (0..5) x k in {8, 32, 64}, 150 training pairs x
5 epochs per (site, k), 60 held-out eval pairs per (site, k), AdamW lr=1e-3 + cosine
schedule — same pilot-scale reduction of the task spec (n=1000, 20 epochs, k in
{4,8,16,32,64}) as `das_codi.py`, same compute-budget reasoning. Teacher-forcing
target simplified to `" ### {donor.answer}"` spliced directly after the last latent
token (Coconut's post-latent continuation is free-form reasoning text ending
`... ### <answer>`, not a single explicit answer-boundary token the way CODI's `eot_id`
is — an approximation of y_donor, not the model's natural generation path).

**Command:**
```
cd /workspace/ai-capstone && .venv_coconut/bin/python scripts/das_coconut.py \
  --checkpoint_path /workspace/coconut_checkpoints/gsm-coconut/checkpoint_33 \
  --data_dir /workspace/coconut_data \
  --slug das-pilot --stage pilot --hardware "RunPod RTX A6000 (secure)" \
  --num_latents 6 --sites 0,1,2,3,4,5 --k_values 8,32,64 \
  --train_n_pairs 150 --eval_n_pairs 60 --epochs 5
```
Same RunPod RTX A6000 pod as the CODI DAS run above (kept warm to avoid a second
provisioning cost), separate `.venv_coconut` pinned to Coconut's own requirements
(`torch==2.5.1`, `transformers==4.46.2`). Setup: checkpoint `hf_hub_download`
(`connordilgren/gpt2-gsm8k-coconut`, `checkpoint_33`, ~cached), gold-trace data file
copied from the local `are-lrms-easily-interpretable` data prep. A throwaway n=5/n=3
smoke test (deleted, not logged) confirmed the training + eval pipeline before the
real pilot. Full 18-combo sweep: ~33 minutes wall time (~$0.29 GPU cost) — faster
than CODI's equivalent sweep since Coconut's splice-based intervention needs no
per-iteration loop restart.

**Headline results:**
- **All 18 of 18 (site, k) combinations: `steered_to_donor_rate=0.000`** (n=60 eval
  pairs each, Wilson CI upper bound ~0.06) — a cleaner, more uniform null than CODI's
  companion run (which had one marginal 1/60 hit). No combination, at any pass 0..5 or
  any subspace size, produced even a single instance of the patched answer matching
  the donor's gold value when the base answer didn't.
- `answer_changed_rate` ranges 0.10–0.55 and, notably, trends upward with k within
  every site (e.g. site=1: 0.367 at k=8 → 0.433 at k=32 → 0.550 at k=64) — the
  larger the swapped subspace, the more it perturbs the output, exactly as expected
  geometrically (k=64 swaps more of the vector than k=8). But this perturbation never
  once resolves into donor-specific steering.
- `intervention_accuracy` (this run's `best_site`/`best_k`, i.e. the best
  `steered_to_donor_rate` found) is exactly 0.0 — there is no "best" combination in any
  meaningful sense; the sweep is flat at floor.

**Interpretation:** Same conclusion as CODI's companion run, and if anything a sharper
version of it: even with the geometric confirmation that larger k perturbs the output
more (`answer_changed` scaling with k is exactly what a correctly-implemented subspace
intervention should show), the donor's specific content is never successfully
transplanted. Combined with `20260920-031246_coconut_decode-patch-pilot`'s finding that
raw full-vector patching already tracks decodability in `answer_changed` but not in
`steered_to_donor`, this DAS result closes off the "intervention was too blunt" reading
for Coconut just as it did for CODI: a rotation trained end-to-end specifically to
maximize donor-answer likelihood, at three subspace sizes and all 6 passes, still finds
no portable subspace. Two independently-implemented mechanisms (horizontal/CODI,
positional-splice/Coconut), two independent DAS implementations, the same null —
this is the strongest evidence yet in this project for "decodable/load-bearing
computation, not portable computation" as a real property of both architectures rather
than an artifact of either one's specific patching mechanics.

**Gotchas hit:**
- Same rsync/git caveat as `20260920-031246_coconut_decode-patch-pilot` and this
  session's CODI DAS run: `.git` wasn't copied to the pod, so `author=""` /
  `git.commit="unknown"` in the raw manifest — hand-corrected after copying results
  back (`author="Henning Lindig"`, `git.commit=0c0f4079af1a0a0119cff22e3d996db3091987ed`,
  `git.dirty=true`).
- `coconut_common.run_passes`'s `@torch.no_grad()` decorator can't be bypassed by
  calling context, so the grad-enabled continuation needed its own resumable
  reimplementation (`run_pass_range` in `das_coconut.py`) rather than reusing the
  shared module's loop directly — documented in the script's own docstring so this
  isn't mistaken for a second, drifted copy of the reference logic.

**Caveats:**
- Same pilot-scale caveat as the CODI run: train_n_pairs=150/epochs=5 vs the spec's
  1000/20, k in {8,32,64} vs {4,8,16,32,64}. The perfect 18/18 null and the
  k-dependent `answer_changed` trend (evidence the intervention is doing something
  geometrically sensible) both argue against undertraining being the reason for the
  `steered_to_donor` floor, but a full-scale rerun would be the way to be certain.
- The `" ### {answer}"` teacher-forcing simplification (skipping the model's natural
  free-form reasoning continuation) means the training signal optimizes a proxy for
  y_donor, not Coconut's actual generation path — plausible this understates what a
  rotation trained against the true continuation could achieve, though the CODI run
  (whose teacher-forcing target IS the natural continuation, just digits after `eot`)
  shows the same null, which weighs against this being the deciding factor.

**Next:** Given the CODI and Coconut DAS pilots agree completely (17/18 and 18/18
null respectively), a full-scale rerun of either seems like a lower-priority use of
remaining budget than moving to the next section of the plan (cross-architecture
mapping / transplant, per `PROPOSAL.md`'s Oct 16 milestone) unless a reviewer
specifically asks for the spec's exact scale.

## Metric B addendum (rescored 2026-09-19, `scripts/rescore_counterfactual.py`)

See `steered_to_donor_audit.md`. `steered_to_donor` as originally logged measures Metric A (`answer_patched == donor.answer` -- already the case for this run except where noted); the table below adds Metric B (`matches_cf`): does the answer equal the counterfactual obtained by substituting the injected value into the RECIPIENT's own remaining chain and re-evaluating.

### DAS learned-subspace patch, unaligned (n=1080)

| group | n | n(cf defined) | matches_cf | 95% CI | perm-null |
|---|---|---|---|---|---|
| site 0 | 180 | 72 | 0.000 | [0.000, 0.051] | 0.021 |
| site 1 | 180 | 60 | 0.000 | [0.000, 0.060] | 0.009 |
| site 2 | 180 | 90 | 0.000 | [0.000, 0.041] | 0.002 |
| site 3 | 180 | 69 | 0.000 | [0.000, 0.053] | 0.032 |
| site 4 | 180 | 60 | 0.033 | [0.009, 0.114] | 0.021 |
| site 5 | 180 | 69 | 0.014 | [0.003, 0.078] | 0.003 |
| total | 1080 | 420 | 0.007 | [0.002, 0.021] | 0.016 |

- **site 0** taxonomy: unchanged 115, other_number 61, donor_intermediate 2, recipient_gold 2
- **site 1** taxonomy: unchanged 99, other_number 68, recipient_gold 11, donor_intermediate 1, recipient_intermediate 1
- **site 2** taxonomy: unchanged 154, other_number 17, recipient_intermediate 4, recipient_gold 4, donor_intermediate 1
- **site 3** taxonomy: unchanged 113, other_number 58, recipient_intermediate 6, donor_intermediate 2, recipient_gold 1
- **site 4** taxonomy: unchanged 117, other_number 53, recipient_intermediate 7, counterfactual 2, donor_intermediate 1
- **site 5** taxonomy: unchanged 131, other_number 40, recipient_intermediate 6, recipient_gold 2, counterfactual 1
- **total** taxonomy: unchanged 729, other_number 297, recipient_intermediate 24, recipient_gold 20, donor_intermediate 7, counterfactual 3
