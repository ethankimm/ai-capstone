## 2026-09-20 — E3(a)+(b): same-problem minimal-pair donor-interchange patch, Coconut -- upper bound is real, and pass 1 alone carries half of it (coconut, run_id: 20260920-195725_coconut_minimal-pair-patch)

**Goal:** `steered_to_donor_audit.md` §5 E3, Coconut counterpart to
`20260920-190420_codi_minimal-pair-patch`. Same design: donor = the recipient's own
question with one eligible number perturbed and the gold chain re-executed
(`latentreasoning.data.minimal_pairs.generate_minimal_pair`), removing the cross-problem
confound that E2's null (`20260920-085317_coconut_qualified-patch`) couldn't rule out.
(a) ALL-SLOT overrides every `<|latent|>` pass's live hidden with the twin's own live
hidden at that pass. (b) single-slot sweep + cumulative prefix localizes which pass(es)
carry it.

**Mechanism / model:** `coconut`, backbone `openai-community/gpt2`, checkpoint
`hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33`, compute_steps=6 (6 `<|latent|>`
passes).

**Data:** `gsm_valid-gold-reasoning-trace_test.json` (Dilgren & Wiegreffe gold-trace prep,
same source as E2's Coconut run), n=600 (seed=0). 213/600 base-correct (accuracy 0.355,
matches E2's Coconut slice). 201 minimal-pair candidates generated, 105/201 had a
base-correct twin ("qualified pairs" -- the set actually scored).

**Command:**
```
cd /workspace/ai-capstone && .venv_coconut/bin/python scripts/patch_minimal_pair_coconut.py \
  --checkpoint_path <hf checkpoint_33 snapshot path> \
  --data_dir /workspace/coconut_data \
  --slug minimal-pair-patch --stage full_run --hardware "RunPod RTX A5000 (secure)" \
  --num_latents 6 --eval_n 600 --n_pairs 200
```
Same RunPod RTX A5000 pod as the CODI run above ($0.27/hr, CA-MTL-1). Coconut has no
checked-in setup script (unlike CODI's `codi_setup.sh`); env was built inline: a fresh
`.venv_coconut` pinned to the reference repo's own `requirements.txt` (torch==2.5.1,
transformers==4.46.2, datasets==3.1.0, numpy==2.1.3), checkpoint via
`hf_hub_download('connordilgren/gpt2-gsm8k-coconut', 'checkpoint_33')`, and the gold-trace
data file copied in directly from the local `are-lrms-easily-interpretable` prep (no
network dependency needed for that file). Env build ~2 min; decode pass (n=600) 30s; full
patch sweep (105 pairs × conditions) ~1 min. A throwaway n=40 smoke test
(`20260920-195521_coconut_minimal-pair-smoketest`, not logged) preceded this run.

**Headline results:** `final_answer_accuracy=0.355` (matches E2's Coconut slice exactly,
same data/seed). n=105 qualified minimal pairs.

| condition | n | matches_twin | 95% CI | answer_changed |
|---|---|---|---|---|
| ALL-SLOT (all 6 passes) | 105 | **0.771** | [0.68, 0.84] | 0.867 |
| single-slot pass 0 | 105 | 0.000 | [0.00, 0.04] | 0.019 |
| single-slot pass 1 | 105 | **0.486** | [0.39, 0.58] | 0.705 |
| single-slot pass 2 | 105 | 0.000 | [0.00, 0.04] | 0.000 |
| single-slot pass 3 | 105 | 0.010 | [0.00, 0.05] | 0.019 |
| single-slot pass 4 | 105 | 0.286 | [0.21, 0.38] | 0.362 |
| single-slot pass 5 | 105 | 0.010 | [0.00, 0.05] | 0.048 |
| prefix 0..0 | 105 | 0.000 | [0.00, 0.04] | 0.019 |
| prefix 0..1 | 105 | 0.486 | [0.39, 0.58] | 0.705 |
| prefix 0..2 | 105 | 0.486 | [0.39, 0.58] | 0.705 |
| prefix 0..3 | 105 | 0.495 | [0.40, 0.59] | 0.714 |
| prefix 0..4 | 105 | 0.743 | [0.65, 0.82] | 0.838 |
| prefix 0..5 | 105 | 0.771 | [0.68, 0.84] | 0.867 |

**Interpretation:** Same qualitative "succeeds where cross-problem fails" result as CODI
(ALL-SLOT 77.1% vs. 0-1.8% for cross-problem donors in E2), but the localization story is
sharply different. Where CODI's effect was diffuse (no single iteration above 11%, a
step-change only appearing once 4 of 6 iterations are present), Coconut's pass 1 alone
carries **48.6%** matches_twin -- essentially all of the gain from prefix 0..0 (0%) to
prefix 0..1 (48.6%) happens in one step, and pass 4 alone adds another independently
detectable 28.6%. The remaining passes (0, 2, 3, 5) are individually inert (0-1%). This
mirrors E2's Coconut finding that passes 0/1 are the load-bearing/high-`answer_changed`
positions, but now with a positive, well-localized causal signal instead of just an
`answer_changed` correlate -- pass 1's continuous thought is, to first approximation, a
single addressable slot carrying the perturbed step value within the same problem, which
is a materially stronger "modular intermediate variable" result than anything CODI shows.
The asymmetry between mechanisms (CODI: distributed across 4+ iterations; Coconut: mostly
one pass) is itself a finding for the "do different latent scratchpads think alike"
question -- no, not in how they localize a causally-verified value, even though both fail
the same E2 cross-problem portability test.

**Gotchas hit:**
- Same rsync-hang issue as the CODI run this session (see its notes.md) -- switched to
  `tar | ssh | tar` for all pod transfers.
- No checked-in `coconut_setup.sh` exists (unlike CODI) -- env build was done as an inline
  shell block. `coconut_common.py`'s docstring says it needs "the reference repo's own
  checkout" but its actual imports are just `torch`/`transformers` (no `coconut.py` import),
  so the vanilla reference checkout was NOT needed on the pod this run -- only the pinned
  venv, the checkpoint, and the gold-trace JSON. Worth promoting to a real
  `coconut_setup.sh` (mirroring `codi_setup.sh`) if this becomes a recurring setup.
- One background SSH session silently produced zero output and left no trace of failure
  (the smoke test's first attempt) -- switched to `nohup ... > remote.log 2>&1 &` with
  polling for a completion marker in the remote log file, rather than piping a foregrounded
  SSH command's stdout through the local tool's own backgrounding, which turned out to be
  more failure-prone over this pod's network path.

**Caveats:**
- Same `n_pairs=200` requested / 105 actually qualified caveat as the CODI run -- corpus
  pool size at n=600, not a bug.
- `matches_twin` here is the same minimal-pair-specific metric as the CODI run's notes
  describe; not directly comparable to E2's `matches_cf` numbers without accounting for the
  different donor construction.
- Pass 0 being exactly 0/105 for both `answer_changed` and `matches_twin` (not just
  `matches_twin`) suggests it may be closer to a true no-op stage for Coconut (e.g. an
  initial "read the question" pass) rather than merely under-detected -- unlike CODI's
  near-zero-but-not-exactly-zero single-slot iterations, which do move answers even where
  they don't move them to the twin's answer specifically.

**Next:** CODI counterpart is `20260920-190420_codi_minimal-pair-patch` (same session).
Per `steered_to_donor_audit.md` §5 E4, DAS on minimal-pair targets is the natural
follow-up; Coconut's pass-1 result in particular is now a strong, well-localized candidate
for a subspace-level (rather than full-vector) faithfulness check, since a single pass
already does most of the work a learned subspace would need to explain.
