## 2026-09-20 — CODI smooth path interpolation patching, all 6 positions (codi, run_id: 20260920-043950_codi_interpolation-pilot)

**Goal:** Task 3 (smooth path interpolation patching). Follow-up to
`20260920-031925_codi_interchange-placeholder-pilot`, whose full alpha=1 donor-swap at
CODI's non-decodable placeholder positions (z0, z3) found an unambiguous null: no
steering toward the donor's answer, despite those positions being load-bearing under
mean-ablation (`20260919-192228_codi_early-termination-ablate-all`). That null is
consistent with two different stories: (a) the slot genuinely carries no portable,
donor-specific content, or (b) a full alpha=1 jump knocks the recipient's residual
stream off-manifold before any donor content could be read out, and the interchange
test never gets a fair look. Walking the continuous path between recipient and donor
at each position, instead of only testing the alpha=1 endpoint, is meant to
distinguish those two.

**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3`, compute_steps=6
(6 continuous-thought loop iterations at inference).

**Data:** gsm8k-aug test, pool_n=400 (seed=0) filtered to the 171 base-correct
examples, then 50 (recipient, donor) pairs sampled with different gold final answers;
run seed=0.

**Command:**
```
cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/interpolation_patch_codi.py \
    --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \
    --slug interpolation-pilot --stage pilot --hardware "RunPod RTX A6000 (secure)" \
    --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \
    --lora_r 128 --lora_alpha 32 --lora_init --greedy True \
    --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
    --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \
    --pool_n 400 --n_pairs 50 --positions 0,1,2,3,4,5
```
(full argv in `eval_command.txt`). 11-point alpha sweep (0.0 to 1.0, step 0.1) x 50
pairs x 6 positions = 300 (pair, position) trajectories, reading `outputs.logits[:, -1, :]`
from the eot-embedding forward pass immediately after the 6-iteration loop at each
alpha (the paper's own vocabulary-projection decode method, same readout
`decode_patch_codi.py` uses for its logit lens). Pilot ran in 447s of actual compute
on an A6000; total pod wall time including env setup, checkpoint download, and a
throwaway n=5/2-position smoke test was under 15 minutes.

**Headline results:**
- Overall classification across all 300 trajectories: `smooth_transition`=2.0% (6/300),
  `off_manifold_collapse`=0%, `step_function_invariance`=0%, **`ambiguous`=98.0%
  (294/300)**.
- Per position (smooth_transition / ambiguous, n=50 each): z0 0%/100%, z1 2%/98%,
  z2 0%/100%, z3 4%/96%, z4 6%/94%, z5 0%/100%.
- Non-decodable {z0,z3} pooled: 2.0% smooth, 98.0% ambiguous (n=100).
  Decodable {z2,z4} pooled: 3.0% smooth, 97.0% ambiguous (n=100).
- **No meaningful non-decodable-vs-decodable contrast** (2% vs 3% smooth is noise at
  n=100 per group) and **no off-manifold-collapse or step-function signature anywhere**
  — every position, decodable or not, is overwhelmingly `ambiguous` by
  `classify_trajectory`'s thresholds.

**Interpretation:** Unlike the Coconut counterpart
(`20260920-041935_coconut_interpolation-pilot`), this readout is the paper's own decode
method and does carry real probability mass on the answer digit at the endpoints (this
is the same readout `decode_patch_codi.py` already validated against the released
checkpoint's headline accuracy), so the near-total `ambiguous` result here is a genuine
finding about the interpolation path, not a readout-placement artifact. It does not
resolve the interchange-patching null's two hypotheses cleanly: there's no evidence of
off-manifold collapse (P(y_A)/P(y_B) don't both crash toward zero with an entropy spike
at intermediate alpha) and no evidence of a clean smooth crossfade or step-function
switch either — the trajectories are just noisy/non-monotonic between the two
endpoints at almost every position, decodable or not. Combined with the interchange
pilot's null, the most consistent reading is that CODI's per-position causal effect on
the *specific* donor answer is weak-to-absent along the whole path, not just at the
alpha=1 extreme — arguing more for "not portable, donor-specific content here" (story
a) than "off-manifold artifact" (story b), though this run alone doesn't prove that;
see Caveats.

**Gotchas hit:**
- **First attempt at this run was silently destroyed.** A prior pod (`opzy7386g1be6r`)
  ran this exact pilot to completion with identical-looking headline numbers, but it
  used ephemeral container disk with no persistent/network volume attached, and got
  stopped and restarted (by a subagent handoff mix-up, not intentionally) before the
  results could be copied off — the container disk wipes on restart, so
  `/workspace` was empty afterward. This run (`20260920-043950_codi_interpolation-pilot`)
  is a clean rerun on a fresh pod (`oycx8vx20imwt7`), executed in the foreground with
  results rsynced back and the pod terminated immediately on completion, before doing
  anything else to it — no persistent volume was needed once the pull-back was made
  synchronous with the run instead of handed off.
- The pod's base image (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`) has
  neither `rsync` nor a `pip` binary inside `codi_setup.sh`'s `uv`-created venv --
  `apt-get install -y rsync` on the pod, and `uv pip install --python .venv/bin/python
  matplotlib` instead of `.venv/bin/pip install`, both fixed before this run.
- Syncing `latentreasoning/` and `scripts/` in one `rsync` call with a shared
  destination flattened both directories' contents together at the top level (rsync
  copies a source dir's *contents* into the destination when given a trailing slash,
  and two sources sharing one destination merge) -- fixed by rsyncing each source into
  its own explicit destination subdirectory.
- No bugs found in `interpolation_patch_codi.py`/`interpolation_common.py` itself --
  both the smoke test (n=5, positions 0/3) and the full pilot ran clean on the first
  try against the real checkpoint.

**Caveats:**
- n=50 pairs, pilot scale, not the full held-out set -- per-position n=50 is enough to
  rule out a large, consistent smooth/collapse/step effect but not a subtle one.
- `classify_trajectory`'s thresholds (collapse probability floor 0.01, entropy jump
  1.5 nats, step-dominance fraction 0.5, monotonic-correlation 0.5) are a first-pass
  heuristic tuned only against synthetic trajectories, not against this checkpoint's
  actual probability/entropy scale -- worth checking `interpolation_pairs.jsonl`
  directly (raw per-alpha P(y_A)/P(y_B)/entropy) before concluding "nothing happens"
  rather than "the classifier's thresholds don't match this signal's scale."
- `intervention_accuracy` in `manifest.json` (0.02 = non-decodable smooth_transition
  rate) is a placeholder headline metric per the shared `RunRecord` schema.

**Next:** Inspect the raw per-alpha curves in `interpolation_pairs.jsonl` /
`interpolation_curves.png` directly (not just the classification labels) to check
whether P(y_A)/P(y_B) show any systematic trend the current thresholds miss. If the
Coconut readout gets fixed (teacher-force through the "###" delimiter first, per that
run's notes), rerun both mechanisms' interpolation sweeps at matched n for a real
cross-mechanism comparison.
