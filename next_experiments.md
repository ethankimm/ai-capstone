# Next experiments (staged this session, not yet run on RunPod)

Everything below is written, syntax-checked, and (where noted) smoke-tested against
the real checkpoint locally. Nothing has been logged as a real run yet -- these are
all candidates for the next RunPod session(s). Update/delete this file as runs land;
it's a scratch planning doc, not part of the generated results index.

## 1. CODI: donor-interchange patching at the non-decodable placeholder positions

**Script:** `scripts/interchange_patch_placeholder_codi.py`
**Status:** written, syntax-checked. Not run.

Follow-up to `20260919-184323_codi_decode-patch-full-eval` (null on the *decodable*
focus iteration) and `20260919-192228_codi_early-termination-ablate-all` (z0/z3 are
load-bearing under mean-ablation, p=6e-5 / p=0.0025). This asks the sharper question
with a content-*bearing* donor instead of a content-free mean vector:

- Single-slot donor-interchange patching at every iteration (1-6), donor/recipient
  pairs chosen by different final answers.
- Grouped: same donor, same recipient, joint-swap {z0,z3} (non-decodable) vs {z2,z4}
  (decodable), paired McNemar comparison.
- Metrics: `answer_changed`, `steered_to_donor` (patched answer matches donor's gold
  answer and base didn't), Wilson CIs + McNemar.

Outcome A (steers) -> placeholder slots carry transferable structured state, logit
lens is just looking at the wrong position. Outcome B (doesn't steer, despite being
load-bearing) -> non-representational / distributed computation there.

Suggested first run: `--full_test False --eval_n 200 --n_pairs_per_site 60
--n_grouped_pairs 100`, pilot stage.

## 2. CODI: linear + MLP probes vs logit lens (basis-drift check)

**Script:** `scripts/probe_codi.py`
**Status:** written, syntax-checked. Ridge regression + signed-log1p transform
verified against synthetic data locally (roundtrip + recovery both exact). Not run
against the model.

Tests whether z0/z3's intermediate values are linearly (or shallowly non-linearly)
recoverable once you stop reading through the unembedding basis W_U -- i.e. whether
the logit-lens null is a basis artifact rather than genuine non-representation.

- Closed-form ridge regression + a small 2-layer MLP (no sklearn/scipy dependency),
  fit per (iteration, step) on `train`, evaluated on the fixed `validation` split.
- Logit-lens accuracy computed on the *same* eval examples in the same run, so the
  probe-vs-lens comparison is apples-to-apples.
- Headline: average probe accuracy (tolerance-based) at non-decodable iterations
  {1,4} vs decodable {3,5}, alongside logit-lens accuracy at the same positions.

**Not implemented:** Distributed Alignment Search (DAS) on the probe-extracted
directions (Priority 2's second half -- verify a probe-aligned subspace has causal
control where raw donor patching at z0/z3 fails). Flagged in the script's
`metrics.extra.das_note`; needs a differentiable subspace-intervention pass, a
separate piece of work from the ridge/MLP readout probes.

Suggested first run: `--train_n 3000 --eval_n 300`, pilot stage.

## 3. Coconut: decode + donor-interchange patching (data-driven decodable split)

**Script:** `scripts/decode_patch_coconut.py` (+ shared primitives in
`scripts/coconut_common.py`)
**Status:** written and **smoke-tested end-to-end** against the real released
checkpoint (`connordilgren/gpt2-gsm8k-coconut`, checkpoint_33) on CPU, pinned venv
(`torch==2.5.1`, `transformers==4.46.2`, matching Coconut's own `requirements.txt`).
2/5 correct on a real slice with the trained 6-latent config, consistent with the
paper's 33.1%. Ran the full script (decode + single-slot + grouped patch) on n=8
without errors; that throwaway output was deleted (n<10, not a logged run).

Extends the same "decodability != causal faithfulness" question to the other
width-based LRM this project studies, specifically to compare against Dilgren &
Wiegreffe (COLM 2026, arXiv:2604.04902) -- whose checkpoint collection this is. Unlike
CODI, does **not** assume a z0/z3-style split: D&W's own backtracking-search finding
is that Coconut encodes gold traces in most latents when correct (54-93%), so the
script empirically ranks passes by decode accuracy (half-A/half-B held-out split) and
groups top-half "most decodable" vs bottom-half "least decodable" for the joint patch.

- Task 1: decode every example, all 6 passes, vocabulary projection on the live
  hidden state (the paper's own method).
- Task 2: single-slot donor-interchange patching at every pass.
- Task 3: grouped patch, most-decodable-pass-group vs least-decodable-pass-group,
  same donor/recipient, paired McNemar.

**Data prerequisite (already generated, local only so far):**
`~/Projects/are-lrms-easily-interpretable/data/gsm_original_test.json` +
`gsm_valid-gold-reasoning-trace_test.json`, produced via that repo's
`preprocessing/prepare_gsm8k.py`. Needs copying onto the pod (`--data_dir`).

**Environment prerequisite:** a venv pinned to Coconut's own `requirements.txt`
(`torch==2.5.1`, `transformers==4.46.2`) -- newer transformers breaks the hand-rolled
KV-cache slicing in `coconut_common.py` (verified locally: modern `transformers`
returns a `Cache` object with no `to_legacy_cache()`, and also refuses legacy tuples
passed back in). Not the same venv as CODI's.

Suggested first run: `--full_test False --eval_n 200 --n_patch_pairs 60
--n_grouped_pairs 100`, pilot stage. Also worth running the **full 1194-example**
gold-trace set once the pilot's numbers look sane, to get a real accuracy number
comparable to Table 1 (33.1%) rather than a small-n estimate.

## 4. CODI + Coconut: smooth path interpolation patching (Task 3)

**Scripts:** `scripts/interpolation_patch_codi.py`, `scripts/interpolation_patch_coconut.py`
(+ shared classification/plotting in `scripts/interpolation_common.py`)
**Status:** Coconut run and logged: `results/20260920-041935_coconut_interpolation-pilot`
(pilot, n=50 pairs x 6 positions). Real finding, but not the intended one -- see that
run's notes.md: the single-forward-step readout right after Coconut's last latent pass
sits on the delimiter, not the answer digit, so P(y_A)/P(y_B) are at floor for every
alpha including the endpoints (no baseline to see move) and 87.7% of trajectories
classify as `ambiguous` for that reason, not because of anything about the
interpolation path itself. Needs a readout fix (teacher-force through `" ###"` first,
or empirically locate the answer-bearing position) before the classification result
means anything for Coconut.

CODI: run and logged, `results/20260920-043950_codi_interpolation-pilot` (pilot,
n=50 x 6 positions). Its first attempt ran to completion with near-identical numbers
but was destroyed by an ephemeral-disk wipe during a subagent handoff mix-up before
results could be pulled back (see that run's notes.md gotchas); this rerun executed
in the foreground with results pulled off synchronously, no persistent volume needed.
Headline: overall `smooth_transition=0.02, off_manifold_collapse=0.0,
step_function_invariance=0.0, ambiguous=0.98`; non-decodable {z0,z3} vs decodable
{z2,z4} show no meaningful difference (2% vs 3% smooth) and no collapse/step signature
anywhere. Unlike Coconut's floor issue, this readout is the paper's own decode method
and does carry real probability mass at the endpoints, so this is a genuine (if null)
finding, not a readout-placement artifact.

Follow-up to `20260920-031925_codi_interchange-placeholder-pilot` (full alpha=1
donor-swap at CODI's z0/z3: unambiguous null, no steering) and
`20260920-031246_coconut_decode-patch-pilot`. That null is consistent with two
different stories: the slot isn't representational, OR the alpha=1 endpoint knocks the
recipient's residual stream off-manifold before any donor content could be read out.
This walks the continuous path between recipient and donor instead of only testing the
endpoint:

- For (Recipient A, Donor B) pairs where the base model gets both right and
  y_A != y_B, at each swept latent position i: z_i(alpha) = (1-alpha) z_i^A + alpha
  z_i^B for an 11-point sweep alpha in {0.0, 0.1, ..., 1.0}, single-slot patch into
  recipient A's own forward pass.
- CODI reads P(y_A)/P(y_B) + entropy off the post-loop pre-generation logits (the
  paper's own vocabulary-projection decode, one readout after all 6 iterations).
  Coconut has no separate final-answer step, so it reads the same three quantities off
  the single forward step that immediately follows the last continuous-thought pass
  (before "### <answer>" would normally be generated) -- a strictly harder readout,
  called out explicitly in that script's docstring.
- Each (pair, position) trajectory is classified as `smooth_transition`,
  `off_manifold_collapse`, `step_function_invariance`, or `ambiguous`
  (`interpolation_common.classify_trajectory` -- collapse checked first as the most
  falsifiable claim, then a concentrated single-alpha-step dominance switch for
  step-function, then a spread-out monotonic trend for smooth; see that function's
  docstring for the exact thresholds and why order matters).
- `interpolation_curves.png`: mean P(y_A), P(y_B), entropy vs alpha, one line per
  swept position, decodable positions solid / non-decodable dashed.

CODI defaults to sweeping all 6 known positions (z0..z5), highlighting the already-
established z0/z3 (non-decodable) vs z2/z4 (decodable) split. Coconut has no fixed
split (per the decode_patch pilot, its passes are mostly decodable data-dependently) --
`--decodable_positions`/`--nondecodable_positions` take that pilot's
`most_decodable_passes`/`least_decodable_passes` output if you want the same
highlighting; left empty it just sweeps every position unhighlighted.

Suggested first run: `--pool_n 400 --n_pairs 50` (matches the task spec's n=50), pilot
stage, both mechanisms.

## Not started

- **Coconut probes** (ridge/MLP vs logit lens, mirroring `probe_codi.py`) -- planned
  but not written this session.
- **DAS** on CODI's probe directions (see #2).
- **Huginn-0125 validation of recurrent_depth** (advisor doc's Priority 3) -- explicit
  hold: different backbone/mechanism swap, bigger separate scope, not part of this
  session's work.
- **Llama-3.2-1B checkpoints** for CODI/Coconut -- out of scope per this session's
  explicit choice to stay GPT-2-only for comparability with everything logged so far.

## Suggested order

1. Coconut decode_patch pilot (#3) -- cheapest sanity check, confirms the released
   checkpoint reproduces Table 1 on a real GPU before trusting the patch numbers.
2. CODI interchange-placeholder pilot (#1) -- directly answers the paper-critique
   framing's Priority 1 question, no new environment needed (reuses the CODI venv).
3. CODI probes (#2) -- independent of #1, can run in parallel/either order.
4. Interpolation patching (#4), both mechanisms -- direct follow-up to #1's null and
   #3's decode+patch pilot, same venvs, no new environment needed.
5. Full-scale reruns of whichever pilots look interesting, then `record-run` skill to
   log each as a real run (manifest + notes.md + `rebuild_index.py`).
