## 2026-09-20 — Coconut smooth path interpolation patching, all 6 passes (coconut, run_id: 20260920-041935_coconut_interpolation-pilot)

**Goal:** Task 3 (smooth path interpolation patching), Coconut counterpart to
`20260920-041329_codi_interpolation-pilot`. Follow-up to
`20260920-031246_coconut_decode-patch-pilot`, whose grouped donor-interchange patch
found `answer_changed` tracking decodability (McNemar p=0.0015) while `steered_to_donor`
stayed at/near floor everywhere — i.e. patching the more-decodable pass group perturbs
the answer more, but essentially never steers it to specifically match the donor's gold
value. Walking the continuous path between recipient and donor at each pass (instead of
only the full alpha=1 swap) asks whether that null is a genuine "no portable content"
result or an artifact of the endpoint knocking the recipient off-manifold.

**Mechanism / model:** `coconut`, backbone openai-community/gpt2, checkpoint
`hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33`, compute_steps=6 (6 `<|latent|>`
passes).

**Data:** gsm8k-aug-equivalent test set (`gsm_valid-gold-reasoning-trace_test.json`,
same corpus/questions as `gsm8k_aug.py`), pool_n=400 (shuffled, seed=0) filtered to the
130 base-correct examples, then 50 (recipient, donor) pairs sampled with different gold
final answers; run seed=0.

**Command:**
```
python scripts/interpolation_patch_coconut.py \
    --checkpoint_path <hf checkpoint_33 snapshot path> \
    --data_dir /workspace/coconut_data \
    --slug interpolation-pilot --stage pilot --hardware "RunPod RTX A6000 (secure)" \
    --num_latents 6 --pool_n 400 --n_pairs 50 --positions 0,1,2,3,4,5 \
    --decodable_positions 0,1,4 --nondecodable_positions 2,3,5
```
(full argv in `eval_command.txt`; `--decodable_positions`/`--nondecodable_positions` are
the pass ranking `decode_patch_coconut.py`'s pilot found by empirical logit-lens
accuracy, used here only for summary grouping/plot highlighting, not for selecting which
positions to sweep — all 6 were swept). 11-point alpha sweep (0.0 to 1.0, step 0.1),
50 pairs x 6 positions = 300 (pair, position) trajectories. Total pod wall time
(env setup + checkpoint download + smoke test + this run) was well under 10 minutes;
the pilot itself ran in 190s.

**Headline results:**
- Overall classification across all 300 trajectories: `smooth_transition`=7.7% (23),
  `off_manifold_collapse`=0% (0), `step_function_invariance`=4.7% (14),
  **`ambiguous`=87.7% (263)**.
- Non-decodable passes {2,3,5}: 90% ambiguous, 8% smooth, 2% step, 0% collapse.
  Decodable passes {0,1,4}: 85.3% ambiguous, 7.3% smooth, 7.3% step, 0% collapse — no
  meaningful decodable/non-decodable split in this classification, unlike CODI's z0/z3
  vs z2/z4 contrast (see `20260920-041329_codi_interpolation-pilot`).
- **The classification result is not the interesting number here — the readout
  strength is.** Inspecting `interpolation_pairs.jsonl` directly: `P(y_A)` and `P(y_B)`
  are at or below floor (`< 1e-2`, median max-over-alpha `~6e-13` for P(y_A) and
  `~1.5e-16` for P(y_B)) at **100% of the 300 trajectories, for every alpha including
  alpha=0 and alpha=1** (i.e. even the *recipient's own* answer digit gets essentially
  zero probability mass at this readout position, unpatched). Entropy is also ~0 at
  every alpha, meaning the distribution is sharply peaked -- just not on either
  candidate answer token. This is exactly the caveat flagged in
  `interpolation_patch_coconut.py`'s own docstring before running: the readout used
  here is the single forward step *immediately following* the last `<|latent|>` pass,
  before "### <answer>" would normally be generated for this checkpoint's GSM8K format
  (per `coconut_common.extract_answer_after_delimiter`). The model is almost certainly
  placing its probability mass on "#" (the start of the delimiter) at this position,
  not on the answer digit -- so `classify_trajectory` is correctly reporting "ambiguous"
  on 300 near-flat-at-floor curves, not measuring anything about the interpolation path
  itself.

**Interpretation:** This run does NOT answer the smooth-transition-vs-collapse question
for Coconut, because the chosen readout position doesn't carry the answer signal at all
regardless of alpha or which pass is patched -- both endpoints (alpha=0, alpha=1) are at
floor, so there's no baseline to see move. It DOES confirm, independently of the
interpolation question, that a single-token readout right after the last latent pass is
the wrong place to look for Coconut's answer probability on this checkpoint's format
(consistent with `decode_patch_coconut.py`'s own logit-lens numbers being modest,
28.8% matched top1, and computed via a *different* method -- `lm_head` on the live
hidden state at each pass, not a full forward step at the sequence's current end).
**Next run should either (a) teacher-force through the literal `" ###"` delimiter
tokens before reading P(y_A)/P(y_B) on the digit position that follows, or (b) locate
the actual highest-probability-mass position empirically per example** rather than
assuming it's the immediate next token. Contrast with CODI
(`20260920-041329_codi_interpolation-pilot`), whose analogous readout (post-loop,
pre-eot-generation logits) is the paper's own decode method and does put real
probability mass on the answer digit at alpha=0/1, so that pilot's "98% ambiguous"
result is a genuine (if still null) finding about the interpolation path, not a
readout-placement artifact.

**Gotchas hit:**
- Wrote and ran an unwitnessed real bug risk (per the script's own docstring, KV-cache
  legacy-normalization / `finish_and_decode` replay edge cases) but the smoke test
  (n=5, positions 0,3) and the full pilot both ran clean on the first try, no code
  changes needed.
- The readout-position issue above was anticipated in the script's docstring before
  running ("if this readout is uniformly near-floor... that's a real and informative
  possible outcome, not necessarily a bug") -- confirmed exactly that outcome.
- Coconut's own env setup (torch==2.5.1 from the PyTorch cu121 index, transformers/
  datasets/etc. from PyPI in a separate `pip install`) worked without the KV-cache
  gotchas `decode_patch_coconut.py`'s notes flagged for a mismatched transformers pin.

**Caveats:**
- n=50 pairs, pilot scale, not the full held-out set.
- `intervention_accuracy` in `manifest.json` (0.08 = non-decodable smooth_transition
  rate) is a placeholder headline metric per the shared `RunRecord` schema; given the
  readout-floor issue above, don't read anything into its specific value here.
- `decodable_positions_in_scope`/`nondecodable_positions_in_scope` reuses
  `decode_patch_coconut.py`'s pilot ranking (fit on n=200, half A) rather than
  re-deriving it on this run's own pool -- a label for grouping, not a re-verified split.

**Next:** Rerun with a fixed readout (teacher-force the delimiter, or empirically locate
the answer-bearing position per example) before drawing any conclusion about Coconut's
interpolation path. Compare directly against `20260920-041329_codi_interpolation-pilot`
once that fix lands.
