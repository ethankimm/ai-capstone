## 2026-09-19 — Causal patching on the step-supervised checkpoint: no faithfulness, symmetric with CODI (recurrent_depth, run_id: 20260919-075647_recurrent_depth_patch-pilot)

**Goal:** The Oct 16 causality item flagged as open in
`results/20260918-214656_recurrent_depth_stepsup-split4-4-4-full/notes.md`'s "Next"
section ("patch s_i from a counterfactual example with a different v_i and check the
read-out at i and the answer at n move as predicted"), designed to mirror
`results/20260919-073312_codi_decode-patch-pilot/notes.md`'s CODI test exactly so the
two mechanisms — the only two in this repo with a working latent state — are measured
the same way. Going in, recurrent_depth already had *correlational* evidence against
faithfulness (full-scale run: answer right 37.5% of the time when all intermediates
decode correctly vs. 8.6% when none do) but no direct intervention test. CODI's
equivalent direct test (patch a donor's latent, see if the answer/read-out move) found
no faithfulness at n=30. This run runs the same test at n=200 for recurrent_depth.

**Mechanism / model:** `recurrent_depth`, gpt2 (125,621,760 params) / the step-supervised
full-run checkpoint (`local:20260918-214656_recurrent_depth_stepsup-split4-4-4-full/ckpt`,
`model.pt`, unchanged — no retraining, no new mechanism), compute_steps=None (this run
reports intervention metrics, not an accuracy sweep). New script
`scripts/patch_recurrent_depth.py`.

**Patch mechanic:** for probe sequence `prompt + " ####"`, `LoopedGPT2` computes
`s_i = iterate(e, s_{i-1})` for i=1..r with full causal self-attention over every
position (`e` fixed per example, from the prelude). For a chosen step `s` (1-indexed,
`s <= n_steps-1`, i.e. an iteration with a trained step target), the recipient's own
honest `s_1..s_s` are computed, then ONLY the last-position slice of `s_s` (the number
position right after " ####") is overwritten with a donor's independently-computed
`s_s[:, -1, :]` — same architecture, different example. Iterations `s+1..n_steps`
continue on this hybrid state (causal masking means nothing already-computed attends
back to the patched position, so this is a clean single-vector intervention, same
spirit as CODI's single-latent-token swap). Two donor conditions, same injection
position, paired to the same recipient/baseline:
  - **real**: donor shares the recipient's step count and has a *different* gold value
    at step `s`.
  - **control**: donor is an unrelated (example, iteration) pair, not matched on step
    count or value.

**Two metrics** (both read out via the trained `coda + lm_head`, same read-out method
as `step_readouts.jsonl`):
  - `readout_next_moved_to_donor`: does iteration `s+1`'s read-out match the donor's
    specific step-`s` value's first token (real only — undefined for an unrelated
    control, since there's no specific value it should push toward)? Compared against
    `baseline_coincidence`: does the *unpatched* recipient's own iteration `s+1`
    read-out already happen to match the donor's value by chance (it's an off-diagonal
    cell of the iteration x step matrix, which the full-run's own data put at ~6-7%
    for adjacent iterations — a real, nonzero prior probability, not zero).
  - `answer_proxy_changed`: does iteration `n_steps`'s read-out — the iteration
    literally trained on the answer, not just correlated with it — change from the
    unpatched baseline? Reported for real and control, paired per recipient (McNemar).
    This is the proxy for "final answer changed"; full free-form generation wasn't
    used because `LoopedGPT2.generate` has no KV cache and recomputes `s_0..s_r` from
    fresh noise at every new token, so a one-iteration patch doesn't persist
    token-to-token the way CODI's cached autoregressive latents do. Restricting to the
    trained probe context sidesteps that architectural mismatch entirely, and the
    full-run notes already established this read-out matches the actual generated
    first token 84% of the time at r=1 (49.7% exact-value match) — a validated proxy,
    not an untested shortcut.

**Data:** gsm8k-aug test n=200 seed=0 (the shared slice, matching every other run
against this checkpoint). Patch pairs: 200 (recipient, step, donor) tuples sampled
without replacement from the 479 valid (example, step) combinations across the 189
test examples with >=2 steps (same population the full-run's step-read-out matrix
was built from), patch seed 42 (also seeds each example's `s_0` deterministically per
index, shared identically across baseline/real/control so the *only* difference
between conditions is the injected vector).

**Command:**
```
uv run python scripts/patch_recurrent_depth.py \
  --ckpt-dir results/20260918-214656_recurrent_depth_stepsup-split4-4-4-full/ckpt \
  --n-pairs 200 --slug patch-pilot --stage pilot --hardware "RunPod RTX A5000 (secure)"
```
(pod `r5rwibiui4937n`, CA-MTL-1, RTX A5000 secure $0.27/hr; created 07:52:32 UTC,
terminated ~07:58 UTC after checksum-equivalent rsync back — ~6 min total including
`uv sync --extra train` and two smoke tests (n=3, n=30) before the logged n=200 run.
The logged run itself took **8.8s** end-to-end on GPU: per-pair, per-condition
batch=1 forward passes on <=8 effective core-layer iterations of a 125M model are
essentially free. Total pod cost **~$0.03**.)

**Headline results:** `intervention_accuracy` (= `readout_next_moved_to_donor.real`)
`=0.0154`.

| metric | real | control | baseline / coincidence |
|---|---|---|---|
| next-iteration read-out matches donor's value (n=130) | 0.0154 (2/130) | — | 0.0154 (2/130) |
| answer-proxy (iteration n_steps read-out) changed from baseline (n=200) | 0.025 (5/200) | 0.035 (7/200) | — |

McNemar 2x2 on answer-proxy-changed (real vs. control, paired): both changed 2,
real-only 3, control-only 5, neither 190 — **p=0.727**, no evidence of a difference.

**Interpretation:**
- **Read-out propagation: real and baseline coincidence are numerically identical**
  (2/130 = 2/130). Patching a donor's specific step-`s` value into the state has *zero*
  measurable effect on the very next iteration's read-out beyond what would happen by
  chance with no patch at all. This is a cleaner null than it might look at first:
  the full-run's own matrix data already showed adjacent-iteration off-diagonal reads
  are low (~6-7%) but nonzero, so this isn't "the test can't detect anything" — it's
  "the specific injected content produces exactly the background rate, not more."
- **Answer-proxy: control changes the answer-proxy *more* than the real, matched patch**
  (3.5% vs 2.5%), and McNemar finds no significant difference (p=0.73, n=200 well
  powered to detect anything but a very small effect at these base rates). Exactly the
  same qualitative pattern CODI's pilot found (CODI: real 60.0% vs. control 63.3%,
  also control-slightly-ahead) — the model's downstream computation is not more
  sensitive to a semantically-matched counterfactual than to an unrelated perturbation
  of the same magnitude at the same position.
- **This makes the cross-mechanism claim symmetric.** Before this run, CODI had a
  formal intervention null (n=30) and recurrent_depth only had the full-run's
  correlational argument (37.5% vs 8.6%, answer accuracy conditional on read-out
  correctness — suggestive but not causal). Both mechanisms now have the same direct
  test, at comparable or better power (recurrent_depth n=200 vs. CODI's n=30), and
  both come back null. **Decodability without demonstrated faithfulness now holds for
  both mechanisms in this repo, under the same causal test, independent of end-task
  accuracy** (CODI 41.5%, recurrent_depth 14.5% ~ no-scratchpad control) — this is the
  finding to bring to Sep 25, not either mechanism's result alone.
**Gotchas hit:**
- Fresh RunPod `runpod/pytorch` image had neither `rsync` nor `git` installed
  (`apt-get install -y rsync git` first) — not a repo bug, just this base image; worth
  a one-line note in `runpod_setup.sh` for the next person who hits it.
- None in the patching logic itself — the n=3 and n=30 smoke tests before the logged
  run confirmed the mechanism produces nonzero, non-identical real/control outcomes
  (ruling out a dead-code patch that silently never applies) before spending the full
  n=200.
**Caveats:**
- `s_start` for the patched continuation is the recipient's own honestly-computed
  `s_s` with only the *last position* overwritten — positions before the answer-number
  slot are never touched, by design (causal masking means nothing already-computed can
  attend back to the last position anyway, so this doesn't leave the test open to a
  "the patch never really applied" objection, but it does mean this tests a
  single-vector intervention, not a broader state transplant).
- `answer_proxy_changed` uses the iteration-`n_steps` read-out (the model's own
  answer-trained target), not full free-form multi-token generation — see the design
  note above for why, and the full-run's 84%/49.7% correspondence for why this proxy
  is trusted rather than assumed.
- Both baseline coincidence and real read-out-movement are near-floor (2/130) — a
  larger n would tighten the CI further but is very unlikely to overturn "real ==
  baseline" given they're already numerically identical at this n.
- Random control donors are drawn from any example/iteration (including iteration 1,
  which is the best-decoded position) — if anything this could make control easier to
  disrupt with, not harder, so it doesn't bias the real≈control finding in the
  "faithful" direction.
**Next:**
- Both mechanisms in this repo now have a matched decode + causal-patch result
  (CODI: `20260919-073312_codi_decode-patch-pilot`; recurrent_depth: this run) --
  the natural Oct 9/16 writeup is these two side by side, not either alone.
- If a third mechanism is added (Coconut per the CODI pilot's "Next" section) it
  should get the identical design (matched-donor real patch + unrelated-donor
  control, paired McNemar on answer-proxy-changed) for a true three-way comparison.
- The one open asymmetry: CODI's control was "harder to tie" per its own notes
  (drawing from otherwise-dead odd iterations too); recurrent_depth's control draws
  from the full iteration range including the best-decoded iteration 1 — if this
  distinction matters, a control restricted to iteration `s` alone (matching donor's
  position exactly) would tighten the comparison further, but is unlikely to change
  the conclusion given how close real and control already are.
