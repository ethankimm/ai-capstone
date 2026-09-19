## 2026-09-19 — Ablation + attention diagnostics: position matters off-manifold, not by content; the answer read-out barely looks at it anyway (recurrent_depth, run_id: 20260919-085911_recurrent_depth_ablate-attn)

**Goal:** Two follow-up diagnostics on `20260919-075647_recurrent_depth_patch-pilot`'s causal
null (real-donor patch vs. random-control patch: no significant difference, McNemar
p=0.73). (1) Ablation: does the injected position matter *at all*, independent of content —
zero it out or replace it with the dataset mean, instead of swapping in a donor's value, and
see if the answer-proxy moves more than content-swaps did. (2) Attention: does the coda's
answer-trained read-out even attend to the position patching targets, or does it get the
answer from elsewhere in the sequence — a direct mechanistic check on *why* patching content
there does nothing.
**Mechanism / model:** `recurrent_depth`, gpt2 (125,621,760 params) / the step-supervised
full-run checkpoint (`local:20260918-214656_recurrent_depth_stepsup-split4-4-4-full/ckpt`,
unchanged, no retraining). New script `scripts/patch_recurrent_depth_diagnostics.py`.
**Data:** gsm8k-aug test n=200 seed=0 (the shared slice). Task 1 replays the exact 200
`(recipient, step, donor, control_donor)` tuples from `20260919-075647_..._patch-pilot`'s
`patch_pairs.jsonl`, so the real/control columns below are that run's own numbers, not
re-measured — only zero/mean are new. Task 2 uses the first 80 test examples with >=2 steps
(fresh honest forward passes, not from the patch run).
**Design — Task 1 (ablation):** same injection mechanic as the original patch run (overwrite
only the last-position slice of `s_s`, continue iterating honestly from there): `zero_vec` =
all-zeros; `mean_vec` = the per-`s` mean of the honestly-computed `s_s[:, -1, :]` across all
200 test examples' valid steps (computed in a first pass: 189/134/90/41/18/5/2 examples
contribute to s=1..7's mean respectively, same support as the original run's step-readout
matrix). Same answer-proxy-changed metric (iteration-`n_steps` read-out vs. baseline),
paired per recipient, McNemar exact on each pair of conditions.
**Design — Task 2 (attention):** `LoopedGPT2`'s state `s_i` lives on the *same* sequence
positions at every iteration (prompt tokens + the number position) — there's no separate
"iteration i's number-position token" for the coda to attend back over; only the *value* at
that one position evolves across iterations. So "attention to the number position at earlier
iterations" isn't well-formed here. What's measured instead: at the point the coda produces
the answer-trained read-out (iteration `n_steps`, the honest, unpatched trajectory), how much
attention mass does the last-position query assign to *itself* (the exact vector patching
overwrites) vs. every other prompt position, averaged over heads, per coda layer.
Implementation note: `GPT2Block.forward` in the installed transformers (5.17.0) always
returns a bare hidden_states tensor — it computes `attn_output, _ = self.attn(...)`
internally and discards the attention weights even with `output_attentions=True` threaded
through `**kwargs`. Attention weights were recovered with a forward hook on `block.attn`
directly (`register_forward_hook(..., with_kwargs=True)`, capturing its own
`(attn_output, attn_weights)` return), with the model loaded via
`attn_implementation="eager"` (SDPA silently declines to materialize weights).
**Command:**
```
uv run python scripts/patch_recurrent_depth_diagnostics.py \
  --ckpt-dir results/20260918-214656_recurrent_depth_stepsup-split4-4-4-full/ckpt \
  --pairs-file results/20260919-075647_recurrent_depth_patch-pilot/patch_pairs.jsonl \
  --attn-n 80 --slug ablate-attn --stage pilot --hardware "RunPod RTX A5000 (secure)"
```
(pod `hfb7u3ucpxmllv`, CA-MTL-1, RTX A5000 secure $0.27/hr; created 08:55 UTC, terminated
~09:05 UTC after result sync — ~10 min total incl. `apt-get install rsync git` (neither was
on the base `runpod/pytorch` image), `uv sync --extra train`, checkpoint transfer, one
mid-run fix to the attention-hook approach (see Gotchas), and two script runs. Total pod
cost **~$0.05**.)
**Headline results:**

*Ablation (n=200 pairs, replayed from the patch-pilot):*

| condition | answer-proxy changed | vs. real (McNemar p) | vs. control (McNemar p) |
|---|---|---|---|
| real donor (known) | 2.5% (5/200) | — | — |
| random control (known) | 3.5% (7/200) | — | — |
| **zero-ablation** | **9.5%** (19/200, CI [6.2%, 14.4%]) | **p=0.00052** | **p=0.0018** |
| mean-ablation | 2.5% (5/200, CI [1.1%, 5.7%]) | p=1.0 | p=0.6875 |

*Attention (n=80 examples, honest unpatched forward pass, coda layers 1→4):*

| coda layer | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| self-attention (the patched position) | 2.3% | 2.3% | 1.5% | 0.9% |
| attention to all other positions | 97.7% | 97.7% | 98.5% | 99.1% |

**Interpretation:**
- **Zero-ablation is the one condition that moves the needle, and it moves it in the
  "this is an anomaly" direction, not the "this position carries the answer" direction.**
  9.5% vs. real/control's 2.5%/3.5% is a real, significant effect (p<0.002 both
  comparisons) — but mean-ablation, which is just as content-free as zero-ablation, ties
  the real/control rate exactly (2.5%, p=1.0 vs. real). The position isn't causally inert —
  push it somewhere GPT-2 activations never naturally sit (a zero vector, after this
  architecture's LayerNorm-normalized adapter input) and the answer changes more often —
  but *which* non-degenerate value sits there (a real donor's content, an unrelated
  perturbation, or the population mean) makes no detectable difference. This sharpens
  rather than overturns the patching null: the model is sensitive to *off-manifold*
  perturbations at this position, not to its *specific decoded content*, which is a
  different and more specific claim than "the position does nothing."
- **The attention result independently explains why.** The coda's answer-trained read-out
  assigns 97-99% of its attention mass to positions *other than* the one it's reading from
  and the one patching targets — self-attention on the patched position itself is under 2.5%
  at every layer and *falls* with depth (2.3% → 0.9%). The model is structurally not looking
  at that position's content when it computes the answer; it's reading the answer largely
  from elsewhere in the sequence (most plausibly the question tokens themselves, not probed
  further here). That the two diagnostics — one behavioral (ablation), one purely
  mechanistic (attention weights, no intervention at all) — converge on the same story from
  completely independent methods is the strongest evidence yet for the faithfulness-spine
  finding: not just "swapping content doesn't move the answer" but "the model isn't even
  looking at that content in the first place."
- **Net effect on the Sep 25 story:** strengthens it, with a more precise claim. Not "the
  recurrent state is causally inert" (zero-ablation shows it isn't, entirely) but "the
  recurrent state's *specific, decodable content* is not what the model reads to produce the
  answer — it is sensitive to gross anomalies there but blind to meaning." Worth stating
  this way rather than the flatter "no faithfulness" framing from the patch-pilot alone.
**Gotchas hit:**
- `GPT2Block.forward` (transformers 5.17.0) discards attention weights internally
  (`attn_output, _ = self.attn(...)`) regardless of `output_attentions`; recovered via a
  forward hook on `block.attn` instead of trusting the block's own return value — see
  Design above. First script attempt assumed the block returned attention weights directly
  and crashed (`RuntimeError`, caught before any bad data was logged) — fixed and re-run
  before this run's numbers were produced, not a data-quality issue.
  `attn_implementation="eager"` is still required; SDPA returns `None` for attention
  weights even to a hook.
- A variable-shadowing bug in the Task 2 loop (`n = len(ex.intermediate_values)`, reusing
  the Task 1 pair-count variable `n`) corrupted two purely cosmetic manifest fields
  (`hyperparams.n_pairs` and the auto-generated `notes` string showed "4" instead of "200")
  — the actual results (`ablation_pairs.jsonl`, `metrics.extra.ablation.n_pairs=200`, every
  number above) were computed and saved correctly before the bug's effect overwrote the
  summary field; corrected by hand in `manifest.json` post-hoc rather than re-running the
  (already-paid-for) job. Fix belongs in the script for next time: don't reuse `n` inside
  the Task 2 loop.
- Base `runpod/pytorch` image had neither `rsync` nor `git`; `apt-get install -y rsync git`
  first, same as `20260919-080349_codi_decode-patch-full` hit independently.
- `git.commit` in the manifest is filled by hand (`f4aab59`, this repo's HEAD when the run
  was launched) since the pod checkout excluded `.git`; `author` likewise filled by hand
  (`git config user.name` returns empty on the git-less pod copy) — same gap noted in
  several earlier runs this session, worth fixing in the sync step for next time.
**Caveats:**
- n=200 ablation pairs (McNemar's discordant-pair counts are small for some comparisons —
  e.g. zero-vs-real has only 16 discordant pairs — but the p-values are still well under
  0.01, so this isn't a marginal call); n=80 for the attention diagnostic (self-attention
  mass has a visible, monotone per-layer trend across only 4 layers — read the trend, not
  precise third-decimal values).
- The attention diagnostic is on the *honest, unpatched* trajectory only — it doesn't
  directly prove the same near-zero self-attention holds inside a patched trajectory too
  (plausible given the mechanism is identical, but not measured here).
- "Other positions" attention isn't broken down by *which* other positions (question tokens
  vs. earlier prompt structure vs. padding) — a natural follow-up if this becomes a writeup
  figure, not done here to stay in budget.
- Zero-ablation's effect size (9.5%) is still small in absolute terms and well below what
  "the model reads the answer from this position" would predict (that would look like a
  much larger, content-*dependent* effect) — it's evidence of anomaly-sensitivity, not of
  the position being where the real computation lives.
**Next:**
- If this goes into the Sep 25 writeup, pair it with whatever the parallel CODI-side
  ablation+attention diagnostic (pod `codi-ablate-attn`, running concurrently this session —
  check `results/` for its run_id) finds, for the same two-mechanism symmetry the causal
  patching results already have.
- A cheap addition if useful later: break down "other positions" attention by token
  identity/position in the question, to see whether the model is literally re-reading the
  relevant operand tokens rather than using the iterative state at all.
