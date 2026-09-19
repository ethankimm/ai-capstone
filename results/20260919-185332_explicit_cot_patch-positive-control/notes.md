## 2026-09-19 — Positive control for activation patching: on explicit CoT, a single-position residual patch is tracked by the next step ~85% of the time and moves the answer ~99%; content-free ablation collapses accuracy (explicit_cot, run_id: 20260919-185332_explicit_cot_patch-positive-control)

**Goal:** Every causal result on the latent mechanisms is a null — real-donor patch ≈
random-control patch, next read-out never moves toward the injected value (CODI
`20260919-080349` / eval-mode `20260919-184323`; recurrent_depth `20260919-075647`), and
content-free ablation of the patched position costs CODI nothing (`20260919-090132` /
`20260919-184716`). Those nulls are only interpretable if the *same kind of intervention*
demonstrably works somewhere on this backbone. This run is that positive control: on the
explicit-CoT GPT-2, where the intermediate value is a visible token, overwrite one result
span — as text, or as the residual stream at layer L — with a donor's, and measure whether
the continuation uses the injected value, whether the answer moves, and whether it moves
to the arithmetically consistent counterfactual. Same controls as the latent runs: random
unrelated donor, zero, mean.
**Mechanism / model:** `explicit_cot`, gpt2 (124.4M) / the full-run baseline checkpoint
`local:20260918-021435_explicit_cot_baseline-full/ckpt` (34.5% on the 200-slice), fp32,
greedy, `attn_implementation="eager"`, `model.eval()` (deterministic — the teacher-forced
continuation path reproduced the free-running baseline generation on 200/200 pairs).
New script `scripts/patch_explicit_cot.py`.
**Data:** gsm8k-aug test, all 1319 examples greedy-generated once (accuracy on all 1319:
35.1%, mean 29.3 generated tokens → `extra.cot_tokens`). A `(recipient, step s)` unit
qualifies if the recipient's *generated* step-s result (i) tokenizes to a clean span,
(ii) equals the gold step-s value, and (iii) is literally an operand of its generated step
s+1 (so a change *can* propagate) — 1040 units from the 1319. 200 pairs sampled with seed
0, ≤2 per recipient → 185 recipients (steps: 114 at s=1, 51 at s=2, 26 at s=3, 8 at s=4, 1
at s=5; 190 single-token spans, 10 two/three-token). Donor: a different example whose
gold-correct step-s result has the same token count and a different value. Random donor:
a random example at a random step, same token count, different value. Recipient baseline
accuracy on these 200: 56.0% (qualification selects for correct intermediate steps).
**Design:** teacher-force `prompt + generated CoT` up to and including the step-s result
span, intervene, continue greedy to EOS (max 160 tokens), extract the answer with the
shared scorer.
- *text*: replace the span's tokens with the donor's value tokens (behavioral ceiling).
- *real_L* (L ∈ {0, 3, 6, 9, 11}): forward hook on `transformer.h[L]` overwrites the
  block-L output at the span positions with the donor's block-L output at *its* span
  (captured from a teacher-forced pass over the donor's own CoT); layers >L at those
  positions are recomputed and the KV cache carries the patch into the continuation. A
  block-L patch therefore reaches later positions only through the K/V of blocks L+1..11
  at the span — L=11 affects only the next-token logits at the span itself.
- *random_L*: same, with the random donor's activations.
- *zero_L / mean_L* (L ∈ {0, 6}): zero vector / mean block-L output over all result-span
  tokens of 300 random examples (5.9 spans each on average).
- Metrics: `tracking_next` = the injected value is an operand of the first `<<…>>` step
  in the continuation (analog of the latent runs' "next read-out moved toward donor";
  None when the continuation has no step); `answer_changed` vs. the recipient's own
  baseline; `matches_cf` = answer equals the counterfactual-consistent value obtained by
  substituting the donor's value into the recipient's own downstream generated
  expressions and re-evaluating the chain (defined for 197/200; `matches_cf_gold` does
  the same on the gold rationale, 186/200); accuracy vs. gold. Wilson CIs; McNemar exact,
  paired per recipient.
**Command:**
```
uv run python scripts/patch_explicit_cot.py \
  --ckpt-dir results/20260918-021435_explicit_cot_baseline-full/ckpt \
  --checkpoint-label "local:20260918-021435_explicit_cot_baseline-full/ckpt" \
  --n-pairs 200 --layers 0,3,6,9,11 --ablate-layers 0,6 --mean-sample-n 300 \
  --device cpu --cache <scratch>/gens_full.jsonl \
  --slug patch-positive-control --stage full_run --hardware "local Apple Silicon CPU, fp32"
```
(Run locally on CPU, not on a pod: GPT-2 small in fp32 does the 1319 generations in 306 s
and the 200 × 15 intervention continuations in 544 s — ~14 min total, **$0**. Code at
commit `a079422` (tree dirty: this script and the concurrent forks' files).)
**Headline results (n=200 pairs):**

| condition | next step uses injected value | answer changed | answer = counterfactual | accuracy (baseline 56.0%) |
|---|---|---|---|---|
| baseline (own value) | 100% (by construction) | — | — | 56.0% |
| **text swap** | **88.5%** [83, 92] | **99.0%** | **44.2%** [37, 51] | 0.5% |
| **real donor, L0** | **86.4%** [81, 91] | 99.0% | 41.1% [34, 48] | 0.5% |
| real donor, L3 | 87.0% | 99.0% | 30.5% | 1.0% |
| real donor, L6 | 83.0% [77, 88] | 97.5% | 25.4% [20, 32] | 1.5% |
| real donor, L9 | 21.0% | 43.0% | 7.1% | 42.5% |
| real donor, L11 | 2.5% | 0.0% | 1.5% | 56.0% |
| random donor, L0 / L3 / L6 / L9 | 71.4% / 89.4% / 81.4% / 20.0% | 98.0% / 99.0% / 97.5% / 44.0% | 33.0% / 25.4% / 19.8% / 4.6% | 2.5% / 1.0% / 1.0% / 42.0% |
| zero, L0 / L6 | — | 76.5% / 89.0% | — | 16.5% / 6.5% |
| mean, L0 / L6 | — | 97.5% / 90.5% | — | 2.5% / 7.0% |

Every accuracy drop vs. baseline is McNemar p < 1e-5 (L0–L6 and both ablations: 100–112
right→wrong vs. 0–3 wrong→right; L9: 31 vs. 4). Restricted to the 112 recipients whose
baseline answer was correct: text swap tracks 94.6% and lands on the counterfactual
answer 62.5%; real L0 92.0% / 60.7%; L3 92.9% / 44.6%; L6 88.4% / 36.6%; L9 19.6% / 9.8%.

Real vs. random donor (paired): tracking real > random at L0 (b=41, c=12, p=1e-4), tied
at L3/L6/L9 (p ≥ 0.36); `matches_cf` real > random at L0 (p=0.048), not significant
elsewhere (p 0.14–0.30); `answer_changed` tied everywhere (p ≥ 0.69). Tracking is
insensitive to how far apart the recipient's and donor's spans sit in absolute position
(L6: 85% at Δpos ≤ 5, 88% at 6–15, 80% at ≥ 16; L0: 95% / 84% / 84%).

**Interpretation:**
- **The intervention works, decisively.** Overwriting the residual stream at one result
  span with another example's activation is enough for the next step to compute with the
  injected number 83–87% of the time at layers 0–6 — within a few points of literally
  editing the text (88.5%) — and to change the final answer 97–99% of the time, landing on
  the exact arithmetically consistent counterfactual 25–41% of the time (37–61% when the
  recipient was solving the problem correctly to begin with; the remainder is the model
  restructuring or mis-computing the rest of the chain, which it does under the text swap
  too — that is the behavioral ceiling, not an intervention failure). The same code path
  reproduces free-running generation exactly when nothing is patched (200/200).
- **This is the positive control the latent-mechanism nulls needed.** The latent runs'
  analog of `tracking_next` — the next iteration's read-out moving toward the injected
  value — is 0.4–0.9% for CODI (eval-mode `20260919-184323`: 0.44%, n=350) and at
  chance for recurrent_depth (2/130, same as baseline coincidence). Here the same kind
  of single-position residual overwrite is tracked ~85% of the time. Content-free
  ablation is the other half of the contrast: zeroing or mean-filling one result span
  drops explicit-CoT accuracy from 56% to 2.5–16.5% (the model either regenerates a
  value — zero at L6 never continues with `>>` — or treats the mean as an unspecified
  number and computes garbage), whereas the same ablation of CODI's best-decoding thought
  input left accuracy at 51–54% vs. 54% (`20260919-090132`) and changed the answer only
  15–21% of the time in eval mode (`20260919-184716`). So the null on the latent
  mechanisms is not "our patching doesn't do anything on GPT-2": the identical
  intervention on a position the model actually reads from is nearly as effective as
  editing the text.
- **The layer profile is the expected one and is itself informative.** L0–L6 patches all
  carry the value; L9 carries it a fifth of the time; L11 not at all — later positions
  read the value out of the K/V of the early-to-mid blocks at that span, so a patch at
  block L only propagates through blocks L+1..11. That the latent mechanisms' "final
  hidden state fed back as the next input embedding" position (CODI) is structurally an
  L=-1/L0-type patch — the *most* effective kind here — makes their null sharper, not
  weaker.
- **Real vs. random donor is (correctly) not the key contrast here.** Both donors inject
  a real number's representation, so both are tracked and both move the answer; the real
  donor's only advantage is being a same-step value, which shows up as a modest tracking
  edge at L0 and nothing at deeper layers. The result that matters for the control is
  *injected content is used* (tracking, matches_cf) and *removing content breaks the
  answer* (ablation) — both of which the latent mechanisms fail on the same axes.
**Gotchas hit:**
- Number regex: the first draft's operand/substitution pattern allowed a leading `-`, so
  in `16-4` the operator was swallowed into `-4` and both operand matching (tracking) and
  chain re-evaluation missed it — caught on the smoke test (a pair whose counterfactual
  should have been 56.25 came out 55), fixed to unsigned matching before this run.
  Signed step results (`<<5-8=-3>>`) are therefore not matched; rare in GSM8K-Aug.
- `scripts/decode_patch_codi.py`'s `wilson_ci` / `mcnemar_exact_p` can't be imported
  outside the CODI checkout (it imports `src.model` at module load), so they're copied
  verbatim into this script.
- A block-L output hook is the right site for "residual stream after layer L"; note it
  leaves blocks 0..L's own K/V at the span untouched, which is why L11 is inert for the
  continuation (only the span's own next-token logits change).
**Caveats:**
- Qualification selects recipients with a correct, downstream-used intermediate — a
  regime where explicit CoT is behaving well (56% baseline vs. 35% overall). That is the
  right regime for a positive control, but the tracking/counterfactual rates shouldn't be
  quoted as properties of explicit CoT on the whole test set.
- `matches_cf` is defined on the recipient's own generated chain; when the same number
  occurs as two different step results (e.g. `…=4>> … <<16*0.25=4>>`), substitution can't
  tell which occurrence a later expression references and follows the model's own
  resolution only by luck. `matches_cf_gold` (on the gold rationale) is a few points
  lower for the same reason plus generated≠gold chain shapes. The tracking metric doesn't
  have this ambiguity.
- GPT-2's absolute position embeddings ride along in a residual patch (the donor's span
  sits at a different index); the Δpos breakdown shows tracking doesn't depend on it here,
  but this would matter more for a model where positional content is larger.
- 10/200 pairs have multi-token spans (patched token-by-token in order); not analysed
  separately.
- Single seed for pair sampling; n=200. CIs above.
**Next:**
- This is the figure-ready control for the Sep 25 / Oct 16 story: one panel per mechanism
  with (tracking of injected value, answer changed, accuracy under content-free ablation)
  for explicit CoT vs. CODI vs. recurrent_depth, same intervention design.
- Optional: repeat with `--layers -1` (embedding output) to pin the text-swap equivalence
  exactly, and on a 200-slice of *all* recipients (no qualification) to report the
  unconditioned rates.
