## 2026-10-14 — Ouro-1.4B step 1: twin patching over (loop, layer) at the answer suffix (recurrent_depth, run_id: 20261014-163349_recurrent_depth_twin-patch-ouro-1.4b)

**Goal:** Step 1 of extending Q4 to a looped model. Ouro has no latent slots, so before any
CODI/Coconut <-> Ouro transplant we need a site that carries the problem's computed value.
Candidate: the residual stream at the post-question answer prompt (`\nAnswer: `, 4 tokens),
per (loop, layer). Q1-style same-problem twins: does patching the twin's suffix states into
the recipient move the recipient's answer to the twin's?
**Mechanism / model:** `recurrent_depth`, released Ouro-1.4B, 4 loops (trained count), 24 layers.
**Data:** recipients = step-0 gate items correct at 4 loops (GSM8K test), shuffled; twin = one
chain operand perturbed (DELTAS ±1..3) and the chain re-executed. 157 candidates, 150 recipients
correct under this script's tokenization, 65 responsive pairs (twin also correct; CFR 0.43),
42 "clean" (neither answer one of the model's 10 most frequent outputs). Fewer than the 200-pair
cap: only 157 of the 242 gate-correct items admit a minimal pair.
**Method:** recipient's prefix + question cached and never patched; patch the decoder-layer
output at (loop u, layer l) at all 4 suffix positions (`twin_all`), the last one only
(`twin_last`), or with another recipient's states (`cross_all`, cross-problem control). Scoring
teacher-forced: ld = logp(twin answer digits + "\n") − logp(recipient answer); `restored` = (ld −
ld_recipient) / (ld_twin_own − ld_recipient); `flip` = ld > 0. Median ld_recipient = −5.3,
ld_twin_own = +4.6. Self-patch sanity passes.
**Command:** see `eval_command.txt` (pod, transformers 4.56.2 venv, A40).

**Headline results** (n=65; `restored`, twin_all; loops 1-indexed, layers 0-indexed (0–23);
manifest `grids` and prediction keys `u_l` are 0-indexed in both):

| loop | layers 0–12 | layer 14 | layer 16 | layer 17 | layer 18 | layers 19–23 |
|---|---|---|---|---|---|---|
| 1 | -0.00 | 0.00 | 0.01 | 0.02 | 0.01 | 0.01 |
| 2 | 0.01 | 0.05 | 0.08 | 0.12 | 0.11 | 0.10–0.11 |
| 3 | 0.10–0.11 | 0.24 | 0.28 | 0.36 | 0.34 | 0.31–0.34 |
| 4 | 0.31–0.32 | 0.36 | 0.38 | 0.41 | 0.40 | 0.38–0.40 |

Flip rate tracks restored closely (max 0.45, loop 4 layer 17). `twin_last` ≈ `twin_all` everywhere: the
effect lives at the last suffix position (the " " that predicts the first digit). Exact 4×24
grids, all and clean subsets, in the manifest. Clean subset (n=42) slightly higher (max 0.45 at
loop 4, layer 17).

**The decisive split — first digit** (at loop 4, layer 18 = key `3_18`):

| subset | n | restored | flip |
|---|---|---|---|
| twin answer's first digit differs from recipient's | 34 | 0.74 | 0.74 |
| twin answer's first digit the same | 31 | 0.03 | 0.03 |
| cross donor, first digit differs | 44 | — | 0.59 |
| cross donor, first digit same | 21 | — | 0.00 |

**Interpretation:** The answer-prompt suffix carries the **next token — the answer's leading
digit — and nothing else causally**. Ouro's tokenizer emits numbers digit by digit after a
fixed " " token; every later digit is computed at its own position by re-reading the question,
so patching the suffix moves an answer only when the twin's first digit differs, and then
~75% of the way. The cross-problem control behaves identically (59% / 0%), so this is not a
problem-specific value either. Where it is computed: nothing at the suffix in loop 1; a first rise in the
upper-middle layers (~13–18) of loop 2 (to 0.12), the main jump in the same layers of loop 3
(0.10 → 0.36), a small gain in loop 4 (→ 0.41). Mirrors the gate's accuracy-by-loops curve
(6.8% → 15.2% → 17.4% → 18.3%) except that the big suffix gain comes one loop later.

**For Q4:** this site is not a scratchpad analogue of CODI's/Coconut's latents (which carried
intermediate step values the decoder then used). Transplanting mapped CODI/Coconut latents into
Ouro's suffix can at best set the leading digit. A Q4 into-Ouro transplant as designed would be
a null by construction — recorded here rather than run. Ouro → CODI/Coconut (Ouro suffix states
as map *source*, like the plain-GPT-2 control) remains possible but would be mapping a
next-digit state, not an intermediate value.

**Gotchas hit:**
- Greedy decoding under a suffix patch never changes (first version of this script: 0 changes
  at all 96 sites): the first generated token is always " ". Fixed by putting the space in the
  suffix and teacher-forced scoring.
- Ouro's `UniversalTransformerCache` breaks transformers' `create_causal_mask` with a prefilled
  cache (mask sized to the new tokens only). The script builds the 4-D mask, position ids and
  cache positions itself. Batched rows share the cached prefix via expanded views (no padding, so
  the left-padding bug from step 0 does not apply).

**Caveats:** n=65 (34 informative first-digit-differs pairs); one prompt; teacher-forced
log-probs rather than generated answers; sites only at the suffix — the question-token positions
(where the twin's number enters) are not tested, since patching them is input patching. The
intermediate value may well be computed at question or digit positions; this run shows only that
it is not handed to the answer through the suffix.

**Next:** if Q4 with Ouro is still wanted, the options are (a) a linear probe for the
intermediate step value at suffix and late-question positions per (loop, layer) — cheap, says
whether the value is *present* even though the suffix does not causally carry it; (b) patch at
the answer-digit positions (teacher-forced twin digits as context would leak the answer, so this
needs care); (c) drop the into-Ouro direction and report Ouro as the vertical mechanism that does
not externalise its intermediate state to a post-question site.
