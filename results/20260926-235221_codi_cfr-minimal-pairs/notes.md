## 2026-09-26 — Counterfactual responsiveness on E3 minimal pairs, CODI: the unpatched model follows a one-number change only 53% of the time (codi, run_id: 20260926-235221_codi_cfr-minimal-pairs)

**Goal:** E3 (`20260920-190420_codi_minimal-pair-patch`) reports all-slot steering of 70.1%, but only
on pairs where the model solves BOTH the original and the perturbed twin. What is the behavioral
reference: how often does the unpatched model's own answer move correctly when one question number
changes? (RESEARCH_PLAN round 1, item 4.)
**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3`, compute_steps=6. No model run.
**Data:** the E3 slice, gsm8k-aug test n=600 seed=0; 263 base-correct; 258 candidate twins decoded.
**Command:** `uv run python scripts/cfr_minimal_pairs.py` (offline; writes this record + the Coconut one).
**Definition:** CFR = P(twin answered correctly | original answered correctly), greedy, over every
candidate twin E3 decoded = `n_qualified_pairs / n_candidates_checked` from E3's manifest (exact).
**Headline results:**

| quantity | value |
|---|---|
| P(original correct), slice | 0.438 |
| **CFR = P(twin correct \| original correct)** | **137/258 = 0.531** [0.470, 0.591] |
| E3 all-slot matches_twin (on the 137 responsive pairs) | 0.701 |
| all-slot × CFR (of all 258 checked originals-correct candidates) | 0.372 |

**Interpretation:** E3's "70% steering" is conditioned on a behaviorally responsive half of the pool.
The model itself only tracks the perturbation in about half of the problems it gets right, so the
patched latents reproduce the model's input-driven answer change in 70% of the cases where that
change exists at all (≈37% of all candidates). Quote the steering number together with CFR, and do
not read 70% as "70% of the time latents carry the value".
**Gotchas hit:** The stratified breakdown (by step, delta, chain length) needs the rejected
candidates, which E3 did not save. Replaying E3's candidate construction needs E3's exact
base-correct set; no saved CODI run reproduces it (E2 has 261 vs 263 correct, the eval-mode full-test
run has 263 but a different set — bf16 greedy flips across pods, and one flip early in the slice
shifts every later `rng.choice`). A small flip search got to 128/137 and was abandoned. So CODI gets
the aggregate only; the E3 rerun (`patch_minimal_pair_codi.py`, 2026-09-26) logs every candidate twin
(`candidates.jsonl`), which gives the stratified CFR directly at n≈1319.
**Caveats:** CFR is over propagation-qualified steps with deltas ±1..±3 only; it says nothing about
larger perturbations.
**Next:** stratified CFR from the E3 rerun's `candidates.jsonl`; Coconut counterpart
`20260926-235221_coconut_cfr-minimal-pairs`.
