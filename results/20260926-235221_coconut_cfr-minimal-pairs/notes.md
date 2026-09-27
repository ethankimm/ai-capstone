## 2026-09-26 — Counterfactual responsiveness on E3 minimal pairs, Coconut: 52%, and it falls with chain length (coconut, run_id: 20260926-235221_coconut_cfr-minimal-pairs)

**Goal:** Behavioral reference for Coconut's E3 steering (77.1%, `20260920-195725_coconut_minimal-pair-patch`):
how often does the unpatched model answer the perturbed twin correctly when it answers the original
correctly? (RESEARCH_PLAN round 1, item 4.)
**Mechanism / model:** `coconut`, gpt2 / `hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33`, 6 latents. No model run.
**Data:** E3 slice, gold-trace test file n=600 seed=0; 213 base-correct; 201 candidate twins decoded.
**Command:** `uv run python scripts/cfr_minimal_pairs.py`.
**Definition:** as in the CODI record: CFR = P(twin correct | original correct), greedy.
Stratified by replaying E3's candidate construction on E2's base correctness
(`20260920-085317_coconut_qualified-patch`, same slice); the replay reproduces E3's logged counts
exactly (201 candidates, 105 qualified), so `predictions.jsonl` holds every checked candidate with its
label.
**Headline results:**

| quantity | value |
|---|---|
| P(original correct), slice | 0.355 |
| **CFR** | **105/201 = 0.522** [0.454, 0.590] |
| E3 all-slot matches_twin (on responsive pairs) | 0.771 |
| all-slot × CFR | 0.403 |

| stratum | CFR |
|---|---|
| perturbed step 0 / 1 / 2 / 3+ | 69/118=0.58, 28/59=0.47, 7/19=0.37, 1/5=0.20 |
| chain length 2 / 3 / 4 / 5+ | 55/83=0.66, 37/73=0.51, 7/30=0.23, 6/15=0.40 |
| \|delta\| 1 / 2 / 3 | 0.51, 0.62, 0.43 |
| delta sign + / − | 99/185=0.54, 6/16=0.38 |

**Interpretation:** Same picture as CODI (0.531): the model follows a one-number change in about
half the problems it solves. Responsiveness drops with chain length and perturbed-step depth, so E3's
qualified pairs over-represent short chains / early steps: the 77% steering number describes that
easier subset.
**Gotchas hit:** none beyond the replay (verified exact here, unlike CODI).
**Caveats:** negative deltas are rare (16/201) because `generate_minimal_pair` requires non-negative
integer chains. Strata beyond step 2 / chain length 4 are small.
**Next:** full-n stratified CFR from the E3 rerun's `candidates.jsonl`.
