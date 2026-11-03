## 2026-11-03 — RuleChain levels 4-7 CODI layerwise probes: rule 1's output is only 0.50 exact (round-1 CODI-0: 0.99) and exact values are gone from rule 2 on (0.10); branch bits stay decodable (1.00 → 0.79 over rules 1-4) and beat the matched CODI-0 at every rule (codi, run_id: 20261103-082542_codi_v2-l4to7-depth2-codi-branch-probes)

**Goal:** Where does the levels 4-7 CODI model (20261103-074958, test 0.071) fail inside the forward pass, and is its +0.028 edge over the matched CODI-0 (20261103-074957) visible in the internal branch/value quantities? Same probes as the round-1 CODI-0 record (20261103-063912), on both levels 4-7 checkpoints (CODI-0 counterpart: 20261103-082746).
**Mechanism / model:** `codi`, GPT-2 small + LoRA r128, 6 latents, `codi_runs_l4to7/gpt2_l4to7_codi/checkpoint-6252` (epoch 4, the checkpoint behind 20261103-074958; local copy at `../codi_runs_l4to7/`).
**Data:** RuleChain (depth2, generator 2.0.0) levels 4-7 test = the canonical L4-L7 test items (2000, 500/level; hash `configs/synthetic/v2_l4to7_data_sha256.json`). Probe split 70/30 by sha256(item id); probes for rule j use items with level ≥ j, so j = 1-4 use all items (n_test 579) and j = 5/6/7 use 437/293/139.
**Hyperparams:** Unchanged from 20261103-063912: one forward per item as eval (question + `<bot>` + 6 latents + `<eot>` "The answer is:"), hidden states at `rule_end` (the "." closing rule j) and `ans` (the ":"), linear softmax probes per (position, j, layer, target), L-BFGS 150 iterations, lambda from {1e-4, 1e-3, 1e-2} on 20% of probe-train. Branch scored as balanced accuracy, values as exact two-digit match. `final_answer_accuracy` in the manifest is the teacher-forced answer-token accuracy from this forward (0.0685; generation gave 0.071).
**Command:** `bash <scratchpad>/run_probes.sh` = `scripts/probe_rulechain_branch_codi.py` from the CODI checkout with `HASH_FILE=configs/synthetic/v2_l4to7_data_sha256.json`, `--max_level 7 --num_latent 6 --inf_latent_iterations 6` and the model flags of the round-1 probe run; full flags in eval_command.txt (gitignored). MPS, 46 s forward.
**Headline results:** Best layer per target at `rule_end` (probe-test split), next to the matched CODI-0 (20261103-082746) and the round-1 CODI-0 (20261103-063912, levels 1-12 training, 1-8 test items):

| rule j | branch b_j: CODI / CODI-0 / round-1 | surface | input s_{j-1} exact: CODI / CODI-0 / round-1 | output s_j exact: CODI / CODI-0 / round-1 |
|---|---|---|---|---|
| 1 | **1.00** / 0.97 / 1.00 | 0.71 | 0.96 / 0.88 / 1.00 (= x0) | **0.50** / 0.44 / 0.99 |
| 2 | **0.94** / 0.87 / 0.96 | 0.65 | 0.19 / 0.12 / 0.89 | 0.10 / 0.07 / 0.51 |
| 3 | **0.88** / 0.77 / 0.89 | 0.64 | 0.09 / 0.05 / 0.83 | 0.08 / 0.03 / 0.34 |
| 4 | **0.79** / 0.70 / 0.87 | 0.60 | 0.05 / 0.05 / 0.58 | 0.04 / 0.04 / 0.27 |
| 5 | 0.77 / 0.71 / 0.80 | 0.61 | ≈ surface | ≈ surface |
| 6 | 0.73 / 0.71 / 0.77 | 0.56 | ≈ surface | ≈ surface |
| 7 | 0.67 / 0.68 / 0.74 | 0.61 | ≈ surface | ≈ surface |

Value majority/surface baselines are 0.02-0.04. At `ans` nothing beyond surface except x0 (j = 1 input: CODI 0.57, CODI-0 0.34), as in round 1.

Related error analysis on the 20261103-074958 / -074957 predictions (no model, scratchpad script): wrong answers that equal "gold with exactly one rule skipped" are 1.5-3.5× chance (predictions shuffled within level) when the skipped rule is early or middle, and at or below chance when it is the last rule, in both models (e.g. CODI L6: rules 1-4 14-18 vs chance ~6; rule 6 8 vs 6.6). Prefix intermediates (stopped early) are at chance.

**Interpretation:**
- **These models never learned exact arithmetic even for rule 1.** s_1 is only 0.50 exact at its own "." (round 1: 0.99 from layer 10). In round 1, level-1 items make s_1 the supervised answer; with levels 4-7 only, no target ever names an intermediate value, and the exact value never forms. This is concrete support for the "easy levels are a scaffold" reading of 20261103-074958, more direct than the step-count argument (which was confounded).
- **Branch decisions still form, from a coarse value.** b_2 at 0.94 while s_1 is 0.50 exact (and s_1 at rule 2's "." only 0.19): the same exact-vs-coarse dissociation as round 1, but here the exact value is missing from the start rather than decaying after rule 2.
- **CODI's edge over CODI-0 is visible internally, at question tokens.** CODI is ahead on every branch bit for j = 1-4 (+0.03 to +0.11) and slightly on s_1 (0.50 vs 0.44). These are read at the rule's own "." in the question, before any latent exists (causal attention), so the difference is in how the shared weights process the question, not in what the latents carry. Training with latents changed the question encoding; the probes don't show the latents doing the computation.
- **Fits the skipped-rule errors.** The last rule is never the dropped one, and the running value is lost after rule 1-2; the answer is mostly "last rule applied to an approximate value".

**Gotchas hit:** First launch passed the shared flags as one zsh string and the parser rejected them (no output); reran from a bash script with an array. A parallel session committed 3eee111 during the run, so manifest git = 3eee111 (dirty); the probe script itself is unchanged since bc915de.
**Caveats:**
- One checkpoint and seed per arm; linear probes; best-over-13-layers is selected on the test split for all arms equally (the comparison is fair, the absolute numbers are slightly optimistic). n_test 579 for j ≤ 4 (binomial SE ≈ 0.01-0.02 on branch), 139 at j = 7.
- Round-1 numbers come from a different item pool (levels 1-8), so per-j comparisons mix models and item sets; the levels 4-7 CODI vs CODI-0 comparison is item-matched.
- Tens-digit (coarse value) accuracy is still not saved, so "coarse value" is inferred, not measured.
- Latent positions themselves are not probed.
**Next:** Probe the 6 latent positions directly (does any latent carry s_j or b_j beyond surface?); save tens-digit accuracy; repeat these probes on the levels 1-7 up-weighted run (3eee111) to test whether easy levels restore exact s_1.
