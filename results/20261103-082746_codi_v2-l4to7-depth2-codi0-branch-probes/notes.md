## 2026-11-03 — RuleChain levels 4-7 CODI-0 layerwise probes (matched reference): rule 1's output 0.44 exact, exact values gone from rule 2 on; branch bits below CODI at every rule 1-4 (codi, run_id: 20261103-082746_codi_v2-l4to7-depth2-codi0-branch-probes)

**Goal:** Matched no-latent reference for the levels 4-7 CODI probes (20261103-082542): same probes, same items, on the CODI-0 checkpoint behind 20261103-074957 (test 0.043).
**Mechanism / model:** `codi`, GPT-2 small + LoRA r128, 0 latents, `codi_runs_l4to7/gpt2_l4to7_codi0/checkpoint-6252` (epoch 4; local copy at `../codi_runs_l4to7/`).
**Data:** RuleChain (depth2) levels 4-7 test, 2000 items (500/level), canonical L4-L7 items; probe split 70/30 by sha256(item id), identical to the CODI run.
**Hyperparams:** As 20261103-082542 with `--num_latent 0 --inf_latent_iterations 0`. Manifest `final_answer_accuracy` is teacher-forced answer-token accuracy (0.0425; generation 0.043).
**Command:** Second half of the same bash driver as 20261103-082542; flags in eval_command.txt (gitignored). MPS, 36 s forward.
**Headline results:** `rule_end`, best layer: branch b_1..b_7 = 0.97 / 0.87 / 0.77 / 0.70 / 0.71 / 0.71 / 0.68 (surface 0.71 / 0.65 / 0.64 / 0.60 / 0.61 / 0.56 / 0.61). Input s_{j-1} exact: 0.88 (x0) / 0.12 / 0.05 / 0.05, then surface. Output s_j exact: 0.44 / 0.07 / 0.03 / 0.04, then surface. At `ans`, only x0 is above surface (0.34). Full side-by-side table with CODI and round 1 in 20261103-082542.
**Interpretation:** Same picture as CODI, one notch weaker everywhere: no exact rule-1 value (round-1 CODI-0 had 0.99), branch bits decodable from a coarse value, exact running value gone by rule 2. Branch bits at j = 3-4 are only 0.10-0.13 above the surface baseline, so beyond rule 2 this model barely executes the chain. See the CODI record for the reading of the CODI vs CODI-0 difference.
**Gotchas hit:** See 20261103-082542 (zsh flag splitting on first launch; manifest git 3eee111 dirty from a parallel session).
**Caveats:** One checkpoint/seed; linear probes; layer selection on the test split; tens-digit accuracy not saved.
**Next:** See 20261103-082542.
