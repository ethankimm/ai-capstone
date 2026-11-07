## 2026-11-07 — Round-1 plain Coconut ignores its latents: all six mean-ablated .158 against .155 intact on L2-4, another item's real latents leave 95-98% of answers unchanged, answers follow the donor .019 (coconut, run_id: 20261107-092430_coconut_v2-depth2-coconut-c1m6-latent-ablation)

**Goal:** First necessity test on any plain Coconut model on RuleChain. Does its poor accuracy come with latents it uses badly, or latents it does not use? Protocol section 3's primary test (resample swap within level) had never been run on this stack.
**Mechanism / model:** `coconut`, GPT-2 small, `gpt2_depth2_coconut_c1m6/checkpoint_10` (test record 20261102-022150), 6 latents, CPU fp32.
**Data:** Canonical depth2 test, first 200 items per level, L1-L4 (n=800). Mean latents from 500 train items, one mean per pass, pooled over levels.
**Command:** `scripts/ablate_synthetic_coconut.py --levels 1-4 --window 2-4 --per_level_n 200 --mean_n 500 --device cpu`; full argv in the manifest.
**Headline results:**
- Pooled L2-4 (n=600), exact: intact .155; mean-ablated .158 (p .63); swap same level .152 (p .63), follows donor .019, keeps gold .153; swap other level .157; truncated to 2 latents .155, to 1 .143, to 0 .137 (p .11). Zero .108 and noise .127 cost more.
- Same answer as intact, L2 / L3 / L4: mean-ablated .96 / .96 / .975; swap same level .98 / .95 / .975; truncated to 2 latents .88 / .90 / .95; truncated to 0 .585 / .755 / .905.
- L1: intact .99, mean-ablated .99, swap .99 with every answer unchanged. Zero .425, noise .90.
- Within 2 of gold, intact vs mean-ablated: L2 .635 / .640, L3 .415 / .395, L4 .265 / .255.

**Interpretation:**
- Plain Coconut does not use latent content at any level tested. This is the opposite of the lr-3e-3 and round-1 CODI models, which follow donors .91-1.00 at L2-3 on edits before the last rule.
- The rough value (within 2 far above chance, about .08) survives ablation and donor swaps, so it is computed outside the latents, in the ordinary pass.
- Latent positions matter a little and their content does not: removing positions moves answers (.585 same at L2) with no significant accuracy cost, while replacing content with the mean or a donor's does almost nothing.
- Zero and noise costs are off-distribution damage, not evidence of use: a real donor latent, and the mean, do nothing.
- So "Coconut null" is not a failed chain. There is no latent route at all, even though the single pass fails from L2 on, so a bypass cannot explain it. Why stage-1 training leaves the latents idle is inferred, not tested here: one rule fits in a single pass, so nothing forces latent 1 to hold v_1.

**Gotchas hit:** `scripts/ablate_synthetic_coconut.py` and `latentreasoning/tasks/twins_v2.py` were untracked in the working tree at run time, so this record's script is not yet in git. The mean is pooled over levels, not per depth as the paper draft specifies; the same applies to the CODI ablations.
**Caveats:** One checkpoint, one seed, final checkpoint only, so stage-1 behaviour is not observed. 200 items per level, and correct items are few at L3-4 (30, 12). L1-4 only. Swap "keeps gold" is capped by intact accuracy, so answer stability is the more useful measure.
**Next:** The same battery on the SIM-Coconut checkpoint (cci-p124), which is the open test of whether step supervision routes the work through the latents. If stage checkpoints survive, run it on the stage-1 checkpoint too.
