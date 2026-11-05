## 2026-11-04 — Stabilised L1-7w CODI barely uses its latents: all six mean-ablated costs 0.018 (0.689 -> 0.671), against 0.42 for the lr-3e-3 CODI (codi, run_id: 20261105-052233_codi_v2-l1to7w-stab-depth2-codi-latent-ablation)

**Goal:** Protocol 10.3: rerun the ablation on the selected stabilised CODI, so "uses its latents" and "gains from its latents" describe the same model.
**Mechanism / model:** `codi`, checkpoint-17820 of 20261105-044444, CPU fp32.
**Data:** canonical L1-L7 test (3,500); means from valid.json.
**Command:** `scripts/ablate_latents_rulechain_codi.py` (`eval_command.txt`).
**Headline results:** Clean 0.689 (L3 0.852, L4 0.728, L5 0.602, L6 0.406, L7 0.238). Any single site, mean or resample: 0.685-0.690. Tail >= 1: 0.673; >= 2: 0.679; >= 3: 0.685. All six: 0.671 (L4 0.708, L5 0.556, L6 0.362, L7 0.218). Centered cosine z_1..z_5 0.94-0.98.
**Interpretation:** This CODI solves the task almost entirely without its latents, which explains the pair's null (20261105-044444). It weakens that null: it shows that a CODI which ignores its latents equals CODI-0. The lr-3e-3 CODI (20261104-022824) depended on its latents (0.744 -> 0.315) and scored higher on test (0.743), but has no matched reference, so the flag change and its effect on latent use cannot be separated here.
**Gotchas hit:** none.
**Caveats:** One checkpoint, one seed. All-ablated keeps the six positions with mean inputs, so they may still add a little generic compute.
**Next:** A CODI claim needs a matched pair where CODI actually uses its latents; low priority behind the SIM-CoT controls.
