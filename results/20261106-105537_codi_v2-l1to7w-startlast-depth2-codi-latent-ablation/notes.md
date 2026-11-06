## 2026-11-06 — Start-last CODI per-site ablation on test: all latents mean-ablated 0.047 against 0.447 clean, any single latent costs at most 0.14 overall, the curve is flat from three latents on (codi, run_id: 20261106-105537_codi_v2-l1to7w-startlast-depth2-codi-latent-ablation)

**Goal:** TRAINING_PROTOCOL section 11.3 reliance, early-answering and filler analogues on the selected CODI.
**Mechanism / model:** `codi`, gpt2, 6 latents, checkpoint-17820 of 20261106-083843, fp32, CPU.
**Data:** Start-last test, n=3500, 500 per level. Means from start-last valid.
**Command:** `scripts/ablate_latents_rulechain_codi.py`; `eval_command.txt`.
**Headline results:** Exact by level L1-L7. Clean 1.00 / 1.00 / .48 / .31 / .16 / .12 / .08. All sites mean-ablated, positions kept: .09 / .05 / .04 / .04 / .04 / .03 / .03. Tail ablated from s, overall: s=1 .147, s=2 .369, s=3 .425, s=4 .441, s=5 .445, clean .447. L2 by the same tail: .05, .86, 1.00. L3: .04, .31, .41, .44, .47. Single sites: `mean_0` removes L1 (.01) and leaves L2 and above unchanged; `mean_1` to `mean_5` cost .00-.06 at L3. Centered cosine between z_1..z_5 .97-.996; z_0 against the rest .57.
**Interpretation:** *Filler analogue:* with latent content removed and positions kept the model is at the floor, so positions alone do nothing, and the content is used. *Reliance without necessity at low levels:* L2 falls from 1.00 to .05 with latents ablated while CODI-0 scores .98 with none. *Early answering:* the curve is flat from three latents on, matching the true-truncation sweep in 20261106-101740. Single-site ablation stays near-free because later latents are near copies, so per-site `N_j(k)` alone would understate reliance.
**Gotchas hit:** None.
**Caveats:** One seed. Mean ablation is off-distribution; resampled latents, which are real ones, cost more per site (L3 .35-.38) and are shown in the manifest.
**Next:** See 20261106-101740.
