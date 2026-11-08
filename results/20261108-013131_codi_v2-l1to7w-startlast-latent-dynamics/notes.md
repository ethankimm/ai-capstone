## 2026-11-08 — Latent dynamics: plain CODI reaches a fixed point after two passes (relative step .01-.03 from t = 2, consecutive latents centred cos .98-1.00), while gated and SIM take large near-orthogonal steps through the pass the answer reads, then stop (gated at once, SIM over 2-3 passes) (codi, run_id: 20261108-013131_codi_v2-l1to7w-startlast-latent-dynamics)

**Goal:** HANDOFF_2026-11-07 step 3. Show plain CODI's attractor (2607.09698 calls CODI "a stable attractor" from geometry alone) next to the gated and SIM staircases, in geometric terms, for the three models whose latent use is known from swaps (`20261108-012524`).
**Mechanism / model:** plain start-last CODI checkpoint-17820 (`20261106-083843`), rule-gated with the gate on (`20261107-204814`), SIM-CODI checkpoint-16335 without its decoder (`20261107-225603`).
**Data:** `synthetic_data_v2_l1to7w_startlast` **valid** L3-L6 (286/286/286/285) per model, the items of step 1. Clean accuracy here equals step 1's answers item by item (checked in the collector): plain .52 / .33 / .14 / .12, gated 1.00, SIM 1.00 / 1.00 / 1.00 / .996.
**Command:** On the laptop (MPS, float32, `run_guarded.sh`, `caffeinate`), in `~/Documents/Penn/CIS5980/codi_audit_artifacts`: `t3_dynamics.py` (collector, CODI venv; per-item metrics in `t3_dynamics.json`), `t3_plot.py` (figure, repo env: `t3_dynamics_z.{png,pdf}`, `t3_dynamics_h.{png,pdf}`; medians in `t3_medians_{z,h}.md`), `record11.py`. Smoke: plain L4, 40 items, answers match t1.
**Metrics:** two trajectories per item, t = 0..6: z_t = prj(h_t), the latent inputs (z_0 from `<bot>`; z_6 is computed but never read), and h_t, the last-layer states before prj. D_t = x_{t+1} - x_t, the output of pass t minus its input. Per item: step size ||D_t||; relative step ||D_t|| / ||x_t||; turning cos(D_t, D_{t-1}); centred consecutive cosine cos(x_t - mu, x_{t+1} - mu), with mu the mean over the level's items and positions 0..5; sensitivity surrogate log(||D_t|| / ||D_{t-1}||). These are the quantities the handoff lists for 2607.09698's Eqs. 1-3; I did not check them against the paper's text.

**Headline results** (z trajectory, medians over items; relative step at t = 0..5):

| L | plain | gated | SIM |
|---|---|---|---|
| 3 | .70 .11 .03 .01 .01 .01 | .82 .87 **1.07** .09 .01 .01 | 1.32 1.30 **1.01** .49 .10 .02 |
| 4 | .63 .09 .02 .01 .01 .01 | .85 .88 .87 **1.07** .09 .01 | 1.34 1.32 1.28 **.93** .51 .14 |
| 5 | .57 .10 .02 .01 .01 .01 | .85 .87 .85 .89 **1.06** .10 | 1.33 1.32 1.32 1.27 **.88** .49 |
| 6 | .54 .09 .02 .01 .01 .01 | .82 .85 .88 .85 .85 **1.05** | 1.27 1.33 1.26 1.31 1.23 **.68** |

(bold: t = L-1, the output of the pass whose input z_{L-1} the answer uses.)
- **Plain: a fixed point after two passes, at every level.** The relative step falls from .54-.70 to about .10 at t = 1 and .02-.03 at t = 2, then .01. Consecutive latents have centred cos .98-.99 at t = 1 and .999-1.000 from t = 2. Turning goes to .9-1.0 as the steps shrink along one direction. Accuracy plateaus after 2 passes too (`20261106-101740`).
- **Gated: equal large steps while rules are read, one larger step out of the used pass, then stop.** Relative step .82-.89 for t <= L-2, 1.05-1.07 at t = L-1, then .09-.10 and .01. During the rollout consecutive latents are near orthogonal (centred cos -.04 to .36) and turning sits at -.5. That is the value expected for steps between roughly independent states of equal norm. The idle latents after the used pass are near copies (centred cos .98-1.00): the probe-readable but unused cells of step 1.
- **SIM: the same staircase with larger steps and a slower stop.** Relative step 1.23-1.34 for t <= L-2, .68-1.01 at t = L-1, then decaying over 2-3 passes (L3: .49, .10, .02). Centred cos -.10 to .17 during the rollout, rising to .56-.99 after it.
- **h (last-layer states before prj):** the same picture. Plain's step falls over t = 0..2 (relative .84-.87, .37-.50, .08-.10), gated and SIM stay at 1.1-1.5 until t = L-1, then drop. The sensitivity surrogate dips for plain at t = 2 (log ratio -1.4 to -1.8) and stays near 0 for gated and SIM during the rollout.

**Predictions (`t_predictions.md`, written before any output):**
- *Plain contracts to near-zero step size after pass 2:* right.
- *Gated and SIM keep large steps through the last rule-reading pass (t = L-2), then drop:* off by one. Large steps continue through t = L-1, the output of the pass the answer reads. Gated drops at once after it; SIM decays over 2-3 passes.
- *Turning low during the rollout, high for plain after the fixed point:* right (-.5 vs .9-1.0).

The figure's reference line marks t = L-1. It was moved there after seeing the data; the pre-registered reference was t = L-2. Said in the plot script's docstring.
**Interpretation:** The geometry separates the two regimes as the swaps do. Plain CODI's latents converge after two passes to a point and stay there, which is 2607.09698's attractor description. Step 1 shows what that means causally: the six latents become near copies that no single swap moves, and only the set matters. Gated and SIM move far each pass, to nearly unrelated states, for exactly as many passes as there are rules to read plus the pass the answer reads. Each of those states is used (step 1). After that both settle into a fixed point too, but a different one: idle latents, readable and unused. So "a fixed point" in the latent geometry does not mean the latents are unused or used. Plain's fixed point is used as a set, gated's and SIM's idle fixed points are not used at all, and the geometry alone does not tell them apart.
**Gotchas hit:** Calibration against the earlier collapse statistic (`20261106-101740`, plain, test: consecutive centred cos .58, .97, .99, .996): ours from t = 1 matches (.98, .999, 1.000). At t = 0 ours is .07-.29 against .58; the earlier centring set (levels, positions, split) is not the same as here, and I did not reconcile the two.
**Caveats:** One seed per model; valid only. Plain's medians include its many wrong items (clean .12-.52); restricting to right items was not done. The step size depends on the norm, which differs by model (SIM's latents move about 1.5x more in relative terms), so compare the shapes across models, not the levels. Descriptive geometry; the causal claims come from step 1.
**Next:** Optional handoff step 4 (does the staircase form before, with, or after the epoch 5-7 accuracy jump?) and step 5's second check (L7 with 7+ passes). Figures for the draft: `codi_audit_artifacts/t3_dynamics_z.pdf` (main), `t3_dynamics_h.pdf` (appendix).
