## 2026-11-08 — RuleChain monitor test: decoders at on-route latents are faithful (gated / SIM: after a swap the decoded value predicts the actual answer .86-1.00), equally accurate decoders at off-route spare latents are decorative (.00-.03); start-last and relay decoders never track a swap (<= .10) (codi, run_id: 20261108-102225_codi_v2-l1to7w-rulechain-monitor-faithfulness)

**Goal:** Monitor steps 2-3 (doc "A Necessity Check for Latent Reasoning Monitors"): decode at the causal location and test the decoded value against own errors and swap outcomes. Main claim: on-route decoders faithful, off-route accurate decoders decorative. Pre-registered (t19, 4ec85a8).
**Mechanism / model:** gated (20261107-204814), SIM (20261107-225603), start-last (20261106-083843), relay = lr-3e-3 CODI (20261103-193425).
**Data:** decoders trained on 1200 train questions per level (L2-L6); tested on valid L3-L5 with t15's items and both_j donors (20261108-073431). On-route (t15 input-only follow >= .5): gated L3 {1}, L4 {0,1,2}, L5 {0..3}; SIM L3 {0,1}, L4 {0,1,2}, L5 {0..3}; start-last, relay none.
**Hyperparams / command:** `rc_monitor.py <model> mps <out> 1200`; linear 100-way decoders per site from the input z_j or the entries (keys + values, layers 3/6/9/12); monitor answer = decoded value pushed through the remaining rules; raw `codi_audit_artifacts/t19_<model>.json`.
**Results (swap at site j, decoder at the same site; "monitor = actual" = monitor answer equals the model's actual answer after the swap):**

| model | clean decode, rule sites | monitor = actual, on-route + answer-read site | spare sites (j >= L): clean decode / monitor = actual (input decoder) |
|---|---|---|---|
| gated | .92-1.00 | .99-1.00 (start value z0 .49-.61) | .18-.98 / .00-.03 |
| SIM | .84-.99 | .86-.99 (z0 .60-.71) | .61-.96 / .00-.01 |
| start-last | .05-.49 | — (no on-route site): <= .10 at every site | — |
| relay | .18-1.00 (L3 .89-1.00) | — : <= .04 (input), <= .19 (entries, z0) | — |

Own errors (monitor answer = the model's wrong answer): gated / SIM have no wrong items (clean 1.000); start-last .04-.31; relay .04-.50 (L4-L5 .36-.50 at sites 1-5).
**Interpretation:** The main claim holds on RuleChain. Gated / SIM latents on the route give faithful monitors; their spare latents read the value just as well (up to .98) but a monitor there predicts the donor's answer while the model keeps its own: an accurate but decorative decoder. Entries decoders at spare sites track the actual answer better (.57-.84) because the swapped input barely changes those entries. Start-last and relay: no single latent's decoder tracks a swap. Prediction miss: relay decoders predict a sizeable share of the relay's own errors (.36-.50 > .3): its pooled copies all hold the value the answer uses, so a decoder correlates with errors without any single latent being causal. Correlation with own errors is therefore not enough to certify a monitor; the swap test separates them.
**Caveats:** one seed per model; decoders linear; gated / SIM own-error test empty.
**Next:** decode GSM8K CODI / Coconut value slots (z1 / z4 inputs and entries) once the check definition is settled.
