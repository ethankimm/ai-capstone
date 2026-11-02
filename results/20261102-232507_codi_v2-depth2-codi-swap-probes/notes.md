## 2026-11-02 — RuleChain CODI latent swap + probes: latent-1 input irrelevant, latents 2-6 essential (0.40 -> 0.11) but not transplantable (follows donor 0.04); intermediate values weakly and flatly decodable (codi, run_id: 20261102-232507_codi_v2-depth2-codi-swap-probes)

**Goal:** Follow-up to `20261102-225018_codi_v2-depth2-codi-latent-diagnostics` (CODI needs 1-2 latents, matches CODI-0). (1) The protocol's primary necessity test, in-distribution: resample swap of latents between items of the same level, plus mean/zero ablation, split by which latents are swapped. (2) Linear probes for the intermediate values v1..v6 and the answer at <bot>, each latent pass and the answer position: do latents hold the chain's values, one per rule? CODI-0 counterpart: `20261102-232815_codi_v2-depth2-codi0-swap-probes`.
**Mechanism / model:** `codi`, `gpt2_depth2_codi_bs64/checkpoint-31280` (epoch 40, behind round-1 run 20261102-000204), 6 latents, pad_invariant.
**Data:** RuleChain (depth2) test, all 6000 items; probes trained on 500 seeded-random train items per level (same sample as the diagnostics run).
**Hyperparams:** see manifest. z_1 = prj(h_<bot>), z_{j+1} = prj(h of latent pass j); latent pass j consumes z_j. Donor = random test item of the same level with a different answer (seed 5980). swap_all: all z_j from the donor, fixed. swap_z1: donor z_1, then the model's own passes. swap_rest: own z_1, donor z_2..z_6 fixed. mean_all: train-sample mean z_j. Readout: teacher-forced answer token at ":" (equals generation within +-0.004/level on these checkpoints). Probes: 100-class linear softmax on standardised features of one position and layer (0, 2, ..., 12), Adam 300 full-batch steps, weight decay 1e-3; target v_j only on items with >= j rules.
**Command:** `scripts/probe_swap_codi_latents.py` from `../codi` (CODI venv), see `eval_command.txt`; MPS, batch 32.
**Headline results:** clean teacher-forced accuracy 0.402. `intervention_accuracy` (swap_all, follows donor) = 0.045.

| condition | keeps own | follows donor | L2 own/donor | L3 own/donor | L4 own/donor |
|---|---|---|---|---|---|
| clean | 0.402 | - | 1.00 | 0.99 | 0.76 |
| swap_z1 | **0.399** | 0.008 | 1.00 / 0.00 | 0.99 / 0.00 | 0.76 / 0.00 |
| swap_rest (z2-z6) | **0.113** | 0.042 | 0.06 / 0.13 | 0.01 / 0.12 | 0.04 / 0.07 |
| swap_all | 0.105 | 0.045 | 0.03 / 0.13 | 0.01 / 0.12 | 0.03 / 0.09 |
| mean_all | 0.181 | - | 0.62 | 0.17 | 0.08 |
| zero_all | 0.198 | - | 0.84 | 0.27 | 0.09 |

Probes, best test accuracy over layers (majority baseline): v1 (0.015): <bot> 0.167, z1-z6 0.152-0.156, ans 0.102. v2 (0.023): <bot> 0.096, z1-z6 0.113-0.117, ans 0.100. v3 (0.023): 0.049 / 0.080-0.087 / 0.083. v4-v6: <= 0.06. answer (0.020): <bot> 0.058, z1-z6 0.10, **ans 0.384** (model accuracy 0.402). CODI-0 (same probes, its run): v1 at <bot> **0.395**, v2 0.184, v3 0.104; answer at ans 0.375.

**Interpretation:**
- **What CODI uses is the computation of the later latent passes, not the latent-1 input.** Swapping z_1 changes nothing: latent pass 1 recomputes from the question. Replacing z_2..z_6 destroys accuracy down to 0.11 (below CODI-0's 0.41, and below CODI with no latents at all, 0.22). Mean and zero ablation also fall below CODI-0. By the protocol's rule this is dependence: the model routes its answer through the latent positions.
- **The latents are not transplantable.** Under a swap only 4% of answers become the donor's (12-13% at L2-L3), most become a third value. The donor's latents only make sense with the donor's question in context; what flows through them is not a self-contained "current value".
- **The latents do not hold a readable chain of values.** v_j is about as (in)decodable from z_1 as from z_6 (v2: 0.113 vs 0.117; v3: 0.080 vs 0.087): no progression one rule per latent. CODI-0's single <bot> position holds v1 and v2 more decodably (0.395, 0.184) than any CODI position (0.167, 0.117). The answer is readable only at the answer position (0.384, matching model accuracy, so the probe works when the information is linearly present).
- Combined with the diagnostics run: CODI's latents are a necessary but opaque channel. They neither add reach over CODI-0 nor carry per-rule intermediate values; the computation that matters seems to be spread over the positions in a way that only works with the item's own question.

**Gotchas hit:** none new (same memory settings as the diagnostics run).
**Caveats:** **Probes overfit heavily** (train 0.96 vs test 0.15 for v1 at z1; 6000 training items for a 768x100 probe, ~60 per class): absolute probe accuracies are lower bounds, and "not decodable" here means "not linearly decodable with this data". The CODI vs CODI-0 comparison uses identical probes, items and training, so the relative statement (CODI-0's <bot> holds v1/v2 better) is fairer than the absolute numbers. Probes are pooled over levels, not split per level. One checkpoint, one seed. One donor per item.
**Next:** Stronger probes (more train items, e.g. 2000/level; PCA or stronger L2; per-level splits; digit-wise targets). Activation patching of single latent positions (which position carries the answer-relevant signal). For retraining: fresh items per epoch (memorisation) and per-latent step supervision, since the unsupervised latents end up as an opaque channel.
