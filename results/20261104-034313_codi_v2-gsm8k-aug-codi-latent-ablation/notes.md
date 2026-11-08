## 2026-11-04 — GSM8K CODI latent ablation: odd sites z_1, z_3, z_5 each carry item content (2-cycle, not RuleChain's fixed point); all latents mean-ablated falls below CODI-0 (codi, run_id: 20261104-034313_codi_v2-gsm8k-aug-codi-latent-ablation)

**Goal:** Henning's RuleChain latent ablation (20261104-022824: z_0 unused, z_1 carries the item, z_1..z_5 near-copies at a fixed point) repeated on the GSM8K-Aug CODI checkpoint, where CODI beats CODI-0 by +0.074 (GSM8K_PILOT.md). Does GSM8K's gain come from serial latent passes (accuracy falls gradually as the mean-ablated tail starts earlier) or from one extra pass (falls only at s = 1)?
**Mechanism / model:** `codi`, gpt2 / checkpoint-31280, compute_steps=6.
**Data:** synthetic-gsm8k_aug test n=1319 seed=None; run seed=None.
**Hyperparams:** {'inference': {'latent_iterations': 6, 'batch_size': 64, 'dtype': 'float32', 'device': 'cuda', 'perm_seed': 0}, 'test_json': '/local-ssd/ktn48/lrm_work/synthetic_data_v2/gsm8k_aug/test.json'}
**Command:** `lrm_work/gsm8k_ablate.sh` (GPU 7, cci-p124 H100 NVL) = `scripts/ablate_latents_rulechain_codi.py` from origin/main (only change: the run record's dataset name and notes follow the task) on `gpt2_gsm8k_aug_codi` checkpoint-31280 (the selected, logged checkpoint, 20261104-012327), same model flags as the GSM8K eval (6 latents, fix_attn_mask + pad_invariant, include_last_cot False, prj 768 + LayerNorm), fp32, means from gsm8k_aug/valid.json (1,000), resample donor = seeded permutation of test items (any step count). Smoke on 64 items first. See eval_command.txt.
**Headline results:** `final_answer_accuracy=0.283`, `unparseable_rate=0.000`, `compute_steps=6`
**Scored items:** accuracy is the teacher-forced answer token at the ':' of "<eot> The answer is:"; items whose answer is more than one GPT-2 token are skipped: **1,195 of 1,319 scored** (124 skipped). Clean = 0.283 on those items. Generated accuracy on the same items is 0.274; teacher-forced and generated agree on 99.2% of items. CODI-0's teacher-forced accuracy on the same 1,195 items is 0.204. All conditions below are compared with this run's own clean 0.283.

| Condition (Δ vs clean 0.283; McNemar p) | s=0 | s=1 | s=2 | s=3 | s=4 | s=5 |
|---|---|---|---|---|---|---|
| mean_s | 0.277 (−0.006; 0.14) | **0.224 (−0.059; 3e-13)** | 0.271 (−0.012; 0.04) | **0.255 (−0.028; 3e-5)** | 0.265 (−0.018; 0.003) | **0.254 (−0.029; 5e-6)** |
| resample_s | 0.280 (−0.003; 0.56) | **0.198 (−0.085; 7e-18)** | 0.267 (−0.016; 0.004) | **0.221 (−0.062; 2e-13)** | 0.256 (−0.027; 3e-5) | **0.220 (−0.063; 6e-14)** |
| mean_tail_ge_s | — | **0.118** | 0.205 | 0.210 | 0.246 | 0.254 |

mean_all 0.115. Adding each site back, from all-ablated to clean: z_1 +0.087 (120 vs 16 items, p 7e-21), z_2 +0.005 (p 0.26), **z_3 +0.036 (62 vs 19, p 2e-6)**, z_4 +0.008 (p 0.02), **z_5 +0.028**. Note that mean_tail_ge_5 = mean_5.

Geometry (clean test latents, means from valid):
- Centered cosine between sites:
  - **2-cycle** among the odd sites: z_1·z_3 0.635, z_1·z_5 0.628, z_3·z_5 0.930.
  - Even sites: z_2·z_4 0.902.
  - **Odd vs even 0.16-0.23.**
  - z_0 is closer to the even sites (0.51 / 0.46) than to the odd sites (~0.19).
- Raw cosine shows the same pattern: z_3·z_5 0.977, z_2·z_4 0.955, odd·even ~0.47-0.51.
- Cross-item cosine per site: 0.872 / 0.677 / 0.706 / 0.698 / 0.702 / 0.702.
- Item-specific energy share: 0.138 / 0.330 / 0.331 / 0.316 / 0.333 / 0.312.
- Norms: about 47-51 at every site.

**Interpretation:**
- **Neither RuleChain's fixed point nor a clean one-step-per-latent chain.** The recurrence alternates between two states. The odd state (z_1, z_3, z_5) carries what the answer reads: every odd site matters on its own (resample costs −0.06 to −0.09), and ablating one is not covered by the other odd copies. The even state (z_2, z_4) costs at most −0.03 alone and adds little to the tail.
- **The tail curve declines in steps:** −0.028 (s=5), −0.037 (s=4), −0.073 (s=3), −0.078 (s=2), −0.165 (s=1). Each odd site adds a significant increment: z_1 +0.087, z_3 +0.036, z_5 +0.028. So on GSM8K later latents keep adding accuracy, unlike RuleChain, where tail >= 3 cost 0.01. That is evidence against "one extra pass" and for something done over several passes. It does not show one step per latent, because z_3 and z_5 are near-copies (centered cosine 0.93) yet each still matters.
- **z_0 is unused** here too: mean −0.006, resample −0.003, both n.s.
- **Mean-ablating every latent (0.115) falls far below CODI-0 (0.204 teacher-forced on the same items; 40 vs 147, p 1e-15).** CODI's answer depends on its latents. It is not a no-latent model with spare positions.
**Gotchas hit:** Our branch lacked the two scripts; checked them out from origin/main (`git checkout origin/main -- scripts/ablate_latents_rulechain_codi.py scripts/diagnose_codi_latents.py`), not merged. The record's dataset name was hard-coded `synthetic-depth2`; it now follows the task. The 64-item smoke test already showed the odd/even structure. Run took ~1 min on the H100.
**Caveats:** One checkpoint, one seed. 124 multi-token-answer items (mostly answers > 999) are not scored, so the scored set is easier than the full test set. Resample donors are drawn from any step count, not within level. Mean ablation uses valid means; valid items are near-duplicates of training items (GSM8K_PILOT.md), which should make the means typical rather than off-distribution. Teacher-forced first token only.
**Next:** Task 2 (probes + logit lens per site) to see what the odd and even states encode; a within-step-count resample swap scored follows-donor / keeps-own.
