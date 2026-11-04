# GSM8K-Aug CODI latent diagnostics: ablation, geometry, probes

Ran 2026-11-03 19:43-19:50 PT on cci-p124 (H100 NVL, GPU 7). Branch `kevin/lrm-training-2`. Context: `GSM8K_PILOT.md`
(CODI beats CODI-0 by +0.074 on GSM8K test). This is the same measurement as Henning's RuleChain run
`20261104-022824_codi_v2-l1to7w-depth2-codi-latent-ablation` (origin/main). That run found z_0 unused, z_1 carrying the
item, and z_1..z_5 as near-copies at a fixed point.

**Checkpoint:** `gpt2_gsm8k_aug_codi` checkpoint-31280, the selected checkpoint behind the logged test eval
`20261104-012327`. Model flags as in that eval: 6 latents, `fix_attn_mask` + `pad_invariant`, `include_last_cot False`,
projection 768 + LayerNorm. fp32.

**Sites:** z_s = the projected vector fed into latent pass s+1; z_0 = prj(hidden at `<bot>`).

**Bottom line:** on GSM8K, CODI's latents do a short serial computation, not one extra pass of parallel compute, and they
do not settle on RuleChain's fixed point.
- The latents alternate in a 2-cycle (odd sites z_1/z_3/z_5 vs even sites z_2/z_4).
- Each odd site carries item content, and ablating the tail from earlier sites costs accuracy gradually.
- The odd sites hold the first step results in order: v_1 at z_1, v_2 at z_3.
- Nothing past step 2 is decodable, and neither is the final answer.

---

## Task 1: ablation and geometry

Run record `20261104-034313_codi_v2-gsm8k-aug-codi-latent-ablation`. Script: `scripts/ablate_latents_rulechain_codi.py`
(from origin/main; only change: the run record's dataset name follows the task). Means from `gsm8k_aug/valid.json`
(1,000). The resample donor is a seeded permutation of test items, from any step count.

**Scoring:** accuracy is the teacher-forced first answer token. Items whose answer is more than one GPT-2 token are
skipped: **1,195 of 1,319 scored**.

| Reference on the scored items | Accuracy |
|---|---|
| Clean (this run) | **0.283** |
| Generated, same items | 0.274 (agrees with teacher-forced on 99.2% of items) |
| CODI-0, teacher-forced, same items | 0.204 |

All conditions are compared with this run's clean 0.283.

### Accuracy by condition

Bold cells are the odd sites.

| Condition | s=0 | s=1 | s=2 | s=3 | s=4 | s=5 |
|---|---|---|---|---|---|---|
| mean_s | 0.277 | **0.224** | 0.271 | **0.255** | 0.265 | **0.254** |
| resample_s | 0.280 | **0.198** | 0.267 | **0.221** | 0.256 | **0.220** |
| mean_tail_ge_s | — | **0.118** | 0.205 | 0.210 | 0.246 | 0.254 |

mean_all = **0.115**.

Δ vs clean, with exact McNemar p:

| Condition | s=0 | s=1 | s=2 | s=3 | s=4 | s=5 |
|---|---|---|---|---|---|---|
| mean_s | −0.006 (0.14) | −0.059 (3e-13) | −0.012 (0.04) | −0.028 (3e-5) | −0.018 (0.003) | −0.029 (5e-6) |
| resample_s | −0.003 (0.56) | −0.085 (7e-18) | −0.016 (0.004) | −0.062 (2e-13) | −0.027 (3e-5) | −0.063 (6e-14) |
| mean_tail_ge_s | — | −0.165 (2e-43) | −0.078 (1e-16) | −0.073 (5e-15) | −0.037 (2e-8) | −0.029 (5e-6) |

What each site adds back, from all ablated to clean (paired between adjacent tail conditions):

| Site restored | z_1 | z_2 | z_3 | z_4 | z_5 |
|---|---|---|---|---|---|
| Δ accuracy | **+0.087** (120 vs 16, p 7e-21) | +0.005 (p 0.26) | **+0.036** (62 vs 19, p 2e-6) | +0.008 (p 0.02) | **+0.028** |

All latents mean-ablated (0.115) vs CODI-0 (0.204), same items: 40 vs 147 discordant, p 1e-15.

### Geometry (clean test latents)

`cos_sites_centered` (each site's valid mean subtracted):

```
      z0    z1    z2    z3    z4    z5
z0  1.00  0.21  0.51  0.19  0.46  0.18
z1  0.21  1.00  0.17  0.64  0.16  0.63
z2  0.51  0.17  1.00  0.22  0.90  0.21
z3  0.19  0.64  0.22  1.00  0.23  0.93
z4  0.46  0.16  0.90  0.23  1.00  0.23
z5  0.18  0.63  0.21  0.93  0.23  1.00
```

`cos_sites_raw`: z_3·z_5 0.977, z_2·z_4 0.955, odd·even about 0.46-0.51.

| Per site | z_0 | z_1 | z_2 | z_3 | z_4 | z_5 |
|---|---|---|---|---|---|---|
| cross_item_cos_per_site | 0.872 | 0.677 | 0.706 | 0.698 | 0.702 | 0.702 |
| item_specific_energy_per_site | 0.138 | 0.330 | 0.331 | 0.316 | 0.333 | 0.312 |
| norm_per_site | 50.9 | 47.1 | 46.6 | 47.7 | 47.4 | 47.9 |

### Comparison with RuleChain (Henning, L1-7w checkpoint)

| | RuleChain | GSM8K |
|---|---|---|
| z_0 | unused | unused (mean −0.006, resample −0.003, n.s.) |
| Between-site structure | fixed point: z_1..z_5 centered cosine 0.97-0.997 | **2-cycle**: odd 0.63-0.93, even 0.90, odd·even ~0.2 |
| Worst single-site ablation | ≤ 0.04 (z_1..z_5 back each other up) | resample z_1 / z_3 / z_5: −0.085 / −0.062 / −0.063 |
| Tail ablation | cliff only at s=1 (tail ≥ 3 costs 0.01) | **gradual:** −0.029, −0.037, −0.073, −0.078, −0.165 for s = 5…1 |
| All latents removed vs matched CODI-0 | CODI-0 not yet available | 0.115 vs 0.204: CODI depends on its latents |

**Key question: gradual, not only at s=1.** Each odd site (z_1, z_3, z_5) adds a significant increment, while the even
sites add almost nothing.

---

## Task 2: what z_1..z_5 encode

Run record `20261104-034608_codi_v2-gsm8k-aug-codi-latent-probes`. Script: `scripts/probe_gsm8k_codi_latents.py` (new).

Why a new script:
- `probe_codi.py` uses the released checkpoint's prompt without `pad_invariant`, and it regresses with a tolerance.
- The new script runs the latent loop of the ablation script's `run()` (checked on 128 items: max |Δz| = 0) and also
  reads the LM head.

Setup:
- Targets: v_1..v_4, the results of the `<<a op b = c>>` steps (only items with at least j steps), plus the final answer.
- Probe: multinomial logistic regression on standardized z_s, one per (site, target).
  - Fit on the first 20,000 items of `gsm8k_aug/train.json`.
  - Weight decay chosen on valid.
  - Evaluated once on test; never fit on test.
- Classes: values seen at least 3 times in the fitting set. Test items with unseen values count as wrong.

### Probe accuracy (test)

| Site | v_1 | v_2 | v_3 | v_4 | answer |
|---|---|---|---|---|---|
| z_0 | 0.150 | 0.042 | 0.032 | 0.021 | 0.030 |
| **z_1** | **0.447** | 0.090 | 0.031 | 0.021 | 0.038 |
| z_2 | 0.197 | 0.054 | 0.032 | 0.014 | 0.029 |
| **z_3** | 0.218 | **0.208** | 0.065 | 0.043 | 0.046 |
| z_4 | 0.095 | 0.074 | 0.032 | 0.033 | 0.035 |
| **z_5** | 0.227 | 0.164 | 0.060 | 0.025 | 0.039 |
| majority baseline | 0.029 | 0.028 | 0.026 | 0.018 | 0.014 |
| coverage (ceiling) | 0.956 | 0.941 | 0.908 | 0.851 | 0.910 |
| n test items | 1,301 | 1,235 | 879 | 515 | 1,319 |
| classes | 355 | 438 | 389 | 260 | 586 |

Fit / valid accuracy at the best site: v_1 0.63 / 0.57, v_2 0.21 / 0.12, answer 0.32 / 0.08. The answer and v_4
probes overfit.

### Logit lens (test): share of items whose top-1 token equals the target

Through the final-layer hidden state whose projection is z_s:

| Site | v_1 | v_2 | v_3 | v_4 | answer | any step | Most frequent top tokens |
|---|---|---|---|---|---|---|---|
| z_0 | 0.006 | 0.002 | 0.003 | 0.008 | 0.005 | 0.013 | "The", "*", "-", "(", "/" |
| **z_1** | **0.433** | 0.077 | 0.035 | 0.035 | 0.021 | 0.510 | "20", "6", "24", "10", "12" |
| z_2 | 0.000 | 0.001 | 0.002 | 0.000 | 0.001 | 0.002 | ">>", ".", "-", "<\|endoftext\|>", "/" |
| **z_3** | 0.202 | **0.210** | 0.059 | 0.043 | 0.027 | 0.425 | "12", "2", "4", "6", "20" |
| z_4 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | ">>", ".", "<\|endoftext\|>", "-", "" |
| **z_5** | 0.203 | 0.164 | 0.077 | 0.031 | 0.022 | 0.397 | "12", "5", "10", "2", "8" |

Through z_s itself (tied weights, so nearest input embedding), the pattern is the same but weaker:
- z_1: v_1 0.34.
- z_3: v_1 / v_2 0.15 / 0.15.
- z_5: v_1 / v_2 0.14 / 0.11.
- Even sites and z_0 decode ":", "the", "...\"".

**Key question: neither one step per latent nor one shared state.**
- The odd sites hold step results that shift with depth: v_1 peaks at z_1, and v_2 peaks at z_3, where it is about equal
  to v_1. z_5 holds about the same mix as z_3 (centered cosine 0.93).
- The even sites decode CoT punctuation (`>>`, `.`) rather than values. The chain alternates value / delimiter like the
  teacher's `<<a op b = c>>` text, about one solution step per odd/even pair.
- v_3, v_4 and the final answer are near the majority baseline at every site. The answer is computed after the latents.

---

## Reading

GSM8K's latent gain over CODI-0 comes from a short serial computation over the odd latent passes, covering about the
first one or two intermediate results (v_1 then v_2). It is not a single extra pass of parallel compute, and it is not a
full chain. This fits where the gain sits in `GSM8K_GAIN_ANALYSIS.md`: 1-3 steps, nothing from 4 steps up.

On RuleChain, the same recipe reaches a fixed point after one pass. There, CODI-0 already solves the depths where one
or two latent steps would help, so nothing is left for them to add.

## Caveats

- One checkpoint, one seed.
- Teacher-forced first answer token only; 124 multi-token-answer items (mostly answers > 999) are excluded, so the
  scored set is easier than the full test set.
- Resample donors come from any step count, not within level.
- Linear class probes cannot reach values unseen in training, and rare classes are hard. Low scores at v_3+ are an upper
  bound on what a linear class probe finds, not proof that the information is absent.
- Lens hits compare a single token, so values of 1,000 and up, and most decimals, can never hit.
- Probe accuracy is not conditioned on whether the model answers correctly.

## Next

- Condition probe and lens hits on items CODI gets right vs wrong.
- Causal test: swap z_1 (or z_3) from a donor with a different v_1 (or v_2), and check whether the answer moves
  accordingly.
- A within-step-count resample swap scored follows-donor / keeps-own (the protocol's primary necessity test).
