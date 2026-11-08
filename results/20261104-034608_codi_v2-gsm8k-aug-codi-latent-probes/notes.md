## 2026-11-04 — GSM8K CODI latent probes + logit lens: odd sites hold step 1-2 results (v_1 peaks at z_1, v_2 at z_3), even sites decode CoT punctuation; v_3+, v_4 and the answer not decodable (codi, run_id: 20261104-034608_codi_v2-gsm8k-aug-codi-latent-probes)

**Goal:** Task 2 of the GSM8K latent diagnostics. What do CODI's latents encode on GSM8K-Aug (checkpoint-31280 of gpt2_gsm8k_aug_codi)? Does site j decode step result v_j (one step per latent), or do all sites decode the same thing (one shared state)? Companion to the ablation run 20261104-034313, which found an odd/even 2-cycle.
**Mechanism / model:** `codi`, gpt2 / checkpoint-31280, compute_steps=6.
**Data:** synthetic-gsm8k_aug test n=1319 seed=None; run seed=None.
**Hyperparams:** {'train_n': 20000, 'min_count': 3, 'wds': [0.0001, 0.001, 0.01, 0.1], 'epochs': 300, 'batch_size': 64, 'dtype': 'float32', 'device': 'cuda'}
**Command:** `lrm_work/gsm8k_probe.sh` (GPU 7, cci-p124 H100 NVL) = `scripts/probe_gsm8k_codi_latents.py` (new), same model flags as the ablation run, fp32. The latents come from `ablate_latents_rulechain_codi.run()`'s loop, extended only to read the LM head; it was checked against `run()` on 128 test items (max |dz| = 0). Probes: multinomial logistic regression on standardized z_s, one per (site, target), fit on the first 20,000 items of gsm8k_aug/train.json, weight decay chosen on valid.json (1e-4..1e-1), then one evaluation on test. Classes = values seen at least 3 times in the fitting set. Test items with an unseen value count as wrong. Smoke run (200 items) first.
**Headline results:** `final_answer_accuracy=None`, `unparseable_rate=None`, `compute_steps=6`
**Probe accuracy on test (linear, class over training values):**

| Site | v_1 (n 1,301) | v_2 (n 1,235) | v_3 (n 879) | v_4 (n 515) | answer (n 1,319) |
|---|---|---|---|---|---|
| z_0 | 0.150 | 0.042 | 0.032 | 0.021 | 0.030 |
| **z_1** | **0.447** | 0.090 | 0.031 | 0.021 | 0.038 |
| z_2 | 0.197 | 0.054 | 0.032 | 0.014 | 0.029 |
| **z_3** | 0.218 | **0.208** | 0.065 | 0.043 | 0.046 |
| z_4 | 0.095 | 0.074 | 0.032 | 0.033 | 0.035 |
| **z_5** | 0.227 | 0.164 | 0.060 | 0.025 | 0.039 |
| majority baseline | 0.029 | 0.028 | 0.026 | 0.018 | 0.014 |
| coverage (ceiling) | 0.956 | 0.941 | 0.908 | 0.851 | 0.910 |

Classes: 355 / 438 / 389 / 260 / 586. Fit/valid accuracy at the best site: v_1 0.63 / 0.57, v_2 0.21 / 0.12, answer 0.32 / 0.08 (answer and v_4 overfit).

**Logit lens on test, top-1 token = target:**
- Hidden state behind z_s, hit rate for v_1 / v_2 / v_3, then the share of items whose top token matches any step value ("any"):
  - z_1: 0.433 / 0.077 / 0.035, any 0.51
  - z_3: 0.202 / 0.210 / 0.059, any 0.43
  - z_5: 0.203 / 0.164 / 0.077, any 0.40
  - z_0, z_2, z_4: ≤ 0.013
  - Answer hit ≤ 0.027 at every site.
- Most frequent top tokens:
  - z_0: "The", "*", "-", "(".
  - Odd sites: numbers ("20", "6", "24", "12").
  - **Even sites: CoT punctuation (">>", ".", "-", "<|endoftext|>").**
- Through z_s itself (tied embedding) the pattern is the same but weaker: z_1 v_1 0.34, z_3 v_1/v_2 0.15/0.15. Even sites and z_0 decode ":", "the", etc.

**Interpretation:**
- **Neither "one step per latent" nor "one shared state."** The odd sites hold step results, and these shift with depth: v_1 peaks at z_1 (0.45 probe, 0.43 lens), v_2 peaks at z_3 (0.21, about equal to v_1 there). z_5 holds about the same mix as z_3 (z_3 and z_5 have centered cosine 0.93 in the ablation run).
- **Even sites carry no step values in the lens.** They decode delimiters: the chain alternates number / delimiter like the teacher's "<<a op b = c>>" text, about one step per odd/even pair. Linear probes still recover some v_1 from z_2 (0.20).
- **Nothing beyond step 2 is decodable:** v_3 ≤ 0.065, v_4 ≤ 0.043, near the 0.02-0.03 majority baseline.
- **The final answer is not linearly decodable from any latent** (≤ 0.046). It is computed after the latents, in the answer pass.
- This matches the ablation: z_1 and z_3 are the sites whose removal costs the most, and the gain over CODI-0 sits at 1-3 steps (GSM8K_GAIN_ANALYSIS.md). On GSM8K the latents compute the first one or two intermediate results in sequence (v_1 then v_2), not the whole chain.
**Gotchas hit:** `probe_codi.py` did not fit: it uses the released checkpoint's prompt without pad_invariant, and it regresses with a tolerance. Hence a new class-probe script on Henning's loop. Values are normalized ("6.0" -> "6"). Lens hits compare the stripped top token to the value string, so multi-token values (>= 1000 and most decimals) can never hit. The run record has `final_answer_accuracy: null` (probe run, no answer accuracy).
**Caveats:** One checkpoint, one seed. Linear class probes over seen values: values unseen in training are unreachable (coverage 0.85-0.96), and rare classes are hard. A digit-wise or MLP probe could find more, so low numbers at v_3+ are an upper bound on linear-class decodability, not proof of absence. Fitting on train (near-duplicates of valid); test never used for fitting or selection. Probe accuracy is not conditioned on the model being right.
**Next:** Condition probe and lens hits on items CODI gets right vs wrong; check whether patching z_1's v_1 content (swap from a donor with a different v_1) changes the answer accordingly (causal test, proposal step 3).
