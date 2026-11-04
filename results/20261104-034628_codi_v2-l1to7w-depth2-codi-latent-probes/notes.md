## 2026-11-03 — RuleChain L1-7w CODI latent probes: every latent holds the same coarse state (v_1 0.38-0.40, v_2 0.31, v_3 0.17, v_4+ <= 0.09 exact); no site-per-step staircase; the answer is only at the readout position (0.66) (codi, run_id: 20261104-034628_codi_v2-l1to7w-depth2-codi-latent-probes)

**Goal:** What do CODI's latent positions encode? If latent j held v_j (the value after rule j) the probes would show a staircase over sites; if every latent holds the same summary, they would be flat. Also tests the idea (Kevin, 2026-11-03) that a latent holds the final answer and the rest copy it.
**Mechanism / model:** `codi`, checkpoint-17820 of 20261103-193425 (levels 1-7, 4-7 up-weighted, 6 latents), CPU fp32. Companion to the ablation run 20261104-022824.
**Data:** probes fit on 2,000 train.json items per level (14,000), scored on the 3,500 canonical L1-L7 test items. Target v_j on items with level >= j.
**Command:** `scripts/probe_latents_rulechain_codi.py` (`eval_command.txt`). Sites = last-layer hidden states (after ln_f): h_0 at <bot>, h_s = output of latent pass s (z_s = prj(h_s) is what is fed on), h_ans = ':' of "<eot> The answer is:". Probe = two 10-way digit heads (linear softmax, L2 on a val slice), exact = both digits right.
**Headline results:** (exact accuracy; majority baseline 0.02-0.03 for every target)

| Target | h_0 | h_1 | h_2 | h_3..h_6 | h_ans |
|---|---|---|---|---|---|
| v_1 | 0.471 | 0.395 | 0.386 | 0.375-0.383 | 0.250 |
| v_2 | 0.356 | 0.311 | 0.310 | 0.302-0.309 | 0.246 |
| v_3 | 0.141 | 0.160 | 0.167 | 0.168-0.170 | 0.219 |
| v_4 | 0.071 | 0.087 | 0.084 | 0.090-0.092 | 0.174 |
| v_5 | 0.042 | 0.071 | 0.075 | 0.089-0.091 | 0.131 |
| v_6 | 0.035 | 0.042 | 0.063 | 0.055-0.056 | 0.132 |
| v_7 | 0.026 | 0.024 | 0.036 | 0.024-0.030 | 0.126 |
| answer | 0.104 | 0.129 | 0.163 | 0.166-0.168 | 0.662 |

Decodes: lm_head(h_s) is almost never a number (0-16%) and equals any v_j or the answer in <= 2.6% of items. z_s . wte^T is a number token 76-78% of the time but equals v_j in only 6-17% (v_2 best) and the answer in < 1%.
**Interpretation:** No step-per-latent structure: h_1..h_6 have the same profile for every target (flat across sites), as expected from their near-identical geometry. What they hold is a partial, early-rule summary: v_1 and v_2 in a third or more of items, v_3 weakly, v_4 onward near floor. v_1 is most readable already at <bot> (0.47, computed while reading the question), and later latents do not add v_2, v_3... in turn. The final answer is not in the latents (0.17, versus 0.66 at the readout position), so the "one latent holds the answer and the rest copy it" picture does not fit: the answer is computed at the readout position, which attends to the latents' keys and values. With the ablation, the latents act as one extra block of compute whose output the readout uses, and nothing in training pushes latent j to advance the value by a rule.
**Gotchas hit:** Some GPT-2 ids have no token string (decode raises), handled as non-number.
**Caveats:** Linear probes on the last layer only, which is what z_s carries forward, but the readout also reads the latents' keys/values at every layer and those are not probed. Linear-probe failure is not proof the value is absent (could be nonlinear). One checkpoint, one seed.
**Next:** Same probe on Kevin's GSM8K-Aug CODI checkpoint (does GSM8K show a staircase?). For RuleChain, a step-level target (SIM-CoT decoder, or per-step distillation / value head) would have to create this structure; currently latent j does not hold v_j even partially beyond j = 2.
