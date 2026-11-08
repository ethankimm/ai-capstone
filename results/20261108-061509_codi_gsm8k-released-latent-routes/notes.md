## 2026-11-08 — Released GSM8K CODI: single latents move the answer (z4 .36), even sites add up (pairs .55-.74, all three .85 = all six), and cutting attention back to earlier latents costs 12.5 points (.425 to .300), concentrated on longer chains; swaps still partly carry under the cut (codi, run_id: 20261108-061509_codi_gsm8k-released-latent-routes)

**Goal:** Does the RuleChain distinction (start-last CODI pools interchangeable estimates; gated and SIM chain through the recurrent input, `20261108-042557`) hold on a natural task? Released GSM8K CODI, with the swap and look-back tests used on RuleChain.
**Mechanism / model:** `zen-E/CODI-gpt2` @ fd641b3 (local HF cache), 6 latents, upstream masking. Loaded with the flags of `20260927-004340`, fp32 on MPS, batch 1 (no padding). Generation path of `scripts/decode_patch_codi.py`.
**Data:** GSM8K-Aug test. The 317 aligned minimal pairs of `20260927-004340`: recipient = test item, twin = the recipient with one number perturbed and the chain re-executed. 314 qualify here (both answered right). Also all 1,319 test items for accuracy.
**Command:** On the laptop (`run_guarded.sh`, `caffeinate`): `~/Documents/Penn/CIS5980/codi_audit_artifacts/t11_gsm8k_routes.py <ai-capstone> mps <out>` (run from the CODI checkout), record `record18.py`. A 20-pair smoke reproduced the record.
**Conditions** (sites as the record: z_s feeds iteration s+1; values sit on even sites):
- swaps on the recipient question: all six latents, single z0 / z2 / z4, pairs of even sites, all even sites, prefix z0..z2 (later latents recomputed);
- the same with the look-back cut: every latent iteration masked from earlier latent iterations, answer tokens see all;
- clean and cut accuracy on the full test.

**Headline results:**
- **Reproduces the September record in fp32.** All .847 (record .839), single z0 / z2 / z4 .201 / .210 / .363 (.199 / .202 / .360), prefix z0..z2 .541 (.539). Full-test accuracy .425 (record .419).
- **Odd sites carry nothing.** Even sites alone give exactly the all-six result (.847).
- **Even sites add up, z4 heaviest.** Pairs z0+z2 .545, z0+z4 .650, z2+z4 .736.
- **Prefix = its even sites.** The prefix z0..z2 (.541) matches z0+z2 alone (.545): recomputed later latents add no twin value. This corrects the earlier reading that "early latents partly carry forward".
- **Look-back cut: accuracy .425 to .300** on all 1,319 items; recipients stay right .567. Swaps under the cut: all .615, even pairs .35-.43, singles .04-.15.
- **The cost grows with chain length** (added after the run, on the 242 unique right recipients): kept right .93 / .52 / .22 / .33 at 2 / 3 / 4 / 5 gold steps (n = 120 / 84 / 32 / 6). Start-last CODI's kept-right under the same cut is flat with depth (.70-.77 at L2-L5). Full-test version: t14.

**Predictions (`t_predictions.md`):**
- *Reproduce within ~.05:* right.
- *The look-back cut costs little (chain-like):* **wrong**, it costs 12.5 points.
- *Swaps still propagate under the cut:* partly.

**Interpretation:** GSM8K CODI is not start-last CODI. Its latents are specialised (odd sites empty; even sites carry different weight; single sites move the answer) where start-last's are interchangeable votes. It leans on attention back to earlier latents, increasingly with chain length. But "prefix = its even sites" shows that later latents do not recompute a twin's value from swapped early latents through the recurrence. So the value reaches the answer by being read where it was written. The route-split test (`20261108-061509_..._route-split`) tests that directly.
**Gotchas hit:** None. The record's bf16/CUDA and this fp32/MPS run differ by at most .01 on every overlapping condition.
**Caveats:**
- One released checkpoint, trained differently from our RuleChain models.
- The look-back cut is an attention mask the model never saw, so off-distribution damage is possible. The odd/even control is in the route-split record.
- The chain-length split uses a selected subset; t14 redoes it on the full test.
**Next:** route split and odd/even control (t12), attention shares (t13), full-test depth split (t14).
