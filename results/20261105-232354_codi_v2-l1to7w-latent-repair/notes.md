## 2026-11-05 — Where do CODI's wrong answers come from? Latents holding a crisp true value fix 99.8% of wrong answers at every level, so the errors are upstream of the latents and answer stage (codi, run_id: 20261105-232354_codi_v2-l1to7w-latent-repair)

**Goal:** Test the relay reading from 20261105-221415 directly. If working latents only carry what prefill computed, giving a wrongly answered item the true value should fix it.
**Mechanism / model:** `codi`, gpt2, 6 latents. Round-1 CODI, lr-3e-3 L1-7w CODI, stabilised CODI as the negative control. fp32, MPS.
**Data:** 600 wrongly answered canonical test items per model, sampled across L3-7. Every latent is patched from a donor the model answers exactly right, with the recipient's last rule. *Short donor*: 2 rules, reaching the true value before the last rule. *Same-length donor*: as many rules as the recipient, same value. *Wrong-value donor*: same length, value 3 away. Same-length donors could not always be found, so n is lower.
**Command:** `scripts/codi_null_audit/b17_repair.py`.
**Headline results:** Exact hit on the target answer (the gold answer; for the wrong-value donor, the last rule applied to the wrong value).

| Donor | Round-1 CODI | lr-3e-3 CODI | Stabilised CODI |
|---|---|---|---|
| Short, true value | 0.998 (n 600) | 0.997 (n 600) | 0.092, unchanged 0.73 |
| Same length, true value | 0.719 (n 438), within 2 0.911 | 0.805 (n 529), within 2 0.960 | 0.074, unchanged 0.82 |
| Same length, value 3 off | 0.715 (n 438) | 0.818 (n 528) | 0.012, unchanged 0.78 |

Same-length donor by level, L4 / L5 / L6 / L7: round 1 0.98 / 0.80 / 0.66 / 0.57; lr-3e-3 0.98 / 0.87 / 0.83 / 0.70. Short donor: 0.99-1.00 at every level.

**Interpretation:**
- **The answer stage is not the bottleneck.** Given a crisp value, working models answer right 99.7-99.8% of the time, at L7 too. So their errors are in the value that reaches the latents.
- **The relay keeps a crisp value crisp** across all six latents, so the blur is unlikely to be added by the latents.
- **The value gets blurrier with chain length, even on items answered right.** Latents from correctly answered long chains transfer exactly only 0.57-0.70 of the time at L7, but within 2 in 0.91-0.96 overall. This corrects my statement that the value is carried faithfully rather than roughly: the carrying looks faithful, but what is carried is rough on long chains, which fits the near-miss errors.
- Stabilised CODI mostly ignores the patch, as expected.

**Gotchas hit:** The first version took the first 900 wrong items, which over-sampled low levels; items are now sampled at random. L3 rows are tiny for the working models (2-3 items).
**Caveats:** One seed per recipe. Selected on model-correct donors, so "blur" is measured on items whose own answers were right. Why long-chain latents transfer worse is inferred: other item-specific information in them is not ruled out. Blur in prefill is inferred from the relay holding crisp values, not read directly.
**Next:** Same test on the start-last pair.
