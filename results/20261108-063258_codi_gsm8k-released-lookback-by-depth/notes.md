## 2026-11-08 — On the full GSM8K test, the look-back cut barely touches short problems and removes most right answers on long ones: kept right .96 / .86 / .51 / .35 / .26 at 1 / 2 / 3 / 4 / 5 gold steps; odd-only and even-only cuts follow the same slope. Start-last CODI's cost is flat with depth (codi, run_id: 20261108-063258_codi_gsm8k-released-lookback-by-depth)

**Goal:** Check the chain-length pattern from t11's selected recipients (kept .93 / .52 / .22 at 2 / 3 / 4 steps, n = 242) on all 1,319 test items, with per-item answers saved.
**Mechanism / model:** `zen-E/CODI-gpt2` @ fd641b3, fp32 MPS, batch 1; t12's model code and `run()`.
**Data:** GSM8K-Aug test, all 1,319 items. Chain length = number of steps in the gold rationale (`Example.intermediate_values`).
**Command:** `~/Documents/Penn/CIS5980/codi_audit_artifacts/t14_gsm8k_lookback_depth.py` (CODI venv, `run_guarded.sh`, `caffeinate`), record `record19.py`. Accuracy reproduces t11 / t12 exactly: clean .425, nb_all .300, nb_even .329, nb_odd .321.

**Headline results** (share of the model's right answers still right under each cut):

| Gold steps | Items | Clean accuracy | Right | Kept: all cut | even sites only | odd sites only |
|---|---|---|---|---|---|---|
| 0 | 18 | .278 | 5 | 1.00 | 1.00 | .80 |
| 1 | 65 | .400 | 26 | .96 | .96 | .96 |
| 2 | 357 | .658 | 235 | .86 | .91 | .89 |
| 3 | 364 | .489 | 178 | .51 | .61 | .65 |
| 4 | 290 | .314 | 91 | .35 | .52 | .41 |
| 5 | 138 | .138 | 19 | .26 | .32 | .37 |
| 6+ | 87 | .069 | 6 | .50 | .67 | .67 |

- **The cost grows with chain length.** One- and two-step problems keep .86-.96 of right answers. Three-step problems keep half, four- and five-step about a third or less (6+ rests on 6 items).
- **Both partial cuts follow the same slope.** Cutting only the value sites or only the punctuation sites each costs part, and together they cost the most at every length.
- **Contrast with start-last CODI** (t10 / b21, RuleChain valid): kept right .70 / .74 / .77 / .77 at L2-L5, flat or slightly rising with depth.

**Predictions:** none written separately for t14. The subset pattern from t11 (written after t11) predicted the slope, and it holds on the full test.
**Interpretation:** In GSM8K CODI, attention back to earlier latents matters more the more steps a problem needs. That fits the route split (`20261108-061509_..._route-split`): values live in latents' keys/values and are fetched by later latents and the answer, so longer chains depend on more such reads. The equal cost of odd-only and even-only cuts at every length says the punctuation sites take part in that process too, not as value holders (t11) but as entries later latents read.
**Gotchas hit:** None.
**Caveats:**
- Chain length co-varies with question length and the number of numbers.
- Accuracy itself falls with length, so the kept share at 5+ steps rests on few items.
- The cut is a mask the model never saw.
- One released checkpoint.
**Next:** GSM8K goes into the regime table as "slots linked by attention". A matched GSM8K pause arm (handoff section 6b) would test whether positions without recurrence suffice there as well.
