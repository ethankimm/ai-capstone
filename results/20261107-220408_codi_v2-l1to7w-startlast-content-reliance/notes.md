## 2026-11-07 — Ungated start-last CODI, content reliance: at L1 latents are ignored (donor latents keep 1.000, same-level mean keeps 1.000); from L2, below reach, the answer follows the latents' content (.994); the earlier ".09 at k=1" was a pooled-mean artefact (codi, run_id: 20261107-220408_codi_v2-l1to7w-startlast-content-reliance)

**Goal:** Separate reliance on latent content from damage done by an off-distribution latent. 20261106-105537 found all latents mean-ablated take k=1 from 1.00 to .09, though CODI-0 needs no latents there, which contradicts the draft's prediction that reliance starts at d50 = 3.
**Mechanism / model:** `codi`, gpt2. Start-last CODI checkpoint-17820 (20261106-083843), ungated, 6 latents. fp32, MPS.
**Data:** Canonical test L1-L7, 500/level. Recipients: items the model answers right. Means come from train: 2,000 per level for the same-level mean, 500 per level over L1-L7 for the pooled mean (seed 21).
**Command:** `codi_audit_artifacts/b29_ungated_resample.py` and `b29b_pooled_mean.py` under `run_guarded.sh` and `caffeinate -dims`; record from `record8.py`. All six latent inputs are replaced at once, because single latents are near copies. Different-value donor: same level, a different v_{L-1}, predicted answer = the recipient's last rule applied to the donor's v_{L-1} (in 0-99 and != gold). Same-value donor: same level, same v_{L-1}. Identity patch keeps every answer (smoke).
**Headline results:**

| Level | Clean | Diff donor: follows / keeps own (n) | Same-value donor keeps own | Same-level mean | Pooled mean |
|---|---|---|---|---|---|
| L1 | 1.000 | .000 / 1.000 (500) | 1.000 | 1.000 | .092 |
| L2 | .998 | .994 / .002 (499) | 1.000 | .042 | .056 |
| L3 | .476 | .374 / .017 (238) | .616 | .026 | .036 |
| L4 | .306 | .275 / .033 (153) | .423 | .038 | – |
| L5 | .160 | .150 / .025 (80) | .325 | .040 | – |
| L6 | .116 | .034 / .017 (58) | .193 | .020 | – |
| L7 | .076 | .105 / .079 (38) | .105 | .034 | – |

Mean columns give accuracy on all items. At L1, z_0's same-level mean has cosine .39 with the pooled mean; at L2-L3 the cosine is .96-.97.

**Interpretation:**
- **L1: latents not relied on.** The answer keeps its own value whatever the latents hold, and the depth-matched mean (the draft's own N_j definition) changes nothing. The .09 in 20261106-105537 came from its pooled mean: L1 latents differ from those of longer chains, and a mean dominated by L4-L7 breaks the model. That is format damage, not reliance.
- **L2: relied on, though not needed.** Donor latents move 99.4% of answers to the donor's value, and the same-value control keeps 1.000. Yet CODI-0 scores .98 at L2 with no latents. Reliance starts at k = 2, the first level with an intermediate value to carry, one level below d50 = 3. The latent gain starts at k = 3 (+.02 at k = 2, +.24 at k = 3; 20261106-083843).
- **L3 and up: relied on, and not a clean carrier.** Donor latents change almost every answer (keeps own .02-.08), but only .37 / .28 / .15 follow the donor's v_{L-1}, and same-value donors keep only .62 / .42 / .33. The latents carry more than v_{L-1}, or carry it imprecisely, which fits the partial reads in 20261107-084635.
- **For the draft:** "reliance near zero below d50" holds at k = 1 and fails at k = 2. Need and use separate by one level, and only content tests show it: mean ablation pooled over depth wrongly puts reliance at k = 1.

**Gotchas hit:** The pooled mean in 20261106-105537 did not follow the draft's per-depth definition (its gotcha line flagged this for the Coconut record and said the CODI ablations shared it). A correction is appended to that record.
**Caveats:** One seed, one checkpoint. Few right recipients at L5-L7 (80 / 58 / 38). Target "follows donor" assumes the latents carry v_{L-1}; at L3 and up they carry something less clean. Recipients are items answered right, which are easier.
**Next:** Run the same test on any new arm, including a latent-only gate (prefill ungated, each latent pass shown one rule), to see whether reliance still starts at k = 2.
