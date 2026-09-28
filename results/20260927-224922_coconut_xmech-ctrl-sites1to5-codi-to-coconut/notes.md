## 2026-09-28 — Experiment A1 complement, CODI z1-z5 → Coconut: post-question-encode latents alone carry 61-98% of the unrestricted map's cf_joint (coconut, run_id: 20260927-224922_coconut_xmech-ctrl-sites1to5-codi-to-coconut)

**Goal:** Complement view of Experiment A1 (`20260927-224224`, CODI z0 only): with z0 (the
pre-reasoning question encoding) held OUT of the map's input, do CODI's remaining five iterations
(z1-z5, where its own iterative "thinking" actually happens) still carry most of the unrestricted
map's transfer power on their own?
**Design:** identical to `20260927-224224` except the ridge map reads CODI z1,z2,z3,z4,z5
concatenated (5x768=3840 dims) instead of z0 alone (`xmech_coconut.py --mode ctrl --ctrl_kind
sites1to5`). Same 259 recipients, same fit split, same carriers (passes 1,4).
**Mechanism / model:** Coconut `hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33` (target), CODI
`hf:zen-E/CODI-gpt2@fd641b3` (source, feature only, z0 excluded).
**Command:** `scripts/xmech_coconut.py ... --mode ctrl --ctrl_kind sites1to5 --slug
xmech-ctrl-sites1to5-codi-to-coconut --stage full_run` (full line in `eval_command.txt`). Pod
`jbzj5cd4diacxb` (RTX A40 secure, $0.49/hr, EU-SE-1); ran alongside 7 other jobs on one GPU.
**Headline results** (259 recipients; siblings `20260927-224224` (z0 only) and `20260927-224642`
(plain GPT-2); reference unrestricted run `20260927-085211`, mapped_all cf_joint L2/L3/L4 =
0.32/0.28/0.24):

| cf_joint (null ≤0.015) | L2 | L3 | L4 |
|---|---|---|---|
| own_carriers (this run) | 0.270 | 0.313 | 0.351 |
| **mapped_carriers (z1-z5)** | **0.278** | **0.274** | **0.147** |
| shuffled_carriers | 0.012 | 0.008 | 0.004 |
| % of unrestricted mapped_all (0.32/0.28/0.24) | 87% | 98% | 61% |

Map fit (held-out R², CODI z1-z5 → Coconut pass): pass0 0.323, pass1 0.185, pass2 0.282, pass3
0.317, pass4 0.180, pass5 0.287 — noticeably higher than z0-only's R² (`20260927-224224`) at every
pass, consistent with the reasoning iterations carrying more total decodable structure once
concatenated. self_mapped_carriers unchanged 76.8% (highest of the three A1/A2 conditions).

**Interpretation:**
- **CODI's iterative latents (z1-z5) alone reach 87-98% of the unrestricted 6-site map at L2/L3,
  and both z0-only and z1-z5-only individually approach or exceed the full 6-site rate at L2/L3** —
  the two halves of CODI's latent trajectory are each close to sufficient on their own, i.e. the
  transferable value information is redundant across the trajectory, not concentrated in one place
  that z0-only or z1-z5-only would miss. This matches the earlier DAS finding
  (`20260927-185312`) that CODI's value isn't confined to a small subspace.
- L4 is the one level where z1-z5 underperforms z0 (61% vs 110%) and both underperform the combined
  6-site map somewhat unevenly across levels — with n=259 and one donor draw per level this could
  be noise, but is also consistent with L4 (random donor, most different problem) needing more of
  the FULL trajectory (including z0's problem framing) than L2/L3's closer donors.
**Caveats:** n=259. This run and z0-only together don't sum to more than the unrestricted map
(they're two overlapping, individually-strong views of the same redundant information, not
independent contributions).
**Next:** See the reverse direction's complement (`20260927-225810_codi_..sites1to5..`) — there the
asymmetry is much sharper (Coconut needs its later passes, unlike CODI where either half suffices).
