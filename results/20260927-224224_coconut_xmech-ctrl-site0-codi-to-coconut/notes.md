## 2026-09-28 — Experiment A1, CODI z0 only → Coconut: pre-reasoning latent alone carries 74-110% of the unrestricted map's cf_joint (coconut, run_id: 20260927-224224_coconut_xmech-ctrl-site0-codi-to-coconut)

**Goal:** Round-3/4 P4 caveat: the unrestricted CODI↔Coconut map reads all six source latents and
writes whole target vectors, so it could route question-derived information rather than the
source's computed values. This is Experiment A (question-only control): does a source
representation that has done NO reasoning yet already carry most of the unrestricted map's
transfer? Here: CODI z0 alone (the vector after only the question-encode pass, before any of
CODI's own iterative "thinking").
**Design** (`xmech_common.run_transplant` with the new `feat`/`feat_sites` params;
`xmech_coconut.py --mode ctrl --ctrl_kind site0`): identical transplant to the unrestricted P4 run
(`20260927-085211_coconut_xmech-codi-to-coconut`) — same 259 eval-split ladder recipients/donors
(base-correct in both models, pair_seed 0), same fit split (8000 train questions), same carriers
(passes 1,4). Only the ridge map's INPUT changes: instead of concatenating all 6 CODI z-sites, it
reads CODI z0 only (768 dims instead of 4608). Recipients/donors are still pinned by the true
CODI↔Coconut correctness intersection (`xmech_common._ladder_pairs`); only the "mapped"/"self_mapped"
conditions' features change.
**Mechanism / model:** Coconut `hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33` (target), CODI
`hf:zen-E/CODI-gpt2@fd641b3` (source, feature only).
**Command:** `scripts/xmech_coconut.py --checkpoint_path .../checkpoint_33 --questions
xmech_questions.jsonl --codi_latents codi_latents.pt --out_latents coconut_latents.pt --mode ctrl
--ctrl_kind site0 --slug xmech-ctrl-site0-codi-to-coconut --stage full_run` (full line in
`eval_command.txt`). Pod `jbzj5cd4diacxb` (RTX A40 secure, $0.49/hr, EU-SE-1); ran alongside 7
other jobs (all 6 Experiment A conditions + both Experiment B `das_ladder` runs) on one GPU.
**Headline results** (259 recipients; sibling runs at the same direction:
`20260927-224922_..sites1to5..` (complement) and `20260927-224642_..gpt2..` (A2 no-reasoning
control); reference unrestricted run `20260927-085211_coconut_xmech-codi-to-coconut`, mapped_all
cf_joint L2/L3/L4 = 0.32/0.28/0.24):

| cf_joint (null ≤0.015) | L2 | L3 | L4 |
|---|---|---|---|
| own_carriers (this run) | 0.270 | 0.313 | 0.351 |
| **mapped_carriers (z0 only)** | **0.236** | **0.205** | **0.263** |
| shuffled_carriers | 0.012 | 0.008 | 0.008 |
| % of unrestricted mapped_all (0.32/0.28/0.24) | 74% | 73% | 110% |

Map fit (held-out R², CODI z0 → Coconut pass): pass0 0.208, pass1 0.290, pass2 0.225, pass3 0.102,
pass4 0.088, pass5 0.062. self_mapped_carriers (recipient's own z0, mapped back into itself)
unchanged 65.6%.

**Interpretation:**
- **CODI's very first latent — computed before any of its own iterative reasoning — already
  carries most of what the unrestricted 6-site map uses to transfer values into Coconut**: 74-110%
  of the all-sites mapped rate, vs 5-8% for a plain-GPT2 question encoding (sibling run
  `20260927-224642`, Experiment A2) and ~87-98% for the complement z1-z5 (sibling
  `20260927-224922`). Per the pre-registered decision rule (≥70% ⇒ explained by the source's
  own pre-reasoning encoding, not a generic question confound — since a plain untrained-for-GSM8K
  GPT-2 encoding of the same text reaches nowhere near this).
- This does NOT mean "no reasoning happened" — z0 is CODI's own trained question encoding, shaped
  by the whole training objective, not raw text. It means CODI concentrates a lot of transferable,
  Coconut-decodable value information very early, consistent with earlier findings that CODI's
  value is redundant across its vector / not confined to a small DAS subspace
  (`20260927-185312`).
- L4 (110%) exceeding 100% is within noise at n=259 (~3pp per few recipients) but also plausible:
  L4's random donor may on average need less problem-specific downstream computation to identify
  than L2/L3's matched-length siblings.
**Caveats:** n=259, one donor draw per level. z0 is still a MODEL-INTERNAL representation (not raw
text), so this doesn't isolate "question information" as cleanly as the plain-GPT2 control
(A2) does — see that run for the cleaner confound test. own_carriers here (0.270/0.313/0.351)
reads a few points lower than the L2/L3 values quoted in round-3/4 notes for the same conditions
(there rounded to "0.30" for all three levels); worth a byte-level dump diff if this resurfaces,
but the comparator this experiment cares about (mapped_all from the unrestricted run) was read
precisely, not rounded, so the ratios above stand.
**Next:** See `20260927-224642` (A2, plain GPT-2) and the reverse direction
(`20260927-225629`/`-225751`/`-225810`) for the full 2-direction x 3-condition picture; Experiment B
(`das_ladder_codi.py`/`das_ladder_coconut.py`) asks the complementary basis question.
