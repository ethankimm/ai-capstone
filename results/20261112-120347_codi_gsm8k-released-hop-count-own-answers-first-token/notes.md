## 2026-11-12 — GSM8K hop count with own answers and first-token scoring (t31 + t32), released CODI and SIM-CODI: usable deep pairs grow 11-13x, hops stay near two at every depth, and answer tokens add no crossings (codi, run_id: 20261112-120347_codi_gsm8k-released-hop-count-own-answers-first-token)

**Goal:** t31: remove t26's selection on correct items on GSM8K. t32: check that generated answer tokens do not carry depth, by scoring only the first answer token, teacher-forced after the answer delimiter.
**Mechanism / model:** gsm8k_codi: hf:zen-E/CODI-gpt2@fd641b3; gsm8k_sim: hf:internlm/SIM_COT-GPT2-CODI. GPT-2 small, fp32 MPS, batch 1.
**Data:** GSM8K-Aug test twins: t22 (1,471 candidates) and t27a long twins (>= 4 steps, 1,305). Qualified (own answers differ): codi 1243 + 1092, sim 1250 + 1148 (t26: 319 + 98, 303 + 89).
**Command:** `QUAL=own FIRST=1 ../codi/.venv/bin/python scripts/necessity/hops.py gsm8k_codi|gsm8k_sim mps <out>`; long twins with `CANDS=t26/gsm8k_long_twins.json`.
**Headline results:**
- Both-right subset reproduces t26 (codi 1.47 vs 1.46, sim 1.54 vs 1.52; long twins identical).
- Mean hops by listed steps s, all qualified: codi s = 2 / 3 / 4 1.40 / 1.73 / 2.04; long twins s = 4-8 1.80 / 1.64 / 1.77 / 1.61 / 1.10. sim 1.59 / 1.98 / 2.26; long 2.02 / 1.93 / 1.93 / 1.69 / 1.75.
- Long twins s = 4, hop shares 0 / 1 / 2 / 3 / 4: codi any-wrong .09 / .32 / .38 / .14 / .06 (both right .01 / .32 / .45 / .22 / 0); sim any-wrong .09 / .31 / .32 / .15 / .06 (both right 0 / .20 / .59 / .19 / .01).
- t32: teacher-forced first token starts the greedy answer on 100% of runs; first-token hops = full-answer hops on .954-.964 of pairs; first-token hops are about .1 lower in every bin.
**Predictions:** t31 any-wrong within +-.30 of both-right at s = 2, 3: codi held (+.20 / -.30), sim missed at s = 2 (+.39). t32 agreement >= .95 held; equal on >= .90 held; by-s within +-.20 held except sim s = 6 (n 20) and codi long s = 7 (n 49).
**Interpretation:** On GSM8K the hop count does not keep rising with depth: about two crossings at s = 4-8 whether or not the model is right. The third crossing (14-15% of s = 4 pairs) is real but rare. Answer tokens do not inflate the count; if anything the full answer needs slightly more refreshes because later digits keep attending to the stale latents.
**Gotchas hit:** none; agreement 1.00.
**Caveats:** released checkpoints only; s is listed steps, not dependency depth (t26 figures use depth); hops is an upper bound.
**Next:** update the paper's GSM8K hop figure with the wrong-answer pairs.
