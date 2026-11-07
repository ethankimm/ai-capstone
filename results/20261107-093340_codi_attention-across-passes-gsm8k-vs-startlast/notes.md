## 2026-11-07 — Attention across latent passes: GSM8K walks forward through the problem, more on longer chains, and stops by the last passes; start-last and the relay freeze after the first latent (codi, run_id: 20261107-093340_codi_attention-across-passes-gsm8k-vs-startlast)

**Goal:** Why does start-last CODI not roll out? One candidate is that nothing tells a pass which rule comes next. GSM8K CODI's harder items settle at later passes, so does its attention move through the problem pass by pass, while start-last's does not?
**Mechanism / model:** `codi`, gpt2, 6 latents. Released GSM8K CODI (`zen-E/CODI-gpt2`), start-last CODI checkpoint-17820, lr-3e-3 CODI as the relay. fp32, MPS, batch 1, so no padding.
**Data:** GSM8K test items with at least 2 steps whose numbers appear in the question: 580 / 236 / 57 / 6 items with 2 / 3 / 4 / 5 mapped steps (5 not reported). RuleChain valid L3-L6 (relay L4-L5), about 286 per level.
**Command:** `codi_audit_artifacts/b24_pointer.py`, tables from `a27_pointer.py` and `a28_pointer_passes.py`. Rows: `<bot>` then passes p0-p5. RuleChain sets are rule tokens; GSM8K sets are question numbers first used in step j. Position 0 is never counted. Centre of mass (COM) runs 0 (first step) to 1 (last).
**Headline results:**
- **Move in COM, p1 to p3, all heads.** GSM8K +.009 / +.028 / +.066 at 2 / 3 / 4 mapped steps. Start-last +.006 / +.005 / +.003 / +.001 at L3-L6. Relay -.003 / -.004.
- **Same, 20 held-out heads.** GSM8K +.036 / +.069 / +.170. Start-last +.022 / +.023 / +.019 / +.015. Relay +.006 / +.005.
- **p3 to p5.** GSM8K +.002 / +.002 / -.005, so the walk has stopped.
- **Mean change in a head's COM, p1 to p3.** GSM8K .10-.12, start-last .01, relay .01.
- **Alternation.** Attention on step tokens at `<bot>`, p1, p3, p5 over p0, p2, p4: GSM8K 2.3-2.6x, start-last .94-1.06x, relay .96-.97x. Those GSM8K passes produce z_0, z_2, z_4 and z_6; z_0, z_2 and z_4 are the sites that carry values under aligned patching.
- **Shares by step, GSM8K 3 steps.** `<bot>` .51 / .25 / .23, p1 .38 / .34 / .27, p3 .38 / .29 / .32, p5 .37 / .30 / .32. **Start-last L4 by rule:** `<bot>` .28 / .23 / .19 / .30, p0 .14 / .12 / .23 / .51, p5 .11 / .11 / .22 / .55. The relay's latents look the same as start-last's.

**Related re-analyses, no new model runs:**
- **Pass sweep by step count**, truncation logged in 20260927-005051, script `codi_audit_artifacts/gsm8k_pass_sweep_by_steps.py`. Accuracy after k = 1-6 latents: 2 steps .37 / .50 / .63 / .63 / .64 / .65; 3 steps .25 / .15 / .43 / .45 / .48 / .49; 4 steps .20 / .11 / .19 / .19 / .29 / .31; 5 steps .10 / .06 / .07 / .05 / .14 / .14. Mean pass where right answers settle: 1.8 / 2.9 / 3.7 / 4.3. Join on `idx`: `load_gsm8k_aug(split="test")` shuffles by default.
- **Twin patches by edited step**, from 20260927-004340, chains of 3+ steps, 61 first-step and 136 later edits. Single z_0 follows the twin .49 against .18 (recipient-bootstrap difference +.21 to +.41), z_2 .15 against .29, z_4 .41 against .46. Post hoc, one checkpoint.

**Interpretation:**
- GSM8K CODI has pointer-like behaviour that start-last lacks. It grows with chain length and stops where accuracy stops rising.
- This does not show that a missing pointer causes the fixed point. Near-copy latents must give near-identical attention, so start-last's freeze follows from the fixed point either way. Only an intervention can separate them.
- GSM8K's three reading passes, the sites that carry values and the jumps in accuracy line up, from three separate measurements.

**Gotchas hit:** My first summary, a slope over all rows, was dominated by the jump from `<bot>` to the first latent, which even the relay has, so it was dropped for same-parity moves between latent passes. The tokenizer here has no offsets, so they come from token bytes, checked against question length. The pass sweep first ran on a misaligned join (1,303 gold mismatches); it was discarded and rerun, with 0 mismatches.
**Caveats:** Attention is not use. GSM8K sets cover digits in the question only, and questions often give numbers in solving order, so position and step are confounded. RuleChain sets are whole rules, so the two tasks are compared on shape, not size. Released model only; 57 items at 4 mapped steps.
**Next:** Rule-gated latents test the cause directly. Kevin could repeat this on his GSM8K checkpoint.
