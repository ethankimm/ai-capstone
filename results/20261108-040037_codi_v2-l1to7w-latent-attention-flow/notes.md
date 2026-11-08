## 2026-11-08 — Attention to earlier latents is not where chaining happens: the known chains (gated, SIM) put only .01-.09 of a working pass's attention output on earlier latents and carry the running value through the latent input; they read back heavily (.2-.5) only in spare passes. Start-last CODI reads back a steady .09-.27, rising smoothly with the pass, spread evenly, like a stronger relay (.02-.18) (codi, run_id: 20261108-040037_codi_v2-l1to7w-latent-attention-flow)

**Goal:** Do start-last CODI's latents build on each other? How much of each latent pass's attention goes to earlier latents vs the question, compared with two models known to chain (gated, SIM) and one known to relay?
**Mechanism / model:**
- start-last CODI (`20261106-083843`);
- rule-gated start-last CODI, gate on (`20261107-204814`);
- SIM-CODI start-last (`20261107-225603`);
- lr-3e-3 CODI with the start value first, the relay (`20261103-193425`).
**Data:** valid L3-L5, 286 items per level and model (relay on its own start-first valid set).
**Command:** On the laptop (MPS, float32, `run_guarded.sh`, `caffeinate`): `~/Documents/Penn/CIS5980/codi_audit_artifacts/t9_attn_flow.py`, tables `t9_table.py` (`t9_table.md`), record `record16.py`. The rows in `predictions.jsonl` are model x level aggregates: this run has no per-item answers.
**Measures:** for the query of latent pass s (input z_s), every layer:
- raw attention, mean over heads;
- a norm-weighted share, ||sum_h a_hk v_hk W_O,h|| (Kobayashi et al. 2020), with W_O taken from the c_proj module so LoRA is included.

Key sets: rules, start sentence, rest of the question, earlier latent passes (also by distance), self, attention sink. Spans are b21's. Shares are reported with the sink removed. The sink takes .38-.60 of raw attention but only .02-.11 of the norm.

**Headline results** (norm share on earlier latents, non-sink, mean over layers, by pass 0-5):

| Model | L3 | L4 | L5 |
|---|---|---|---|
| start-last CODI | 0 / .13 / .19 / .22 / .24 / .27 | 0 / .11 / .16 / .19 / .21 / .23 | 0 / .09 / .14 / .17 / .19 / .21 |
| gated (rule passes 0..L-2) | 0 / .02 / **.22 / .40 / .44 / .47** | 0 / .02 / .04 / **.29 / .47 / .51** | 0 / .01 / .04 / .05 / **.35 / .51** |
| SIM (trained steps 0..L-2) | 0 / .04 / .08 / **.19 / .24 / .28** | 0 / .04 / .05 / .09 / **.22 / .27** | 0 / .04 / .04 / .09 / .11 / **.26** |
| relay | 0 / .03 / .10 / .13 / .16 / .18 | 0 / .02 / .08 / .11 / .13 / .15 | 0 / .02 / .06 / .09 / .11 / .13 |

(bold: passes from L-1 on, after the last rule-reading pass.)
- **The chains barely look back while they compute.** In gated and SIM, passes that read a rule put .01-.09 of their attention output on earlier latents; the rest goes to the question (the rule) and to the pass itself. The running value arrives through the latent input z_s; earlier records show that one swapped z_j moves the answer .98-1.00.
- **They look back once the work is done.** From the pass whose input carries v_{L-1} onward, gated reads earlier latents .29-.51 and SIM .19-.28: spare passes copy the value. These are the "readable, not used" latents of `20261108-012524`.
- **Start-last looks back steadily, with no break.** .09-.13 at pass 1, rising smoothly to .21-.27 at pass 5, about 1.5x the relay at every pass, and never as low as a chain's working pass. The look-back is spread evenly over earlier passes; the largest single share goes to pass 0 (`<bot>`'s projection: .34-.64 of the look-back). The chains' working passes prefer the immediately previous latent (.76-.77 of the look-back at d = 1, pass 2, L4).
- **Raw attention shows the same ordering.** Start-last .12-.30 at L4, relay .02-.11, gated and SIM working passes .01-.03.

**Predictions (`t_predictions.md`, written before t9 output):**
- *Norm share similar to raw for start-last, far above the relay:* similar to raw, right. Above the relay by about 1.5x, not "far".
- *Chains weight the immediately previous latent more, start-last spread:* right.

**Interpretation:** Attention to earlier latents is a poor sign of chaining. The two models that chain by construction carry their running value through the latent input and hardly attend back while computing; heavy look-back marks spare passes copying a finished value. Start-last CODI's look-back is moderate and uniform, and it does not carry content: a donor's first one to three latents are overwritten (`20261108-032531_..._prefix-suffix-pooling`), and its single-latent swaps never move the answer (.00, against .98-1.00 in the chains).

So do start-last latents build on each other?
- Weakly, and not as a chain. Each pass re-derives a value mostly from the question.
- It takes some input from the earlier passes, which is causally useful at L2-L3 (b21 blocking: .30 / .31 of answers change, net accuracy cost .30 at L2 and .07 at L3).
- The answer then pools the passes.

The passes are better described as repeated, partly informed reads of the question than as steps.
**Gotchas hit:** Gated passes that read no rule (pass >= L-1) show 0 attention on rules by construction of the gate; the shares are correct, not a bug.
**Caveats:**
- Descriptive. The norm measure ignores the MLP and later layers, and attention shares are not causal.
- One seed per model.
- The relay is on start-first data, so its question composition differs.
**Next:** These descriptive results agree with the causal ones (t6 prefix swaps, b21 blocking). A direct causal comparison would be to block the look-back in gated and SIM and check that their accuracy holds while the latent input carries the value (not run).
