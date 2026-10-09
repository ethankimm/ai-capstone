## 2026-10-09 — Coconut Llama-3.2-1B: reproduction gate + E3 twin transplant (coconut, run_id: 20261009-094526_coconut_llama1b-minimal-pair-patch)

**Goal:** Same as the paired CODI run (`20261009-102724_codi_llama1b-minimal-pair-patch`): (1) reproduction gate
for the released Llama-3.2-1B Coconut checkpoint; (2) does the GPT-2 E3 twin transplant
(`20260927-003331_coconut_minimal-pair-patch-full`: all-slot 0.827, value in passes 1 and 4) hold on Llama?
Same script, same caps.
**Mechanism / model:** `coconut`, backbone `unsloth/Llama-3.2-1B-Instruct@5a8abab` (tokenizer patched to Meta's
defaults) + `connordilgren/llama32-1b-gsm8k-coconut@49dc864/checkpoint_10`, 6 latent passes.
**Data:** gsm8k-aug test, gold-trace set n=1194 (same filter as the GPT-2 run).
**Hyperparams:** Coconut env pins (torch 2.5.1, transformers 4.46.2), greedy; pair caps 1000 / 300; deltas ±1..3.
**Command:** `eval_command.txt`. Code change: `coconut_common.finish_and_decode` also stops on `<|eot_id|>`,
as the reference `Coconut.generate` does for Llama (GPT-2 behaviour unchanged).
**Headline results:**
- **Gate passes:** base accuracy **0.364** vs 0.357 in Dilgren & Wiegreffe Table 1 (Coconut paper 0.453 used
  a different setup). Loaded with missing=0 unexpected=0.
- Twin transplant, n=232 qualified pairs:

| pass | single | leave-one-out | prefix 0..p | GPT-2 single / LOO |
|---|---|---|---|---|
| 0 | **0.190** | **0.159** | 0.190 | 0.013 / 0.818 |
| 1 | 0.147 | 0.211 | 0.194 | 0.558 / 0.398 |
| 2 | 0.091 | 0.211 | 0.203 | 0.000 / 0.827 |
| 3 | 0.056 | 0.211 | 0.207 | 0.026 / 0.818 |
| 4 | 0.017 | 0.211 | 0.211 | 0.338 / 0.628 |
| 5 | 0.009 | 0.211 | 0.216 | 0.004 / 0.784 |
| **ALL-SLOT** | **0.216** [0.167, 0.273] (changed 0.293) | | | 0.827 [0.773, 0.870] |

- Non-responsive pairs (n=215): all-slot moves the recipient to the twin's wrong answer 0.102 [0.069, 0.150]
  (GPT-2 0.489).
- CFR: 232/447 = 0.519 [0.473, 0.565] (GPT-2 0.559); P(twin correct | original wrong) = 0.082.

**Interpretation:** **The Coconut transplant result does not port.** Replacing all six continuous thoughts with
the twin's changes the answer in only 29% of pairs and steers it to the twin's answer in 22% (GPT-2: 83%).
Whatever steering there is comes almost entirely from pass 0. Passes 1–5 add ~2.5 pp in the prefix curve, and
leaving any of passes 1–5 out costs nothing. The answer is mostly determined by the question tokens, which the
answer positions can still attend to. That fits two readings, which this run cannot yet separate:
(a) Llama-1B Coconut does most of its computation outside the latents (latents mostly unused, a known worry
for Coconut at scale), or (b) the value is in the latents but redundantly re-derivable from the question, so
overwriting them is "corrected". The decisive control is ablation: zero or mean-replace all latents and check
whether accuracy drops. If it barely moves, (a) holds.
This makes the Q1/Q3 picture method-dependent across families: CODI's finding ports (0.84 → 0.69), Coconut's does not (0.83 → 0.22).
**Gotchas hit:** without the `<|eot_id|>` stop, generation runs past the answer turn. The smoke test caught it
only indirectly; checking `finish_and_decode` against the reference generate did.
**Caveats:** single seed, greedy. Different pair sets per model. Before trusting the null, confirm with the ablation
control above. A pass-indexing bug is unlikely, since it is the same code that gives the GPT-2 table and the
checkpoint loads cleanly with gate accuracy matching the paper, but the ablation would also expose one.
**Next:** latent-ablation control on Llama Coconut (cheap: one decode pass, ~6 min); then the Q2 decode control.
