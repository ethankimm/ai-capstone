## 2026-11-08 — Released GSM8K CODI stores latent values in attention, not in the recurrence: a twin's value placed only in a latent's keys/values moves the answer as much as a full swap (z4 .35 vs .36), placed only in the recurrent input it does nothing (.007). The first value (z0) reaches the answer mostly through later latents (.17 of .20 with the answer blind to z0); z4 is read directly. Cutting look-back to odd or even sites costs about the same (codi, run_id: 20261108-061509_codi_gsm8k-released-route-split)

**Goal:** Which route carries a GSM8K CODI latent's value: the recurrent input (z_s feeds the next iteration, the vertical route used by gated and SIM on RuleChain, `20261108-042557`) or the latent's own keys and values, read by attention? And is t11's look-back cost about values?
**Mechanism / model:** `zen-E/CODI-gpt2` @ fd641b3, fp32 MPS, batch 1, as t11.
**Data:** the 317 aligned pairs of `20260927-004340`. 314 qualify; 307 have twin and recipient questions of equal token length (needed so keys/values line up by position) and are used for the route split. Look-back controls also on all 1,319 test items.
**Command:** `~/Documents/Penn/CIS5980/codi_audit_artifacts/t12_gsm8k_route_split.py` (CODI venv, `run_guarded.sh`, `caffeinate`), record `record18.py`. Smoke (20 pairs): writing a run's own keys/values back changed no answer.
**Conditions**, site s = the iteration whose input is z_s; its keys/values sit at position Q + s:
- both_s: twin's z_s fed in (t11 single_s).
- input_s: twin's z_s fed in, then that iteration's keys/values reset to the recipient's own. Only the next input z_{s+1} carries the twin's value.
- kv_s: clean input, that iteration's keys/values set to the twin's. Only attention carries the twin's value.
- kv_noans_s: kv_s with the answer tokens masked from position Q + s. The twin's value reaches the answer only through later latents.
- noans_s: clean, answer tokens masked from site s (control).
- Look-back controls: nb_even (each iteration masked from earlier even sites), nb_odd (earlier odd sites), nb_all (t11's cut).

**Headline results** (n = 307; follows twin):

| Site | both | input only | keys/values only | keys/values, answer blind to the site | answer blind to the site, clean: keeps own |
|---|---|---|---|---|---|
| z0 | .199 | .007 | .202 | .169 | .860 |
| z2 | .215 | .007 | .205 | .068 | .951 |
| z4 | .358 | .007 | .349 | .010 | .795 |

- **The recurrent input carries nothing.** In every site the full effect of a swap sits in the latent's keys/values; the input it hands to the next iteration carries none of the twin's value. This is the opposite of gated and SIM, where the input carries everything and look-back is unnecessary.
- **The first value is relayed by later latents, the last is read directly.** With the answer blind to z0, its value still reaches the answer .169 of .202: later latents read it by attention and pass it on. For z2 the share passed on is about a third (.068 of .205); for z4 it is nil (.010), so the answer reads z4 itself.
- **Look-back controls (all 1,319 test items):** clean .425; nb_even .329, nb_odd .321, nb_all .300. Recipients right: .637 / .672 / .567. Cutting look-back to the punctuation sites costs as much as cutting it to the value sites.

**Predictions (`t_predictions.md`):**
- *nb_even costs far more than nb_odd:* **wrong**, they cost the same.
- *kv_noans well above 0 for z0 and z2 (horizontal chain), input_s weaker than in the RuleChain chains:* right for z0, partly for z2. input_s is not weaker but nil.
- *Vertical-chain alternative (input carries it):* ruled out.

**Interpretation:** GSM8K CODI uses its latents as memory slots. Each even latent writes a value into its keys/values, and later latents and the answer fetch it by attention. The recurrent input does not carry values forward at all. The earliest value is read by later latents and relayed to the answer, a horizontal chain for the first step. The latest value is read by the answer directly. This is a third regime, distinct from start-last's interchangeable estimates (width) and from gated / SIM's recurrence (vertical depth): specialised slots linked by attention.

The odd/even result qualifies t11's look-back cost. Odd sites, which hold no value (t11), still matter to later latents. So the look-back cost is not only about values: either the odd sites' keys/values serve as structure or scratch space, or masks the model never saw do part of the damage. The route split does not depend on that cut: it moves one site's keys/values and keeps every mask intact, so it is the clean evidence.
**Gotchas hit:** None. Equal-length pairs only for the keys/values patches (307 of 314).
**Caveats:**
- One released checkpoint.
- Patching keys/values at one position replaces all 12 layers' entries for that position.
- Twin and recipient differ in one number, so their keys/values elsewhere match closely; the patch is minimal.
- "Relayed by later latents" is measured with the answer blind to the site. Later latents might also read the twin's value from the question rather than from z0, but the question is the recipient's, so the value can only come from the patched site.
**Next:** attention shares (t13) and the full-test depth split (t14). Then place GSM8K in the regime table: specialised slots read by attention.
