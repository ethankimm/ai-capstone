## 2026-11-08 — The chains carry their value through the recurrent input, not attention: cutting every latent pass's look-back to earlier passes leaves gated and SIM at .997-1.000, and a swapped z_j still moves the answer .99-1.00. Start-last CODI loses .30 at L2 to the same cut and never follows a swapped latent: it is not a chain on either route (codi, run_id: 20261108-042557_codi_v2-l1to7w-startlast-latent-routes)

**Goal:** Make the depth-vs-width reading causal. Two routes can carry information between latent passes:
- the recurrent input, where z_s = prj(last layer of pass s-1) enters pass s at layer 0 (vertical: adds 12 layers on top of the previous result);
- attention from pass s to earlier passes' keys and values at the same layer (horizontal: like attending to extra tokens).

`20261108-040037_..._latent-attention-flow` found that the known chains barely attend back while computing. Test: cut the look-back and see whether the chains still work, and whether a swapped latent still propagates.
**Mechanism / model:** start-last CODI (`20261106-083843`); rule-gated start-last CODI with the gate on (`20261107-204814`); SIM-CODI start-last (`20261107-225603`).
**Data:** `synthetic_data_v2_l1to7w_startlast` **valid** L2-L5 (286/286/286/286; L2 swaps n = 285 where one item lacks a donor).
**Command:** On the laptop (MPS, float32, `run_guarded.sh`, `caffeinate`): `~/Documents/Penn/CIS5980/codi_audit_artifacts/t10_routes.py` (copy of `t1_single_use.py`, `MODEL=startlast|gated|sim`; start-last outputs are named `t10_plain.json`), record `record17.py`. Smokes (L4, 40 items): identity patch and the noback=False path both reproduce the clean answers.
**Conditions:**
- noback: latent pass s >= 1 cannot attend to latent passes 0..s-1, for that pass only. The answer tokens still see every pass; the question stays visible, through the gate for gated.
- swap_j, j = 1, 2: z_j from a same-level donor (t1's donors, seeds and ks), later latents recomputed, with and without noback. Scored: follows donor / keeps own, on items answered right.

**Headline results:**

| Model | L | clean | noback | swap_1 follows (noback) | swap_2 follows (noback) |
|---|---|---|---|---|---|
| gated | 2 / 3 / 4 / 5 | .997 / 1 / 1 / 1 | .997 / 1 / 1 / .997 | .996 (.996) / 1 (1) / 1 (1) / 1 (1) | idle at L2 / 1 (1) / 1 (1) / 1 (1) |
| SIM | 2 / 3 / 4 / 5 | 1 / 1 / 1 / 1 | 1 / 1 / 1 / 1 | .993 (.993) / 1 (1) / 1 (1) / 1 (1) | idle at L2 / 1 (1) / 1 (1) / 1 (1) |
| start-last | 2 / 3 / 4 / 5 | .997 / .521 / .329 / .136 | .699 / .455 / .297 / .126 | .00 (.03) / .01 (.00) / .03 (.01) / .08 (.03) | .00 (.01) / .01 (.03) / .04 (.03) / .03 (.03) |

- **Chains: the recurrent input does all the carrying.** Without any look-back, gated and SIM keep .997-1.000. A swapped z_1 or z_2 still makes the answer follow the donor's value .99-1.00. Later passes compute on the swapped value through the input alone. ("Idle at L2": z_2 is a spare latent there, so the swap keeps own 1.000, as in t1.)
- **Start-last: look-back matters, content does not propagate.** noback reproduces b21 exactly (L2 .699; net -.07 / -.03 / -.01 at L3-L5). Swapped latents are followed .00-.08 with or without look-back. Cutting look-back makes swaps more disruptive (keeps own .49-.60 at L2, from 1.00) without making them followed.

**Predictions (`t_predictions.md`):**
- Chains keep accuracy and follow swaps under noback: right.
- Start-last reproduces b21 and follows ~.00: right.
- The originally proposed second test was dropped before running. It would have fed start-last fixed inputs for passes 2-5 while keeping look-back. t6 showed that a start-last pass exposes its input's value to the answer, so fixed inputs would hurt start-last as well and not separate the routes.

**Interpretation:** This is the causal version of depth vs width.
- **Gated and SIM: depth.** Their computation runs through the recurrence: each pass takes the previous result as input and applies one rule on top of it, which stacks layers. Attention back to earlier passes is unnecessary.
- **Start-last CODI: not a chain on either route.** A value placed in its input is not built on but replaced by a fresh estimate from the question (t6 prefix swaps). Its look-back helps on short chains (L2), but it carries no item-specific value forward either.

So start-last's extra passes act as repeated estimates, which is width, plus about a hop of depth that sits within the first two passes and that these tests do not localise further. The forced models use the latents as a recurrence.
**Gotchas hit:** none.
**Caveats:**
- One seed per model; valid only.
- noback removes attention to latent positions at every layer of a pass, a coarse cut.
- The swap targets are model-specific (t1's ks), so "follows" means each model's own trained carrier.
**Next:** The decisive comparison for "unforced latents = width" is a matched no-recurrence control: the start-last CODI recipe with 6 pause positions whose inputs do not depend on the previous pass (learned constant embeddings). If it matches start-last CODI, the latents' gain is positions, not recurrence. Not run; it needs a training run.
