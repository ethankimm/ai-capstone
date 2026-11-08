## 2026-11-08 — The necessity check on five RuleChain models: only gated and SIM pass (input-only patch carries every rule step at 1.00); start-last, relay and stabilised fail at every single latent. Relay and start-last carry their value only as a group, mostly through attention entries; stabilised carries nothing (codi, run_id: 20261108-073431_codi_v2-l1to7w-route-split-five-models)

**Goal:** Run the paper's proposed necessity check (an input-only patch per latent), with the full route split, on five RuleChain CODI models. Fills the RuleChain rows of the validation table in `HANDOFF_2026-11-08_NECESSITY.md`.
**Mechanism / model:**
- rule-gated start-last CODI, gate on (`20261107-204814`);
- SIM-CODI start-last (`20261107-225603`);
- start-last CODI (`20261106-083843`);
- lr-3e-3 L1-7w CODI with the start value first, the relay (`20261103-193425`);
- stabilised L1-7w CODI, start first (`20261105-044444`).

**Data:** valid L3-L5, 286 / 286 / 286 (285 at L6 not used). Start-last data for gated / SIM / start-last, start-first data for relay / stabilised. Scored on recipients answered right with a usable donor (n per cell in the manifest).
**Command:** On the laptop (MPS, float32, `run_guarded.sh`, `caffeinate`): `~/Documents/Penn/CIS5980/codi_audit_artifacts/t15_rc_route_split.py` (`MODEL=...`), record `record20.py`. Smokes (gated, start-last, L4, 40 items): writing a run's own entries or own input back changed no answer.
**Conditions (site j = latent pass j):**
- both: donor z_j fed in.
- input only: donor z_j fed in, pass j's keys/values reset to the recipient's own.
- entries only: own z_j, pass j's keys/values set to the donor's.
- entries hidden from the answer.

Each per site and for all six sites at once. Donors follow t1's rule (same level, differ at the carried value, predicted answer in range) with equal question token length.

**Headline results** (follows donor; L3 / L4 / L5):

| Model | Rule-step latents: input only | Latent the answer reads: entries only (hidden from answer) | Single sites otherwise | All sites: both / entries |
|---|---|---|---|---|
| gated | 1.00 at every rule step; z0 .49 / .53 / .61 | z_{L-1}: 1.00 (.01-.02) | spares 0 | 1.00 / 1.00 |
| SIM | 1.00 at every trained step; z0 .60-.71 | z_{L-1}: .98 (.00) | spares 0 | 1.00 / .98-.99 |
| start-last | .00-.03 | - | <= .05 everywhere | .46 / .35, .22 / .16, .15 / .13 |
| relay | .00-.01 | - | <= .03 everywhere | .98 / .85, .83 / .75, .57 / .55 |
| stabilised | .00 | - | .00 everywhere | .00-.01 / .00-.01 |

- **Gated and SIM pass the check.** For every latent that applies a rule, the value travels only through the recurrent input (input only = full swap = 1.00, entries only = 0). The last carried value (z_{L-1}) is read by the answer directly from its attention entries. Spare latents after it carry nothing on any route.
- **Start-last and relay fail the check at every latent**, yet both use their latents as a group. Swapping all six moves the answer, and most of that effect survives when only the entries are swapped (relay .85 of .98 at L3, start-last .35 of .46). Pooled values reach the answer mainly through attention.
- **Stabilised CODI uses no route.** Every condition .00-.01, consistent with its ignored latents (`20261105-052233`).

**Predictions (`t_predictions.md`, written before t15 output):** all right. Gated / SIM: input carries rule steps, entries carry the read latent. Start-last: single sites ~0, all sites carry. Relay: single ~0, all high. Stabilised: ~0. The all-sites values for start-last (.46 / .35) were below the predicted ~.9; relay's (.98 / .85) matched.
**Interpretation:** On RuleChain the necessity check separates the models exactly as the regime profiles predicted. It passes only where training gives each pass a job (vertical chain), and fails for every plainly trained model, whether it relays (copies), pools (width) or ignores. The pass/fail line matches the depth result: only gated and SIM reach far past single-pass reach (1.00 at L6 vs CODI-0 .06). This is the first record that runs the paper's headline check end to end on the RuleChain models. With GSM8K CODI (`20261108-061509_..._route-split`: input only .007, entries = full swap) it gives five failing and two passing models so far.
**Gotchas hit:** None. Equal-length donors only, so some recipients lack a donor (n in the manifest).
**Caveats:**
- One seed per model; valid only.
- Patching entries replaces all 12 layers at that position.
- "Input only" at the last site (z5) is 0 by construction (no pass reads it).
- Start-last's all-sites follow rate is bounded by its low clean accuracy.
**Next:** decode at the causal location and the decoded-value donor test (monitor steps 2-3); the cross-dataset rows (ProsQA / ProntoQA, Coconut, Kevin's GSM8K, Ouro) per `HANDOFF_2026-11-08_NECESSITY.md`.
