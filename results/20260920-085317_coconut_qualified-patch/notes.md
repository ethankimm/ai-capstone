## 2026-09-20 — E2: step-aligned, base-correct, controlled raw single-slot patch at every Coconut pass -- null survives the fixed pair design (coconut, run_id: 20260920-085317_coconut_qualified-patch)

**Goal:** `steered_to_donor_audit.md` E2, Coconut counterpart to `20260920-085206_codi_qualified-patch`.
The earlier Coconut raw-patch pilot (`20260920-031246_coconut_decode-patch-pilot`) picked
donor/recipient pairs by differing final answers only, at whichever pass decoded a step
best (leaving passes 2/3/5 completely untested, `n=0`, since they never won "best" for any
step), with no base-correctness requirement. This run tests EVERY pass with qualified pairs
(recipient AND donor base-correct; recipient's step-k gold value is an operand of step
k+1) and adds a proper mean-ablation control alongside the random-donor control.

**Mechanism / model:** `coconut`, backbone openai-community/gpt2, checkpoint
`hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33`, compute_steps=6 (6 `<|latent|>`
passes).

**Data:** `gsm_valid-gold-reasoning-trace_test.json` (Dilgren & Wiegreffe's gold-trace
data prep, same underlying corpus as our own `gsm8k_aug.py`), n=600 (shuffled, seed=0).
Site→step assignment is POSITIONAL (`site_to_step`), splitting the 6 passes into
`max_step`=7 contiguous groups (pass 0→step 0, pass 1→step 1, pass 2→step 2, pass 3→step
4, passes 4/5→steps 5/6) -- not a decoding-accuracy fit, so every pass gets a target step
regardless of how well it happens to decode.

**Command:**
```
cd /workspace/ai-capstone && .venv_coconut/bin/python scripts/patch_qualified_coconut.py \
  --checkpoint_path <hf checkpoint_33 snapshot path> \
  --data_dir /workspace/coconut_data \
  --slug qualified-patch --stage full_run --hardware "RunPod RTX A6000 (secure)" \
  --num_latents 6 --eval_n 600 --n_pairs_per_site 200
```
Same RunPod RTX A6000 pod as the CODI run above ($0.53/hr, EU-SE-1, kept warm to avoid a
second provisioning cost). Setup: fresh `.venv_coconut` pinned to Coconut's own
`requirements.txt` (torch==2.5.1, transformers==4.46.2, datasets==3.1.0), checkpoint
`hf_hub_download` (~cached), gold-trace data file copied from the local
`are-lrms-easily-interpretable` data prep. A throwaway n=40/3-pair smoke test (logged as
`20260920-085123_coconut_qualified-patch-smoketest`, not a real run — kept only to confirm
the pipeline before spending on the full n=600/200) preceded this. Decode pass (n=600)
took seconds; full patch sweep (6 sites, 282 qualified pairs, 4 forward trajectories/pair)
under 2 min.

**Headline results:** `final_answer_accuracy=0.355` (213/600) — close to paper Table 1's
33.1%. 282 qualified pairs across 6 sites; pool sizes shrink sharply at higher steps
(164/89/28/1/0/0 — passes 4 and 5 had zero qualifying base-correct examples in this slice).

| site (pass) | step | n | real matches_cf | random matches_cf | real answer_changed | random | mean | McNemar real-vs-random (cf) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 164 | 0.018 | 0.012 | 0.628 | 0.622 | 0.348 | b=3,c=2,p=1.0 |
| 1 | 1 | 89 | 0.011 | 0.000 | 0.899 | 0.876 | 0.730 | b=1,c=0,p=1.0 |
| 2 | 2 | 28 | 0.000 | 0.000 | 0.071 | 0.179 | 0.071 | b=0,c=0,p=1.0 |
| 3 | 4 | 1 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | — |
| 4 | 5 | 0 | — | — | — | — | — | — |
| 5 | 6 | 0 | — | — | — | — | — | — |

**Overall (n=282):** `matches_cf` real=**1.42%** [0.55%, 3.59%] vs random=**0.71%**
[0.19%, 2.55%] — same floor, no site's McNemar test approaches significance (b/c small,
p=1.0 everywhere there's any discordance). `matches_donor_final` (Metric A) real=0.71%,
random=0.35%. `answer_changed`: real=66.0%, random=66.0% (identical), mean-ablation=44.0%
(clearly lower, as with CODI — a content-free mean vector perturbs less than a real
donor's activation, confirming the intervention itself is working mechanically even though
it never steers toward donor content). Outcome taxonomy for real: other_number 168,
unchanged 96, recipient_intermediate 9, counterfactual 4, donor_intermediate 4,
donor_final 1 — again, changed answers mostly scramble toward the recipient's own operand
recombinations, not the donor's content.

**Interpretation:** Passes 0 and 1 are the load-bearing/high-`answer_changed` passes
(62.8% and 89.9% — pass 1 patches nearly always change the answer), exactly mirroring the
prior pilot's finding that `answer_changed` tracks decodability/position. But even at
these two passes, with the pair design fully fixed (base-correct, propagation-qualified,
random-donor AND mean-ablation controls), `matches_cf` stays indistinguishable from chance
(1.1-1.8% real vs 0-1.2% random). This directly answers `steered_to_donor_audit.md` §3.2's
open question about the earlier pilot's one uncontrolled lead (pass 1: 5/60, 8.3%,
"a lead, not a result" without a matched control) — with base-correct qualified pairs and
a random-donor control at the same pass, pass 1's `matches_cf` drops to 1/89 (1.1%),
statistically indistinguishable from its own random-donor control (0/89). The earlier
uncontrolled lead does not survive proper controls at scale.

**Gotchas hit:**
- Passes 4 and 5 (steps 5/6 under the positional assignment) had zero qualifying
  base-correct pairs in this 600-example slice — gsm8k-aug problems rarely have 6+ real
  steps, so the highest-indexed steps have almost no qualifying pool regardless of sample
  size; same shape of gap as CODI's sites 4-6.
- Same proxy-SSH-needs-a-PTY issue as the CODI run; used direct SSH throughout.

**Caveats:**
- Same positional (not decoding-fit) site→step caveat as the CODI notes: the per-site
  table is the number that matters, the manifest's pooled `intervention_accuracy` averages
  very unequal site sizes.
- `matches_cf` is undefined for mean-ablation by construction (no injected value to check
  propagation of).

**Next:** Same as the CODI counterpart — `steered_to_donor_audit.md` §5 E3 (same-problem
minimal-pair donors) is the next step if this cross-problem null is worth chasing further.
