## 2026-09-18 — Paper-faithful filler pilot: 50/50 CoT/filler mix, unmasked filler loss, compute_steps=32 (filler_tokens, run_id: 20260918-011303_filler_tokens_budget-32-faithful-pilot)

**Goal:** Re-run the `compute_steps=32` arm under paper-faithful training (50/50
CoT/filler data mixture + unmasked filler-span loss, matching
`github.com/JacobPfau/fillerTokens`'s reference training code rather than the
simplified default) to test whether the two identified methodological deviations
explain the full-scale null result
(`20260917-163049_filler_tokens_budget-0-control-full` vs
`20260917-190640_filler_tokens_budget-32-full`, 13.0% vs 12.0%). See sibling control run
[20260918-010048_filler_tokens_budget-0-control-faithful-pilot](../20260918-010048_filler_tokens_budget-0-control-faithful-pilot/notes.md).
**Mechanism / model:** `filler_tokens`, gpt2 / results/20260918-011303_filler_tokens_budget-32-faithful-pilot/ckpt, compute_steps=32.
**Data:** gsm8k-aug test n=200 seed=0; run seed=42. Train: pilot subset, n=20000 from train split (`--seed 42`), 50% real CoT rationales / 50% filler-formatted with 32 forced filler tokens and unmasked filler-span loss.
**Hyperparams:** {'lr': 5e-05, 'epochs': 3, 'batch_size': 16, 'filler_token': '.', 'train_n': 20000, 'faithful': True, 'cot_rate': 0.5, 'mask_filler_loss': False}
**Command:** `uv run python scripts/train_filler_tokens.py --compute-steps 32 --train-n 20000 --faithful --slug budget-32-faithful-pilot --stage pilot`
**Headline results:** `final_answer_accuracy=0.025`, `unparseable_rate=0.000`, `compute_steps=32`, `train_loss=0.4595`
**Interpretation:** Paper-faithful training does **not** flip the result at pilot
scale: this arm (2.5%) is flat vs. the original non-faithful compute_steps=32 pilot
(2.5%, `20260917-074839_filler_tokens_budget-32`) and now trails its own faithful
control sibling (5.0%, `20260918-010048_...-control-faithful-pilot`) rather than
beating it — reversing the pilot-scale direction seen in the non-faithful runs (there,
filler 2.5% > control 1.0%) but matching the direction of the full-scale non-faithful
result (control 13.0% > filler 12.0%). Train loss is much lower here (0.4595 vs 0.9031
for the control) because unmasked filler-span loss gives the model 32 extra easy-to-predict
positions per filler example to drive loss down on, which is not directly comparable to
the control's train_loss — don't read the loss gap as "this arm learned more."
**Gotchas hit:** Same `author`-field patch as the sibling run (empty in the raw
manifest, hand-patched to `Henning Lindig`).
**Caveats:** Pilot scale (n=20000), same caveat as sibling control run — non-faithful
filler_tokens reversed from beating control at pilot scale to losing at full scale, so
this pilot result alone should not be read as confirming or refuting the mechanism;
a full-scale faithful run is the natural next step before drawing conclusions for the
write-up. `unparseable_rate=0.000` for both faithful arms at pilot scale (vs. some
unparseable outputs in earlier non-faithful pilots) suggests the CoT half of the
training mixture is teaching the `#### <answer>` format more reliably regardless of
the filler arm — a plausible secondary effect of `--faithful` worth noting if it
recurs at full scale.
**Theory (added 2026-09-18):** see `20260917-190640_filler_tokens_budget-32-full`'s
notes for the full writeup of why this null result is expected — short version: the
paper's own reference code tests a serial/instance-adaptive task variant matching
GSM8K's sequential-arithmetic shape, and reports filler tokens fail on it too
(baseline performance), for the same reason (no channel to carry an intermediate
value forward between content-free filler positions). This pilot result is consistent
with that, not a contradiction of it.
**Next:** Decided to run this pair at full scale (`--train-n -1`) — see
`20260918-*_filler_tokens_budget-*-faithful-full` (or `results/README.md` if those
runs haven't landed as of this reading) for the result.
