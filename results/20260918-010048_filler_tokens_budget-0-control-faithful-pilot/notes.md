## 2026-09-18 — Paper-faithful control pilot: 50/50 CoT/filler mix, no filler tokens (filler_tokens, run_id: 20260918-010048_filler_tokens_budget-0-control-faithful-pilot)

**Goal:** Re-run the `compute_steps=0` control under paper-faithful training (see
`latentreasoning/mechanisms/filler_tokens.py`'s "Paper-faithful mode" note) after
auditing against Pfau et al.'s released code (`github.com/JacobPfau/fillerTokens`)
found two deviations in the original simplified implementation: (1) filler-only
training data instead of their 50/50 CoT/filler mixture (`cot_rate=0.5`), and (2)
masked filler-span loss instead of their unmasked scheme. This is the control half of
that faithful pair; see the sibling run
[20260918-011303_filler_tokens_budget-32-faithful-pilot](../20260918-011303_filler_tokens_budget-32-faithful-pilot/notes.md)
for the `compute_steps=32` arm.
**Mechanism / model:** `filler_tokens`, gpt2 / results/20260918-010048_filler_tokens_budget-0-control-faithful-pilot/ckpt, compute_steps=0.
**Data:** gsm8k-aug test n=200 seed=0; run seed=42. Train: pilot subset, n=20000 from train split (same subset draw as the earlier non-faithful pilots, `--seed 42`).
**Hyperparams:** {'lr': 5e-05, 'epochs': 3, 'batch_size': 16, 'filler_token': '.', 'train_n': 20000, 'faithful': True, 'cot_rate': 0.5, 'mask_filler_loss': False}
**Command:** `uv run python scripts/train_filler_tokens.py --compute-steps 0 --train-n 20000 --faithful --slug budget-0-control-faithful-pilot --stage pilot`
**Headline results:** `final_answer_accuracy=0.050`, `unparseable_rate=0.000`, `compute_steps=0`, `train_loss=0.9031`
**Interpretation:** Note that even with `compute_steps=0` (no forced filler tokens), the
`--faithful` flag still applies the 50/50 mixture: half the examples are real
explicit-CoT sequences, half are direct question→answer with zero filler splice. So
this is **not** a no-training control — it's "control conditioned on the same CoT
co-training as the filler arm." Under that matched comparison, this control
(5.0%) beats the faithful compute_steps=32 arm (2.5%, sibling run) at pilot scale — the
same direction as the full-scale non-faithful result (13.0% control vs 12.0% filler,
see `20260917-163049_filler_tokens_budget-0-control-full`). It's also a large jump over
the original non-faithful compute_steps=0 pilot (1.0%,
`20260917-074147_filler_tokens_budget-0-control`) — expected, since that arm had zero
CoT exposure at all while this one gets 50% real rationales in training.
**Gotchas hit:** None beyond what's already documented in `filler_tokens.py` and
`HANDOFF.md`. `author` field was empty in the raw manifest (pod never had
`git config user.name` set) — patched by hand post-hoc, same fix as the prior 5 runs.
**Caveats:** Pilot scale only (n=20000 train, not the full ~385k-example train split) —
same caveat as the earlier pilot-vs-full reversal (filler beat control at pilot scale,
lost at full scale, in the non-faithful runs). Whether the faithful control's lead over
faithful filler holds, narrows, or reverses at full scale is untested — see Task #2 in
the active session tracking (decide with the team whether to spend the ~2.6-3.5h+
budget on a full-scale faithful run).
**Theory (added 2026-09-18):** this control-beats-filler ordering, even under
paper-faithful training, turns out to match the paper's own reported result on its own
serial/instance-adaptive task variant (filler tokens fail there too, staying at
baseline) — see `20260917-190640_filler_tokens_budget-32-full`'s notes for the full
writeup, or `HANDOFF.md`'s "Paper-faithful methodology audit" section.
**Next:** Decided to run this pair at full scale — see
`20260918-*_filler_tokens_budget-*-faithful-full` (or `results/README.md` if those
runs haven't landed as of this reading) for the result.
