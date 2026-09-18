## 2026-09-17 — Pilot: explicit_cot baseline (explicit_cot, run_id: 20260917-075729_explicit_cot_baseline-pilot)

**Goal:** Pilot reproduction of the explicit-CoT baseline (visible calculator-annotated
rationale + answer) — the reference every other latent-reasoning mechanism gets
compared against.
**Mechanism / model:** `explicit_cot`, gpt2 / results/20260917-075729_explicit_cot_baseline-pilot/ckpt, compute_steps=- (unbudgeted).
**Data:** gsm8k-aug test n=200 seed=0; run seed=42.
**Hyperparams:** {'lr': 5e-05, 'epochs': 3, 'batch_size': 16, 'train_n': 20000, 'max_new_tokens': 256}
**Command:** `uv run python scripts/train_explicit_cot.py --train-n 20000 --stage pilot --hardware "RunPod RTX A4000 (community)" --slug baseline-pilot`
**Headline results:** `final_answer_accuracy=0.045`, `unparseable_rate=0.000`, `extra.cot_tokens=34.3`
**Interpretation:** 9/200 correct — clearly ahead of both filler_tokens pilot
conditions at the same scale (control 1.0%, compute_steps=32 2.5%), consistent with
explicit CoT being a stronger mechanism than filler tokens for this task, as expected
going in.
**Gotchas hit:** Depends on the same `load_gsm8k_aug` fixes described in the
filler_tokens pilot runs' notes (pyarrow streaming bug + test-split file format).
**Caveats:** Pilot scale only (20,000/384,620 train examples, ~5%, 3 epochs). No
full-scale explicit_cot run has been done yet, so this number isn't yet on equal
footing with the full-scale filler_tokens runs (`*-full`, trained on the entire
384,620-example split) — don't quote 4.5% against 13.0%/12.0% as a fair comparison.
**Theory (added 2026-09-18):** explicit CoT's advantage over filler_tokens on this task
isn't just "more training signal" — it's the natural contrast case for the theory that
explains filler_tokens' null result (see
`20260917-190640_filler_tokens_budget-32-full`'s notes). GSM8K arithmetic chains are
sequential/instance-adaptive (each step needs the *previous step's specific numeric
result*); explicit CoT tokens carry that content forward explicitly (the rationale
literally writes out each intermediate value), while filler tokens are content-free
and have no channel to relay it. That's consistent with the paper's own reference code
(`github.com/JacobPfau/fillerTokens`) reporting filler tokens fail specifically on
their serial/instance-adaptive task variant while presumably still being comparable to
CoT on their parallel/dense variant (not verified against their parallel numbers this
session — see `HANDOFF.md`).
**Next:** Run explicit_cot at full scale (`--train-n -1`) to match the filler_tokens
full runs before drawing a three-way comparison. (As of this writing, a full-scale
explicit_cot run is in progress on a separate pod — check `results/README.md` for
whether it has landed.)
