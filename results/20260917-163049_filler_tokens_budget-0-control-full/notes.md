## 2026-09-17 — Full-scale: filler_tokens no-filler control (filler_tokens, run_id: 20260917-163049_filler_tokens_budget-0-control-full)

**Goal:** Full-scale (matching `configs/mechanisms/filler_tokens.yaml` exactly: entire
384,620-example train split, 3 epochs) run of the no-filler control, after the pilot
run (5% of the data) suggested — inconclusively — that filler tokens might help. This
is the number actually comparable to the paper.
**Mechanism / model:** `filler_tokens`, gpt2 / results/20260917-163049_filler_tokens_budget-0-control-full/ckpt, compute_steps=0.
**Data:** gsm8k-aug test n=200 seed=0; run seed=42.
**Hyperparams:** {'lr': 5e-05, 'epochs': 3, 'batch_size': 16, 'filler_token': '.', 'train_n': 384620}
**Command:** `uv run python scripts/train_filler_tokens.py --compute-steps 0 --train-n -1 --stage full_run --hardware "RunPod RTX 2000 Ada (secure)" --slug budget-0-control-full`
**Headline results:** `final_answer_accuracy=0.130`, `unparseable_rate=0.000`, `compute_steps=0`
**Interpretation:** 13.0% at full scale, vs. 1.0% at pilot scale (20k examples) — full
data clearly matters a lot, as expected. Compare against
`20260917-190640_filler_tokens_budget-32-full` (compute_steps=32, full scale: 12.0%)
— the no-filler control is now *slightly ahead* of the filler-token condition, which
reverses the pilot's apparent ordering. See that run's notes for the fuller
interpretation of what this means for the mechanism overall.
**Gotchas hit:** Ran on a **Secure Cloud** pod (RTX 2000 Ada, `EU-RO-1`), not the
Community Cloud A4000 used for the pilots — the community pod had already silently
restarted mid-job once during an unrelated CODI attempt earlier the same day (host
preemption with zero warning/traceback), and this run's ~2.6h+3.5h combined
unattended duration felt too risky to trust on preemptible hardware. Took ~2h36m to
train (72,117 steps, ~8.3 it/s on this GPU tier — slower than the A4000's ~11 it/s).
**Caveats:** Still only `eval_n=200` — see the compute_steps=32 full run's notes for
why the 13.0% vs 12.0% gap isn't statistically distinguishable at this n.
**Theory (added 2026-09-18):** see `20260917-190640_filler_tokens_budget-32-full`'s
notes for the full explanation — this run's control number is the anchor the filler
condition is compared against there.
**Next:** See `20260917-190640_filler_tokens_budget-32-full` for the combined
interpretation and next-steps recommendation.
