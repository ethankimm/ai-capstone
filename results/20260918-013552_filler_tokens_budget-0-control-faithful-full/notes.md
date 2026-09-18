## 2026-09-18 — Paper-faithful control, full scale: 50/50 CoT/filler mix, no filler tokens (filler_tokens, run_id: 20260918-013552_filler_tokens_budget-0-control-faithful-full)

**Goal:** Full-scale (`--train-n -1`, entire 384,620-example train split) counterpart
to the faithful pilot control
[20260918-010048_filler_tokens_budget-0-control-faithful-pilot](../20260918-010048_filler_tokens_budget-0-control-faithful-pilot/notes.md)
— does the faithful control's pilot-scale lead over faithful filler
(5.0% vs 2.5%) hold, narrow, or reverse once trained on the full dataset? See sibling
run [20260918-045031_filler_tokens_budget-32-faithful-full](../20260918-045031_filler_tokens_budget-32-faithful-full/notes.md)
for the `compute_steps=32` arm and the full interpretation.
**Mechanism / model:** `filler_tokens`, gpt2 / results/20260918-013552_filler_tokens_budget-0-control-faithful-full/ckpt, compute_steps=0.
**Data:** gsm8k-aug test n=200 seed=0; run seed=42. Train: full split, n=384620, 50%
real CoT rationales / 50% direct question→answer (zero filler splice, `compute_steps=0`).
**Hyperparams:** {'lr': 5e-05, 'epochs': 3, 'batch_size': 16, 'filler_token': '.', 'train_n': 384620, 'faithful': True, 'cot_rate': 0.5, 'mask_filler_loss': False}
**Command:** `uv run python scripts/train_filler_tokens.py --compute-steps 0 --train-n -1 --faithful --slug budget-0-control-faithful-full --stage full_run`
**Headline results:** `final_answer_accuracy=0.105`, `unparseable_rate=0.000`, `compute_steps=0`, `train_loss=0.4981`
**Interpretation:** 10.5% (21/200), still ahead of the faithful filler full run (9.5%,
sibling run) — the control-beats-filler direction holds at full scale under
paper-faithful training too, the fourth pair (pilot × full × non-faithful × faithful)
to land in the same direction. See the sibling run's notes for the full "why" writeup
(cross-referencing `20260917-190640_filler_tokens_budget-32-full`'s theory section and
`HANDOFF.md`). One notable wrinkle: both faithful full numbers (10.5%/9.5%) are
*lower* than the non-faithful full numbers (13.0%/12.0%,
`20260917-163049_filler_tokens_budget-0-control-full` /
`20260917-190640_filler_tokens_budget-32-full`) despite faithful training adding real
CoT supervision. Plausible explanation: in the faithful mixture, each epoch only shows
this arm's "half" (either CoT-formatted or filler-formatted) at 50% the frequency a
dedicated single-format run would get — effectively less exposure to any one format
per epoch, even though total training examples are unchanged. Not confirmed, just the
leading hypothesis; worth flagging rather than ignoring.
**Gotchas hit:** `author` field empty in the raw manifest (pod never had
`git config user.name` set) — patched by hand, same as every prior run this session.
Also: the first `rsync` attempt to pull this run's checkpoint back silently died
partway (`Connection reset by peer` mid-transfer, backgrounded shell still reported
exit code 0) — caught by checking `du -sh`/`ls ckpt/` after the "completed"
notification and finding `model.safetensors` missing; re-ran with `--partial` and
verified file presence this time. Same class of gotcha as documented in `HANDOFF.md`
("Results checkpoints are large... verify file sizes... don't just trust that the
command didn't error") — recurred even after being documented once already.
**Caveats:** Single seed, `eval_n=200` — see the full-scale non-faithful runs' notes
for the binomial CI caveat at this base rate (still applies here).
**Next:** See sibling run for next steps (positive-control experiment, explicit_cot
full-scale comparison, etc.) — not repeated here to avoid drift between the two.
