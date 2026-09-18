## 2026-09-18 — Paper-faithful filler, full scale: 50/50 CoT/filler mix, unmasked filler loss, compute_steps=32 (filler_tokens, run_id: 20260918-045031_filler_tokens_budget-32-faithful-full)

**Goal:** Full-scale (`--train-n -1`, entire 384,620-example train split) counterpart
to the faithful pilot filler run
[20260918-011303_filler_tokens_budget-32-faithful-pilot](../20260918-011303_filler_tokens_budget-32-faithful-pilot/notes.md)
— the last of four filler_tokens data points (pilot × full, non-faithful × faithful)
needed to see whether paper-faithful training methodology changes the null result at
full scale. Sibling control run:
[20260918-013552_filler_tokens_budget-0-control-faithful-full](../20260918-013552_filler_tokens_budget-0-control-faithful-full/notes.md).
**Mechanism / model:** `filler_tokens`, gpt2 / results/20260918-045031_filler_tokens_budget-32-faithful-full/ckpt, compute_steps=32.
**Data:** gsm8k-aug test n=200 seed=0; run seed=42. Train: full split, n=384620, 50%
real CoT rationales / 50% filler-formatted with 32 forced filler tokens and unmasked
filler-span loss.
**Hyperparams:** {'lr': 5e-05, 'epochs': 3, 'batch_size': 16, 'filler_token': '.', 'train_n': 384620, 'faithful': True, 'cot_rate': 0.5, 'mask_filler_loss': False}
**Command:** `uv run python scripts/train_filler_tokens.py --compute-steps 32 --train-n -1 --faithful --slug budget-32-faithful-full --stage full_run`
**Headline results:** `final_answer_accuracy=0.095`, `unparseable_rate=0.000`, `compute_steps=32`, `train_loss=0.2466`

**Interpretation — the complete picture across all four filler_tokens conditions:**

| scale | methodology | control | filler (32) | control wins? |
|---|---|---|---|---|
| pilot (20k) | non-faithful | 1.0% | 2.5% | no |
| full (385k) | non-faithful | 13.0% | 12.0% | yes (marginal) |
| pilot (20k) | faithful | 5.0% | 2.5% | yes |
| full (385k) | faithful | **10.5%** | **9.5%** | yes |

Three of four conditions have the control ahead; the sole exception is the original
non-faithful pilot, which is exactly the smallest/noisiest data point and the one most
plausibly explained by chance (n=200, single-digit correct counts) rather than a real
effect. Paper-faithful training does not rescue filler tokens at any scale tested.
**This is not a new discovery — it replicates a failure mode the paper's own reference
code (`github.com/JacobPfau/fillerTokens`) already reports on its serial/instance-
adaptive Match3 task variant**: filler tokens stay at baseline there too, because
serial/instance-adaptive computation needs an actual intermediate value cached and
carried forward token-to-token, and filler tokens are content-free, so there's no
channel to carry that forward. GSM8K-Aug arithmetic chains need exactly that (each
step depends on the previous step's specific numeric result). What's new here is that
this holds outside their synthetic/from-scratch setting — natural-language word
problems, a pretrained 124M-param GPT-2 instead of their from-scratch ~30M-param
model. Full citation trail, structural differences not controlled for (task class,
model, scale, filler-budget-to-difficulty coupling), and recommended write-up framing
are in `20260917-190640_filler_tokens_budget-32-full`'s notes and `HANDOFF.md`'s
"Paper-faithful methodology audit and rerun" section — this note intentionally doesn't
duplicate that content, just points to it.

One open wrinkle, not fully explained: both faithful full numbers (10.5%/9.5%) are
*lower* in absolute terms than the non-faithful full numbers (13.0%/12.0%), despite
faithful training adding real CoT supervision that the non-faithful runs never see.
Leading hypothesis (unconfirmed): the faithful mixture halves each format's effective
per-epoch exposure (only 50% of examples in any given epoch are the format that
matters for a given arm), so it may need more epochs to reach parity with a
single-format-only run of the same total size — not tested.
**Gotchas hit:** Same `author`-field patch and same partial-rsync-silently-reported-
success issue as the sibling control run (see its notes for detail) — that one hit the
checkpoint transfer, this run's transfer completed cleanly.
**Caveats:** Single seed, `eval_n=200` per condition throughout — not powered to
detect a small true effect, only large ones, at any of the four (scale × methodology)
cells above.
**Next:**
- Run `explicit_cot` at full scale for the three-way comparison (separate
  in-progress work, not this session — check `results/README.md`).
- The positive-control experiment (a parallelizable/combinatorial task on GPT-2,
  matched in domain/format to a sequential variant — "graph reasoning" per
  `PROPOSAL.md`'s Oct 30 sweep candidates is a good fit) is the next step if the
  write-up wants to isolate sequentiality as the causal variable independently rather
  than relying on citing the paper's own serial-vs-parallel comparison.
- If anyone wants to chase the faithful-vs-non-faithful absolute-accuracy gap above,
  an epoch-matched-per-format ablation would test the "halved exposure" hypothesis.
