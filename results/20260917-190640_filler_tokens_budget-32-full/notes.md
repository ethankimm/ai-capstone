## 2026-09-17 — Full-scale: filler_tokens compute_steps=32 (filler_tokens, run_id: 20260917-190640_filler_tokens_budget-32-full)

**Goal:** Full-scale counterpart to the compute_steps=0 full run — the config's
default nonzero sweep point (32 forced filler tokens), trained on the entire
384,620-example split to get a number actually comparable to Pfau, Merrill & Bowman
2024's own setting.
**Mechanism / model:** `filler_tokens`, gpt2 / results/20260917-190640_filler_tokens_budget-32-full/ckpt, compute_steps=32.
**Data:** gsm8k-aug test n=200 seed=0; run seed=42.
**Hyperparams:** {'lr': 5e-05, 'epochs': 3, 'batch_size': 16, 'filler_token': '.', 'train_n': 384620}
**Command:** `uv run python scripts/train_filler_tokens.py --compute-steps 32 --train-n -1 --stage full_run --hardware "RunPod RTX 2000 Ada (secure)" --slug budget-32-full`
**Headline results:** `final_answer_accuracy=0.120`, `unparseable_rate=0.000`, `compute_steps=32`
**Interpretation:** 12.0% (24/200 correct), essentially tied with the no-filler
control's 13.0% (26/200) from `20260917-163049_filler_tokens_budget-0-control-full`
run at the same full scale. At the pilot scale (20k examples) filler tokens appeared
to help (2.5% vs. 1.0%), but that ordering does **not** hold once trained on the full
dataset — if anything the control is marginally ahead. Binomial 95% CI at this base
rate (n=200) is roughly ±4.5pp, so the 1pp gap is well within noise either direction.
**Honest conclusion: filler tokens show no measurable benefit over no scratchpad at
all on GSM8K-Aug at this scale — a null result, not a negative one.** This actually
matches Pfau et al.'s own caveat (see `latentreasoning/mechanisms/filler_tokens.py`'s
docstring): they found filler tokens mainly help under dense/parallel synthetic-task
supervision, and a null result on a real reasoning benchmark like GSM8K is plausible
and expected, not a sign anything is broken. Per `CLAUDE.md`, we plan for null
findings — this is one.
**Gotchas hit:** Same Secure Cloud move as the compute_steps=0 full run (see its
notes). Took ~3h28m to train (72,117 steps, ~5.8 it/s average — noticeably slower per
step than compute_steps=0's ~8.3 it/s, since every batch now carries 32 extra filler
positions through the forward/backward pass).
**Caveats:** Single seed, single eval_n=200 per condition — not enough to rule out a
*small* real effect in either direction, just enough to rule out the pilot's
apparently large one. `unparseable_rate=0.000` for both full runs, so the null result
isn't hiding behind formatting/extraction failures — it's genuinely about correctness.
**Theory — why the null result is expected, not just plausible (added 2026-09-18):**
Prompted by the question "isn't this exactly what the paper did?", I audited this
implementation against the paper's actual released training code
(`github.com/JacobPfau/fillerTokens`, not just its abstract/prose) and found two
fixable methodology gaps (training-data mixture, filler-span loss masking — see the
faithful pilot runs below) plus several **structural** differences that explain the
persistent null more directly than any training-recipe bug could:

1. **Task class mismatch — the likely dominant factor.** The paper's tasks are a
   synthetic "Match3" problem (`src/match3.py`): find an index triple among rows of
   small integers whose values sum to zero mod `m` — a vector 3SUM. Their code has
   both a parallel/dense variant (`dot_filler_parallel`, any-match search) and a
   serial/instance-adaptive variant (`dot_filler_serial`, `serial_cot`). **On the
   serial variant, their own reported result is that filler tokens fail — stay at
   baseline**, same as no intermediate tokens at all. Their stated mechanism: serial/
   instance-adaptive computation needs an actual intermediate result cached and
   carried forward token-to-token, and filler tokens are content-free, so there's no
   channel to carry that forward. GSM8K-Aug arithmetic chains are exactly this shape
   (each step needs the previous step's specific numeric result — "she had 3, bought
   5, gave away 2..."). **So this null result is a replication of an effect the paper
   already demonstrated on its own synthetic serial task, not a new finding** — what's
   new is that it holds in a different domain (natural language, not symbolic
   strings) and a different model class (pretrained GPT-2, not a from-scratch
   synthetic-task model).
2. **Model.** They train a randomly-initialized tiny Llama (4 layers, hidden=384,
   ~30M params, `misc/llama_d384l4h6.json` + `scripts/run_match3.py`,
   `AutoModelForCausalLM.from_config`, no `from_pretrained`) from scratch, directly on
   the synthetic task. We fine-tune pretrained GPT-2 (124M, 12 layers). Their model
   has zero prior to unlearn; GPT-2 already has a strong pretrained prior for `.`
   specifically (sentence-final punctuation, decimal point) that plausibly resists
   being repurposed as a content-free compute placeholder.
3. **Scale.** Their data-generation default is `--train_samples 1e7`, 5 epochs, batch
   256 — up to ~195k optimizer steps on a narrow, low-diversity task. Our full run is
   ~385k examples, batch 16, 3 epochs ≈ 72k steps, on a much higher-diversity natural-
   language task — comparable or fewer steps on a harder problem, and "filler tokens
   require specific, dense supervision to converge" per their own abstract.
4. **Filler-budget coupling.** Theirs: `filler_length = length**2`, derived per-
   instance from the task's own difficulty parameter. Ours: a fixed `compute_steps=32`
   for every GSM8K problem regardless of whether it needs 1 op or 8+ — no per-example
   scaling.

Full citation trail (exact code excerpts, argparse defaults, the fetched
`misc/llama_d384l4h6.json`) is in `HANDOFF.md`'s "Paper-faithful methodology audit and
rerun" section and `latentreasoning/mechanisms/filler_tokens.py`'s module docstring —
both written 2026-09-18, before `HANDOFF.md` is folded in and deleted per its own
convention. **Recommended write-up framing:** "replicating the paper's own serial-task
failure mode in a new domain and model class," not "reproducing a paper result" (no
paper-reported number exists for GSM8K-Aug/GPT-2 to match) and not "we discovered
filler tokens fail on sequential tasks" (the paper already reported that on its own
synthetic serial task — we're extending it, not originating it). One live gap: we
have not run a positive control (a parallelizable task on GPT-2, the kind filler
tokens *do* help with per the paper) ourselves, so the causal claim about
sequentiality specifically rests on citing the paper's own serial-vs-parallel
comparison, not on an ablation we ran independently.

**Paper-faithful reruns (added 2026-09-18):** fixed the two methodology gaps (50/50
CoT/filler training mixture, unmasked filler-span loss — see
`latentreasoning/mechanisms/filler_tokens.py`'s "Paper-faithful mode" docstring and
`scripts/train_filler_tokens.py --faithful`) and reran at pilot scale:
`20260918-010048_filler_tokens_budget-0-control-faithful-pilot` (5.0%) vs.
`20260918-011303_filler_tokens_budget-32-faithful-pilot` (2.5%) — control still wins,
now even at pilot scale, matching this full-scale run's direction rather than
overturning it. Full-scale faithful pair was launched next; check those runs'
`notes.md` / `results/README.md` for whether it landed and what it showed.

**Next:**
- Run `explicit_cot` at full scale (`--train-n -1`) for a fair three-way comparison —
  right now only the pilot number (4.5%) exists for it.
- If the team wants a real dose-response curve rather than two points, sweep more
  `compute_steps` values (e.g. 8, 16, 64) at full scale.
- Consider a larger `eval_n` or multiple seeds before treating any future
  compute_steps comparison as conclusive at these low base rates.
- If the write-up wants to isolate sequentiality as the causal variable independently
  (not just by citing the paper), the next experiment is a positive-control
  parallelizable/combinatorial task on the same GPT-2 setup — not attempted this
  session.
- CODI: paper number verified from released weights (43.67% on full test, matches
  paper's 43.7% exactly) — see `HANDOFF.md` and
  `results/20260918-021217_codi_released-weights-6lat/` (separate agent's work, not
  detailed further here).
