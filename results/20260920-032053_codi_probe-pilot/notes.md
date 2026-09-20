## 2026-09-20 — CODI: linear ridge + 2-layer MLP probes on z0..z5 vs logit lens (basis-drift check) (codi, run_id: 20260920-032053_codi_probe-pilot)

**Goal:** Test whether the CODI logit-lens null on the non-decodable continuous-thought
positions (z0/z3, load-bearing under mean-ablation per
`20260919-192228_codi_early-termination-ablate-all`, p=6e-5 / p=0.0025) is a basis
artifact rather than genuine non-representation. If the intermediate arithmetic values
are encoded in a rotated subspace of `z`, a closed-form ridge regression or small MLP
reading the raw 768-d vector (no `W_U` unembedding basis involved) should recover them
even where the logit lens reads nothing.

**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3` (released checkpoint,
same one used in every other logged CODI run), compute_steps=6, paper inference protocol
(LoRA r=128/α=32, projection 768+LN, greedy).

**Data:** Probes fit on `gsm8k-aug` `train` (n=3000, `train_seed=0`), reported on the
fixed `validation` split (n=300, seed=0, the same 1000-example held-out set everyone
uses — disjoint from `train` and untouched `test`). Logit-lens accuracy computed on the
identical 300 validation examples in the same run (same decode call, same `num_match`
scorer as `decode_patch_codi.py`), so probe vs. lens is apples-to-apples.

**Hyperparams:** ridge_lambda=1.0, mlp_hidden=64 (2-layer, ReLU, ~50k params),
mlp_epochs=200, mlp_lr=1e-3, tol=0.01 (predictions within 1% relative error of gold
count as a probe hit), max_step=3 (per-step n shrinks: 299/252/147 examples have a
1st/2nd/3rd calculator step). Target transform: signed-log1p, inverted before the
tolerance check. Features standardized (mean/std from train) before both ridge and MLP.

**Command:**
```
cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/probe_codi.py \
    --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \
    --slug probe-pilot --stage pilot --hardware "RunPod RTX A4500 (secure)" \
    --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \
    --lora_r 128 --lora_alpha 32 --lora_init --greedy True \
    --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
    --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \
    --output_dir /tmp/unused --train_n 3000 --eval_n 300
```
(pod `vtdo237s362exw`, EU-RO-1, RTX A4500 secure $0.25/hr — A5000 and A6000 both showed
zero stock on SECURE/COMMUNITY when checked twice ~1 min apart just before provisioning;
A4500 was the nearest available substitute and is the same GPU
`20260919-184323_codi_decode-patch-full-eval` already validated for this exact
checkpoint/venv. Setup: `codi_setup.sh` clone+patch+pinned venv build (~1 min, cached
wheels), checkpoint `snapshot_download` (~3s, 3 files), a throwaway smoke test at
`--train_n 40 --eval_n 20` to catch errors before spending on the real pilot (ran clean,
deleted, not logged per the n<10 exception's spirit — it was a code-path check, not a
result). Real pilot: feature extraction 281s (train, 3000 examples) + 27s (eval, 300
examples) — one `run_thoughts` forward pass per example, all 6 iterations cached at
once — then 18 (iteration, step) probe fits (ridge closed-form + 200-epoch MLP each),
negligible additional time. Total pod uptime 768s (~12.8 min) ≈ **$0.053**. Pod
terminated immediately after rsyncing results back and confirming they landed locally.)

**Headline results** (`metrics.extra`, tol=0.01 relative-error hit rate):

| | avg ridge acc | avg MLP acc | avg logit-lens top-1 |
|---|---|---|---|
| **non-decodable iters {1,4}** (z0,z3) | 0.0059 | 0.0053 | 0.1158 |
| **decodable iters {3,5}** (z2,z4, per plan's 0-indexed naming) | 0.0048 | 0.0059 | 0.0000 |

Per-(iteration,step) detail in `manifest.json`/`predictions.jsonl`. Ridge and MLP are
both at floor everywhere: 0.0–1.6% across all 18 (iteration, step) cells, no position
stands out, "decodable" vs "non-decodable" makes no visible difference to either probe.

**Interpretation:**
- **Basis-drift hypothesis is not supported.** If the non-decodable slots' arithmetic
  were legible in a rotated subspace, ridge/MLP reading the raw 768-d vector should beat
  the logit-lens floor at iterations 1 and 4 (z0/z3) specifically — instead both probes
  sit at 0.4–1.3% there, statistically indistinguishable from their own floor everywhere
  else (Wilson CIs in `manifest.json` all include each other). The result is consistent
  with the earlier reading: z0/z3 are load-bearing (per the ablation run) but not
  representing the intermediate calculator values in any form this pilot can recover —
  genuine non-representation (or at least non-linear/non-shallow-recoverable
  representation) rather than a logit-lens basis artifact.
- **A labeling note that matters for how to read the table above:** by *logit-lens*
  accuracy in this same run, the empirically "live" iterations are 2/4/6 (27–35% top-1
  at step 1) and the empirically "dead" ones are 1/3/5 (0% top-1 everywhere) — matching
  the FULL_POPULATION decoding matrix from `20260919-184323_codi_decode-patch-full-eval`.
  The script's `nondecodable_iters=(1,4)` / `decodable_iters=(3,5)` labels come from the
  plan's zero-indexed z-naming (z0=iter1, z3=iter4 are the ablation-load-bearing slots;
  z2=iter3, z4=iter5 are their paired comparison group), not from this run's own
  logit-lens numbers — so "decodable iters {3,5}" in the headline table is *not* the same
  set as "iterations where the logit lens actually decodes" (that would be {2,4,6}, i.e.
  the table's "non-decodable" set already includes iteration 4, one of the two iterations
  where logit lens gets ~28-29%). Read the averaged table as testing the ablation-derived
  z0/z3-vs-z2/z4 grouping, not a decodability-sorted grouping.
- **A sharper, possibly more informative fact than the averaged headline:** at iteration
  4 specifically, logit-lens top-1 is 27–29% (steps 1–2) while ridge/MLP sit at 0.4–1.3%
  on the exact same eval examples. Ridge regression on the full unrestricted 768-d vector
  should in principle be at least as capable as a restricted linear read through `W_U`
  (logit lens), so probe << logit-lens at a position where logit-lens clearly succeeds is
  worth flagging as a possible metric-comparability issue rather than a clean "logit lens
  wins" result: `logit_lens_top1` is an exact-string match against a small effective
  vocabulary of plausible integers (`num_match`), while the ridge/MLP hit rate requires a
  *continuous* prediction landing within 1% relative error of the gold value — a much
  stricter target for regression than "did the argmax token match one of a few likely
  small integers". The two metrics are computed on the same rows but are not
  equal-difficulty tasks, so "probe accuracy < logit-lens accuracy" at iteration 4 should
  not by itself be read as "the probe found less than the logit lens did" — it may just
  be a harder-to-hit criterion. The floor-level *non-decodable* comparison (iterations 1
  and 4 vs. logit lens' own floor there) is the cleaner test and is the one the
  basis-drift interpretation above rests on.
- **DAS not implemented** (`metrics.extra.das_note`), as scoped — this pilot answers
  "is it linearly/shallowly recoverable at all", not "does a probe-aligned subspace have
  causal control"; that's flagged as explicit future work, not attempted here.

**Gotchas hit:**
- None new. Same pinned CODI venv (torch 2.7.1 / transformers 4.52.4 / peft 0.15.2) and
  checkpoint as every prior CODI run this week; base `runpod/pytorch` image again lacked
  `rsync`/`git` (installed via `apt-get`, same as every other run this session).
- `next_experiments.md`'s "decodable_iters" label is a plan-level 0-indexed z-naming
  convention, not this run's own logit-lens output — flagged above so a future reader
  doesn't misquote the averaged table as "probe accuracy at logit-lens-decodable
  positions".

**Caveats:**
- Single seed (`train_seed=0`, model `seed=11`) for both feature extraction and probe
  fitting; no cross-validation of `ridge_lambda`/`mlp_hidden` — this is a pilot, not a
  tuned probe.
- `tol=0.01` (1% relative error) is a strict, somewhat arbitrary continuous-recovery
  threshold; the near-zero probe accuracy could partly reflect that strictness rather
  than "no information at all" — a looser tolerance sweep (5%, 10%) would help separate
  "no recoverable signal" from "signal exists but not to 1% precision," especially since
  GSM8K intermediate values span a wide range and errors are measured relative to
  magnitude.
- `max_step=3` caps the per-example step count; the per-step n shrinks fast (299 → 252 →
  147) since most GSM8K-Aug problems have 2-3 calculator steps — step-3 numbers are the
  noisiest (widest Wilson CIs).
- Ridge/MLP predict the *pooled per-iteration* z-vector value for a given step in
  isolation; no attempt to combine information across iterations or steps.

**Next:**
- If pursuing this further: loosen `tol` to see whether probe accuracy separates from
  floor at a coarser precision, and/or report R²/MAE on the log-transformed target
  alongside the tolerance hit-rate (a continuous metric doesn't collapse to zero the way
  a hard threshold does, and would better distinguish "no signal" from "some signal, not
  to 1%").
- DAS on the probe-extracted directions (out of scope for this script, per `das_note`) —
  the natural follow-up if a future looser-tolerance or continuous-metric re-run finds
  more signal than this pilot did.
- Companion pilot `#1` in `next_experiments.md` (donor-interchange patching at z0/z3 with
  content-bearing donors) tests the causal-faithfulness side of the same question and can
  run independently of this probe result.
