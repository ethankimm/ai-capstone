## 2026-09-19 — Early termination + ablate-all/ablate-single: CODI's thoughts are load-bearing collectively (−18 pp without them, worse than nothing with content-free ones), the load sits on the *non-decodable* slots, and the budget needed tracks the problem's step count (codi, run_id: 20260919-192228_codi_early-termination-ablate-all)

**Goal:** The question left open by `20260919-090132_codi_ablate-attn` / `20260919-184716_codi_ablate-attn-eval`:
if wiping the best-decoding thought's input costs ≤4 pp and the answer read-out puts 1.8% of its
attention there, what are the six continuous thoughts *for*? CODI's 41-44% vs. the 13.0%
no-scratchpad control says they do something. Three measurements on the released checkpoint:
(i) **early termination** — run only k∈{0..6} thoughts at inference, then eot → answer, on the
full-budget model (this is `early_termination_necessity` from PROPOSAL.md / `manifest.py`);
(ii) **ablate-all** — all six thought inputs replaced by a zero vector or by per-iteration
population means, positions kept; (iii) **ablate-single** — one iteration's input replaced by
its population mean, for each iteration, to see whether any position matters on its own.
If truncation/ablate-all don't hurt, the thoughts are decorative and self-distillation trained
the answer-phase weights; if they do, the chain matters collectively even though no single
decodable position is causally faithful.
**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3` (released checkpoint,
unchanged), 6 continuous thoughts, paper inference protocol (LoRA r=128/α=32, projection
768+LN, greedy), batch 1, **eval mode** (the `build_model` fix from commit `9485853`; the script
asserts no submodule is in training mode, and a 5-example decode-twice check confirmed
byte-identical outputs). New script `scripts/early_termination_codi.py`, reusing
`run_thoughts` / `decode_answer` from `scripts/decode_patch_codi.py` so the intervention mechanic
is the exact one used by the patch/ablation runs.
**Data:** gsm8k-aug test, all 1319 examples for truncation and ablate-all; the shared 200-slice
(`load_gsm8k_aug(split="test", n=200, seed=0)`) for ablate-single (projected everything-on-full
was ~49 min, over the ~40 min budget set for this run). Every condition is also reported on the
200-slice subset. Population means: per-iteration mean of the latent that normally feeds
iteration i (i=1: the bot-position latent; i≥2: iteration i−1's post-projection output) over 300
seeded random test examples (norms 62-68, comparable to individual latents).
**Design:**
- Truncation k: the honest chain is run once, the KV cache snapshotted after each iteration,
  and the answer greedy-decoded from every snapshot (eot fed as the next input, exactly as
  `eval_codi.py` does after `inf_latent_iterations`). Checked at startup to be byte-identical
  to running `run_thoughts(n_iters=k)` from scratch for k∈{0,3,6}. k=0 = question + bot + eot.
- Ablate-all / ablate-single: `run_thoughts(..., override_input_at={i: vec})` — the
  post-projection latent fed *into* iteration i is replaced; all six positions still run.
  `ablate_single_mean_i` therefore removes the content of thought z_{i−1} (i=1: the bot latent
  z_0) at the point it is consumed.
- Metrics: accuracy (Wilson CI), unparseable rate, answer-changed vs. the k=6 baseline, and
  paired McNemar exact on correctness vs. k=6 (b = condition right & baseline wrong, c = the
  reverse). `early_termination_necessity` **:= acc(k=6) − acc(k=0)**, the accuracy the
  full-budget model loses when its scratchpad is truncated to nothing at inference; the whole
  curve is in `metrics.extra.truncation_curve_accuracy_by_k`.
**Command:**
```
cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/early_termination_codi.py \
  --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \
  --slug early-termination-ablate-all --stage full_run --hardware "RunPod RTX A4500 (secure)" \
  --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \
  --lora_r 128 --lora_alpha 32 --lora_init --greedy True \
  --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
  --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \
  --smoke_n 0 --single_n 200 --mean_sample_n 300
```
(pod `14cm635d5aj73b`, EU-RO-1, RTX A4500 secure $0.25/hr — no A5000 stock anywhere at launch;
created 18:31 UTC, `codi_setup.sh` + checkpoint download ~2 min, n=3 smoke test, full run
18:46-19:22 UTC (pass 1: 1924 s, 1.46 s/ex for 7 truncation decodes + 2 ablate-all chains; pass
2: 194 s), results synced and pod terminated 19:25 UTC. ≈55 min ≈ **$0.23** (billing record not
yet posted at write-up time). Code at local commit `9485853`; pod copy has no `.git`, so
`author`/`git.commit` in `manifest.json` are filled by hand.)
**Headline results:** `final_answer_accuracy=0.419` (553/1319 at k=6 — identical count to the
eval-mode batch-1 baseline in `20260919-184323_codi_decode-patch-full-eval`, run on a different
GPU), `unparseable_rate=0.000`, **`early_termination_necessity=0.181`** (0.419 → 0.238).

*Truncation + ablate-all (full test set, n=1319; 200-slice in the last column):*

| condition | accuracy (95% CI) | answer changed | McNemar vs. k=6 (b, c, p) | 200-slice acc |
|---|---|---|---|---|
| k=0 (no thoughts) | 23.8% [21.6, 26.2] | 74.5% | 67 / 306, p=1e-37 | 23.5% |
| k=1 | 24.3% [22.1, 26.7] | 72.7% | 63 / 295, p=5e-37 | 23.5% |
| k=2 | 23.2% [21.0, 25.6] | 64.7% | 37 / 284, p=2e-48 | 21.5% |
| k=3 | 37.2% [34.7, 39.9] | 38.0% | 35 / 97, p=6e-8 | 37.0% |
| k=4 | 36.8% [34.3, 39.5] | 36.1% | 33 / 100, p=5e-9 | 37.0% |
| k=5 | 41.0% [38.4, 43.7] | 10.0% | 9 / 21, p=0.043 | 38.0% |
| **k=6 (baseline)** | **41.9%** [39.3, 44.6] | — | — | 40.0% |
| ablate-all, zero | 23.3% [21.1, 25.6] | 75.5% | 66 / 312, p=2e-39 | 24.0% |
| ablate-all, mean | **10.6%** [9.1, 12.4] | 80.7% | 27 / 440, p=3e-97 | 13.5% |

Reference points: no-scratchpad control (plain GPT-2 fine-tuned answer-only,
`20260917-163049_filler_tokens_budget-0-control-full`) 13.0% on the 200-slice; explicit CoT 34.5%.

*Truncation curve by gold step count (full set):*

| steps | n | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 |
|---|---|---|---|---|---|---|---|---|
| 1 | 65 | .262 | .308 | .369 | .477 | .462 | .446 | .415 |
| 2 | 357 | .339 | .375 | .510 | .630 | .627 | .639 | .653 |
| 3 | 364 | .266 | .250 | **.148** | .434 | .437 | .478 | .486 |
| ≥4 | 515 | .144 | .140 | **.078** | .140 | .134 | .204 | .216 |

*Ablate-single with the population mean (200-slice, n=200; baseline 40.0%):*

| input replaced (iteration i ← its mean) | content removed | decodes to a number? | accuracy | Δ | McNemar (b, c, p) |
|---|---|---|---|---|---|
| i=1 | z₀ (bot-position latent) | no | 29.0% | **−11.0** | 4 / 26, **p=6e-5** |
| i=2 | z₁ | no (odd) | 39.0% | −1.0 | 5 / 7, p=0.77 |
| i=3 | z₂ (best-decoding thought) | **yes** | 36.5% | −3.5 | 5 / 12, p=0.14 |
| i=4 | z₃ | no (odd) | 32.0% | **−8.0** | 5 / 21, **p=0.0025** |
| i=5 | z₄ | **yes** | 37.5% | −2.5 | 7 / 12, p=0.36 |
| i=6 | z₅ | no (odd) | 39.5% | −0.5 | 2 / 3, p=1.0 |

**Interpretation:**
- **The thoughts are load-bearing, collectively.** Removing them costs 18 pp (41.9% → 23.8%,
  p=1e-37); zeroing all six inputs gives exactly the no-thought number (23.3%); feeding six
  on-manifold but content-free mean vectors is *worse than having no thoughts at all* (10.6%,
  below even the 13.0% no-scratchpad control). So "the thoughts are decorative and
  self-distillation just trained the answer-phase weights" is wrong — although the answer-phase
  weights do carry a real chunk: with no thoughts the model still gets 23.8%, 11 pp above the
  answer-only control, so roughly 40% of CODI's advantage over the control is in the weights and
  60% is in the chain.
- **Zero vs. mean is a format story.** With zero inputs (or no thoughts) the model drops the
  "The answer is:" template and emits a bare number on every example (0/1319 use the template
  at k=0 and under zero-ablation) — it "knows" no reasoning happened and falls back to direct
  answering at ~24%. With mean inputs it keeps the template on every example (1319/1319) and
  answers confidently and wrongly: on-manifold, content-free thoughts are actively misleading,
  not ignorable. Template use recovers with k (3.7% at k=1, 67.7% at k=2, 99.5% at k=3) while
  accuracy at k=2 is still at the floor, so the k≤2 collapse is not just a format artifact. This is the CODI counterpart of recurrent_depth's
  off-manifold sensitivity, with the sign flipped — there, the off-manifold zero was the
  disruptive one; here the on-manifold mean is.
- **The chain does sequential, depth-dependent work.** The truncation curve is stepwise, not
  linear: k≤2 is at the no-thought floor, k=3 recovers ~75% of the gap, k=5-6 the rest. Broken
  down by step count, 2-step problems saturate at k=3 (63%), 3-step problems need k≥3 and gain
  again at k=5, and ≥4-step problems only move at k=5-6 (14% → 22%) — and truncating to k=2
  *hurts* multi-step problems well below their no-thought accuracy (3-step: 26.6% → 14.8%;
  ≥4-step: 14.4% → 7.8%): a half-finished chain is worse than none. The number of thoughts the
  model needs tracks the number of reasoning steps, which is the behavioral signature of a
  chain that actually computes something step by step.
- **But the load is not where the decodable content is.** Per position, the two mean-ablations
  that significantly hurt remove z₀ (the bot latent, −11 pp, p=6e-5) and z₃ (−8 pp, p=0.0025) —
  neither of which decodes to a number under the logit lens (odd iterations score 0.000 in
  every decoding run). Removing z₂ (the strongest decode position, 18% matched top-1) costs
  3.5 pp (p=0.14) and removing z₄ (the other live position) 2.5 pp (p=0.36); z₁ and z₅ cost
  nothing. Together with the causal-patching null on z₂
  (`20260919-184323_codi_decode-patch-full-eval`) and the explicit-CoT positive control
  (`20260919-185332_explicit_cot_patch-positive-control`, where the same intervention is tracked
  ~85% and moves the answer ~99%), this is the sharpest form of the paper's claim so far: **what
  the logit lens reads out of CODI's thoughts is not what the model computes with.** The
  content that matters sits at the "placeholder" positions the paper's own case study called
  "seemingly meaningless"; the content that reads out as intermediate values is, at the single
  position level, close to dispensable.
- **Caveat on that reading (see Caveats):** ablate-single is n=200 with ±7 pp CIs, and every
  single-position effect is small next to the collective 18 pp — the chain is redundant, with
  no single position necessary. The z₀/z₃ vs. z₂/z₄ contrast is a difference between
  significant and non-significant effects at n=200, not a demonstrated difference between
  effects; it is a hypothesis to confirm at full n before it goes in the paper as a claim.
**Gotchas hit:**
- No RTX A5000 stock in any secure data centre at launch; used an RTX A4500 (20 GB, $0.25/hr)
  in EU-RO-1. Slightly slower per forward than the A5000 runs (1.46 s/ex for 9 conditions);
  results are deterministic across the two GPU types (553/1319 at k=6 here and in the A5000
  eval-mode re-run).
- The `runpod/pytorch:1.0.2-cu1281-torch280` image ships no `git`/`rsync`; `apt-get install`
  first, then `codi_setup.sh` (pinned torch 2.7.1+cu126 / transformers 4.52.4 / peft 0.15.2 in
  its own venv) worked unchanged. Whole setup ≈2 min.
- One script bug caught by the smoke test (unpacking `run_thoughts`' 3-tuple as 2), fixed before
  any logged run; the KV-snapshot shortcut for truncation was verified byte-identical to the
  plain path on 5 examples × k∈{0,3,6} before the full pass.
- `author`/`git.commit` filled by hand (git-less pod copy), same as every other pod run.
**Caveats:**
- Ablate-single is on the 200-slice only (n=200, Wilson CIs ±6-7 pp); the truncation and
  ablate-all numbers are full-set (n=1319). The per-position contrast above is the part that
  most needs the full-set rerun (~18 extra minutes, ~$0.08).
- Truncation at k<6 is off-distribution for a model trained with exactly 6 thoughts then eot
  (as is any early-termination measurement on a fixed-budget model); the format collapse at
  k≤2 shows the model notices. The step-count dependence is nevertheless the cleanest evidence
  in the repo that the chain's *length* does work, independent of any decoding claim.
- `ablate_single_mean_i` replaces the input to iteration i, so the KV entries at position i and
  every downstream thought are recomputed — "removing z_{i−1}" is exact, but the effect size
  includes downstream propagation, not just position i's own contribution.
- Step-count buckets use gold `intermediate_values` counts; 18 examples have zero parsed steps
  and are omitted from the by-step table (present in all headline numbers).
- Single seed / greedy; deterministic, so no seed variance to report.
**Next:**
- Rerun ablate-single on the full test set (cheap) to confirm or dissolve the z₀/z₃-vs-z₂/z₄
  contrast at n=1319 before citing it.
- Pairwise ablations (e.g. z₂+z₄ together, z₁+z₃+z₅ together) to test the redundancy reading
  directly: if removing both decodable thoughts costs ~5 pp and removing the three
  non-decodable ones costs ~15 pp, the "content is at the placeholders" story is confirmed.
- Truncation *with the eot slot kept in place* (feed eot at position 7 after k honest thoughts
  and 6−k mean thoughts) would separate "chain too short" from "eot arrives in an unexpected
  slot" in the k≤2 collapse.
- For the writeup: pair the by-step truncation table with the causal-patching null and the
  positive control — that triple is the paper's argument in one figure.
