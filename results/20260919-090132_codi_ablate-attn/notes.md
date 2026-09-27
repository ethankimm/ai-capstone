## 2026-09-19 — Ablation + attention diagnostics: wiping the best-decoding thought's input costs nothing, injecting content hurts, and the answer barely attends to it — with a LoRA-dropout caveat that applies to all three CODI interp runs (codi, run_id: 20260919-090132_codi_ablate-attn)

**Goal:** The CODI mirror of `20260919-085911_recurrent_depth_ablate-attn`, on top of
`20260919-080349_codi_decode-patch-full`'s causal null (real-donor patch ≈ random-control
patch at the focus iteration: 35.7% vs. 37.0% answer-changed, McNemar p≈0.9). (1) Ablation:
does the focus position matter *at all* independent of content — overwrite it with a zero
vector or the population mean instead of a donor's value, paired against the already-known
real/control outcomes on the same recipients. (2) Attention: during honest answer
generation, how much attention mass lands on the focus thought position vs. the question,
the other thoughts, eot, and prior answer tokens — a direct mechanistic check on whether
the model is even reading that position when it answers.
**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3` (released checkpoint,
unchanged), 6 continuous thoughts, paper inference protocol (LoRA r=128/α=32, projection
768+LN, greedy). New script `scripts/ablate_attention_codi.py`, importing the loop helpers
from `scripts/decode_patch_codi.py`.
**Data:** gsm8k-aug test, half B (idx odd) only, same as the prior run. Task 1 replays the
exact 227 focus-iteration (iteration 2) `(recipient, step)` pairs from the prior run's
`patch_pairs.jsonl` in file order — the real-donor / any-control columns below are that run's
own outcomes carried over, not re-measured; only zero/mean are new. **Those 227 pairs cover
152 unique recipients** (102 appear once, 33 twice, 11 three times, 4 four, 2 five — several
steps map to iteration 2, so a recipient recurs once per matched step); the zero/mean
interventions depend only on the recipient, so the duplicated rows are repeat measurements
of the same intervention, and n=227 overstates the independent sample (dedup numbers given
below). Task 2 uses 80 fresh half-B examples (seeded shuffle, honest unpatched generation).
Mean vector: per-dim mean of iteration 2's post-projection output over 300 random half-B
examples.
**Design — Task 1 (ablation):** identical injection mechanic to the patch run
(`run_thoughts(..., override_input_at={2: vec})`): the post-projection latent that would
normally be fed as the input embedding to iteration 2's forward pass is replaced by `vec`,
and iterations 2-6 plus the answer decode continue honestly from there. Same
answer-changed-from-baseline metric, paired per row, McNemar exact. Also recorded: answer
*correctness* under each condition (a cleaner view here — see Caveats), and whether the
read-out one iteration downstream (iteration 3's top-1 token) changed at all.
**Design — Task 2 (attention):** honest generation with `output_attentions=True` during the
answer phase; per generated answer token, attention mass averaged over all 12 layers × 12
heads, bucketed by key position: question+bot (`L0` positions, mean 55, range 23-121), the
focus thought (iteration 2's sequence position — the exact position whose input the
patch/ablation overwrites), the other 5 thought positions, eot, and previously generated
answer tokens. Reported over all 422 generated tokens and for the first answer token alone
(no prior-answer context yet).
**Command:**
```
cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/ablate_attention_codi.py \
  --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \
  --slug ablate-attn --stage full_run --hardware "RunPod RTX A5000 (secure)" \
  --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \
  --lora_r 128 --lora_alpha 32 --lora_init --greedy True \
  --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
  --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \
  --n_attn_examples 80 --mean_sample_n 300
```
(pod `codi-ablate-attn`, RTX A5000 secure $0.27/hr, created ~08:51 UTC; ablation pass 135 s
+ attention pass 12 s, results synced 09:02 UTC. The background agent driving the pod then
stalled and the pod sat idle until it was found and terminated ~11:45 UTC — pod billing for
the day shows **$0.77** for it (`8g1r61p96r2wml`), of which roughly $0.05 was the actual
job and the rest idle time. Code at local commit `f4aab59`; pod checkout had no `.git`, so
`author` / `git.commit` in `manifest.json` are filled by hand.)
**Headline results:**

*Ablation (n=227 rows = 152 unique recipients, all at iteration 2):*

| condition | answer changed (95% CI) | vs. real (McNemar p) | vs. control (McNemar p) | accuracy | Δ vs. own baseline (paired p) |
|---|---|---|---|---|---|
| real donor (prior run) | 35.7% (81/227) [29.7, 42.1] | — | — | 48.9% (111/227) | −6.2 pp (18 r→w / 4 w→r, **p=0.004**) |
| random any-iter control (prior run) | 37.0% (84/227) [31.0, 43.5] | — | — | 43.6% (99/227) | −11.5 pp (27 / 1, **p=2e-7**) |
| **zero-ablation** | 28.2% (64/227) [22.7, 34.4] | **p=0.027** | **p=0.012** | 51.1% (116/227) | −2.6 pp (9 / 3, p=0.15) |
| **mean-ablation** | 25.6% (58/227) [20.3, 31.6] | **p=0.004** | **p=0.0007** | 54.2% (123/227) | +0.4 pp (4 / 5, p=1.0) |

Baselines: this run's honest decode 53.7% (122/227); the prior run's honest decode of the
same rows 55.1% (125/227) — they differ, see Caveats. Zero vs. mean: p=0.43 (no
difference). Deduplicated to one row per recipient (n=152): zero 29.6% changed / accuracy
51.3%→49.3%; mean 27.6% / 51.3%→52.6% — same picture. Of the answer changes, almost all are
wrong→wrong churn (zero: 52 of 64; mean: 49 of 58).

Iteration-3 read-out (one step downstream of the injection, a structurally dead iteration
under the prior run's mapping) top-1 token changed in 57.7% (zero) / 100% (mean) of rows —
recorded for completeness; it's "changed at all", not "moved toward a target", and a dead
iteration's top-1 token shifting under a fixed injected input says little.

*Attention (n=80 examples, 422 generated answer tokens, mean over 12 layers × 12 heads):*

| key positions | all answer tokens | first answer token only |
|---|---|---|
| question + bot (≈55 positions) | 75.3% (≈1.55% per token) | 81.8% (≈1.70% per token) |
| **focus thought (iteration 2)** | **1.78%** | **2.07%** (median 2.0%, max 3.4%, never >5%) |
| other 5 thoughts | 12.5% (≈2.5% per thought) | 13.5% (≈2.7% per thought) |
| eot | 1.4% | 2.7% |
| prior answer tokens | 9.1% | — |

**Interpretation:**
- **Wiping the input to the best-decoding thought costs the model nothing measurable.** A
  zero vector or the population mean in place of the recipient's own latent leaves
  accuracy at 51-54% vs. a 54% baseline (neither paired test significant), and changes the
  answer *less* often than either content swap (p≤0.027 in all four comparisons). Every
  content swap, by contrast — matched donor or random draw — significantly *lowers*
  accuracy (−6 to −12 pp, p≤0.004). So the pattern at this position is: content-free →
  harmless; someone else's content → harmful; the *right* someone else's content (the
  matched donor) → no better than a random one on the answer-changed metric the prior run
  used, and less harmful than a random draw on accuracy (p=0.065) only in the sense that a
  same-step donor vector is presumably more on-manifold for that slot. None of that is
  faithfulness: a faithful position would be one where the answer *tracks* the injected
  value, and where wiping it costs accuracy. Here it's a position the model can do without
  entirely but is disturbed by when it holds structured-but-wrong content.
- **The attention check independently says the answer isn't read from there.** Three
  quarters of the answer read-out's attention goes to the question tokens; the six thoughts
  together get ~14%, and the focus thought — the single most decodable position — gets
  1.8%, about the same as one question token and *less* than the average other thought
  (2.5%). No example's first answer token puts more than 3.4% of its attention there. The
  decodable position is not privileged in the read-out at all.
- **Cross-mechanism comparison with `20260919-085911_recurrent_depth_ablate-attn`:** same
  headline, different signature. Both mechanisms: decoded content at the patched position
  isn't what drives the answer, and the answer read-out barely attends to that position
  (recurrent_depth 0.9-2.3% self-attention; CODI 1.8%). But the ablation direction is
  reversed: recurrent_depth was sensitive to the *off-manifold* zero vector (9.5% vs.
  2.5% for content swaps) and indifferent to the on-manifold mean, i.e. "anomaly-sensitive,
  content-blind"; CODI is indifferent to both content-free vectors and disturbed by any
  *content*, i.e. "content-sensitive but not content-faithful". Worth stating both
  signatures separately in the writeup rather than collapsing them to "position doesn't
  matter" — CODI's position does matter in the sense that wrong content propagates through
  thoughts 3-6 and costs accuracy; it just doesn't carry the answer.
- **How much to lean on this before the eval-mode re-run (next bullet):** the *paired
  orderings* (mean ≤ zero < real ≈ control on answer-changed; content-free ≈ baseline <
  content swaps on accuracy) and the attention picture are the robust parts. The absolute
  answer-changed rates are not — see Caveats.
**Gotchas hit:**
- **The model was never put in eval mode — LoRA dropout (p=0.1, 48 layers) was active in
  every forward pass of this run and of both prior CODI interp runs (`..._codi_decode-patch-
  pilot`, `..._codi_decode-patch-full`).** Found while writing this up: the honest greedy
  baseline decode of the same recipient disagrees with itself 13.6% of the time within this
  run (15/110 duplicate-recipient pairs) and 20.3% of the time between this run and the
  prior run (46/227 rows); greedy decoding should be deterministic. Cause, verified locally:
  `eval_codi.build_model` never calls `model.eval()` — `eval_codi.py` calls it in its own
  main (so the accuracy reproduction in `20260918-021217_codi_released-weights-6lat` is
  fine), but `decode_patch_codi.py` and this script import `build_model` and never do.
  `from_pretrained` returns GPT-2 in eval mode, but `get_peft_model` injects fresh LoRA
  layers whose `Dropout(0.1)` modules default to training mode (checked on peft 0.21.0:
  48/48 LoRA dropouts in training mode after injection, GPT-2's own 37 dropouts in eval;
  two passes of one input differ once `lora_B` is nonzero, argmax agreement 0.90; identical
  after `.eval()`). Fixed in `build_model` in the same commit as this run so every caller
  gets a deterministic model. The recurrent_depth scripts all call `.eval()` — that side is
  unaffected.
- Base `runpod/pytorch` image lacks `rsync`/`git`, same as every other run this session.
- The 227 focus pairs are not 227 independent recipients (152 unique) — a property of the
  prior run's pair sampling that this run inherited by replaying its pairs. Dedup numbers
  are reported above; the qualitative result is the same either way.
**Caveats:**
- **Dropout noise floor (see Gotchas).** Every "answer changed" rate in this run and the
  two prior CODI runs includes a ~14-20% floor of pure re-decode noise, so the absolute
  rates (e.g. 48% in the full run, 36% at iteration 2) are inflated and the power to see a
  small real-vs-control difference is reduced. The comparisons themselves remain valid as
  paired contrasts (all conditions share the noise), and the accuracy view is more robust
  (noise averages out across recipients instead of adding to every row), which is why this
  writeup leans on accuracy-under-intervention. Decoding accuracy (18.3% matched top-1 in
  the full run) was also measured under dropout and is if anything an underestimate. The
  attention weights themselves are exact softmax outputs (GPT-2's attention dropout was in
  eval mode) but were measured on a LoRA-dropout-noised trajectory. **All three CODI interp
  runs should be re-run in eval mode before any of their numbers go in the writeup** —
  decode+patch full ≈ $0.11 and this diagnostic ≈ $0.05 on an A5000, so ~$0.20 total.
- The injection point is the *input* to iteration 2 (the slot that normally carries
  iteration 1's output), exactly as in the prior run — so "the focus thought" throughout
  means that sequence position, whose input is overwritten and whose output is the
  best-decoding z_2. The attention bucket targets that same position, so the two tasks are
  aligned with each other and with the patch run.
- The mean vector is the population mean of iteration 2's *output* (z_2), injected into
  iteration 2's *input* slot — on-manifold for the projection's output space, but not the
  mean of what that slot normally receives (z_1). Zero and mean behave the same (p=0.43),
  so this doesn't change the reading; a z_1-mean would be the tidier control next time.
- n=80 / 422 tokens for attention; "question" mass isn't broken down by which question
  tokens (operands vs. filler) — the natural follow-up if this becomes a figure.
- Attention is on honest, unpatched trajectories only.
**Next:**
- **Re-run in eval mode** (`decode_patch_codi.py --full_test True --mapping_split True
  --n_patch_pairs 350`, then this script) with the `build_model` fix in place — ~$0.20 —
  and log both as follow-up runs; expect deterministic baselines, lower absolute
  answer-changed rates, and the same paired orderings. Until then, cite the recurrent_depth
  side (unaffected) as the well-powered one and this side as provisional.
- Append the corresponding correction to the two prior CODI runs' `notes.md` (done in this
  commit).
- If the eval-mode re-run holds: the writeup's two-mechanism figure is (a) patch ≈ control,
  (b) ablation signature per mechanism (off-manifold-sensitive vs. content-sensitive), (c)
  attention mass on the patched position, side by side.


---
**Correction (2026-09-27): CODI site indexing was shifted by one position.** `run_thoughts(override_input_at={i: v})`
replaces the latent fed INTO iteration i, which is normally z_{i-1} (z_0 = latent-0 for i=1); z_6 is never
consumed. The `mean` condition fed the population mean of iteration it's OUTPUT (z_it) into iteration it; the `mean_in` condition and zero ablation are aligned. So a site labelled "iter i" / "z_i" here transplanted the donor's z_i into the slot where the
recipient's z_{i-1} lives: one position early, z_0 never transplanted, and the (normally unused) z_6 fed into
iteration 6. Only the `mean` rows are affected; prefer `mean_in`. Aligned rerun of E3: `20260927-004340_codi_minimal-pair-patch-aligned` (steering sits on
z0/z2/z4, all-slot 0.839 vs 0.713 legacy on the same 317 pairs; the "diffuse" localization does not hold).
Numbers above are left as logged; they describe the shifted intervention.
