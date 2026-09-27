## 2026-09-19 — CODI decode+patch full run, re-run in eval mode: decoding unchanged, the injected value still never propagates, and with the noise floor gone the answer-changed contrasts resolve into slot sensitivity, not content faithfulness (codi, run_id: 20260919-184323_codi_decode-patch-full-eval)

**Goal:** Exact re-run of `20260919-080349_codi_decode-patch-full` after the finding (logged in
`20260919-090132_codi_ablate-attn`) that `eval_codi.build_model` never called
`model.eval()`, leaving PEFT's injected LoRA `Dropout(0.1)` layers active in every forward
pass of the three earlier CODI interp runs. Same script, same arguments, same checkpoint,
same held-out split and pair-selection seed; only difference is the `model.eval()` fix in
`build_model` (commit `9485853`). Question: which of the earlier numbers were dropout
artefacts?
**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3` (released checkpoint),
compute_steps=6, paper inference protocol (LoRA r=128/α=32, projection 768+LN, greedy).
**Data:** gsm8k-aug test, all 1319 examples; half A (idx even, n=660) fits
`best_iter_for_step`, half B (idx odd, n=659) is where every decoding number and every patch
pair comes from. Train-corpus baseline from 20,000 train examples. 350 patch pairs (227 at
the focus iteration, 123 at other live iterations), `random.Random(0)` as before.
**Determinism check (before spending on the run):** `build_model` now returns a model with
0 modules in training mode (0 Dropout modules in training mode, checked on the pod's peft
0.15.2 / transformers 4.52.4); decoding 5 test questions twice through
`run_thoughts`+`decode_answer` gave 5/5 byte-identical outputs. In the ablation companion
run (`20260919-184716_codi_ablate-attn-eval`) the 110 duplicate-recipient baseline pairs
disagree 0/110 times (was 15/110 under dropout).
**Command:**
```
cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/decode_patch_codi.py \
  --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \
  --slug decode-patch-full-eval --stage full_run --hardware "RunPod RTX A4500 (secure)" \
  --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \
  --lora_r 128 --lora_alpha 32 --lora_init --greedy True \
  --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
  --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \
  --full_test True --mapping_split True --n_patch_pairs 350
```
(pod `52nc4qt7mfykv7`, EU-RO-1, RTX A4500 secure $0.25/hr — A5000 had no secure stock;
created 18:27 UTC, setup ≈4 min (`apt-get install rsync git`, `codi_setup.sh`, checkpoint
download), determinism check + n=12/3-pair smoke test, decode pass 226 s + patch pass 284 s,
then the ablation companion, results synced and pod terminated ≈18:49 UTC — ≈22 min ≈
**$0.09** for both runs; billing record not yet posted when this was written. Pod checkout has
no `.git`, so `author` / `git.commit` in `manifest.json` are filled by hand.)
**Headline results:** `final_answer_accuracy=0.419` (553/1319, batch=1 decode; 0.409 under
dropout; the paper-protocol batch=128 number is 0.4367). `decoding_accuracy` (held-out
matched top-1) **=0.1839** [0.168, 0.201], n=2126; top-5 0.2658. `intervention_accuracy`
(next-iteration read-out moved toward the injected value) **=0.0044** (1/227).

*Side by side with the dropout-affected run (`20260919-080349`):*

| | dropout run | **eval-mode run** |
|---|---|---|
| accuracy (batch=1) | 0.409 | 0.419 |
| matched top-1 / top-5 (held out, n=2126) | 0.1834 / 0.2634 | **0.1839 / 0.2658** |
| train-corpus baseline (same population) | 0.0282 | 0.0282 (×6.5) |
| paper-style Table-3 metric, 1/2/3-step (n) | 0.545 (11) / 0.060 (116) / 0.0 (78) | 0.667 (9) / 0.067 (119) / 0.0 (82) |
| `best_iter_for_step` (half A) | {1:2, 2:2, 3:4, 4:4, 5:2, 6:4, 7:4, 8:1} | {1:2, 2:2, 3:6, 4:6, 5:2, 6:2, 7:4, 8:1} |
| live iterations | [1, 2, 4] | [1, 2, 4, 6] |
| read-out moved toward donor: real / any-ctrl / live-ctrl | 0.86% (3/350) / 2.1% (2/97) / 0.5% (1/184) | **0.44% (1/227)** / 1.6% (1/63) / 4.4% (4/91) |
| answer changed, all 350: real / any-ctrl / live-ctrl | 48.0% / 48.9% / 47.4% | **40.9% / 30.6% / 37.4%** |
| McNemar real-vs-any / real-vs-live / live-vs-any | p=0.84 / 0.91 / 0.68 | **p=0.0012 / 0.27 / 0.037** |
| answer changed, focus iter 2 (n=227): real / any / live | 35.7% / 37.0% / 34.4% | **31.3% / 29.1% / 28.6%** (p=0.60 / 0.51) |
| answer changed, other iters (n=123): real / any / live | 70.7% / 70.7% / 71.5% (iter 4) | 58.5% / 33.3% / 53.7% (iter 6; real-vs-live p=0.44) |

*Accuracy under intervention, focus iteration (n=227; paired McNemar vs. baseline,
right→wrong / wrong→right):*

| condition | accuracy | Δ | r→w / w→r | p |
|---|---|---|---|---|
| baseline (honest) | 55.5% | — | — | — |
| **real donor patch** (same slot, same step count, different value) | 52.4% | −3.1 pp | 15 / 8 | **0.21** |
| any-iteration control | 48.0% | −7.5 pp | 22 / 5 | 0.0015 |
| live-iteration control | 45.4% | −10.1 pp | 24 / 1 | <0.0001 |

Real patch vs. live control on accuracy: 25 / 9, p=0.009 — the matched donor is
significantly *less* harmful than an unrelated live-iteration vector. Of the 71 focus-iter
answers the real patch changed, 3 equal the donor's value or the donor's gold answer; 10
equal the live-control's answer.

**Interpretation:**
- **Decoding was never a dropout artefact.** Matched top-1 0.1839 vs. 0.1834, the full
  iteration×step matrix agrees to the third decimal (even iterations 2/4/6 each decode
  steps 1–2 at ≈0.20–0.28 top-1, odd iterations at 0.000), the train-corpus baseline ratio
  is the same 6.5×, and the paper-style Table-3 metric is still far below the paper's
  97/84/75% (67%/6.7%/0% at n=9/119/82). The only mapping change is that steps 3–4 now
  argmax to iteration 6 instead of 4 — the two were within 0.005 of each other both times
  (0.115 vs. 0.111 for step 3), so this is a tie-break flipping, not a different picture. The
  Table-3 gap therefore is not explained by dropout either; the "value appears in ANY
  iteration's top-5" hypothesis from the earlier notes is still the open thread.
- **Faithfulness (does the injected value propagate?) is still at floor.** The next
  iteration's read-out moves toward the injected value 1/227 times; the patched answer
  lands on the donor's value or the donor's answer 3/71 times among the answers that
  changed. Neither control is lower than the real condition. Injecting a vector that
  *decodes to a wrong intermediate value* into the single most decodable slot leaves
  accuracy at 52.4% vs. 55.5% (p=0.21). If the answer were computed from that slot's
  decodable content, a plausible-but-wrong value there should be the most damaging
  intervention available, and the answer should move toward it; it is instead the *least*
  damaging of the content-carrying conditions and the answer essentially never follows it.
- **What removing the noise floor did change is the sensitivity contrasts, and they now
  make mechanistic sense.** Under dropout every condition sat at ≈48% answer-changed and
  looked identical. In eval mode the rates drop to 31–41% and separate: vectors taken from a
  *dead* iteration (half of the any-iteration control's draws) perturb the answer less
  (30.6%) than vectors from live iterations (real 40.9%, live-control 37.4%), so real-vs-any
  is now significant (p=0.0012) while real-vs-live is not (p=0.27, and p=0.51 at the focus
  iteration). The right reading is slot-kind sensitivity — what kind of vector sits in the
  slot matters — not content faithfulness: the real donor is by construction a same-slot
  vector, and the live control is mostly an other-slot vector (iterations 1/4/6 injected into
  slot 2), so the extra disruption from the live control is off-slot structure, not
  "wronger" content. The tracking metrics above are the ones that speak to faithfulness, and
  they are unchanged from the dropout run.
- **Net for the writeup:** the earlier CODI claim survives with the caveat removed, but
  should be phrased as "a decodably-wrong value in the best-decoding slot neither propagates
  nor costs accuracy; the slot is sensitive to off-slot vectors" rather than the flatter
  "real ≈ any control" (which was partly a dropout artefact — under dropout the any-control
  looked as disruptive as everything else). The companion ablation run
  (`20260919-184716_codi_ablate-attn-eval`) completes the picture with content-free
  conditions.
**Gotchas hit:**
- None new. Base `runpod/pytorch` image lacks `rsync`/`git` (installed); ssh commands that
  launch a `nohup … &` job hang the local `ssh` call even with all three fds redirected —
  harmless, the job runs; poll the log in a second session.
- `--hardware` with spaces has to be quoted in the shell wrapper (the pod's
  `run_full.sh` passed it with underscores; corrected by hand in `manifest.json`).
**Caveats:**
- Neither control is "same iteration, unrelated example". That control would be nearly the
  same intervention as the real patch (a same-slot z_2 decoding to a different step-1 value),
  which is why the changed-rate contrast can't decide faithfulness on its own — the
  read-out-tracking and answer-tracking metrics are the faithfulness evidence; the
  changed-rate contrasts measure slot sensitivity.
- Read-out tracking is only defined at the focus iteration (n=227); the other-iteration
  pairs now sit at iteration 6, where there is no next thought to read out (`it >= n_latents`).
- The same recipient recurs across focus pairs (152 unique recipients in 227 pairs) — the
  real/control rows differ per pair (different donors), but the baseline rows repeat; the
  companion run reports deduplicated numbers for the recipient-only conditions.
- Single seed for pair selection (`Random(0)`); n=350 / 227 / 123 as stated.
**Next:**
- The paper-style Table-3 gap is now dropout-free and well-powered — try the
  "any-iteration top-5" variant from cached logits before treating it as a discrepancy.
- Writeup: replace the dropout-affected CODI numbers (`20260919-073312`, `20260919-080349`,
  `20260919-090132`) with this run and its companion everywhere; keep the earlier runs as the
  record of the artefact.
- A "same-iteration, unrelated-example" control is cheap to add to `decode_patch_codi.py` if
  a reviewer wants the changed-rate contrast to isolate content; it would sit between the
  real patch and the live control by construction.

## Metric B addendum (rescored 2026-09-19, `scripts/rescore_counterfactual.py`)

See `steered_to_donor_audit.md`. `steered_to_donor` as originally logged measures Metric A (`answer_patched == donor.answer` -- already the case for this run except where noted); the table below adds Metric B (`matches_cf`): does the answer equal the counterfactual obtained by substituting the injected value into the RECIPIENT's own remaining chain and re-evaluating.

### Step-aligned single-slot patch (n=227 focus-iter pairs)

| group | n | n(cf defined) | matches_cf | 95% CI | |
|---|---|---|---|---|
| real donor | 227 | 69 | 0.000 | [0.000, 0.053] |
| control: random example, random iter | 227 | 69 | 0.000 | [0.000, 0.053] |
| control: random example, live iter | 227 | 69 | 0.000 | [0.000, 0.053] |

- **real donor** taxonomy: unchanged 156, other_number 61, recipient_gold 8, recipient_intermediate 2
- **control: random example, random iter** taxonomy: unchanged 161, other_number 58, recipient_gold 5, recipient_intermediate 3
- **control: random example, live iter** taxonomy: unchanged 162, other_number 61, recipient_intermediate 3, recipient_gold 1

## ANY-iteration decodability addendum (rescored 2026-09-22, `scripts/rescore_any_iter_top5.py`)

Tests the hypothesis flagged in this run's own "Next" section: the paper's Table-3 metric
doesn't commit to one "correct" iteration per step the way `best_iter_for_step` does; a gold
value appearing in ANY of the 6 iterations' top-5 (not just the mapped one) might close most
of the reproduction gap. No new model run — recomputed locally from this run's own cached
`predictions.jsonl` (`per_iter` top-1/top-5 were already saved per example), same held-out
half-B population (n=2126 example-step pairs, n=659 examples) as the logged matched-iteration
number, so it's directly comparable. `decode_patch_codi.py` now also computes this metric by
default for future runs.

| | matched-iteration (logged) | ANY-iteration (this addendum) |
|---|---|---|
| top1 | 0.1839 | **0.2281** (485/2126) |
| top5 | 0.2658 | **0.3358** (714/2126) |

Paper-style metric (Table 3, correct-only, by step count), matched-iteration vs. ANY-iteration:

| steps | matched-iteration (logged) | ANY-iteration | paper (Shen et al.) |
|---|---|---|---|
| 1 | 0.667 (n=9) | 0.667 (n=9) | 0.971 |
| 2 | 0.067 (n=119) | **0.126** (n=119) | 0.839 |
| 3 | 0.000 (n=82) | 0.012 (n=82) | 0.750 |
| 4 | — | 0.023 (n=43) | — |
| 5 | — | 0.000 (n=8) | — |

**Interpretation:** ANY-iteration decoding does surface real additional signal — top1/top5
both rise (18.4%→22.8%, 26.6%→33.6%), and the 2-step paper-style bucket nearly doubles
(6.7%→12.6%). But it does **not** close the reproduction gap to the paper's reported
97.1%/83.9%/75.0% — every bucket is still far below the paper's numbers, and the 1-step bucket
(the best-powered one at n=9) is completely unchanged, since it was already saturated under
the matched mapping. The "not committing to one fixed iteration" hypothesis explains part of
the gap, not most of it; the larger remaining gap is more likely n (paper evaluates on their
full 1319-example test set with presumably far more 1/2/3-step examples than this 659-example
half-B slice gives, esp. at n=8-9) and/or a genuinely different reading rule than "value in
top-5 of a forward-pass logit lens" that Appendix E's case study doesn't fully specify.

**Caveat:** this doesn't affect any of the causal-patching or faithfulness conclusions in this
run or elsewhere in the repo — those are about whether a *specific* injected value propagates,
which ANY-iteration doesn't touch. It only revises the decodability side of the story upward,
modestly.


---
**Correction (2026-09-27): CODI site indexing was shifted by one position.** `run_thoughts(override_input_at={i: v})`
replaces the latent fed INTO iteration i, which is normally z_{i-1} (z_0 = latent-0 for i=1); z_6 is never
consumed. The patch sweep used `{it: donor_recs[it-1]["post"]}`. So a site labelled "iter i" / "z_i" here transplanted the donor's z_i into the slot where the
recipient's z_{i-1} lives: one position early, z_0 never transplanted, and the (normally unused) z_6 fed into
iteration 6. Decoding numbers are unaffected; only the patch-sweep numbers describe shifted sites. Aligned rerun of E3: `20260927-004340_codi_minimal-pair-patch-aligned` (steering sits on
z0/z2/z4, all-slot 0.839 vs 0.713 legacy on the same 317 pairs; the "diffuse" localization does not hold).
Numbers above are left as logged; they describe the shifted intervention.
