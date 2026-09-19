## 2026-09-19 — CODI ablation + attention, re-run in eval mode: content-free vectors in the best-decoding slot cost ≤4 pp, a decodably-wrong value costs the same, and the answer read-out still puts 1.8% of its attention there (codi, run_id: 20260919-184716_codi_ablate-attn-eval)

**Goal:** Eval-mode re-run of `20260919-090132_codi_ablate-attn` (which, like the other CODI
interp runs before commit `9485853`, ran with LoRA dropout active), paired against the
eval-mode patch run `20260919-184323_codi_decode-patch-full-eval` instead of the
dropout-affected one. Two changes to the script: `--prior_run` replaces the hard-coded
prior run id, and a third content-free condition `mean_in` — the population mean of what the
focus slot normally *receives* (iteration 1's output), matched to the slot being overwritten,
alongside the original `mean` (mean of iteration 2's output). Deduplicated-per-recipient
numbers are now in `manifest.json` (`extra.dedup_per_recipient`).
**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3`, 6 thoughts, paper
inference protocol. `scripts/ablate_attention_codi.py`.
**Data:** the 227 focus-iteration (iteration 2) pairs from `20260919-184323`'s
`patch_pairs.jsonl` in file order (152 unique recipients: 102×1, 33×2, 11×3, 4×4, 2×5),
half B only; real/any-control columns are that run's outcomes carried over. Means over 300
random half-B examples. Attention on 80 fresh half-B examples (same seeded shuffle as
before; 429 generated answer tokens).
**Design:** as in the earlier run — overwrite the post-projection latent fed into iteration
2 (`override_input_at={2: vec}`), continue honestly. Conditions: `zero`, `mean` (mean of z_2),
`mean_in` (mean of z_1). Answer-changed vs. baseline, paired McNemar; answer correctness,
paired McNemar; attention mass from each generated answer token to the focus slot / question /
other thoughts / eot / prior answer tokens, averaged over 12 layers × 12 heads.
**Command:**
```
cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/ablate_attention_codi.py \
  --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \
  --slug ablate-attn-eval --stage full_run --hardware "RunPod RTX A4500 (secure)" \
  --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \
  --lora_r 128 --lora_alpha 32 --lora_init --greedy True \
  --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
  --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \
  --prior_run 20260919-184323_codi_decode-patch-full-eval --n_attn_examples 80 --mean_sample_n 300
```
(same pod `52nc4qt7mfykv7` as the patch run, immediately after it: means 54 s, ablation
pass 135 s, attention pass 15 s; the two runs together ≈ $0.09 of pod time. Baseline
determinism on this run: the 110 duplicate-recipient baseline pairs agree 110/110 — was
95/110 under dropout.)
**Headline results:**

*Answer changed vs. baseline (n=227 rows; dedup n=152 in parentheses):*

| condition | eval mode | dropout run | vs. real (p) | vs. any-ctrl (p) |
|---|---|---|---|---|
| real donor (prior run) | 31.3% | 35.7% | — | — |
| any-iteration control (prior run) | 29.1% | 37.0% | — | — |
| zero | **21.1%** (23.0%) | 28.2% | 0.0032 | 0.025 |
| mean (of z_2) | **15.4%** (15.8%) | 25.6% | <0.0001 | <0.0001 |
| mean_in (of z_1) | **14.1%** (14.5%) | — | <0.0001 | <0.0001 |

zero vs. mean p=0.09; mean vs. mean_in p=0.76.

*Accuracy on the same 227 rows (baseline 55.5%; paired McNemar r→w / w→r):*

| condition | accuracy | Δ | r→w / w→r | p |
|---|---|---|---|---|
| real donor patch (prior run) | 52.4% | −3.1 | 15 / 8 | 0.21 |
| zero | 51.5% | −4.0 | 10 / 1 | 0.012 |
| mean_in | 52.4% | −3.1 | 8 / 1 | 0.039 |
| mean | 54.6% | −0.9 | 3 / 1 | 0.63 |
| any-iteration control (prior run) | 48.0% | −7.5 | 22 / 5 | 0.0015 |
| live-iteration control (prior run) | 45.4% | −10.1 | 24 / 1 | <0.0001 |

*Attention (80 examples, 429 answer tokens; dropout-run values were within 0.001):*

| key positions | all answer tokens | first answer token |
|---|---|---|
| question + bot (≈55 positions) | 75.0% (1.54%/token) | 81.8% (1.70%/token) |
| **focus slot (iteration 2)** | **1.78%** | **2.09%** (median 2.05%, max 3.6%) |
| other 5 thoughts | 12.5% (2.5%/thought) | 13.5% (2.7%/thought) |
| eot | 1.4% | 2.6% |
| prior answer tokens | 9.3% | — |

**Interpretation:**
- **Content-free conditions are the least disruptive thing you can put in the slot**, and
  the ordering is now clean and all-significant: mean_in ≈ mean (14–15%) < zero (21%) <
  any-control (29%) ≈ real donor (31%). A vector with no example-specific content
  changes the answer half as often as one carrying some other example's content.
- **But the slot is not inert.** With the noise gone, zero and mean_in cost a small,
  significant 3–4 pp of accuracy (p=0.012 / 0.039; the earlier run's "n.s." for zero was a
  power loss from dropout, not a different result). The mean of the slot's own output
  (`mean`) costs nothing (p=0.63). So: something at that position is used a little, and a
  generic on-manifold vector supplies it.
- **The decodable content is not what's used.** The real donor — a same-slot vector that
  decodes to a wrong intermediate value for this problem — costs the same 3 pp as the
  content-free vectors (p=0.21, not distinguishable from zero/mean_in), never moves the next
  read-out toward its value (1/227) and almost never the answer (3/71). If the 18% top-1
  decodability at this slot were on the causal path to the answer, the one intervention
  that specifically corrupts the decoded value should hurt more than wiping the slot; it
  hurts the same.
- **Off-slot vectors are what actually disrupt**: the live-iteration control (mostly
  iteration 1/4/6 vectors placed into slot 2) costs 10 pp and is significantly worse than
  the real donor (p=0.009 in the patch run). That is the sensitivity the earlier
  dropout-affected runs mistook for "any content disrupts equally".
- **Attention is unchanged**: 75% of the answer read-out's attention goes to the question,
  the six thoughts together get ~14%, and the best-decoding slot gets 1.8% — one question
  token's worth, less than the average other thought. This was already dropout-robust
  (softmax weights are exact; only the trajectory carried LoRA-dropout noise) and the
  numbers agree to three decimals.
- **Against `20260919-085911_recurrent_depth_ablate-attn`:** both mechanisms now show the
  same shape — decodable content that the answer neither tracks nor depends on, a slot
  that carries a little generic signal, and a read-out that barely attends to it. The
  signatures still differ in detail: recurrent_depth's zero-vector was the *most*
  disruptive condition (off-manifold sensitivity), CODI's zero is *less* disruptive than
  any content and only slightly worse than the mean.
**Gotchas hit:** none new; `--hardware` underscores corrected by hand in `manifest.json`
(see the patch run's notes).
**Caveats:**
- Effect sizes for the content-free conditions are small (3–4 pp on n=227, 10 and 8
  discordant pairs); read them as "small but real", not as a large ablation effect.
- 152 unique recipients behind 227 rows — the dedup column gives the recipient-level
  rates; the paired tests treat rows as units and therefore slightly overstate n for the
  recipient-only conditions.
- `readout_changed_at_all` (iteration 3's top-1 token changed: zero 58%, mean 100%) is a
  dead-iteration read-out and carries no faithfulness information; kept for continuity.
- Attention is on honest trajectories only; "question" mass is not broken down by token.
**Next:**
- Writeup figure: (a) tracking metrics at floor, (b) accuracy cost by condition —
  content-free ≈ wrong-content ≪ off-slot, (c) attention mass on the slot — side by side
  with recurrent_depth's version.
- If the slot's small generic contribution matters for the story, the natural follow-up is
  the all-thoughts ablation / early-termination sweep (in progress separately as
  `*codi_early-termination*`).
