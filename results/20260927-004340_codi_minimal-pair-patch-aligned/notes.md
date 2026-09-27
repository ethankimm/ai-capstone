## 2026-09-27 — E3 rerun with aligned sites, CODI: the value is NOT diffuse — it sits on z0/z2/z4, and the old "diffuse" result was a site-indexing artifact (codi, run_id: 20260927-004340_codi_minimal-pair-patch-aligned)

**Goal:** RESEARCH_PLAN round 1, item 1: rerun E3 on CODI with the latent-0 site, at n toward the
~210 power target, before the "CODI stores values diffusely, Coconut locally" claim is written anywhere.
**What changed vs the committed E3 (20260920-190420): a site-indexing bug.** CODI feeds six latents into
its loop: z_0 (latent-0, bot position) into iteration 1, and z_s (iteration s's output) into iteration
s+1; z_6 is never consumed (`decode_answer` continues from the KV cache with eot).
`run_thoughts(override_input_at={i: v})` replaces the vector fed INTO iteration i. The committed E3,
E2 (`patch_qualified_codi.py`), E4 (`das_minimal_pair_codi.py`), `decode_patch_codi.py` and
`interchange_patch_placeholder_codi.py` all patched `{i: donor z_i}`: the donor's z_i placed where the
recipient's z_{i-1} goes, one position early, never transplanting z_0 and feeding the unused z_6 into
iteration 6. The ablation scripts (`early_termination_codi.py`, mean/zero) were already aligned; Coconut's
`run_passes` is aligned. This run patches site s as `{s+1: twin z_s}` (s = 0..5, same numbering as
Coconut passes 0..5), and keeps the old conditions as `*_legacy` on the same pairs.
(The uncommitted latent-0 edits to `patch_minimal_pair_codi.py` / `patch_qualified_codi.py` /
`probe_codi.py` found on 2026-09-26 were on this machine only — no teammate branch touches them — and
added a site 0 but left sites 1..6 on the shifted indexing; this script supersedes the E3 one.)
**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3`, 6 latents, eval mode, greedy, batch 1.
**Data:** gsm8k-aug test, all 1319; candidates from every example (1471), 546 with a correct original,
**317 qualified pairs** (original and twin both correct), 229 non-responsive pairs (original correct,
twin answered wrong), every twin logged in `candidates.jsonl`.
**Command:**
```
cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/patch_minimal_pair_codi.py \
  --ckpt_dir /workspace/codi_released --checkpoint_label hf:zen-E/CODI-gpt2@fd641b3 \
  --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 --lora_r 128 --lora_alpha 32 \
  --lora_init --greedy True --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
  --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True --output_dir /tmp/o \
  --slug minimal-pair-patch-aligned --stage full_run --hardware "RunPod RTX A5000 (secure)" \
  --eval_n 0 --n_pairs 1000 --n_nonresponsive 300
```
Pod `k030n8z4jrafxm` (RTX A5000 secure, CA-MTL-1, $0.27/hr), run in parallel with the two necessity runs
and the Coconut E3 rerun; decode 271 s, twins ~5 min, patching 317 pairs × 25 conditions ~25 min.
**Headline results:** base accuracy 0.419 (553/1319, identical to the eval-mode baseline).

| condition (n=317) | matches_twin | answer_changed |
|---|---|---|
| **ALL-SLOT aligned (z0..z5)** | **0.839** [0.795, 0.875] | 0.953 |
| ALL-SLOT legacy (old E3 intervention) | 0.713 [0.661, 0.760] | 0.937 |

Aligned vs legacy on the same pairs: 41 pairs steer only aligned, 1 only legacy (McNemar p = 2e-11).

| site | single-slot | leave-one-out | prefix z0..s |
|---|---|---|---|
| z0 (latent-0) | 0.199 | 0.729 | 0.199 |
| z1 | 0.003 | 0.839 | 0.202 |
| z2 | 0.202 | 0.659 | 0.539 |
| z3 | 0.000 | 0.836 | 0.539 |
| z4 | **0.360** | **0.546** | 0.842 |
| z5 | 0.000 | 0.842 | 0.839 |

Legacy single-slot "iter i" on the same pairs: 0.025, 0.041, 0.000, 0.041, 0.000, 0.107 (reproduces the
old E3 table: max 10.9% there, 10.7% here).

Non-responsive pairs (n=229, twin answered wrong): all-slot moves the recipient to the twin's **wrong**
answer 53.3% [0.468, 0.596] of the time, to the twin's gold answer 2.2%.
CFR (from `candidates.jsonl`): P(twin correct | original correct) = 317/546 = 0.581; by chain length
2/3/4/5+: 0.70 / 0.62 / 0.48 / 0.27; by perturbed step 0/1/2/3: 0.63 / 0.59 / 0.46 / 0.21.
P(twin correct | original wrong) = 0.070.

**Interpretation:**
- **The "CODI is diffuse" claim is withdrawn.** It came from the shifted indexing: legacy single-slot
  never exceeds 11%, but aligned single sites reach 36% (z4), 20% (z2) and 20% (z0 = latent-0). The
  perturbed value sits on the EVEN latents z0, z2, z4; the odd latents z1, z3, z5 are inert both
  alone (≤0.3%) and when left out (no drop from 0.839). Prefix jumps exactly at z0, z2 and z4.
- This is the same shape as Coconut (`20260927-003331`: passes 1 and 4 carry it, 0/2/3/5 inert) — both
  mechanisms localize the value to a few specific latents with inert slots between. The cross-mechanism
  difference is WHICH positions (CODI 0/2/4 vs Coconut 1/4), not diffuse vs local.
- The even latents are also CODI's decodable positions (logit lens: z0 per
  `20260923-052604_codi_decode-patch-full-eval-latent0`; z2/z4 per the earlier decode runs, whose
  "{z0,z3} non-decodable vs {z2,z4} decodable" labels were on output indices). So for CODI the
  "decodable but not causally used" story from the shifted patch runs needs rechecking too.
- Non-responsive pairs: the latents carry the model's own (wrong) computation for the twin, not the
  correct value — 53% follow the twin's wrong answer. That is a faithfulness result: what the latents
  carry is what the model actually computed.
- Behavioral reference: the unpatched model follows a one-number change in 58% of problems it solves,
  dropping to 27% for 5+-step chains; quote the 84% alongside it.
**Gotchas hit:** 8 processes on one A5000 made the GPU the bottleneck (98% util); fine for this size.
**Caveats:**
- **Every committed CODI donor-patch run used the shifted indexing**: E2 `20260920-085206`, E4
  `20260920-235312` and `20260920-050602`, `20260919-080349`/`-184323` patch sweeps,
  `20260920-031925` interchange. Their CODI single-site numbers describe a one-position-early transplant,
  not the named site; all-slot style conditions (full shifted chain) are less affected. Correction
  notes should be appended to those runs and E2/E4 rerun aligned before any CODI site-level claim.
- Pairs are not the same as the old E3 (full test set, candidates from all examples, different rng draw).
**Next:** append correction notes to the shifted-indexing runs; rerun E2 (and E4 DAS) with aligned
sites; bucket analysis on this run is `20260927-004433_codi_nonsteered-buckets-e3full`.
