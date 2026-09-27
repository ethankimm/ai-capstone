## 2026-09-20 — E2: step-aligned, base-correct, controlled raw single-slot patch at every CODI iteration -- null survives the fixed pair design (codi, run_id: 20260920-085206_codi_qualified-patch)

**Goal:** `steered_to_donor_audit.md` E2. The earlier raw-patch nulls
(`20260919-184323_codi_decode-patch-full-eval`, `20260920-031925_codi_interchange-placeholder-pilot`)
used cross-problem donor pairs chosen only by differing final answers, at whichever
iteration decoded a step best, with no requirement that the recipient itself be
base-correct or that the targeted step actually be able to propagate downstream. This run
fixes all three: qualified pairs (recipient AND donor base-correct; recipient's step-k
gold value is an operand of step k+1, so a perturbation there CAN reach the final answer),
tested at EVERY iteration (not just the best-decoding one, so z4/z5/z6 -- never a "best"
step in the earlier decode-based mapping -- get tested too), with a proper mean-ablation
control alongside the random-donor control. Scored with the shared `score_patch` module
(Metric B `matches_cf` primary) from the start, not retrofitted.

**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3` (released checkpoint),
compute_steps=6, paper inference protocol (LoRA r=128/α=32, projection 768+LN, greedy,
`model.eval()` fix from `20260919-090132_codi_ablate-attn` in place).

**Data:** gsm8k-aug test, n=600 (seed=0). Site→step assignment is POSITIONAL
(`latentreasoning.eval.counterfactual.site_to_step`), not decoding-accuracy-fit: the 6
iterations are split into `max_step`=8 contiguous groups (iter 1→step 0, iter 2→step 1,
iter 3→step 2, iters 4/5/6→steps 4/5/6 respectively) — deliberately not
`decode_patch_codi.py`'s `best_iter_for_step`, so every iteration gets a target step
instead of only the ones that happen to decode a step best.

**Command:**
```
cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/patch_qualified_codi.py \
  --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \
  --slug qualified-patch --stage full_run --hardware "RunPod RTX A6000 (secure)" \
  --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \
  --lora_r 128 --lora_alpha 32 --lora_init --greedy True \
  --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
  --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \
  --eval_n 600 --n_pairs_per_site 200
```
RunPod RTX A6000 (secure, EU-SE-1, $0.53/hr — A5000/A4500 both showed no stock at
provision time). Fresh pod: `codi_setup.sh` (clone+patch+pinned venv, torch 2.7.1 /
transformers 4.52.4 / peft 0.15.2), checkpoint `snapshot_download` (~3s). Decode pass
(n=600) 73.8s; full patch sweep (6 sites, 370 qualified pairs total, 4 forward
trajectories/pair) ~3 min. Ran concurrently with the Coconut E2 run below on the same
pod. Total pod time including setup/downloads for both mechanisms: ~72 min ≈ **$0.64**
(idle SSH-wait time dominates; actual GPU compute for this run was well under 5 min).

**Headline results:** `final_answer_accuracy=0.435` (261/600), base-correct pool used for
all pairing. 370 qualified pairs found across 6 sites (site pool sizes shrink sharply at
higher steps since few examples have >3 steps at all: 190/121/52/2/3/2).

| site (iter) | step | n | real matches_cf | random matches_cf | real answer_changed | random | mean | McNemar real-vs-random (cf) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 190 | 0.005 | 0.000 | 0.263 | 0.274 | 0.274 | b=1,c=0,p=1.0 |
| 2 | 1 | 121 | 0.025 | 0.008 | 0.413 | 0.421 | 0.099 | b=3,c=1,p=0.625 |
| 3 | 2 | 52 | 0.000 | 0.000 | 0.308 | 0.327 | 0.308 | b=0,c=0,p=1.0 |
| 4 | 4 | 2 | 0.000 | 0.000 | 0.500 | 0.500 | 0.500 | — |
| 5 | 5 | 3 | 0.000 | 0.000 | 0.333 | 0.333 | 0.000 | — |
| 6 | 6 | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |

**Overall (n=370):** `matches_cf` real=**1.10%** [0.43%, 2.80%] vs random=**0.28%**
[0.05%, 1.54%] — both at floor, not distinguishable (the two site-level McNemar tests with
any discordant pairs, sites 1 and 2, are non-significant: p=1.0, p=0.625).
`matches_donor_final` (Metric A) real=0.81%, random=0.54% — same floor. `answer_changed`:
real=31.9%, random=33.0% (statistically indistinguishable), mean-ablation=21.9%
(noticeably lower, as expected — a content-free mean vector perturbs less than another
example's real activation). Outcome taxonomy for real: unchanged 252, other_number 111,
counterfactual 4, recipient_intermediate 2, donor_intermediate 1 — same "answers scramble
toward recombinations of the recipient's own operands, not the donor's" pattern as every
prior patching run on this mechanism.

**Interpretation:** This is the cleanest version of the CODI raw-patch null run so far —
base-correct-only pairs (removes the confound of patching an already-wrong recipient),
propagation-qualified steps (removes the confound of patching a step whose value doesn't
even matter downstream), and every iteration tested including the ones a decoding-fit
mapping would never select. The null holds unchanged: real donor content is
indistinguishable from a random donor's under Metric B (`matches_cf`) at every site with
enough pairs to test (sites 4-6 have only 2-3 qualifying base-correct examples each — the
positional site→step assignment runs out of qualifying pool at high step counts, since
gsm8k-aug problems rarely have more than 3-4 real steps; `max_step=8` is driven by a small
number of long/noisy rationales). Confirms `steered_to_donor_audit.md`'s prediction: fixing
the metric AND the pair-qualification design does not surface a portable-value signal that
the earlier, looser designs were hiding.

**Gotchas hit:**
- No RunPod pod was running at session start (prior pods had been terminated); this run
  required a from-scratch pod provision + `codi_setup.sh` + checkpoint download, all
  included in the ~72 min pod time above.
- Proxy SSH (`ssh.runpod.io`) requires a PTY and does not support one-shot non-interactive
  commands from a scripted client; used the pod's direct SSH port (`ssh root@<ip> -p
  <port>`) for everything instead, which behaves like normal sshd.
- Sites 4/5/6 (steps 4/5/6) have only 2-3 qualifying pairs each — `n_pairs_per_site=200`
  requested but the qualified, base-correct pool at high step indices is tiny at this
  problem-length distribution; not a bug, just the corpus's step-count tail.

**Caveats:**
- Site→step assignment is a POSITIONAL heuristic (`site_to_step`), not a
  decoding-accuracy fit — chosen deliberately so non-decodable iterations get tested, but
  it means a given iteration's "assigned step" may not be the step it is most naturally
  associated with computationally. `intervention_accuracy` in the manifest is a global
  average across sites of very unequal size (n=190 down to n=2); the per-site table above
  is the number that matters, not the pooled `intervention_accuracy` scalar.
- `matches_cf` is undefined (`None`) for the mean-ablation condition by construction — there
  is no "value" a mean vector injects, so Metric B has nothing to check propagation of;
  only `answer_changed` and the outcome taxonomy are meaningful for that condition.

**Next:** Coconut counterpart is `20260920-085317_coconut_qualified-patch` (same session,
same design, same pod). Per `steered_to_donor_audit.md` §5, E3 (same-problem minimal-pair
donors, `latentreasoning/data/minimal_pairs.py`) is the next step if this cross-problem
null is worth chasing further — it removes the "donor's remaining program lives in
question text the recipient never saw" confound entirely.


---
**Correction (2026-09-27): CODI site indexing was shifted by one position.** `run_thoughts(override_input_at={i: v})`
replaces the latent fed INTO iteration i, which is normally z_{i-1} (z_0 = latent-0 for i=1); z_6 is never
consumed. This run patched `{it: donor z_it}` (`thought_cache[donor][it-1]["post"]`). So a site labelled "iter i" / "z_i" here transplanted the donor's z_i into the slot where the
recipient's z_{i-1} lives: one position early, z_0 never transplanted, and the (normally unused) z_6 fed into
iteration 6. Per-site real/random/mean rates describe shifted sites; the cross-problem null itself is not expected to depend on the shift, but should be confirmed with an aligned rerun. Aligned rerun of E3: `20260927-004340_codi_minimal-pair-patch-aligned` (steering sits on
z0/z2/z4, all-slot 0.839 vs 0.713 legacy on the same 317 pairs; the "diffuse" localization does not hold).
Numbers above are left as logged; they describe the shifted intervention.

Aligned rerun of this experiment: `20260927-062443_codi_qualified-patch-aligned` (z0 matches_cf 7.5% vs 0.5% here; see its notes on why real≈random is not a null).
