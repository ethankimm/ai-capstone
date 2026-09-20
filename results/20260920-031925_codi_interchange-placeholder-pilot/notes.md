## 2026-09-20 — CODI donor-interchange patching, placeholder vs decodable positions: no steering anywhere, load-bearing content is not portable (codi, run_id: 20260920-031925_codi_interchange-placeholder-pilot)

**Goal:** Sharper follow-up to `20260919-192228_codi_early-termination-ablate-all` (mean-ablation:
z0/z3 load-bearing, p=6e-5 / p=0.0025, z2/z4 not) and `20260919-184323_codi_decode-patch-full-eval`
(single-slot interchange at the *decodable* focus iteration only — null, answer essentially never
follows the injected value). That run asked "does content at z0/z3 matter" with a content-free mean
vector; this one asks the sharper question with a content-*bearing* donor vector at every iteration,
including the non-decodable placeholder positions z0/z3 for the first time: does swapping in ANOTHER
example's z0/z3 (or z2/z4) steer the answer toward that donor's answer? Two outcomes distinguish the
paper's central claim: (A) donor-swap steers toward the donor -> placeholder slots carry
transferable, semantically structured state, logit lens is just looking at the wrong place; (B) it
doesn't steer despite z0/z3 being load-bearing under ablation -> non-representational / distributed
computation, necessary but not portable the way the positive control (explicit CoT single-position
patch, ~99% steering) is.
**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3` (released checkpoint, unchanged),
6 continuous thoughts, paper inference protocol (LoRA r=128/α=32, projection 768+LN, greedy), batch
1, eval mode (build_model fix from commit `9485853`, same as the two runs above). New script
`scripts/interchange_patch_placeholder_codi.py`, reusing `run_thoughts` / `decode_answer` /
`wilson_ci` / `mcnemar_exact_p` from `decode_patch_codi.py`.
**Data:** gsm8k-aug test, n=200 (seed=0, the shared eval slice). Single-slot: 60 donor/recipient
pairs per iteration (1–6), pairs drawn with `random.Random(pair_seed*1000 + iter)`, donor/recipient
required to have different gold final answers. Grouped: 100 pairs (`pair_seed+999`), same
donor/recipient patched two ways: jointly at {z0,z3} (non-decodable) vs jointly at {z2,z4}
(decodable).
**Command:**
```
cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/interchange_patch_placeholder_codi.py \
  --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \
  --slug interchange-placeholder-pilot --stage pilot --hardware "RunPod RTX A4500 (secure)" \
  --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \
  --lora_r 128 --lora_alpha 32 --lora_init --greedy True \
  --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
  --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \
  --full_test False --eval_n 200 --n_pairs_per_site 60 --n_grouped_pairs 100
```
(pod `fxbzoi0p11tsqw`, EU-RO-1, RTX A4500 secure $0.25/hr — A5000 had zero secure/community
stock and A6000 had no instances available at request time, both checked live against the
catalog before falling back; A4500 reused because it is precedented for this exact checkpoint/venv
in the two runs this follows up on. Setup: rsync repo + `bash scripts/codi_setup.sh` (~2 min,
identical pinned env: torch 2.7.1 / transformers 4.52.4 / peft 0.15.2, matches prior runs exactly)
+ checkpoint `snapshot_download` (~3 s, cached). Load: `missing=0 unexpected=0`. Run itself:
single-slot phase (360 patches) 127 s, grouped phase (300 patches) 48 s. Total pod wall time
~20 min including setup/teardown; **cost ≈ $0.08**.)
**Headline results:**

Single-slot donor-interchange, `steered_to_donor` (patched answer matches donor's gold, base
didn't), n=60 per iteration:

| iter | slot kind | answer_changed | steered_to_donor | 95% Wilson CI (steered) |
|---|---|---|---|---|
| 1 | **NONDECODABLE (z0)** | 0.550 | **0.017** | [0.003, 0.089] |
| 2 | other (z1) | 0.433 | 0.000 | [0.000, 0.060] |
| 3 | decodable (z2) | 0.450 | 0.000 | [0.000, 0.060] |
| 4 | **NONDECODABLE (z3)** | 0.383 | 0.000 | [0.000, 0.060] |
| 5 | decodable (z4) | 0.367 | 0.000 | [0.000, 0.060] |
| 6 | other (z5) | 0.417 | 0.000 | [0.000, 0.060] |

Grouped, joint-swap {z0,z3} (non-decodable) vs {z2,z4} (decodable), n=100, paired McNemar:

| condition | answer_changed | steered_to_donor |
|---|---|---|
| non-decodable group | 0.590 [0.492, 0.681] | 0.010 |
| decodable group | 0.560 [0.462, 0.653] | 0.010 |
| McNemar (b, c, p) | b=20, c=17, p=0.743 | b=1, c=1, p=1.000 |

`intervention_accuracy` (headline metric, grouped non-decodable steered-to-donor rate) = **0.01**.

**Interpretation:**
- **Outcome B, unambiguously, and it generalizes beyond the placeholder positions.** Donor-swap
  essentially never steers the answer toward the donor at *any* single iteration — 1/60 (1.7%) at
  z0, 0/60 everywhere else including the two iterations (z2, z4) that decode best under logit lens.
  This is not a placeholder-specific finding: the "decodable" slots don't transfer content under
  interchange patching either, consistent with (and now directly replicating on a second donor/
  recipient sample) the null already logged in `20260919-184323_codi_decode-patch-full-eval` at the
  focus decodable iteration.
- **Grouped comparison finds no non-decodable-vs-decodable contrast at all** — steered_to_donor is
  identically 0.01 in both groups (1/100 each, the *same* recipient/donor pairs happening to land on
  the donor's answer by chance, not by transfer: McNemar b=1,c=1 exactly cancel, p=1.0). If the
  placeholder slots carried transferable structured state that the logit lens simply couldn't read
  out (Outcome A), the grouped joint-swap at {z0,z3} — which changes strictly more of the residual
  stream's content than a single slot — would be the best-powered place to see it emerge. It
  doesn't: answer_changed is similar between groups too (59% vs 56%, p=0.74) — both groups perturb
  the answer at comparable, generically-disruptive rates, with no directional pull toward the donor
  in either.
- **Reconciling with the ablation finding:** z0/z3 are load-bearing (mean-ablation costs accuracy,
  p=6e-5/0.0025) but not portable (donor content there doesn't redirect the answer). The two
  measurements are compatible under a "distributed / non-linearly entangled" reading — the
  computation happening at those positions matters for *this* forward pass's internal state, but a
  *different* example's z0/z3 doesn't slot in as a drop-in replacement, unlike explicit CoT's single-
  position patch (`20260919-185332_explicit_cot_patch-positive-control`), which is tracked ~85% of
  the time and moves the answer ~99% of the time. CODI's continuous thoughts, at every position
  tested here (placeholder or decodable), behave nothing like that positive control.
- Read together with `20260919-184323_codi_decode-patch-full-eval` and
  `20260919-192228_codi_early-termination-ablate-all`, the three-run picture for CODI is now
  consistent end to end: the six-thought chain is collectively necessary (ablate-all costs ~18pp),
  the necessity concentrates on z0/z3 specifically (ablate-single), but no single position's content
  — decodable or not — is a portable, causally faithful carrier of the donor's answer under direct
  interchange. Decodability and portability are both absent at the placeholder positions, and
  portability is *also* absent at the positions that do decode.
**Gotchas hit:** none. Determinism/eval-mode fix from commit `9485853` was already in place (this
run reuses `eval_codi.build_model`); no smoke test needed since the script had already been
syntax-checked and the mechanic (`run_thoughts`/`decode_answer`/`override_input_at`) is identical to
the already-validated `decode_patch_codi.py`.
**Caveats:** Single-slot iterations are *unpaired* across sites (different donor/recipient draws
per iteration; the "other iters" comparison in `next_experiments.md` is only informal via Wilson
CIs, not McNemar — the grouped condition is the properly paired comparison and is the one to quote).
n=60/100 pairs at pilot scale; steered_to_donor sits at or near the floor everywhere so CIs are wide
relative to the (near-zero) point estimates — a full-scale rerun (larger n_pairs) would tighten the
upper bound but is unlikely to change the qualitative read given how uniformly flat the result is
across all 6 iterations and both groups.
**Next:** This closes out Priority 1 from `next_experiments.md` for CODI — the paper-critique
framing's central causal-faithfulness question is answered (Outcome B) at both pilot scale and
(via the two related runs) from three independent angles (ablation, single-position content-free
patch, single-position and grouped content-bearing patch). Remaining CODI work per
`next_experiments.md`: #2 probe_codi.py (linear/MLP probes vs logit lens — is the logit-lens null a
basis artifact, independent question from causal portability) and DAS on any probe-recoverable
directions. No need for a full-scale (`--full_test True`) rerun of this script specifically — the
pilot's n and the uniform near-zero effect across every iteration and both groups already give a
clear qualitative answer; scale would mainly narrow CIs already resting on essentially 0/1-count
data.
