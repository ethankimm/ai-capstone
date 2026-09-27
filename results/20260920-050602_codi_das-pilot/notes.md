## 2026-09-20 — Distributed Alignment Search on CODI: learned subspace still doesn't steer (codi, run_id: 20260920-050602_codi_das-pilot)

**Goal:** Follow-up to `20260920-031925_codi_interchange-placeholder-pilot` (raw full-vector
donor-interchange patching at all 6 iterations is a clean null: swapping in another
example's z never steers the answer toward the donor, even at the load-bearing
placeholder positions z0/z3) and `20260920-032053_codi_probe-pilot` (which flagged DAS
as the unimplemented next step: is the raw-vector null an artifact of an unconstrained
intervention that also overwrites context/attention state, rather than evidence that no
transferable subspace exists?). This run implements Distributed Alignment Search: learn
an orthogonal rotation R (`torch.nn.utils.parametrizations.orthogonal`) that isolates a
k-dim subspace of z, trained end-to-end (transformer + LoRA frozen) to maximize the
donor's own gold-answer likelihood under the intervention, then evaluates whether that
learned subspace steers the recipient's answer where the raw full vector could not.

**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3` (released checkpoint,
unchanged), 6 continuous thoughts, paper inference protocol (LoRA r=128/α=32,
projection 768+LN, greedy). New script `scripts/das_codi.py`, reusing
`encode_question`/`decode_answer`/`wilson_ci`/`mcnemar_exact_p` from
`decode_patch_codi.py` and `sample_pairs` from
`interchange_patch_placeholder_codi.py`. DAS hook per the task spec: inside
`run_thoughts`'s per-iteration loop, replacing the INPUT fed to iteration `site` (same
site-indexing convention as the interchange script).

**Data:** Trained on `gsm8k-aug` `split="train"` (donor/recipient pairs with different
gold answers), evaluated on `split="validation"` (the fixed 1000-example held-out set) —
train/eval split convention matches `probe_codi.py`, keeping `test` untouched.

**Hyperparams:** all 6 iteration sites (1..6) x k in {8, 32, 64}, 150 training pairs x
5 epochs per (site, k), 60 held-out eval pairs per (site, k), AdamW lr=1e-3 + cosine
schedule. Pilot-scale reduction of the task spec (n=1000 training pairs, 20 epochs,
k in {4,8,16,32,64}) — see script docstring for the compute-budget accounting that
motivated the reduction (batch-size-1 rollout, same constraint as every other CODI
script in this repo).

**Command:**
```
cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/das_codi.py \
  --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \
  --slug das-pilot --stage pilot --hardware "RunPod RTX A6000 (secure)" \
  --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \
  --lora_r 128 --lora_alpha 32 --lora_init --greedy True \
  --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
  --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \
  --sites 1,2,3,4,5,6 --k_values 8,32,64 --train_n_pairs 150 --eval_n_pairs 60 --epochs 5
```
Fresh RunPod RTX A6000 pod ($0.53/hr, EU-RO-1, secure cloud — A5000 showed zero secure
stock at provision time, confirmed live against the catalog). Setup: rsync repo +
`bash scripts/codi_setup.sh` (~2 min, pinned env: torch 2.7.1 / transformers 4.52.4 /
peft 0.15.2) + checkpoint `snapshot_download` (~7s, cached). Load: `missing=0
unexpected=0`. A throwaway n=5/n=3 smoke test (deleted, not logged, per the n<10
exception) confirmed the training + eval pipeline ran end-to-end before the real pilot.
Full 18-combo sweep: ~65 minutes wall time (~$0.57 GPU cost).

**Headline results:**
- **17 of 18 (site, k) combinations: `steered_to_donor_rate=0.000`** (n=60 eval pairs
  each, Wilson CI upper bound ~0.06). Training loss did decrease within each combo
  (e.g. site=1/k=8: 35.0 → 26.0 over 750 steps), so R was learning *something* about the
  donor's answer likelihood, but that did not translate into steering the recipient's
  greedy-decoded answer toward the donor's gold value.
- **One marginal exception:** site=4, k=32 — `steered_to_donor_rate=0.017` (1/60,
  Wilson CI [0.003, 0.089]) — a single pair, not distinguishable from noise at this n.
- `answer_changed_rate` ranges 0.10–0.32 across combos (patching does perturb the
  output sometimes), but essentially never toward the donor's specific answer — same
  qualitative signature as `20260919-192228_codi_early-termination-ablate-all`'s
  ablation result and the raw-vector interchange null: perturbation without
  donor-specific steering.
- No systematic k-dependence (k=8, 32, 64 give statistically indistinguishable nulls)
  and no clean placeholder-vs-decodable split emerges in `steered_to_donor` (unlike
  `answer_changed`, which trends slightly higher at some sites, e.g. site=4: 0.25 (k=8)
  to 0.32 (k=64)) — best_site=4 only because of the single k=32 hit above.

**Interpretation:** DAS was the natural strongest-case test of the "unconstrained
intervention" reading of the raw-patch null — a rotation trained specifically to
maximize donor-answer likelihood, at every subspace size tested, still fails to
transfer a donor's calculation state into a recipient. This weighs against "the
intervention was just too blunt" and toward the harder conclusion already emerging
from ablation + attention diagnostics: CODI's continuous thoughts are collectively
load-bearing but not *portable* — not a subspace-basis artifact fixable by a smarter
linear intervention. Consistent with "we plan for null findings" — this null is itself
informative for the causality section of the writeup (a negative DAS result, at pilot
scale, on the mechanism whose raw-patch null most needed this follow-up).

**Gotchas hit:**
- Same rsync/git caveat as `20260920-031246_coconut_decode-patch-pilot`: only
  `latentreasoning/`, `scripts/`, `pyproject.toml`, `configs/`-equivalent paths were
  synced (no `.git`), so `RunRecord`'s `git_info()`/`default_author()` produced
  `author=""` / `git.commit="unknown"` — hand-corrected in `manifest.json` after
  copying results back (`author="Henning Lindig"`, `git.commit` set to the local HEAD
  at run time `0c0f4079af1a0a0119cff22e3d996db3091987ed`, `git.dirty=true`).
- `torch.nn.utils.parametrizations.orthogonal` (not `.parametrization.orthogonal`,
  a typo in the task spec this followed) is the correct current PyTorch API.
- The teacher-forced training loss (CE against the donor's gold-answer digits right
  after CODI's `eot` token) started high (30s-40s, well above the ~10.8 nats a uniform
  random guess over GPT-2's vocab would give) and only partially came down within 750
  steps — plausibly undertrained at this pilot's epoch budget (5 vs the spec's 20), not
  necessarily a bug; the eval-time `steered_to_donor` null is the metric that matters
  and is unambiguous regardless.

**Caveats:**
- Pilot scale throughout (train_n_pairs=150 vs spec's 1000, epochs=5 vs 20, k in
  {8,32,64} vs {4,8,16,32,64}) — a full-scale rerun could in principle still find a
  steerable subspace this pilot missed, though the uniformity of the null across all 18
  combos (no combo showed even a directional trend toward steering) makes that less
  likely than for a single ambiguous result.
- `evaluate_rotation`/eval-time compute reuses a fresh per-(site,k) cache, so donor
  z-vectors are recomputed for every combo rather than shared across the sweep — a
  performance detail, not a correctness one.

**Next:** If DAS is worth pursuing further, a full-scale rerun
(`--train_n_pairs 1000 --epochs 20 --k_values 4,8,16,32,64`) would rule out
undertraining as the explanation, though given the uniformity of this pilot's null and
budget constraints, the coordinated CODI+Coconut null (see
`20260920-053935_coconut_das-pilot`) is likely a stronger signal to write up than a
larger, more expensive rerun of the same result.

## Metric B addendum (rescored 2026-09-19, `scripts/rescore_counterfactual.py`)

See `steered_to_donor_audit.md`. `steered_to_donor` as originally logged measures Metric A (`answer_patched == donor.answer` -- already the case for this run except where noted); the table below adds Metric B (`matches_cf`): does the answer equal the counterfactual obtained by substituting the injected value into the RECIPIENT's own remaining chain and re-evaluating.

### DAS learned-subspace patch, unaligned (n=1080)

| group | n | n(cf defined) | matches_cf | 95% CI | perm-null |
|---|---|---|---|---|---|
| site 1 | 180 | 117 | 0.000 | [0.000, 0.032] | 0.002 |
| site 2 | 180 | 84 | 0.000 | [0.000, 0.044] | 0.003 |
| site 3 | 180 | 96 | 0.010 | [0.002, 0.057] | 0.001 |
| site 4 | 180 | 93 | 0.000 | [0.000, 0.040] | 0.005 |
| site 5 | 180 | 105 | 0.000 | [0.000, 0.035] | 0.002 |
| site 6 | 180 | 90 | 0.000 | [0.000, 0.041] | 0.001 |
| total | 1080 | 585 | 0.002 | [0.000, 0.010] | 0.002 |

- **site 1** taxonomy: unchanged 159, other_number 18, unparseable 2, recipient_gold 1
- **site 2** taxonomy: unchanged 145, other_number 23, recipient_gold 9, unparseable 2, donor_intermediate 1
- **site 3** taxonomy: unchanged 156, other_number 21, recipient_gold 2, counterfactual 1
- **site 4** taxonomy: unchanged 132, other_number 37, unparseable 8, recipient_gold 2, donor_final 1
- **site 5** taxonomy: unchanged 147, other_number 29, recipient_gold 3, unparseable 1
- **site 6** taxonomy: unchanged 152, other_number 21, recipient_gold 5, unparseable 2
- **total** taxonomy: unchanged 891, other_number 149, recipient_gold 22, unparseable 15, donor_intermediate 1, counterfactual 1, donor_final 1


---
**Caveat (2026-09-27): the DAS training recipe used here is suspect.** In `20260927-072659_codi_das-minimal-pair-aligned-bigk` the same recipe (lr 1e-3, 5 epochs, teacher-forced CE) trained at k=512 steers 22% while an UNTRAINED random rotation steers 71% on the same pairs. This run had no untrained reference, so its null may be an optimization failure; do not cite it as evidence against a linear subspace until rerun with a fixed recipe.

**Resolved (2026-09-27):** the DAS teacher-forcing target was wrong (CODI: bare number instead of "The answer is: N"; Coconut: " ###" instead of "###"), and the Coconut minimal-pair script trained and evaluated on overlapping test examples. With both fixed, DAS finds a ~16-dim value subspace in both mechanisms: `20260927-094215_codi_das-minimal-pair-fixed` (z0+z2+z4, k=16: 0.70 vs raw 0.83), `20260927-084449_coconut_das-minimal-pair-fixed` (passes 1+4, k=32: 0.67 vs raw 0.75). This run's null is an artifact.
