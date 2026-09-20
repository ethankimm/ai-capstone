## 2026-09-20 — E3(a)+(b): same-problem minimal-pair donor-interchange patch, CODI -- upper bound is real, and it's diffuse (codi, run_id: 20260920-190420_codi_minimal-pair-patch)

**Goal:** `steered_to_donor_audit.md` §5 E3. E2 (`20260920-085206_codi_qualified-patch`)
found a clean null on **cross-problem** donors even with base-correct, propagation-qualified
pairs. §4.3 argued the cross-problem design confounds two failure modes: "the value isn't
in the thoughts at all" vs. "the value is there but entangled with the donor's problem
context, which the recipient never saw." E3 removes the second confound entirely: the donor
is the recipient's **own question** with one eligible number perturbed and the gold chain
re-executed (`latentreasoning.data.minimal_pairs.generate_minimal_pair`), so the twin's
final answer literally IS the counterfactual value (Metric A == Metric B by construction,
`matches_twin`). (a) ALL-SLOT patches every latent iteration with the twin's own live
latents -- an upper bound on whether the thought chain carries the value at all. (b)
single-slot sweep + cumulative prefix localizes which iteration(s) carry it if ALL-SLOT
succeeds.

**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3` (released checkpoint),
compute_steps=6, same inference protocol as E2 (LoRA r=128/α=32, projection 768+LN, greedy).

**Data:** gsm8k-aug test, n=600 (seed=0), same slice as E2's CODI run. 263/600 base-correct
(accuracy 0.438, matches E2). 258 minimal-pair candidates generated (base-correct +
propagation-qualified original), 137/258 had a twin the model also solved correctly
("qualified pairs" -- the set actually scored).

**Command:**
```
cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/patch_minimal_pair_codi.py \
  --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \
  --slug minimal-pair-patch --stage full_run --hardware "RunPod RTX A5000 (secure)" \
  --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \
  --lora_r 128 --lora_alpha 32 --lora_init --greedy True \
  --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
  --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \
  --eval_n 600 --n_pairs 200
```
Fresh RunPod RTX A5000 (secure, CA-MTL-1, $0.27/hr). CODI env setup (`codi_setup.sh`) took
~3 min once the repo actually made it onto the pod -- the file transfer, not the env build,
was the bottleneck this session (see Gotchas). Decode pass (n=600) 75s; full patch sweep
(137 pairs × 4 conditions: all-slot, 6 single-slot, 6 prefix) ~3.5 min. Total pod time for
this run + a throwaway n=40 smoke test (`20260920-185840_codi_minimal-pair-smoketest`, not
logged): well under 10 min GPU.

**Headline results:** `final_answer_accuracy=0.438` (matches E2's CODI slice exactly, same
data/seed). n=137 qualified minimal pairs.

| condition | n | matches_twin | 95% CI | answer_changed |
|---|---|---|---|---|
| ALL-SLOT (all 6 iters) | 137 | **0.701** | [0.62, 0.77] | 0.927 |
| single-slot iter 1 | 137 | 0.036 | [0.02, 0.08] | 0.416 |
| single-slot iter 2 | 137 | 0.080 | [0.05, 0.14] | 0.292 |
| single-slot iter 3 | 137 | 0.000 | [0.00, 0.03] | 0.146 |
| single-slot iter 4 | 137 | 0.036 | [0.02, 0.08] | 0.168 |
| single-slot iter 5 | 137 | 0.000 | [0.00, 0.03] | 0.007 |
| single-slot iter 6 | 137 | 0.109 | [0.07, 0.17] | 0.212 |
| prefix 1..1 | 137 | 0.036 | [0.02, 0.08] | 0.416 |
| prefix 1..2 | 137 | 0.080 | [0.05, 0.14] | 0.445 |
| prefix 1..3 | 137 | 0.080 | [0.05, 0.14] | 0.445 |
| prefix 1..4 | 137 | 0.467 | [0.39, 0.55] | 0.774 |
| prefix 1..5 | 137 | 0.489 | [0.41, 0.57] | 0.803 |
| prefix 1..6 | 137 | 0.701 | [0.62, 0.77] | 0.927 |

**Interpretation:** This is the "succeeds where cross-problem fails" outcome
`steered_to_donor_audit.md` §4.3 called out as itself a finding. ALL-SLOT moves the answer
to the twin's answer 70% of the time (vs. 0.5–2.5% for cross-problem donors in E2, same
model/data) -- the thought chain unambiguously carries the perturbed value when the donor
shares the recipient's problem context. But the localization result is the more interesting
half: no single iteration carries it (iter 6 alone is the best at 10.9%; iters 3 and 5 are
exactly 0), yet the cumulative prefix jumps sharply between prefix 1..3 (8.0%) and prefix
1..4 (46.7%), then climbs to 70.1% by prefix 1..6. The value isn't stored in a single
addressable slot the way a "scratchpad variable" framing would predict -- it's distributed
across the sequence of latents, and needs iterations 4-6 present together to reconstruct.
This qualitatively matches the ablate-all finding referenced in `next_experiments.md`
(thoughts are collectively load-bearing, load sits on the non-decodable placeholder slots)
but now shown as a *positive* causal effect, not just a decrement from removal. Net: "not
portable across problems" (E2) and "not stored in an addressable single-iteration slot even
within the same problem" (E3 single-slot) both hold, but "not present in the thought chain
at all" is now ruled out -- CODI's thoughts do encode the perturbed value, just as a
joint/distributed function of iterations rather than a localized one.

**Gotchas hit:**
- The real bottleneck this session was **not** the CODI env setup (fast once it ran) but
  getting the repo onto the pod: macOS's built-in `/usr/bin/rsync` is `openrsync` (protocol
  version 29), which hangs/stalls for very long periods (confirmed 10+ min with zero bytes
  transferred, twice) talking to the pod's modern rsync 3.2.7 -- almost certainly what made
  last session's CODI setup churn for 1h19m+ before the pod disappeared with nothing
  produced. Switched to `tar -cz | ssh ... tar -xz` for the transfer, which worked
  immediately. Worth fixing properly (e.g. pin transfers to `tar` or install real rsync via
  homebrew) rather than rediscovering this each session.
- First `tar` transfer accidentally included the full local `results/` tree (672MB --
  `--exclude='results/*/predictions*.jsonl'`-style patterns don't reliably match through
  `tar`'s exclude matching the way they do for `rsync`/`.gitignore`); fixed by excluding
  `results/` wholesale from the pod transfer (it doesn't need historical results, only to
  write new ones) and creating an empty `results/` dir on the pod instead.
- `--out_dir` is not a flag on `patch_minimal_pair_codi.py` (only on some other scripts) --
  `RunRecord.save()` always writes to this repo's own `results/<run_id>/`, which then has to
  be pulled back off the pod explicitly (this run + its notes were generated on the pod and
  tarred back to the local repo).

**Caveats:**
- `n_pairs=200` was requested but only 137 qualified pairs existed in this 600-example
  slice (base-correct twin AND base-correct original AND a valid non-negative-integer
  perturbation existed among deltas ±1/±2/±3) -- not a bug, just the corpus's qualifying
  pool size at this n.
- `matches_twin` is a *minimal-pair-specific* metric, not directly comparable to E2's
  `matches_cf` numbers in the table above without noting the different donor construction --
  the audit doc's whole point was that these two designs answer different questions
  (portability vs. presence).
- Single-slot iter 3 and 5 being *exactly* 0/137 is consistent with (not proof of) those
  being genuinely non-load-bearing positions in isolation; distinguishing "truly zero
  effect" from "effect below what n=137 single-pair patches can detect" would need a larger
  n or a grouped-ablation design (flagged in `next_experiments.md`).

**Next:** Coconut counterpart is `20260920-195725_coconut_minimal-pair-patch` (same
session). Both mechanisms now have the E3(a) upper-bound result; per
`steered_to_donor_audit.md` §5 E4, DAS on minimal-pair targets is the natural follow-up
now that ALL-SLOT succeeds and single-slot localizes weakly -- the prefix-jump pattern
(3→4 for CODI) is a concrete hypothesis a learned subspace could test more precisely than
raw single-slot patching.
