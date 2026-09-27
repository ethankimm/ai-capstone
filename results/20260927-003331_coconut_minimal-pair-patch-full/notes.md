## 2026-09-27 — E3 rerun at full n, Coconut: passes 1 and 4 carry the value (leave-one-out confirms), non-responsive pairs follow the twin's wrong answer (coconut, run_id: 20260927-003331_coconut_minimal-pair-patch-full)

**Goal:** RESEARCH_PLAN round 1, item 1 (power): Coconut E3 had n=105 pairs; rerun on the full gold-trace
set to pass the ~210 target, and add leave-one-out, non-responsive pairs and a full CFR log, so it
matches the aligned CODI rerun (`20260927-004340_codi_minimal-pair-patch-aligned`) condition-for-condition.
Coconut's sites were already aligned (`run_passes` splices the twin's pass-p vector into the recipient's
pass-p slot).
**Mechanism / model:** `coconut`, gpt2 / `hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33`, 6 latents.
**Data:** gold-trace test file, all 1194 (seed-0 order); 1412 candidates from every example, 413 with a
correct original, **231 qualified pairs**, 182 non-responsive pairs; every twin in `candidates.jsonl`.
**Command:**
```
cd /workspace/ai-capstone && .venv_coconut/bin/python scripts/patch_minimal_pair_coconut.py \
  --checkpoint_path /workspace/coconut_checkpoints/checkpoint_33 --data_dir /workspace/coconut_data \
  --slug minimal-pair-patch-full --stage full_run --hardware "RunPod RTX A5000 (secure)" \
  --eval_n 0 --n_pairs 1000 --n_nonresponsive 300
```
Same pod as the CODI rerun ($0.27/hr A5000), in parallel; decode 232 s, whole run ~20 min.
**Headline results:** base accuracy 0.336 on the gold-trace set (paper 0.331–0.341).

| pass (n=231) | single-slot | leave-one-out | prefix 0..p |
|---|---|---|---|
| 0 | 0.013 | 0.818 | 0.013 |
| 1 | **0.558** | **0.398** | 0.550 |
| 2 | 0.000 | 0.827 | 0.550 |
| 3 | 0.026 | 0.818 | 0.584 |
| 4 | **0.338** | **0.628** | 0.784 |
| 5 | 0.004 | 0.784 | 0.827 |
| ALL-SLOT | **0.827** [0.773, 0.870] | | |

Non-responsive pairs (n=182): all-slot moves the recipient to the twin's **wrong** answer 48.9%
[0.417, 0.561], to the twin's gold 2.7%.
CFR = 231/413 = 0.559; by chain length 2/3/4/5+: 0.73 / 0.57 / 0.31 / 0.26; P(twin correct | original
wrong) = 0.056.
**Interpretation:** Replicates the n=105 run (all-slot 0.771 → 0.827, pass 1 0.486 → 0.558, pass 4 0.286
→ 0.338) with tighter CIs. Leave-one-out shows passes 1 and 4 are also *necessary* for steering (dropping
pass 1 costs 43 pp, pass 4 costs 20 pp); 0, 2, 3, 5 cost ≤4 pp. With CODI's aligned rerun, both
mechanisms localize the transplanted value to a few latents (Coconut 1/4, CODI 0/2/4), with inert
latents between. Non-responsive pairs: the latents carry what the model computed for the twin, right or
wrong.
**Gotchas hit:** none.
**Caveats:** gold-trace set (1194) is a filtered subset of the 1319 test questions. Multi-use operands
still limit steering (0.59 vs 0.89 single-use; see `20260927-004443_coconut_nonsteered-buckets-e3full`).
**Next:** aligned CODI E2/E4 reruns so the two mechanisms are compared on identical site semantics.
