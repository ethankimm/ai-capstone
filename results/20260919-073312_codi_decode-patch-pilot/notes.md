## 2026-09-19 — CODI decodability + causal-patching pilot: decodable at ~4x baseline, but patching shows no faithfulness (codi, run_id: 20260919-073312_codi_decode-patch-pilot)

**Goal:** The faithfulness-spine contrast case to `recurrent_depth`'s step-supervised run
(`20260918-214656_recurrent_depth_stepsup-split4-4-4-full`, read-out 31.3% but final
answer = the no-scratchpad control). CODI's own paper (Shen et al. 2025, Sec. 5.1 /
Table 3) reports its continuous thoughts decode to intermediate values at 97.1%/83.9%/75.0%
(1/2/3-step problems) — but only an associational, correctness-conditioned metric via
logit-lens; they never test whether the decoded content is *causally used*. This run adds
that test: (1) an unconditional decodability number actually comparable to recurrent_depth's
31.3%, and (2) activation patching — swap one continuous thought for a donor example's,
see if the read-out and final answer move as the donor's value predicts.
**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3` (the authors' released
checkpoint, same one verified in `20260918-021217_codi_released-weights-6lat` to reproduce
43.67% on the full GSM8K test / paper protocol), compute_steps=6 (6 continuous-thought
iterations).
**Data:** gsm8k-aug test n=200 seed=0 (the shared slice); run seed=None (greedy, no RNG).
**Hyperparams:** paper protocol, same as the released-weights eval run — LoRA r=128/α=32,
6 latents, projection 768+LN, greedy decode.
**Command:** `scripts/decode_patch_codi.py` (new script this run, see its docstring) —
```
cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/decode_patch_codi.py \
  --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \
  --slug decode-patch-pilot --stage pilot --hardware "RunPod RTX A5000 (secure)" \
  --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \
  --lora_r 128 --lora_alpha 32 --lora_init --greedy True \
  --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
  --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \
  --eval_n 200 --eval_seed 0 --n_patch_pairs 30
```
(pod `3xldc5s3rskdnb`, CA-MTL-1, RTX A5000 secure $0.27/hr; ~25 min wall time incl. env
setup and checkpoint download; inference-only, well under $0.15 of pod time.)
**Headline results:** `final_answer_accuracy=0.400` (80/200), `unparseable_rate=0.000`,
`decoding_accuracy=0.186` (matched-iteration top1; 0.254 top5), `intervention_accuracy`
(next-readout moved toward donor) `=0.04`.

**Task 1 — decoding.** `outputs.logits[:, -1, :]` at each of the 6 loop iterations IS the
paper's own decode method (GPT-2 ties `wte`/`lm_head`, LoRA doesn't touch either, so
`logits = lm_head(hidden_state)` exactly) — no extra work needed to reproduce it.

*Iteration↔step structure (the first finding, and a genuine surprise):* only the **even**
iterations (2, 4, 6) ever decode to a number matching any gold step value — iterations
1, 3, 5 score **0.000** at every step, top1 and top5, across all 200 examples (one
exception: iter 5/step 1 top5 = 0.005, noise). This matches the paper's own case study
description ("each intermediate result is separated by a seemingly meaningless continuous
token") — confirmed here as a clean, load-bearing structural fact rather than an
anecdote: CODI's 6 latents are 3 content slots interleaved with 3 placeholders, not 6
independent thoughts. Full iteration×step matrix in `metrics.extra` and `manifest.json`.

Within the 3 live iterations, the mapping is *not* a clean diagonal (iter2→step1,
iter4→step2, iter6→step3 the way recurrent_depth's loop was diagonal-dominant) — iter4
and iter6 both peak at **step1**, not step2/step3 respectively (iter4: step1=0.227 top1
vs step2=0.111; iter6: step1=0.207 vs step3=0.164). `best_iter_for_step` (empirical
argmax): `{1:2, 2:2, 3:4, 4:6, 5:4, 6:2, 7:1, 8:1}` (steps 6-8 have too little support —
≤13 examples — to trust). Reading this as "iteration 2 is the strongest single decode
position for early steps, and later live iterations decode a *mix* of early and later
steps" is a better description than "iteration 2i reads step i."

*Two decoding numbers, not comparable to each other:*
- **Unconditional, full 6×8 grid** (every iteration × every step index, including the
  structurally-dead odd iterations and step counts beyond what 6 latents can reach):
  top1=0.074, top5=0.110. This is *not* the number to compare to recurrent_depth's
  31.3% — it's diluted by cells that can never be anything but 0 by construction.
- **Matched** (only `(best_iter_for_step[s], s)` cells — the fair comparison): **top1=0.186,
  top5=0.254**. Baseline (always guess `"6"`, the single most common intermediate value
  in this 200-slice, computed same-slice since a train-corpus pass needs network access
  this session didn't spend — recurrent_depth's analogous train-corpus baseline was
  3.5%, same order of magnitude): **0.043** on the identical matched population. CODI's
  matched top1 is ~4.3x that floor, top5 ~5.9x — a real, above-chance decodability
  signal, smaller than recurrent_depth's 31.3%/3.5%≈8.9x but the same qualitative
  story: genuinely decodable, not baseline noise.
- **Paper's own metric, reproduced** (top-5, all steps present, correct-answers-only,
  by step count, using the empirical `best_iter_for_step` mapping since the paper doesn't
  publish which iteration they read): 1-step **0.40**, 2-step **0.059**, 3-step **0.0**
  (n=5, 17, 8 correct-and-in-that-step-count-bucket respectively — too few to take at
  face value). This is well below the paper's reported 97.1%/83.9%/75.0%. Two candidate
  reasons, neither resolved here: (a) small-n at this bucket size (paper likely
  evaluates over their full 1319-example test set, not 200), (b) our `best_iter_for_step`
  mapping is an empirical stand-in for whatever positions/rule the paper actually used —
  Appendix E's per-example case study, not a formula, so an exact replication of their
  metric wasn't possible from the paper text alone. **Flagging this rather than
  papering over it**, per the pilot's own instructions: this gap is large enough that
  the paper-style number here should not be quoted as a discrepancy with Table 3 without
  first checking on the full test set / trying alternate position rules.
- Accuracy by step count (this slice): 1-step 5/9, 2-step 34/55, 3-step 22/44, 4-step
  16/49, 5-step 2/23, 6-step 0/13, 7-step 0/3, 8-step 1/2 — monotone decline, same shape
  as every other mechanism in this repo.

**Task 2 — causal patching (the decisive result).** 30 pairs, matched on step count and a
differing gold value at one step `s`, patched at `iter = best_iter_for_step[s]`: recipient
keeps its own KV cache through `iter-1`, the donor's independently-computed post-projection
latent at `iter` is substituted for the recipient's own, iterations `iter+1..6` and the
answer decode proceed on the recipient's (now-patched) trajectory. Control: same injection
position, but the substituted latent comes from a random unrelated (example, iteration)
pair rather than a matched donor.

- **Final answer changed from baseline: real patch 18/30 = 0.600, random control 19/30 =
  0.633.** Statistically indistinguishable — if anything the control (unrelated content)
  disrupts the answer marginally *more* than the semantically-matched donor value. The
  model's answer is about as sensitive to *any* perturbation at these live positions as
  to the specific counterfactual value that should, if the mechanism were faithful, push
  it toward a different — and predictable — answer.
- **Next-iteration read-out moved toward the donor's specific value: 1/25 = 0.04**
  (pairs patched at iteration 6 have no next iteration to check, n=25 not 30). This is
  the most direct test available — literally the next forward pass reading out the
  patched position — and it's at floor: the model's own subsequent decode essentially
  never reflects the specific content just injected.
- Checked whether the patched/control answer lands on the *donor's own* gold answer (a
  weak proxy, since recipient and donor share only one step, not the full problem): 0/30
  for base, patch, *and* control. Uninformative here — not surprising given the proxy is
  weak — dropped as a metric, keeping the two above.

**Interpretation:**
- CODI's continuous thoughts are genuinely decodable — 4-6x a same-slice frequency
  baseline, a real structural finding (the alternating content/placeholder pattern) not
  previously quantified anywhere I could find, including the paper itself. That much
  replicates and sharpens the paper's own interpretability claim.
- **But the causal-patching evidence shows no faithfulness**: swapping in the specific
  counterfactual content a human reading the logit-lens output would call "step i = X"
  does not move the model's own next read-out, or its final answer, any more than an
  unrelated perturbation of the same magnitude at the same position. Decodability here
  is not evidence the model is *using* that content the way it looks like it's using it.
- **This changes the recurrent_depth-vs-CODI contrast the faithfulness-spine plan was
  built around.** Going in, the open question was whether CODI (higher end-task
  accuracy, 41.5%/43.67% vs recurrent_depth's 14.5%) would show the *opposite* pattern —
  decodable *and* faithful. It doesn't, on this evidence: CODI's causal test is if
  anything a **cleaner, more decisive null** than recurrent_depth's (which at least
  showed a real correlation between read-out correctness and answer correctness — 37.5%
  vs 8.6% — even without direct patching). **Both mechanisms currently in this repo
  dissociate decodability from faithfulness; neither yet supports "decodable → used."**
  That's a stronger, if less flattering, unifying finding for Sep 25 than the
  recurrent_depth result alone, and it's a genuine result either way per the project's
  null-findings framing.
**Gotchas hit:**
- `scripts/decode_patch_codi.py`'s first draft copied `eval_codi.py`'s batched
  `next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)` verbatim for a batch-size-1
  decode loop — harmless when batch>1 (their script always runs batch=128), but at
  batch=1 `.squeeze(-1)` collapses the size-1 batch dim to a 0-d scalar and breaks the
  subsequent embed+unsqueeze, crashing generation. Fixed by dropping the squeeze (fixed
  before the logged run — see the script's inline comment).
- `final_answer_accuracy=0.400` (80/200) here vs. **0.415** (83/200) in
  `20260918-021217_codi_released-weights-6lat` on the identical checkpoint and slice —
  same greedy/deterministic protocol, so this should be bit-identical, and isn't. Cause:
  the documented upstream attention-mask quirk in `latentreasoning/mechanisms/codi.py`
  ("outputs therefore depend on batch composition") — the released-weights run uses the
  paper's exact batch-128 protocol; this run decodes one example at a time (needed for
  patching, which requires per-example KV-cache control), so left-padding-attention
  effects differ. Expected and explained, not a bug — but it means this run's 40.0% and
  the paper-protocol run's 41.5%/43.67% aren't the same measurement; don't quote 40.0%
  as *the* CODI number, use 41.5%/43.67% from the released-weights run for that.
- Two of 200 examples parse to `n_steps=0` (rationale had no `<<...=x>>` markers) — both
  wrong, excluded from all step-indexed analysis automatically.
**Caveats:**
- n=200 for decoding (±~7pp on the headline numbers), n=30 patch pairs (±~18pp) — the
  patch result (control ≥ real) is clean enough at this n that a larger run seems
  unlikely to overturn it, but it hasn't been tried.
- `best_iter_for_step` is an empirical argmax on this same 200-example slice, not a
  held-out fit — some circularity in using it to also select patch targets. A cleaner
  design would fit the mapping on one split and patch-test on another; not done here
  given pilot scope.
- The same-slice most-common-value baseline (4.3%) is a weaker floor than
  recurrent_depth's train-corpus baseline (3.5%) methodologically, though numerically
  close — a train-corpus version would need network access not spent this session.
- Random control's donor iteration is drawn from all 6 iterations (including the
  "dead" odd ones), so it can inject an out-of-distribution activation relative to the
  live-iteration donor — if anything this makes the control a *harder* bar to tie, so it
  doesn't undermine the real≈control finding.
- Paper-style metric (Table 3 reproduction) is unreliable at this n and mapping — see
  Task 1 above; not a claim of discrepancy with the paper, a flagged gap.
**Next:**
- If the team wants to push this further before Sep 25: repeat the patch test at
  iteration 2 only (largest single-iteration hit rate, 14/30 of this run's pairs already
  landed there) with a larger n to tighten the CI, rather than spreading budget across
  iterations 2/4/6 evenly.
- Oct 9 decoding writeup: use the matched-iteration numbers (0.186/0.254 vs baseline
  0.043) as CODI's entry, explicitly alongside recurrent_depth's 0.313/baseline 0.035 —
  same shape of finding, different mechanism.
- Oct 16 causality writeup: this pilot *is* the causality result for CODI — the
  recurrent_depth equivalent (patch `s_i` from a counterfactual example, per that run's
  own "Next" section) is still open and should use the same real-vs-random-control
  design used here.
- Coconut (paper checkpoint, not self-trained) is the natural next mechanism for this
  same decode+patch treatment — same continuous-thought representation family as CODI,
  different training regime (curriculum vs. self-distillation), and would test whether
  the faithfulness gap found here is specific to self-distillation or general to
  horizontal continuous-thought mechanisms.
