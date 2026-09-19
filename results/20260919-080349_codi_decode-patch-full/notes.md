## 2026-09-19 — CODI decode+patch, scaled up: full test set, held-out mapping split, McNemar-tested null (codi, run_id: 20260919-080349_codi_decode-patch-full)

**Goal:** Tighten the pilot (`20260919-073312_codi_decode-patch-pilot`) on every caveat its
own notes flagged: the 200-slice → full 1319-example test set; the circular
`best_iter_for_step` mapping (fit and evaluated on the same examples) → fit on a held-out
half; the same-slice baseline → a train-corpus one, matching recurrent_depth's
methodology; 30 patch pairs spread over 3 iterations → 350 pairs stratified toward the
single strongest iteration, with two control variants (any-iteration / live-iteration-only)
and proper paired statistics (McNemar exact, Wilson CIs) instead of raw fractions.
**Mechanism / model:** `codi`, gpt2 / `hf:zen-E/CODI-gpt2@fd641b3` (same released
checkpoint as the pilot and `20260918-021217_codi_released-weights-6lat`), compute_steps=6.
**Data:** gsm8k-aug test, all 1319 examples (`--full_test True`); train-corpus baseline
sampled from 20,000 train examples. Held-out split: half A (idx even, n=660) fits
`best_iter_for_step`; half B (idx odd, n=659) is where every headline decoding number and
every patch pair comes from — half A's mapping is used, never refit on half B.
**Hyperparams:** paper protocol (LoRA r=128/α=32, 6 latents, projection 768+LN, greedy),
same as the pilot; `n_patch_pairs=350`, `patch_iter_focus=2` (auto-selected =
`best_iter_for_step[1]`), `patch_focus_frac=0.65`, `mapping_split=True`.
**Command:**
```
cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/decode_patch_codi.py \
  --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \
  --slug decode-patch-full --stage full_run --hardware "RunPod RTX A5000 (secure)" \
  --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \
  --lora_r 128 --lora_alpha 32 --lora_init --greedy True \
  --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \
  --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \
  --full_test True --mapping_split True --n_patch_pairs 350
```
(pod `725ecit98cwjyw`, CA-MTL-1, RTX A5000 secure $0.27/hr; created 07:52 UTC, decode pass
268s + patch pass 237s + ~3 min setup/checkpoint-download ≈ 12 min compute, pod kept ~25
min total incl. result sync before termination ≈ **$0.11**. Code at local commit `3e06224`
— pod checkout has no `.git` (excluded from the tar transfer), so `manifest.json` records
`git.commit=unknown`, same gap noted in the recurrent_depth full run.)
**Headline results:** `final_answer_accuracy=0.409` (540/1319; batch=1 decoding, so not
bit-comparable to the paper-protocol batch=128 run's 41.5%/43.67% — same documented
attention-mask-under-padding quirk as the pilot). `decoding_accuracy` (held-out matched
top1) **=0.1834** (95% CI [0.168, 0.200], n=2126), top5=0.2634. `intervention_accuracy`
(next-readout moved toward donor) **=0.0086** (n=350).

**Task 1 — decoding, held out.**
- `best_iter_for_step` (fit on half A): `{1:2, 2:2, 3:4, 4:4, 5:2, 6:4, 7:4, 8:1}`,
  `live_iterations=[1, 2, 4]`. The well-supported steps (1-3) land on the same iterations
  as the pilot's circular fit (1→2, 2→2); steps 4-8 shift around (pilot: `{...4:6, 5:4,
  6:2, 7:1, 8:1}`, live={2,4,6}) — those buckets have little support (step 5+ is rare in
  GSM8K-Aug), so treat the mapping as reliable through step ~3 and noisy beyond that, not
  as a contradiction of the pilot's "even iterations are live" story (which was itself
  drawn from the same small counts).
- **Held-out matched top1=0.1834 [0.168, 0.200], top5=0.2634, n=2126** — reassuringly
  close to the pilot's circular estimate (0.186/0.254): the circularity flagged in the
  pilot's caveats didn't materially inflate the number.
- **Train-corpus baseline** (mirrors recurrent_depth's methodology): most common step
  value in a 20k-example train sample is `"20"` (1,599/51,944 step-positions, 3.08%);
  matched against the same held-out population the decoding number uses, its hit rate is
  **0.0282** (n=2126). Matched top1 (0.1834) is **~6.5x** this baseline — same qualitative
  finding as the pilot's same-slice baseline (~4.3x), now on firmer methodological ground
  and comparable to recurrent_depth's ~8.9x (31.3%/3.5%) on the same kind of baseline.
- **Paper's own metric (Table 3), now well-powered:** correct-answers-only, all gold steps
  in the mapped iteration's top-5 — 1-step **0.545** (n=11), 2-step **0.060** (n=116),
  3-step **0.0** (n=78) — vs. the paper's reported 97.1%/83.9%/75.0%. At n=116 and n=78
  this is no longer dismissible as small-sample noise the way the pilot's n=17/8 buckets
  were. **Flagging a methodological hypothesis, not tested here:** the paper's own
  description (Sec. 5.1 case study) checks whether a value appears among a thought's
  top-tokens without committing to one fixed "correct" iteration per step the way this
  run's `best_iter_for_step` does — an "appears in ANY of the 6 iterations' top-5" version
  of this metric might close most of the gap, since CODI's mapping isn't a clean diagonal
  (iteration 4 alone decodes steps 1-4 reasonably at top1 0.20/0.17/0.11/0.06 per the full
  matrix in `manifest.json`). Worth a cheap follow-up before reading this as a reproduction
  failure.

**Task 2 — causal patching, scaled and stratified.**
350 pairs from half B only (227 at the focus iteration [2], 123 spread across the other
live iterations [1, 4]), each with baseline / real-donor-patch / any-iteration-control /
live-iteration-control outcomes on the same recipient (paired design).
- **Answer changed from baseline:** real patch **48.0%** (168/350, CI [0.428, 0.532]),
  any-iteration control **48.9%** (CI [0.437, 0.541]), live-iteration control **47.4%**
  (CI [0.423, 0.527]). **McNemar exact, patch vs. any-control: b=47, c=50, p=0.839**; vs.
  live-control: b=43, c=41, p=0.913. Neither is remotely significant — the real
  counterfactual value moves the answer no more than an unrelated one, now with a formal
  paired test instead of eyeballing two raw fractions.
- **Next-iteration read-out moved toward the injected value:** real **0.86%** (3/350),
  any-control **2.06%** (2/97 — qualifying pairs only, see Caveats), live-control **0.54%**
  (1/184). All three are at floor and statistically indistinguishable from each other on
  these tiny counts; the real condition is *not* higher than either control.
- **By iteration:** at the focus iteration (2, n=227, the strongest single decode
  position) patch=35.7% vs. control=37.0% — still no advantage for the real patch even
  where decodability is highest. At the other live iterations (n=123) patch=70.7% vs.
  control=70.7% — an exact tie in raw counts (87/123 both), likely reflecting that
  iterations 1 and 4 are simply high-variance under any perturbation rather than anything
  specific to the injected content; not investigated further here.

**Interpretation:**
- This is a tightened confirmation of the pilot, not a new finding: CODI's continuous
  thoughts are genuinely decodable (6.5x a train-corpus baseline, tight CI) but show
  **no causal faithfulness** under patching, now with a properly powered (n=350, not 30)
  paired design and a formal significance test rather than two raw percentages that
  happened to be close.
- **This independently corroborates a companion result already logged this session**:
  `results/.../` for recurrent_depth's own causal-patching test (commit `3e06224`, "Log
  recurrent_depth causal-patching pilot: symmetric null with CODI", n=200 pairs, the same
  real-vs-unrelated-control design) found the identical pattern there — next-iteration
  read-out movement at chance (2/130 both conditions), answer-proxy change 2.5% (real) vs.
  3.5% (control), McNemar p=0.73. **Both mechanisms in this repo — horizontal (CODI,
  continuous thoughts) and vertical (recurrent_depth, looped state) — now show decodable-
  but-not-faithful under matched causal-patching designs, independent of end-task accuracy**
  (CODI 41-44% vs. recurrent_depth tying its no-scratchpad control). That convergence,
  from two structurally different mechanisms tested the same way, is the strongest form of
  this finding available in the repo right now.
- The Table 3 reproduction gap (well-powered now, not just small-n) is the one open thread
  from this run — worth a cheap follow-up (the "any-iteration" metric variant above)
  before the Sep 25 writeup treats it as a real discrepancy with the paper.
**Gotchas hit:**
- Neither the local machine nor the RunPod `runpod/pytorch` image had `rsync` installed
  (both push and pull failed with "command not found"); used `tar czf - | ssh ... tar xzf -`
  instead for both directions. Cheap, no functional difference, but worth noting in
  `runpod_setup.sh` / the record-run skill so the next session doesn't rediscover it.
- Otherwise none new — the pilot's batch-size-1 `.squeeze(-1)` fix was already in place.
**Caveats:**
- `readout_moved_toward_donor`'s absolute counts are tiny (3/350, 2/97, 1/184) — the
  qualitative conclusion (at floor, real not better than either control) is robust, but
  don't quote the specific percentages as precise; the answer-changed metric (n=350,
  proper CIs and McNemar) is the stronger evidence here.
- `best_iter_for_step` is unstable at step counts ≥4 (little data); only steps 1-3 are on
  firm footing in both this run and the pilot.
- Live-iteration control's "qualifying pairs" for the readout check (n=97 for any-control)
  is smaller than n_pairs because a random any-iteration draw can land on a structurally
  dead iteration with no defined target value — expected, not a bug (see
  `scripts/decode_patch_codi.py`'s `value_for_iter`).
- Single seed (`greedy=True`, deterministic decode); n=659 for decoding, n=350 for
  patching — good power for the headline claims, as noted above where it isn't.
**Next:**
- Try the "value appears in ANY of the 6 iterations' top-5" variant of the Table 3 metric
  — cheap (reuses this run's cached per-iteration logits pattern), would clarify whether
  the paper-reproduction gap is real or a mapping-methodology artifact.
- Oct 9 / Sep 25 writeup: cite this run's CODI causal null together with
  recurrent_depth's (commit `3e06224`) as the same finding under two independent
  mechanisms and matched methodology — this is now the strongest evidence in the repo for
  the faithfulness-spine pivot.
- Coconut (paper checkpoint doesn't exist — confirmed this session; needs actual training,
  see `scripts/decode_patch_codi.py`'s sibling discussion in chat) remains the natural next
  test of whether this dissociation is specific to self-distillation (CODI) or general to
  continuous-thought mechanisms trained differently (curriculum learning).

**Correction (2026-09-19, appended by `20260919-090132_codi_ablate-attn`'s write-up):** this
run's model was never put in eval mode — `eval_codi.build_model` didn't call `model.eval()`,
and unlike `eval_codi.py`'s own main (which does, so the accuracy reproduction stands),
`decode_patch_codi.py` never added it. `get_peft_model` injects fresh LoRA layers whose
`Dropout(0.1)` modules default to training mode, so **LoRA dropout was active in every
forward pass here** (verified locally on peft 0.21.0; GPT-2's own dropouts were in eval).
Symptom: the greedy baseline decode of the same recipient disagrees with itself ~14-20% of
the time across repeats. Consequences: every "answer changed" rate above sits on a ~14-20%
re-decode noise floor (absolute rates inflated, power reduced); the real-vs-control
contrasts remain valid as paired comparisons (both conditions share the noise) and the
qualitative null is unchanged; decoding accuracy is if anything underestimated. Numbers
above are left as run. Fixed in `build_model` in commit for `20260919-090132_codi_ablate-attn`;
re-run in eval mode before citing these numbers (~$0.11 on an A5000).
