## 2026-09-18 — explicit CoT baseline, full-scale (explicit_cot, run_id: 20260918-021435_explicit_cot_baseline-full)

**Goal:** The unbudgeted visible-reasoning baseline at full scale, on the same recipe
as the full-scale filler_tokens runs (3 epochs, full fine-tune, lr 5e-5, batch 16),
so the three same-recipe numbers — no-CoT control / filler / explicit CoT — are
finally on equal footing. Replaces the 20k-example pilot (4.5%,
`20260917-075729_explicit_cot_baseline-pilot`) as the number to quote.
**Mechanism / model:** `explicit_cot`, gpt2 (full fine-tune, 124,439,808 params) /
`results/20260918-021435_explicit_cot_baseline-full/ckpt` (HF `save_pretrained`,
`model.safetensors` sha256 `3c3ed6cd…68e9`, 498 MB — synced to this machine, also
still on pod `kld6kfwefo73vf` until it's terminated). Unbudgeted: no `compute_steps`;
reports `extra.cot_tokens=30.07` (avg. generated rationale+answer length).
**Data:** gsm8k-aug train, all 384,620 examples (the 1000-example fixed validation
carve-out excluded), 3 epochs = 72,117 steps; eval gsm8k-aug test n=200 seed=0; run
seed=42.
**Hyperparams:** {'lr': 5e-05, 'epochs': 3, 'batch_size': 16, 'train_n': 384620,
'max_new_tokens': 256}; bf16, HF Trainer defaults otherwise (linear decay, no warmup,
AdamW); greedy decoding at eval.
**Command:** `uv run python scripts/train_explicit_cot.py --train-n -1 --stage full_run --hardware "RunPod RTX A5000 (secure)" --slug baseline-full`
(via `/workspace/run_explicit_full.sh` on pod `kld6kfwefo73vf`; 02:12–03:48 UTC,
train_runtime 5596 s ≈ 93 min at 12.9 steps/s, ~$0.45 of pod time)
**Headline results:** `final_answer_accuracy=0.345` (69/200), `unparseable_rate=0.000`,
`train_loss=0.3089` (epoch-avg; logged loss 1.74 → 0.23 over training),
`sec_per_example=0.264` (unbatched greedy), `extra.cot_tokens=30.07`.
**Interpretation:** Visible CoT at full scale gets 34.5%, vs 13.0% for the same-recipe
no-CoT control (`20260917-163049_filler_tokens_budget-0-control-full`) and 12.0% for
32 filler tokens (`20260917-190640_filler_tokens_budget-32-full`) — so on this
recipe, GPT-2 gets ~2.7x from actually writing the intermediate steps and nothing
from content-free extra positions. Every output is well-formed (calculator rationale
then `#### N`; 0 unparseable), and the errors are wrong arithmetic-chain choices, not
formatting (e.g. `<<3.5*8=28>> <<52-28=20>>` where the gold chain needed one more
subtraction). Against CODI's released weights on the identical 200 examples
(`20260918-021217_codi_released-weights-6lat`, 41.5%): 52 both right, 17 CoT-only,
31 CODI-only, 100 neither. CODI's edge is real on this slice but the recipes differ
(40-epoch LoRA vs 3-epoch full FT) — see Caveats. The paper's own CoT-SFT GPT-2
number is ≈44% (their recipe), so 34.5% is in the expected range for a 3-epoch
full fine-tune, not a red flag.
**Gotchas hit:**
- The script's `RunRecord.notes` string said "pilot" regardless of `--stage`
  (hardcoded); this manifest's `notes` field therefore reads "explicit_cot pilot
  fine-tune on 384620 train examples". It IS the full run (`stage=full_run`,
  `train_n=384620`). Script fixed for future runs; manifest left as written.
- Nothing else — the A5000 ran at 2x the RTX 2000 Ada's step rate for 3 cents/hr
  more; worth defaulting to it for these ~100-min runs.
**Caveats:**
- n=200 → ±~6.5 points 95% CI; the 34.5 vs 41.5 CODI gap is within noise at this n
  (the full-split CODI number is 43.7%; we have no full-split number for this model
  yet — cheap to add, ~6 min unbatched).
- Recipe asymmetry vs CODI: CODI = 40 epochs LoRA r=128 lr 3e-3; this = 3 epochs
  full FT lr 5e-5. Same-recipe comparisons are only within {this, filler_tokens
  full-scale runs}. Budget-matching (30 visible tokens here vs 6 latents vs 32
  fillers) is a separate decision — see the record-run skill's compute-budget rule.
- Not deduplicated against the filler_tokens faithful runs' 50/50 CoT mixture — the
  faithful `compute_steps=0` run trains on CoT half the time and is a different
  control; compare this against the non-faithful control (13.0%).
**Next:**
- Full-split (1319) eval of this checkpoint so it can be quoted against CODI's 43.7%
  on equal n.
- Oct 9: this checkpoint's rationale tokens are the "visible scratchpad" reference
  for the decoding probes — `Example.intermediate_values` are the targets.
