#!/usr/bin/env python3
"""P1 (RESEARCH_PLAN §4): transfer-ladder levels 2-4 for CODI. The same recipients get a
donor at each level (`ladder_common.py`: L2 same operator sequence, L3 same length but
different operators, L4 any problem = random-donor control), and the donor's latents are
transplanted with the aligned sites of the E3 rerun
(`20260927-004340_codi_minimal-pair-patch-aligned`): site s = z_s, fed into iteration s+1.

Conditions per (recipient, level): all_slot (z_0..z_5), carriers (default z0,z2,z4 -- the
sites that carry the value in aligned E3), and each carrier alone. Scored into
`ladder_common.BUCKETS` against a permutation null. Level 1 is E3 itself.

Run inside the CODI venv, from the CODI checkout:

  cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/patch_ladder_codi.py \\
      --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \\
      --slug ladder-patch --stage full_run --hardware "RunPod RTX A5000 (secure)" \\
      --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \\
      --lora_r 128 --lora_alpha 32 --lora_init --greedy True \\
      --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \\
      --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \\
      --output_dir /tmp/o --eval_n 0 --n_recipients 1000
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())  # the CODI checkout: `src.model`

from src.model import DataArguments, ModelArguments, TrainingArguments  # noqa: E402
from eval_codi import build_model  # noqa: E402
from decode_patch_codi import decode_answer, run_thoughts  # noqa: E402
from ladder_common import LEVELS, build_ladder_pairs, format_row, summarize, targets  # noqa: E402

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.eval.metrics import extract_final_number, is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402


@dataclass
class LadderArguments:
    slug: str = field(default="ladder-patch")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: Optional[str] = field(default=None)
    eval_n: int = field(default=0, metadata={"help": "0 = full test set"})
    eval_seed: int = field(default=0)
    n_recipients: int = field(default=1000)
    carriers: str = field(default="0,2,4")
    pair_seed: int = field(default=0)
    reps: int = field(default=200)
    max_new_tokens: int = field(default=64)
    smoke_n: int = field(default=0, metadata={"help": ">0: first n examples only, don't log"})


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LadderArguments))
    model_args, data_args, training_args, la = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = "cuda"

    model, tokenizer, load_result = build_model(model_args, training_args)
    assert not model.training, "model must be in eval mode"
    n_params = sum(p.numel() for p in model.parameters())
    n_latents = training_args.inf_latent_iterations
    sites = list(range(n_latents))
    carriers = [int(s) for s in la.carriers.split(",")]
    conditions = {"all_slot": sites, "carriers": carriers, **{f"z{s}": [s] for s in carriers}}

    if la.eval_n:
        examples = load_gsm8k_aug(split="test", n=la.eval_n, seed=la.eval_seed)
    else:
        examples = load_gsm8k_aug(split="test", n=None, seed=None)
    if la.smoke_n:
        examples = examples[:la.smoke_n]
    print(f"n_examples={len(examples)}, conditions={conditions}")

    def prep(question: str) -> str:
        return question.strip().replace("  ", " ")

    t0 = time.perf_counter()
    base, correct, z_cache = {}, {}, {}
    for ex in examples:
        pkv, thoughts, latent = run_thoughts(model, tokenizer, prep(ex.question), device, n_latents, include_latent0=True)
        pred = extract_final_number(decode_answer(model, tokenizer, pkv, latent, device, la.max_new_tokens))
        base[ex.idx], correct[ex.idx] = pred, is_correct(pred, ex.answer)
        z_cache[ex.idx] = {rec["iter"]: rec["post"] for rec in thoughts if rec["iter"] < n_latents}
    decode_elapsed = time.perf_counter() - t0
    accuracy = sum(correct.values()) / len(correct)
    print(f"decode pass {decode_elapsed:.0f}s, base accuracy={accuracy:.3f}")

    pairs = build_ladder_pairs(examples, correct, random.Random(la.pair_seed), la.n_recipients)
    print(f"ladder recipients (L2, L3 and L4 donors available): {len(pairs)}")

    rows = []
    t1 = time.perf_counter()
    for pi, p in enumerate(pairs):
        r = p["recipient"]
        row = {"recipient_idx": r.idx, "recipient_answer": r.answer, "base": base[r.idx]}
        for level in LEVELS:
            d = p[level]
            z = z_cache[d.idx]
            preds = {}
            for name, site_set in conditions.items():
                pkv, _, latent = run_thoughts(model, tokenizer, prep(r.question), device, n_latents,
                                              override_input_at={s + 1: z[s] for s in site_set})
                preds[name] = extract_final_number(decode_answer(model, tokenizer, pkv, latent, device, la.max_new_tokens))
            row[level] = {"donor_idx": d.idx, "donor_answer": d.answer, "targets": targets(r, d), "preds": preds}
        rows.append(row)
        if (pi + 1) % 25 == 0:
            print(f"patched {pi + 1}/{len(pairs)} recipients ({time.perf_counter() - t1:.0f}s)", flush=True)
    patch_elapsed = time.perf_counter() - t1

    rng = random.Random(la.pair_seed + 1)
    summary = {level: {name: summarize(rows, name, level, la.reps, rng) for name in conditions} for level in LEVELS}
    for level in LEVELS:
        for name in conditions:
            print(format_row(f"{level} {name}", summary[level][name]))

    if la.smoke_n:
        print("smoke run -- not logging")
        return

    record = RunRecord(
        run_id=new_run_id("codi", la.slug),
        mechanism="codi",
        stage=la.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=la.checkpoint_label or model_args.ckpt_dir, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(examples),
                            seed=la.eval_seed if la.eval_n else None),
        metrics={
            "final_answer_accuracy": accuracy,
            "compute_steps": n_latents,
            "sec_per_example": decode_elapsed / len(examples),
            "intervention_accuracy": summary["L2"]["all_slot"]["rates"]["donor_final"],
            "extra": {
                "design": "P1 transfer ladder L2 (same ops) / L3 (same length, other ops) / L4 (any problem), "
                          "same recipients at every level, aligned sites; buckets + permutation null "
                          "(scripts/ladder_common.py)",
                "site_definition": "site s = z_s = latent fed into loop iteration s+1; z_0 = latent-0",
                "conditions": {k: v for k, v in conditions.items()},
                "n_recipients": len(rows), "summary": summary,
                "decode_elapsed_sec": decode_elapsed, "patch_elapsed_sec": patch_elapsed,
                "related_runs": ["20260927-004340_codi_minimal-pair-patch-aligned"],
            },
        },
        hyperparams={"inf_latent_iterations": n_latents, "carriers": carriers, "n_recipients_cap": la.n_recipients,
                     "pair_seed": la.pair_seed, "reps": la.reps, "max_new_tokens": la.max_new_tokens},
        seed=la.pair_seed,
        hardware=f"{la.hardware} / {torch.cuda.get_device_name(0)}",
        notes="P1 transfer ladder, CODI: donors at L2/L3/L4 transplanted into the same recipients "
              "(all-slot, carriers z0/z2/z4, each carrier alone).",
    )
    manifest = record.save(predictions=rows)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
