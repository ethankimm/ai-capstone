#!/usr/bin/env python3
"""CODI latent-KV eviction: which of the 6 continuous-thought positions, at which layers,
does the answer actually read through attention? The thoughts are computed honestly
(unpatched); only the *answer phase* (eot + every generated answer token) is blocked from
attending to chosen thought positions at chosen layers, via an additive -inf mask injected
into each GPT2Attention call. This is KV-cache eviction used as a causal test -- unlike
the necessity/ablation runs, the latents themselves are untouched, so a drop isolates the
answer -> latent attention read.

Slot convention matches ablate_attention_codi.py / override_input_at: slot i (1..6) is the
sequence position of iteration i's input, absolute position L0 + i - 1 (L0 = question+bot).
The 6th iteration's output is never written to the cache (decode starts from eot).

Conditions: full; evict_all; keep_only_i / evict_only_i (i=1..6); keep_135 (the slots the
patching runs found carry the value) and keep_246 (control); evict all slots at one layer
band only (L0-3 / L4-7 / L8-11) and keep one band only; merge_mean (each layer's 6 slot
K/V vectors averaged into slot 1, slots 2-6 evicted -- the compression variant).

Runs on CPU in fp32 from a local CODI checkout (same convention as the other CODI scripts):

  cd <codi checkout> && .venv/bin/python <repo>/scripts/evict_kv_codi.py \\
      --ckpt_dir <hf snapshot of zen-E/CODI-gpt2> --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \\
      --model_name_or_path gpt2 --seed 11 --model_max_length 512 \\
      --lora_r 128 --lora_alpha 32 --lora_init --greedy True \\
      --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \\
      --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \\
      --device cpu --n 0
"""
from __future__ import annotations

import copy
import os
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
from evict_kv_common import N_LAYERS, AttnMasker, conditions, merge_slots, summarize  # noqa: E402

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.eval.metrics import extract_final_number, is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402


@dataclass
class EvictArguments:
    slug: str = field(default="evict-kv")
    stage: str = field(default="full_run")
    hardware: str = field(default="local Apple Silicon CPU")
    checkpoint_label: Optional[str] = field(default=None)
    device: str = field(default="cpu")
    n: int = field(default=0, metadata={"help": "first n test examples in file order; 0 = all 1319"})
    max_new_tokens: int = field(default=64)
    no_save: bool = field(default=False)


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, EvictArguments))
    model_args, data_args, training_args, ea = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = ea.device
    torch.manual_seed(training_args.seed)

    model, tokenizer, load_result = build_model(model_args, training_args, device=device, dtype=torch.float32)
    print(f"loaded {model_args.ckpt_dir}: missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}")
    masker = AttnMasker(model)
    assert masker.n_hooked == N_LAYERS, masker.n_hooked
    n_slots = training_args.inf_latent_iterations
    conds = conditions(n_slots, {"keep_135": {1, 3, 5}, "keep_246": {2, 4, 6}})

    examples = load_gsm8k_aug(split="test", n=None, seed=None)
    if ea.n:
        examples = examples[:ea.n]

    preds = []
    t0 = time.perf_counter()
    for n_done, ex in enumerate(examples, 1):
        q = ex.question.strip().replace("  ", " ")
        pkv, _, latent = run_thoughts(model, tokenizer, q, device, n_slots)
        L0 = pkv[0][0].shape[2] - n_slots
        pos = {i: L0 + i - 1 for i in range(1, n_slots + 1)}
        masker.ans_start = L0 + n_slots  # eot
        row = {"idx": ex.idx, "gold": ex.answer, "L0": L0, "outputs": {}, "correct": {}}
        for name, spec in conds.items():
            cache = copy.deepcopy(pkv)
            if spec is None:  # merge_mean
                merge_slots(cache, [pos[i] for i in range(1, n_slots + 1)])
                spec = {l: set(range(2, n_slots + 1)) for l in range(N_LAYERS)}
            masker.cols = {l: sorted(pos[i] for i in s) for l, s in spec.items() if s}
            try:
                out = decode_answer(model, tokenizer, cache, latent, device, ea.max_new_tokens)
            finally:
                masker.cols = {}
            row["outputs"][name] = out
            row["correct"][name] = is_correct(extract_final_number(out), ex.answer)
        preds.append(row)
        if n_done % 50 == 0 or n_done == len(examples):
            el = time.perf_counter() - t0
            acc = sum(r["correct"]["full"] for r in preds) / n_done
            print(f"{n_done}/{len(examples)} ({el:.0f}s, {el / n_done:.2f}s/ex, full acc {acc:.3f})", flush=True)
    elapsed = time.perf_counter() - t0

    n = len(preds)
    summary = summarize(preds, conds, extract_final_number)
    print(f"elapsed {elapsed:.0f}s")
    if ea.no_save:
        return

    record = RunRecord(
        run_id=new_run_id("codi", ea.slug),
        mechanism="codi",
        stage=ea.stage,
        model=ModelInfo(backbone=model_args.model_name_or_path, checkpoint=ea.checkpoint_label or model_args.ckpt_dir,
                        n_params=sum(p.numel() for p in model.parameters())),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=n, seed=None),
        metrics={"final_answer_accuracy": summary["full"]["acc"], "unparseable_rate":
                 sum(extract_final_number(r["outputs"]["full"]) is None for r in preds) / n,
                 "compute_steps": n_slots, "sec_per_example": elapsed / n,
                 "extra": {"conditions": summary, "dtype": "float32", "command": argv}},
        hyperparams={"num_latent": n_slots, "greedy": True, "max_new_tokens": ea.max_new_tokens,
                     "mask": "additive -inf on thought-slot key columns, answer phase only (eot + generated tokens)"},
        seed=training_args.seed,
        hardware=ea.hardware,
        notes="Answer-phase KV eviction of CODI's thought slots (by slot and layer band) as a causal test of which latents the answer reads.",
    )
    path = record.save(predictions=preds)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
