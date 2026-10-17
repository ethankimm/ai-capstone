#!/usr/bin/env python3
"""Coconut counterpart of evict_kv_codi.py: answer-phase KV eviction of the 6 continuous-thought
slots, by slot and by layer band (see evict_kv_common.py for the mask and the condition grid).

The latent passes run unmasked (`coconut_common.run_passes`). The answer phase then starts
at <|end-latent|>: the last pass (6th thought slot + <|end-latent|>) and greedy decoding run with a
KV cache, and only queries at or after <|end-latent|> are masked. `finish_and_decode` (the
reference `Coconut.generate` tail) re-runs the whole sequence each step with no cache. With no
mask, the cached decode here must give the same outputs; `--check_reference` asserts that for
the "full" condition.

Slot i = the i-th <|latent|> position = pass i-1's vector (`override_at_pass` key i-1). So the
necessity/patching runs' passes 1 and 4 are slots 2 and 5.

  .venv/bin/python scripts/evict_kv_coconut.py \\
      --checkpoint_path <hf snapshot>/checkpoint_33 --data_file <...>/gsm_original_test.json --n 0
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import transformers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coconut_common import (  # noqa: E402
    encode_question, extract_answer_after_delimiter, finish_and_decode, load_coconut, run_passes,
)
from evict_kv_common import N_LAYERS, AttnMasker, conditions, merge_slots, summarize  # noqa: E402

from latentreasoning.eval.metrics import is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402


@dataclass
class EvictArguments:
    checkpoint_path: str = field(metadata={"help": "the downloaded checkpoint_33 file"})
    data_file: str = field(metadata={"help": "gsm_original_test.json (Dilgren & Wiegreffe prep)"})
    model_id: str = field(default="openai-community/gpt2")
    checkpoint_label: str = field(default="hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33")
    num_latents: int = field(default=6)
    slug: str = field(default="evict-kv")
    stage: str = field(default="full_run")
    hardware: str = field(default="local Apple Silicon CPU")
    device: str = field(default="cpu")
    seed: int = field(default=0)
    n: int = field(default=0, metadata={"help": "first n test examples in file order; 0 = all 1319"})
    max_new_tokens: int = field(default=48)
    check_reference: bool = field(default=False, metadata={"help": "assert full == finish_and_decode per example"})
    no_save: bool = field(default=False)


@torch.no_grad()
def decode_cached(base_model, embedding, tokenizer, inputs_embeds, kv_cache, next_compute_range,
                  device, max_new_tokens: int, eos_token_id: int) -> str:
    """`finish_and_decode` with a KV cache: same final pass, then one token per forward."""
    lo, hi = next_compute_range
    seq_len = inputs_embeds.shape[1]
    out = base_model(
        inputs_embeds=inputs_embeds[:, lo:hi, :],
        attention_mask=torch.ones(1, hi, dtype=torch.long, device=device),
        position_ids=torch.arange(lo, hi, device=device).unsqueeze(0),
        past_key_values=[(k[:, :, :lo, :], v[:, :, :lo, :]) for k, v in kv_cache],
        use_cache=True,
    )
    pkv = out.past_key_values
    next_token = torch.argmax(out.logits[0, -1]).item()
    tokens = [next_token]
    for step in range(max_new_tokens - 1):
        out = base_model(
            inputs_embeds=embedding(torch.tensor([[next_token]], device=device)),
            position_ids=torch.tensor([[seq_len + step]], device=device),
            past_key_values=pkv, use_cache=True,
        )
        pkv = out.past_key_values
        next_token = torch.argmax(out.logits[0, -1]).item()
        if next_token == eos_token_id:
            break
        tokens.append(next_token)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def main() -> None:
    (ea,) = transformers.HfArgumentParser((EvictArguments,)).parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = ea.device
    torch.manual_seed(ea.seed)

    tokenizer, base_model, embedding, special_ids = load_coconut(ea.model_id, ea.checkpoint_path, device)
    masker = AttnMasker(base_model)
    assert masker.n_hooked == N_LAYERS, masker.n_hooked
    eos_id = tokenizer.eos_token_id
    n_slots = ea.num_latents
    conds = conditions(n_slots, {"keep_25": {2, 5}, "keep_1346": {1, 3, 4, 6},
                                 "keep_135": {1, 3, 5}, "keep_246": {2, 4, 6}})

    rows = json.loads(Path(ea.data_file).read_text())
    examples = [{"idx": i, "question": r["question"], "answer": r["answer"].strip()} for i, r in enumerate(rows)]
    if ea.n:
        examples = examples[:ea.n]

    preds = []
    n_ref_mismatch = 0
    t0 = time.perf_counter()
    for n_done, ex in enumerate(examples, 1):
        input_ids, attn = encode_question(tokenizer, special_ids, ex["question"], n_slots, device)
        _, ie, kv, ncr = run_passes(base_model, embedding, input_ids, attn, device, n_slots,
                                    latent_token_id=special_ids["latent"])
        latent_pos = (input_ids[0] == special_ids["latent"]).nonzero().squeeze(1).tolist()
        pos = {i: latent_pos[i - 1] for i in range(1, n_slots + 1)}
        masker.ans_start = latent_pos[-1] + 1  # <|end-latent|>
        row = {"idx": ex["idx"], "gold": ex["answer"], "outputs": {}, "correct": {}}
        for name, spec in conds.items():
            cache = copy.deepcopy(kv)
            if spec is None:  # merge_mean; slot 6's K/V is computed in the answer-phase pass, so merge 1-5 only
                merge_slots(cache, [pos[i] for i in range(1, n_slots)])
                spec = {l: set(range(2, n_slots + 1)) for l in range(N_LAYERS)}
            masker.cols = {l: sorted(pos[i] for i in s) for l, s in spec.items() if s}
            try:
                out = decode_cached(base_model, embedding, tokenizer, ie, cache, ncr, device, ea.max_new_tokens, eos_id)
            finally:
                masker.cols = {}
            row["outputs"][name] = out
            row["correct"][name] = is_correct(extract_answer_after_delimiter(out), ex["answer"])
        if ea.check_reference:
            ref = finish_and_decode(base_model, embedding, tokenizer, ie, kv, ncr, attn, device, ea.max_new_tokens, eos_id)
            if ref != row["outputs"]["full"]:
                n_ref_mismatch += 1
                print(f"  ref mismatch idx {ex['idx']}: cached={row['outputs']['full']!r} ref={ref!r}")
        preds.append(row)
        if n_done % 50 == 0 or n_done == len(examples):
            el = time.perf_counter() - t0
            acc = sum(r["correct"]["full"] for r in preds) / n_done
            print(f"{n_done}/{len(examples)} ({el:.0f}s, {el / n_done:.2f}s/ex, full acc {acc:.3f})", flush=True)
    elapsed = time.perf_counter() - t0

    n = len(preds)
    summary = summarize(preds, conds, extract_answer_after_delimiter)
    print(f"elapsed {elapsed:.0f}s; reference mismatches {n_ref_mismatch}" if ea.check_reference else f"elapsed {elapsed:.0f}s")
    if ea.no_save:
        return

    record = RunRecord(
        run_id=new_run_id("coconut", ea.slug),
        mechanism="coconut",
        stage=ea.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=ea.checkpoint_label,
                        n_params=sum(p.numel() for p in base_model.parameters())),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=n, seed=None),
        metrics={"final_answer_accuracy": summary["full"]["acc"], "unparseable_rate":
                 sum(extract_answer_after_delimiter(r["outputs"]["full"]) is None for r in preds) / n,
                 "compute_steps": n_slots, "sec_per_example": elapsed / n,
                 "extra": {"conditions": summary, "dtype": "float32", "command": argv,
                           "data_file": "gsm_original_test.json (Dilgren & Wiegreffe prep)",
                           "reference_mismatches": n_ref_mismatch if ea.check_reference else None}},
        hyperparams={"num_latents": n_slots, "greedy": True, "max_new_tokens": ea.max_new_tokens,
                     "mask": "additive -inf on thought-slot key columns, queries at/after <|end-latent|> only"},
        seed=ea.seed,
        hardware=ea.hardware,
        notes="Answer-phase KV eviction of Coconut's thought slots (by slot and layer band), the Coconut mirror of the CODI evict-kv run.",
    )
    path = record.save(predictions=preds)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
