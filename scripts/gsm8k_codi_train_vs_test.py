#!/usr/bin/env python3
"""Memorisation check for the released GSM8K CODI (`zen-E/CODI-gpt2` @ fd641b3): accuracy on items it was trained on
(the repo's fixed 1000-item `split="validation"` carve-out of GSM8K-Aug train; CODI trained on all 385,620 train rows,
so these are seen items) vs the GSM8K test split, same generation code, batch size 1 (no padding, so upstream
masking is exact), greedy, the paper's last-number metric and the shared `score_outputs`. Stratified by the number of
`<<...>>` steps in the rationale. Counterpart of the RuleChain train-vs-test check in
results/20261102-225018_codi_v2-depth2-codi-latent-diagnostics.

Caveat built into the design: GSM8K-Aug train is model-generated augmentation of GSM8K train, test is original GSM8K
test, so a gap mixes memorisation with distribution shift. A small gap is informative; a large one is not, on its own.

Run from the CODI checkout in its venv (flags as gsm8k_codi_lens_collapse.py, --batch_size 1).
"""
from __future__ import annotations

import collections
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.model import DataArguments, ModelArguments, TrainingArguments  # noqa: E402

from eval_codi import build_model, extract_answer_number, generate_batches  # noqa: E402

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.eval.harness import score_outputs  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402

if not torch.cuda.is_available():
    _torch_load = torch.load
    torch.load = lambda *a, **kw: _torch_load(*a, **{"map_location": "cpu", **kw})


@dataclass
class Args:
    n: int = field(default=0, metadata={"help": "cap per split (0 = 1000 seen / 1319 test)"})
    slug: str = field(default="gsm8k-released-train-vs-test")
    stage: str = field(default="full_run")
    hardware: str = field(default="local")
    checkpoint_label: Optional[str] = field(default="hf:zen-E/CODI-gpt2@fd641b3")
    no_log: bool = field(default=False)


def evaluate(model, tokenizer, exs, data_args, training_args):
    qs = [ex.question.strip().replace("  ", " ") for ex in exs]
    outs = generate_batches(model, tokenizer, qs, data_args, training_args)
    # gold parsed like the prediction (last number): a few GSM8K-Aug train answers carry units, e.g. "$165"
    paper = [extract_answer_number(o) == extract_answer_number(ex.answer) for o, ex in zip(outs, exs)]
    res = score_outputs(exs, outs)
    by_steps = collections.defaultdict(list)
    for ex, ok in zip(exs, paper):
        by_steps[min(len(ex.intermediate_values), 6)].append(ok)
    return outs, paper, res, {s: (round(sum(v) / len(v), 4), len(v)) for s, v in sorted(by_steps.items())}


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, Args))
    model_args, data_args, training_args, a = parser.parse_args_into_dataclasses()
    assert data_args.batch_size == 1, "batch size 1: no padding, so upstream (unmasked) latent passes are exact"
    argv = " ".join(sys.argv)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = build_model(model_args, training_args, device=device, dtype=torch.float32)
    seen = load_gsm8k_aug(split="validation", n=a.n or None, seed=None)
    test = load_gsm8k_aug(split="test", n=None, seed=None)
    if a.n:
        test = test[:a.n]
    t0 = time.perf_counter()
    out = {}
    for name, exs in (("seen_train", seen), ("test", test)):
        outs, paper, res, steps = evaluate(model, tokenizer, exs, data_args, training_args)
        acc = sum(paper) / len(paper)
        out[name] = {"n": len(exs), "paper_metric_accuracy": round(acc, 4),
                     "shared_metric_accuracy": round(res.final_answer_accuracy, 4),
                     "unparseable_rate": round(res.unparseable_rate, 4), "accuracy_by_steps": steps,
                     "records": [{**r, "split": name, "paper_metric_correct": p, "n_steps": len(ex.intermediate_values)}
                                 for r, p, ex in zip(res.records, paper, exs)]}
        print(f"{name}: n={len(exs)} paper-metric acc {acc:.3f} shared {res.final_answer_accuracy:.3f} | by #steps "
              f"{steps} ({time.perf_counter() - t0:.0f}s)", flush=True)
    gap = {s: round(out["seen_train"]["accuracy_by_steps"][s][0] - out["test"]["accuracy_by_steps"][s][0], 4)
           for s in out["test"]["accuracy_by_steps"] if s in out["seen_train"]["accuracy_by_steps"]}
    print(f"gap (seen - test) overall {out['seen_train']['paper_metric_accuracy'] - out['test']['paper_metric_accuracy']:+.3f}"
          f" | by #steps {gap}", flush=True)
    if a.no_log:
        return
    records = out["test"].pop("records") + out["seen_train"].pop("records")
    record = RunRecord(
        run_id=new_run_id("codi", a.slug), mechanism="codi", stage=a.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=a.checkpoint_label or model_args.ckpt_dir,
                        n_params=sum(p.numel() for p in model.parameters())),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=out["test"]["n"], seed=None),
        metrics={"final_answer_accuracy": out["test"]["shared_metric_accuracy"],
                 "unparseable_rate": out["test"]["unparseable_rate"],
                 "compute_steps": training_args.inf_latent_iterations,
                 "sec_per_example": (time.perf_counter() - t0) / (out["test"]["n"] + out["seen_train"]["n"]),
                 "extra": {**out, "gap_by_steps": gap,
                           "seen_split_note": "repo split='validation' = 1000 fixed items of GSM8K-Aug train, all in "
                                              "CODI's training data"}},
        hyperparams={"latent_count": training_args.inf_latent_iterations, "batch_size": 1, "greedy": True,
                     "masking": "upstream (no padding at batch 1)", "device": device},
        seed=None, hardware=f"{a.hardware} / {device}",
        notes="Released GSM8K CODI: accuracy on seen train items vs test (memorisation check).",
    )
    manifest = record.save(predictions=records)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id}")


if __name__ == "__main__":
    main()
