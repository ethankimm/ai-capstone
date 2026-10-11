#!/usr/bin/env python3
"""Per-level test eval of a no-CoT or CoT GPT-2 checkpoint trained by the Coconut repo
(`run.py`, configs in `configs/synthetic/`) on the synthetic depth / width tasks.

Runs in the Coconut venv (torch 2.5.1 / transformers 4.46.2), like the other
`*_coconut.py` scripts:
  cd /workspace/ai-capstone && .venv_coconut/bin/python scripts/eval_synthetic_coconut.py \\
      --method cot --checkpoint_path /workspace/coconut/checkpoints/gpt2_depth_cot/checkpoint_12 \\
      --test_json /workspace/synthetic_data/depth/test.json \\
      --train_log /workspace/logs/gpt2_depth_cot.log \\
      --slug depth-cot --stage full_run --hardware "RunPod RTX A5000"
  # smoke (first N examples, no logging):  ... --smoke_n 8 --verify_batched_n 8 --device cpu

Prompt = exactly what Coconut's `run.py` feeds `generate` at eval for cot / no_cot:
`tokenizer.encode(question + "\\n", add_special_tokens=True)` (`dataset.get_dataset`'s
`question_tokenized`; `get_question_latent_dataset(..., no_special_marker=True)` adds no
<|start-latent|>/<|end-latent|> markers for cot/no_cot), greedy, stop at EOS.
Differences from run.py's own val loop, on purpose:
  * batched (left-padded, length-bucketed) instead of batch 1 -- `--verify_batched_n`
    checks equality with the batch-1 path;
  * `max_new_tokens` defaults to longest gold CoT target + 16 (cot) / 16 (no_cot)
    instead of run.py's hardcoded 128 for non-"gsm" val paths -- a 16-step width CoT is
    ~131 tokens, which run.py's own per-epoch val accuracy truncates;
  * scoring: answer = text after the last "###" (`coconut_common.extract_answer_after_delimiter`),
    numeric compare (`latentreasoning.eval.metrics.is_correct`); no "###" = unparseable
    (we do NOT fall back to the last number: for width the max product appears in
    the steps, so a truncated CoT would be scored right by accident). run.py's own
    criterion (`full_text.split("#")[-1].replace(",", "").strip() == answer`) is also
    recorded per example as `coconut_native_correct`.

Per-example records: idx, id, level, task, question, gold_answer, raw_output,
predicted_answer, correct, coconut_native_correct, n_generated_tokens.
metrics.extra: task, task_level_range, seconds_per_epoch (from `--train_log` -- the
captured run.py output -- or `--seconds_per_epoch`), per-level accuracy / n / CI,
4PL midpoint, AUC (`scripts/depth_curve.py`), cot_tokens.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coconut_common import extract_answer_after_delimiter  # noqa: E402
from depth_curve import curve_metrics  # noqa: E402
from swap_random_coconut import load_model  # noqa: E402
from synthetic_common import annotate_records, coconut_cot_text, load_synthetic, parse_coconut_log  # noqa: E402

from latentreasoning.eval.metrics import is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402

MECHANISM = {"cot": "explicit_cot", "no_cot": "baseline_no_cot"}


@dataclass
class EvalArguments:
    checkpoint_path: str = field(metadata={"help": "Coconut run.py checkpoint_N (state dict)"})
    test_json: str = field(metadata={"help": "synthetic test.json"})
    method: str = field(default="cot", metadata={"help": "cot | no_cot"})
    model_id: str = field(default="openai-community/gpt2")
    checkpoint_label: Optional[str] = field(default=None)
    config_path: Optional[str] = field(default=None, metadata={"help": "Coconut yaml used for training (recorded)"})
    train_log: Optional[str] = field(default=None, metadata={"help": "captured run.py stdout+stderr"})
    seconds_per_epoch: Optional[float] = field(default=None, metadata={"help": "override if no train_log"})
    max_new_tokens: int = field(default=0, metadata={"help": "0 = auto (see docstring)"})
    batch_size: int = field(default=64)
    verify_batched_n: int = field(default=0)
    slug: Optional[str] = field(default=None)
    stage: str = field(default="full_run")
    hardware: str = field(default="RunPod GPU")
    smoke_n: int = field(default=0, metadata={"help": ">0: first n examples only, don't log"})
    device: str = field(default="cuda")


@torch.no_grad()
def greedy(model, prompts: list[list[int]], max_new_tokens: int, eos: int, device: str, batch_size: int):
    """(continuation ids before EOS, per prompt). batch_size 1 = run.py-style unpadded call
    (input_ids + all-ones attention_mask, as run.py's collator passes)."""
    out: list[list[int] | None] = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), batch_size):
        idxs = order[s:s + batch_size]
        L = max(len(prompts[i]) for i in idxs)
        ids = torch.tensor([[eos] * (L - len(prompts[i])) + prompts[i] for i in idxs], device=device)
        attn = torch.tensor([[0] * (L - len(prompts[i])) + [1] * len(prompts[i]) for i in idxs], device=device)
        gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=eos, pad_token_id=eos)
        for row, i in zip(gen, idxs):
            new = row[L:].tolist()
            out[i] = new[:new.index(eos)] if eos in new else new
    return out


def run_eval(model, tokenizer, ds, ea: EvalArguments, max_new: int, batch_size: int):
    eos = tokenizer.eos_token_id
    q_texts = [r["question"] + "\n" for r in ds.rows]
    prompts = [tokenizer.encode(q, add_special_tokens=True) for q in q_texts]
    t0 = time.perf_counter()
    conts = greedy(model, prompts, max_new, eos, ea.device, batch_size)
    elapsed = time.perf_counter() - t0
    records = []
    for ex, q, cont in zip(ds.examples, q_texts, conts):
        text = tokenizer.decode(cont, skip_special_tokens=True)
        pred = extract_answer_after_delimiter(text)
        native = (q + text).split("#")[-1].replace(",", "").strip()  # run.py decodes question+output
        records.append({
            "idx": ex.idx, "question": ex.question, "gold_answer": ex.answer, "raw_output": text,
            "predicted_answer": pred, "correct": is_correct(pred, ex.answer),
            "coconut_native_correct": native == ex.answer.replace(",", "").strip(),
            "n_generated_tokens": len(cont),
        })
    annotate_records(records, ds)
    return records, elapsed


def main() -> None:
    (ea,) = transformers.HfArgumentParser((EvalArguments,)).parse_args_into_dataclasses()
    assert ea.method in MECHANISM, ea.method
    argv = " ".join(sys.argv)
    tokenizer, model, _, _ = load_model(ea.model_id, ea.checkpoint_path, ea.device)
    n_params = sum(p.numel() for p in model.parameters())

    ds = load_synthetic(ea.test_json).subset(ea.smoke_n)
    task = ds.task
    if ea.max_new_tokens:
        max_new = ea.max_new_tokens
    elif ea.method == "cot":
        max_new = max(len(tokenizer.encode(coconut_cot_text(r), add_special_tokens=False)) for r in ds.rows) + 16
    else:
        max_new = 16
    print(f"method={ea.method} task={task} n={len(ds.rows)} levels={ds.level_range} max_new_tokens={max_new}", flush=True)

    if ea.verify_batched_n:
        sub = ds.subset(ea.verify_batched_n)
        r1, _ = run_eval(model, tokenizer, sub, ea, max_new, 1)
        rb, _ = run_eval(model, tokenizer, sub, ea, max_new, ea.batch_size)
        mism = [(a["id"], a["raw_output"], b["raw_output"]) for a, b in zip(r1, rb) if a["raw_output"] != b["raw_output"]]
        print(f"verify batched (bs={ea.batch_size}) vs batch-1 on {len(sub.rows)}: "
              f"{len(sub.rows) - len(mism)}/{len(sub.rows)} identical", flush=True)
        for m in mism[:5]:
            print("  MISMATCH", m)
        if mism:
            raise SystemExit("batched decode differs from batch-1 -- rerun with --batch_size 1")

    records, elapsed = run_eval(model, tokenizer, ds, ea, max_new, ea.batch_size)
    n = len(records)
    acc = sum(r["correct"] for r in records) / n
    unparseable = sum(r["predicted_answer"] is None for r in records) / n
    native_acc = sum(r["coconut_native_correct"] for r in records) / n
    curve = curve_metrics(records)
    print(f"accuracy={acc:.3f} (run.py criterion {native_acc:.3f}) unparseable={unparseable:.3f} "
          f"midpoint={curve['midpoint']} ({curve['midpoint_status']}) auc={curve['auc']:.3f} "
          f"{elapsed / n:.3f}s/example", flush=True)
    print("per-level:", json.dumps(curve["per_level_accuracy"]), flush=True)
    for r in records[:3]:
        print(f"  [{r['id']} L{r['level']}] gold={r['gold_answer']} pred={r['predicted_answer']} "
              f"out={r['raw_output'][:120]!r}", flush=True)

    if ea.smoke_n:
        print("smoke run -- not logging")
        return

    train_info = parse_coconut_log(Path(ea.train_log).read_text()) if ea.train_log else None
    secs = list(train_info["seconds_train_epoch"].values()) if train_info else []
    sec_per_epoch = ea.seconds_per_epoch if ea.seconds_per_epoch is not None else (sum(secs) / len(secs) if secs else None)
    if sec_per_epoch is None:
        print("WARNING: no --train_log / --seconds_per_epoch -> seconds_per_epoch=None")

    train_cfg = None
    if ea.config_path:
        import yaml  # in the Coconut venv via run.py's own requirements (datasets -> pyyaml)
        train_cfg = yaml.safe_load(Path(ea.config_path).read_text())

    mechanism = MECHANISM[ea.method]
    label = ea.checkpoint_label or ea.checkpoint_path
    record = RunRecord(
        run_id=new_run_id(mechanism, ea.slug or f"{task}-coconut-{ea.method.replace('_', '')}"),
        mechanism=mechanism,
        stage=ea.stage,
        model=ModelInfo(backbone=ea.model_id, checkpoint=label, n_params=n_params),
        dataset=DatasetInfo(name=f"synthetic-{task}", split="test", n_examples=n, seed=None),
        metrics={
            "final_answer_accuracy": acc,
            "unparseable_rate": unparseable,
            "sec_per_example": elapsed / n,
            "extra": {
                "task": task,
                "task_level_range": ds.level_range,
                "seconds_per_epoch": sec_per_epoch,
                "seconds_per_epoch_all": train_info["seconds_train_epoch"] if train_info else None,
                "monitor_accuracy_by_epoch": [v["accuracy"] for v in train_info["val_accuracy"]] if train_info else None,
                "coconut_native_accuracy": native_acc,
                "cot_tokens": sum(r["n_generated_tokens"] for r in records) / n,
                **curve,
            },
        },
        hyperparams={"method": ea.method, "trainer": "facebookresearch/coconut run.py @27273cb",
                     "config_path": ea.config_path, "max_new_tokens": max_new, "batch_size": ea.batch_size,
                     "test_json": ea.test_json, "decoding": "greedy", "train_config": train_cfg},
        seed=train_cfg.get("seed") if train_cfg else None,
        hardware=f"{ea.hardware} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        notes=f"Coconut-repo {ea.method} GPT-2 on synthetic {task}, per-level test eval.",
    )
    manifest = record.save(predictions=records)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    if train_info:
        (manifest.parent / "train_log.json").write_text(json.dumps(train_info, indent=2) + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
