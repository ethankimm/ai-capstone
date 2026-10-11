#!/usr/bin/env python3
"""Per-level test eval of a CODI checkpoint trained on the synthetic depth / width tasks
(`configs/synthetic/train_gpt2_depth_codi.sh`, CODI repo + scripts/codi_synthetic.patch).

Runs INSIDE the CODI venv, from the CODI checkout, like `eval_codi.py` (whose model
construction and the reference test.py generation loop it reuses unchanged):

  cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/eval_synthetic_codi.py \\
      --ckpt_dir /workspace/codi_runs/gpt2_depth_codi/gpt2/ep_10/lr_0.003/seed_11 \\
      --test_json /workspace/synthetic_data/depth/test.json --slug synthetic-depth-codi \\
      --stage pilot --hardware "RunPod A40 (secure)" --output_dir /tmp/codi_eval_unused \\
      --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \\
      --lora_r 128 --lora_alpha 32 --lora_init --batch_size 256 --greedy True \\
      --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \\
      --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True
  # monitor a mid-training checkpoint (no RunRecord): add --no_log True, --test_json .../valid_monitor.json

Prompt = the training format of the patch's `depth`/`width` branch: question.strip() + "\\n",
then <bot>, 6 latent iterations, <eot>, greedy decode. Scoring: last number in the output
(CODI test.py's metric, the target is "The answer is: N"), compared with
`latentreasoning.eval.metrics.is_correct`. Per-level accuracy / midpoint / AUC via
`scripts/depth_curve.py`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers

sys.path.insert(0, os.getcwd())  # the CODI checkout: `src.model`
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.model import DataArguments, ModelArguments, TrainingArguments  # noqa: E402

from depth_curve import curve_metrics  # noqa: E402
from eval_codi import build_model, generate_batches, hyperparams_from_ckpt  # noqa: E402
from synthetic_common import annotate_records, load_synthetic  # noqa: E402

from latentreasoning.eval.metrics import is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402


@dataclass
class SynthEvalArguments:
    test_json: str = field(default="")
    slug: str = field(default="synthetic-codi")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: Optional[str] = field(default=None)
    train_log_from: Optional[str] = field(default=None, metadata={"help": "trainer_state.json"})
    seconds_per_epoch: Optional[float] = field(default=None)
    smoke_n: int = field(default=0)
    no_log: bool = field(default=False, metadata={"help": "print accuracy only, no RunRecord"})


def last_number(text: str) -> Optional[str]:
    import re
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, SynthEvalArguments))
    model_args, data_args, training_args, ea = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    model, tokenizer, load_result = build_model(model_args, training_args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {model_args.ckpt_dir}: missing={len(load_result.missing_keys)} "
          f"unexpected={len(load_result.unexpected_keys)}", flush=True)

    ds = load_synthetic(ea.test_json).subset(ea.smoke_n)
    questions = [r["question"].strip() + "\n" for r in ds.rows]
    t0 = time.perf_counter()
    outputs = generate_batches(model, tokenizer, questions, data_args, training_args)
    elapsed = time.perf_counter() - t0

    records = []
    for ex, out in zip(ds.examples, outputs):
        pred = last_number(out)
        records.append({"idx": ex.idx, "question": ex.question, "gold_answer": ex.answer, "raw_output": out,
                        "predicted_answer": pred, "correct": is_correct(pred, ex.answer)})
    annotate_records(records, ds)
    n = len(records)
    acc = sum(r["correct"] for r in records) / n
    unparseable = sum(r["predicted_answer"] is None for r in records) / n
    curve = curve_metrics(records)
    print(f"accuracy={acc:.3f} unparseable={unparseable:.3f} midpoint={curve['midpoint']} "
          f"({curve['midpoint_status']}) auc={curve['auc']:.3f} {elapsed / n:.4f}s/example", flush=True)
    print("per-level:", json.dumps(curve["per_level_accuracy"]), flush=True)
    for r in records[:3]:
        print(f"  [{r['id']} L{r['level']}] gold={r['gold_answer']} pred={r['predicted_answer']} "
              f"out={r['raw_output'][:120]!r}", flush=True)
    if ea.smoke_n or ea.no_log:
        print("not logging (smoke_n / no_log)")
        return

    log_history = None
    if ea.train_log_from:
        log_history = json.loads(Path(ea.train_log_from).read_text()).get("log_history", [])
    hp = hyperparams_from_ckpt(model_args.ckpt_dir) or {}
    hp["inference"] = {"inf_latent_iterations": training_args.inf_latent_iterations, "greedy": training_args.greedy,
                       "batch_size": data_args.batch_size, "max_new_tokens": 256,
                       "prompt": "question.strip() + '\\n' + <bot>"}
    hp["test_json"] = ea.test_json
    record = RunRecord(
        run_id=new_run_id("codi", ea.slug),
        mechanism="codi",
        stage=ea.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=ea.checkpoint_label or model_args.ckpt_dir, n_params=n_params),
        dataset=DatasetInfo(name=f"synthetic-{ds.task}", split="test", n_examples=n, seed=None),
        metrics={
            "final_answer_accuracy": acc,
            "unparseable_rate": unparseable,
            "compute_steps": training_args.inf_latent_iterations,
            "sec_per_example": elapsed / n,
            "extra": {"task": ds.task, "task_level_range": ds.level_range,
                      "seconds_per_epoch": ea.seconds_per_epoch, **curve},
        },
        hyperparams=hp,
        seed=hp.get("seed"),
        hardware=f"{ea.hardware} / {torch.cuda.get_device_name(0)}",
        notes=f"CODI GPT-2 on synthetic {ds.task}, per-level test eval.",
    )
    manifest = record.save(predictions=records)
    if log_history is not None:
        (manifest.parent / "train_log.json").write_text(json.dumps(log_history, indent=2))
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
