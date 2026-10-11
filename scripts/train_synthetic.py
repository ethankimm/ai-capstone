#!/usr/bin/env python3
"""Filler-token / explicit-CoT fine-tune + per-level eval on the synthetic depth / width
tasks (GPT-2). A thin sibling of `train_filler_tokens.py` / `train_explicit_cot.py`:
it imports their dataset builders (`FillerDataset`, `MixedFillerCotDataset`,
`CotDataset`, `collate`) and the mechanism prompt builders unchanged, so the token
format, loss masking, and the 50/50 faithful mixture are byte-identical to the GSM8K
runs. Those two scripts are untouched (logged GSM8K results stay reproducible); what's
new lives here:

  * data from synthetic JSON (`--train-json/--val-json/--test-json`, `steps` joined
    with " " into the GSM8K-style `<<a*b=c>>` rationale -- see `synthetic_common.py`);
  * per-epoch GENERATION accuracy on the monitor set (`valid_monitor.json`, 30/level),
    logged into train_log.json next to the Trainer loss history;
  * `seconds_per_epoch` = wall-clock of the training part of each epoch only
    (CUDA-synchronized; monitor eval and checkpoint writes excluded);
  * checkpoints in `--output-dir`: `best/` (highest monitor accuracy so far, strict
    improvement) and `last/`; the test set is scored with `--eval-checkpoint`
    (default best);
  * `--early-stop-patience N`: stop when monitor accuracy hasn't improved by >=
    `--early-stop-min-delta` (default 0.005 = 0.5 pt) over the best-so-far for N
    consecutive epochs (Keras semantics: the reference only moves on a >= min-delta gain).
    The LR schedule is still the Trainer's linear decay over `--epochs` -- an early-stopped
    run ends with LR not fully decayed;
  * batched left-padded greedy decode, length-bucketed (`--gen-batch-size`; 1 = the
    old unpadded batch-1 path). `--verify-batched-n K` decodes the first K test
    examples both ways and fails loudly on any mismatch;
  * per-example records carry `id`, `level`, `task`; metrics.extra carries
    `task`, `task_level_range`, `seconds_per_epoch`, per-level accuracy, 4PL midpoint
    and AUC (`scripts/depth_curve.py`).

Faithful filler mode trains on the same 50/50 CoT/filler mixture as
`train_filler_tokens.py --faithful`; eval is filler-only (as in the paper).

Usage (RunPod, main venv with `uv sync --extra train`):
  D=~/Documents/Penn/CIS5980/synthetic_data/depth
  uv run python scripts/train_synthetic.py --method filler_tokens --faithful --compute-steps 6 \\
      --train-json $D/train.json --val-json $D/valid_monitor.json --test-json $D/test.json \\
      --epochs 20 --early-stop-patience 3 --output-dir /workspace/ckpt/depth-filler6 \\
      --stage full_run --hardware "RunPod RTX A5000"
  # smoke (no logging, nothing under results/):
  uv run python scripts/train_synthetic.py --method filler_tokens --faithful --compute-steps 6 \\
      --train-json ... --val-json ... --test-json ... --smoke-n 8 --epochs 2 --verify-batched-n 8
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_curve import curve_metrics  # noqa: E402
from synthetic_common import SyntheticSet, annotate_records, load_synthetic  # noqa: E402
from train_explicit_cot import CotDataset  # noqa: E402
from train_filler_tokens import FAITHFUL_COT_RATE, FillerDataset, MixedFillerCotDataset, collate  # noqa: E402

from latentreasoning.eval.harness import score_outputs  # noqa: E402
from latentreasoning.mechanisms import explicit_cot, filler_tokens  # noqa: E402
from latentreasoning.runlog.manifest import RESULTS_DIR, DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402
from latentreasoning.utils import set_seed  # noqa: E402

MODEL_ID = "gpt2"


# ---------------------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------------------
def prompt_ids(method: str, tokenizer, question: str, compute_steps: int, filler_token: str) -> list[int]:
    if method == "filler_tokens":
        return filler_tokens.build_eval_prompt_ids(tokenizer, question, compute_steps, filler_token)
    return explicit_cot.build_eval_prompt_ids(tokenizer, question)


@torch.no_grad()
def greedy_decode(model, tokenizer, prompts: list[list[int]], max_new_tokens: int, device: str,
                  batch_size: int) -> tuple[list[str], list[int]]:
    """Greedy continuation of each prompt; returns (texts, n_generated_tokens_before_eos).
    batch_size == 1: the old per-example path (no padding, no attention mask -- exactly
    `train_*.py`'s `generate`). batch_size > 1: prompts sorted by length, left-padded
    with EOS (attention_mask 0 over the pad; GPT-2 position ids are derived from the
    mask by `generate`)."""
    eos = tokenizer.eos_token_id
    texts: list[str | None] = [None] * len(prompts)
    n_new: list[int] = [0] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for start in range(0, len(order), batch_size):
        idxs = order[start:start + batch_size]
        L = max(len(prompts[i]) for i in idxs)
        if batch_size == 1:
            ids = torch.tensor([prompts[idxs[0]]], device=device)
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=eos)
        else:
            ids = torch.tensor([[eos] * (L - len(prompts[i])) + prompts[i] for i in idxs], device=device)
            attn = torch.tensor([[0] * (L - len(prompts[i])) + [1] * len(prompts[i]) for i in idxs], device=device)
            out = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=eos, eos_token_id=eos)
        for row, i in zip(out, idxs):
            new = row[L:].tolist()
            n = new.index(eos) if eos in new else len(new)
            texts[i] = tokenizer.decode(new, skip_special_tokens=True)
            n_new[i] = n
    return texts, n_new  # type: ignore[return-value]


def evaluate(model, tokenizer, ds: SyntheticSet, args, device: str, batch_size: int | None = None):
    was_training = model.training
    model.eval()
    prompts = [prompt_ids(args.method, tokenizer, ex.question, args.compute_steps, args.filler_token)
               for ex in ds.examples]
    t0 = time.perf_counter()
    texts, n_new = greedy_decode(model, tokenizer, prompts, args.max_new_tokens, device,
                                 batch_size or args.gen_batch_size)
    elapsed = time.perf_counter() - t0
    result = score_outputs(ds.examples, texts)
    result.sec_per_example = elapsed / len(ds.examples) if ds.examples else 0.0
    annotate_records(result.records, ds)
    for rec, n in zip(result.records, n_new):
        rec["n_generated_tokens"] = n
    if was_training:
        model.train()
    return result, elapsed


# ---------------------------------------------------------------------------------------
# per-epoch monitor eval + early stopping
# ---------------------------------------------------------------------------------------
class EpochMonitor(TrainerCallback):
    def __init__(self, tokenizer, monitor: SyntheticSet, args, device: str, ckpt_dir: Path):
        self.tokenizer, self.monitor, self.args, self.device, self.ckpt_dir = tokenizer, monitor, args, device, ckpt_dir
        self.t0 = 0.0
        self.epochs: list[dict] = []
        self.best_acc = -1.0          # best monitor accuracy (any strict improvement) -> best/ ckpt
        self.best_epoch = None
        self.patience_ref = -1.0      # reference for early stopping (moves only on >= min_delta)
        self.bad_epochs = 0
        self.early_stopped = False

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_epoch_begin(self, args, state, control, **kw):
        self._sync()
        self.t0 = time.perf_counter()

    def on_epoch_end(self, args, state, control, model=None, **kw):
        self._sync()
        train_secs = time.perf_counter() - self.t0
        epoch = len(self.epochs) + 1
        result, eval_secs = evaluate(model, self.tokenizer, self.monitor, self.args, self.device)
        acc = result.final_answer_accuracy
        per_level = curve_metrics(result.records)["per_level_accuracy"]
        improved = acc > self.best_acc
        if improved:
            self.best_acc, self.best_epoch = acc, epoch
            if self.args.save_checkpoints:
                model.save_pretrained(self.ckpt_dir / "best")
                self.tokenizer.save_pretrained(self.ckpt_dir / "best")
        if acc >= self.patience_ref + self.args.early_stop_min_delta:
            self.patience_ref, self.bad_epochs = acc, 0
        else:
            self.bad_epochs += 1
        entry = {"epoch": epoch, "global_step": state.global_step, "monitor_accuracy": acc,
                 "monitor_unparseable_rate": result.unparseable_rate, "monitor_n": result.n,
                 "monitor_per_level_accuracy": per_level, "seconds_train_epoch": train_secs,
                 "seconds_monitor_eval": eval_secs, "is_best": improved, "epochs_without_improvement": self.bad_epochs}
        self.epochs.append(entry)
        state.log_history.append(dict(entry))
        print(f"[epoch {epoch}] train {train_secs:.1f}s | monitor acc={acc:.3f} (n={result.n}, {eval_secs:.1f}s) "
              f"best={self.best_acc:.3f}@{self.best_epoch} no-improve={self.bad_epochs}", flush=True)
        if self.args.early_stop_patience and self.bad_epochs >= self.args.early_stop_patience:
            self.early_stopped = True
            control.should_training_stop = True
            print(f"early stop: no >= {self.args.early_stop_min_delta:.3f} gain for {self.bad_epochs} epochs", flush=True)
        return control


# ---------------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--method", required=True, choices=["filler_tokens", "explicit_cot"])
    p.add_argument("--train-json", required=True)
    p.add_argument("--val-json", required=True, help="monitor set for per-epoch generation accuracy")
    p.add_argument("--test-json", required=True)
    p.add_argument("--compute-steps", type=int, default=0, help="filler tokens (filler_tokens only)")
    p.add_argument("--filler-token", default=".")
    p.add_argument("--faithful", action="store_true", help="filler_tokens: 50/50 CoT/filler mix, unmasked filler loss")
    p.add_argument("--epochs", type=float, default=3, help="max epochs (LR schedule length)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-new-tokens", type=int, default=0, help="0 = longest gold target in train/test + 16")
    p.add_argument("--gen-batch-size", type=int, default=64)
    p.add_argument("--verify-batched-n", type=int, default=0, help="check batched == batch-1 decode on first N test")
    p.add_argument("--early-stop-patience", type=int, default=0, help="0 = off")
    p.add_argument("--early-stop-min-delta", type=float, default=0.005)
    p.add_argument("--eval-checkpoint", choices=["best", "last"], default="best")
    p.add_argument("--output-dir", default=None, help="checkpoints (best/, last/); default results/<run_id>/ckpt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage", default="pilot", choices=["smoke_test", "pilot", "full_run"])
    p.add_argument("--hardware", default="RunPod GPU")
    p.add_argument("--slug", default=None)
    p.add_argument("--smoke-n", type=int, default=0, help=">0: first N of every split, no RunRecord / results dir")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    if args.method == "explicit_cot" and (args.compute_steps or args.faithful):
        p.error("--compute-steps/--faithful only apply to --method filler_tokens")
    argv = " ".join(sys.argv)
    smoke = args.smoke_n > 0

    seed = set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train = load_synthetic(args.train_json).subset(args.smoke_n)
    monitor = load_synthetic(args.val_json).subset(args.smoke_n)
    test = load_synthetic(args.test_json).subset(args.smoke_n)
    task = test.task

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)

    if args.method == "explicit_cot":
        train_ds = CotDataset(train.examples, tokenizer)
    elif args.faithful:
        train_ds = MixedFillerCotDataset(train.examples, tokenizer, args.compute_steps, args.filler_token, args.seed)
    else:
        train_ds = FillerDataset(train.examples, tokenizer, args.compute_steps, args.filler_token)

    if not args.max_new_tokens:
        tmpl = (explicit_cot.TARGET_TEMPLATE if args.method == "explicit_cot"
                else filler_tokens.ANSWER_TEMPLATE)
        longest = max(len(tokenizer(tmpl.format(rationale=ex.rationale, answer=ex.answer),
                                     add_special_tokens=False)["input_ids"])
                      for ex in train.examples + test.examples)
        args.max_new_tokens = longest + 16
    print(f"task={task} method={args.method} compute_steps={args.compute_steps} faithful={args.faithful} "
          f"train={len(train.examples)} monitor={len(monitor.examples)} test={len(test.examples)} "
          f"max_new_tokens={args.max_new_tokens} device={device}", flush=True)

    slug = args.slug or (f"{task}-{'faithful-' if args.faithful else ''}budget-{args.compute_steps}"
                         if args.method == "filler_tokens" else f"{task}-cot")
    run_id = new_run_id(args.method, slug)
    tmp_root = Path(tempfile.mkdtemp(prefix="train_synthetic_"))
    if args.output_dir:
        ckpt_dir = Path(args.output_dir)
    elif smoke:
        ckpt_dir = tmp_root / "ckpt"
    else:
        ckpt_dir = RESULTS_DIR / run_id / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.save_checkpoints = True

    monitor_cb = EpochMonitor(tokenizer, monitor, args, device, ckpt_dir)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_root / "trainer_tmp"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            logging_steps=50,
            save_strategy="no",
            report_to=[],
            bf16=device == "cuda",
            seed=args.seed,
            use_cpu=device == "cpu",
        ),
        train_dataset=train_ds,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
        callbacks=[monitor_cb],
    )
    train_result = trainer.train()
    model.save_pretrained(ckpt_dir / "last")
    tokenizer.save_pretrained(ckpt_dir / "last")

    eval_ckpt = ckpt_dir / args.eval_checkpoint
    if args.eval_checkpoint == "best" and monitor_cb.best_epoch != len(monitor_cb.epochs):
        model = AutoModelForCausalLM.from_pretrained(eval_ckpt).to(device)
    model.eval()

    if args.verify_batched_n:
        sub = test.subset(args.verify_batched_n)
        r1, _ = evaluate(model, tokenizer, sub, args, device, batch_size=1)
        rb, _ = evaluate(model, tokenizer, sub, args, device, batch_size=args.gen_batch_size)
        mism = [(a["id"], a["raw_output"], b["raw_output"]) for a, b in zip(r1.records, rb.records)
                if a["raw_output"] != b["raw_output"]]
        print(f"verify batched (bs={args.gen_batch_size}) vs batch-1 on {len(sub.examples)} test examples: "
              f"{len(sub.examples) - len(mism)}/{len(sub.examples)} identical", flush=True)
        for m in mism[:5]:
            print("  MISMATCH", m)
        if mism:
            raise SystemExit("batched decode differs from batch-1 decode -- use --gen-batch-size 1")

    result, test_secs = evaluate(model, tokenizer, test, args, device)
    curve = curve_metrics(result.records)
    secs = [e["seconds_train_epoch"] for e in monitor_cb.epochs]
    for rec in result.records:
        rec["compute_steps"] = args.compute_steps if args.method == "filler_tokens" else None
        if args.method == "filler_tokens":  # faithful mode also trains on CoT: did it write one after the filler?
            rec["emitted_cot"] = "<<" in rec["raw_output"]
    extra = {
        "task": task,
        "task_level_range": test.level_range,
        "seconds_per_epoch": sum(secs) / len(secs) if secs else None,
        "seconds_per_epoch_all": secs,
        "epochs_trained": len(monitor_cb.epochs),
        "best_epoch": monitor_cb.best_epoch,
        "best_monitor_accuracy": monitor_cb.best_acc,
        "monitor_accuracy_by_epoch": [e["monitor_accuracy"] for e in monitor_cb.epochs],
        "early_stopped": monitor_cb.early_stopped,
        "eval_checkpoint": args.eval_checkpoint,
        "test_eval_seconds": test_secs,
        **curve,
    }
    if args.method == "filler_tokens":
        extra["cot_leak_rate"] = sum(r["emitted_cot"] for r in result.records) / max(1, result.n)
    if args.method == "explicit_cot":
        extra["cot_tokens"] = sum(r["n_generated_tokens"] for r in result.records) / max(1, result.n)
    print(f"test acc={result.final_answer_accuracy:.3f} unparseable={result.unparseable_rate:.3f} "
          f"midpoint={curve['midpoint']} ({curve['midpoint_status']}) auc={curve['auc']:.3f} "
          f"sec/epoch={extra['seconds_per_epoch']}", flush=True)
    print("per-level:", json.dumps(curve["per_level_accuracy"]), flush=True)

    if smoke:
        if not args.output_dir:
            shutil.rmtree(tmp_root, ignore_errors=True)
        else:
            shutil.rmtree(tmp_root / "trainer_tmp", ignore_errors=True)
        print("smoke run -- not logging")
        return
    shutil.rmtree(tmp_root, ignore_errors=True)

    record = RunRecord(
        run_id=run_id,
        mechanism=args.method,
        stage=args.stage,
        model=ModelInfo(backbone=MODEL_ID, checkpoint=str(eval_ckpt),
                        n_params=sum(p.numel() for p in model.parameters())),
        dataset=DatasetInfo(name=f"synthetic-{task}", split="test", n_examples=result.n, seed=None),
        metrics={
            "final_answer_accuracy": result.final_answer_accuracy,
            "unparseable_rate": result.unparseable_rate,
            **({"compute_steps": args.compute_steps} if args.method == "filler_tokens" else {}),
            "train_loss": train_result.training_loss,
            "sec_per_example": result.sec_per_example,
            "extra": extra,
        },
        hyperparams={
            "lr": args.lr, "max_epochs": args.epochs, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "train_n": len(train.examples), "train_json": args.train_json, "val_json": args.val_json,
            "test_json": args.test_json, "max_new_tokens": args.max_new_tokens, "gen_batch_size": args.gen_batch_size,
            "early_stop_patience": args.early_stop_patience, "early_stop_min_delta": args.early_stop_min_delta,
            **({"filler_token": args.filler_token, "faithful": args.faithful} if args.method == "filler_tokens" else {}),
            **({"cot_rate": FAITHFUL_COT_RATE, "mask_filler_loss": False} if args.faithful else {}),
        },
        seed=seed,
        hardware=f"{args.hardware} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        notes=f"{args.method} on synthetic {task} (levels {test.level_range[0]}-{test.level_range[1]}), "
              f"{len(train.examples)} train examples, per-level eval on {args.test_json}",
    )
    manifest = record.save(predictions=result.records)
    (manifest.parent / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={run_id} -> fill in {manifest.parent / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
