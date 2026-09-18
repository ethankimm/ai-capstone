#!/usr/bin/env python3
"""Explicit CoT baseline: ordinary GPT-2 fine-tune on GSM8K-Aug, generating the visible
calculator-annotated rationale then the answer. See
`latentreasoning/mechanisms/explicit_cot.py` for the mechanism definition -- this
script is just the training loop and CLI around it, owned independently per CLAUDE.md
("How we work").

Needs the `train` extra (`uv sync --extra train`) and a GPU in practice -- run on RunPod.

Usage:
  uv run python scripts/train_explicit_cot.py --train-n 20000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from latentreasoning.data.gsm8k_aug import Example, load_gsm8k_aug
from latentreasoning.eval.harness import score_outputs
from latentreasoning.mechanisms.explicit_cot import build_eval_prompt_ids, build_example_ids
from latentreasoning.mechanisms.explicit_cot import name as mechanism_name
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id
from latentreasoning.utils import set_seed

MODEL_ID = "gpt2"


class CotDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer):
        self.rows = [build_example_ids(tokenizer, ex.question, ex.rationale, ex.answer) for ex in examples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        input_ids, labels = self.rows[i]
        return {"input_ids": input_ids, "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict:
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


@torch.no_grad()
def generate(model, tokenizer, question: str, max_new_tokens: int, device: str) -> tuple[str, int]:
    ids = build_eval_prompt_ids(tokenizer, question)
    input_ids = torch.tensor([ids], device=device)
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), len(new_tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-n", type=int, default=20000, help="pilot subset of train split; -1 = full")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--eval-n", type=int, default=200)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", default="pilot", choices=["smoke_test", "pilot", "full_run"])
    parser.add_argument("--hardware", default="RunPod GPU")
    parser.add_argument("--slug", default="baseline")
    args = parser.parse_args()
    train_n = None if args.train_n < 0 else args.train_n

    seed = set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)

    train_examples = load_gsm8k_aug(split="train", n=train_n, seed=args.seed)
    train_ds = CotDataset(train_examples, tokenizer)

    run_id = new_run_id(mechanism_name, args.slug)
    out_dir = Path("results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir / "trainer_tmp"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            logging_steps=50,
            save_strategy="no",
            report_to=[],
            bf16=torch.cuda.is_available(),
        ),
        train_dataset=train_ds,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    train_result = trainer.train()

    ckpt_dir = out_dir / "ckpt"
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    eval_examples = load_gsm8k_aug(split="test", n=args.eval_n, seed=args.eval_seed)
    model.eval()
    start = time.perf_counter()
    raw_outputs, cot_lens = [], []
    for ex in eval_examples:
        text, n_tok = generate(model, tokenizer, ex.question, args.max_new_tokens, device)
        raw_outputs.append(text)
        cot_lens.append(n_tok)
    elapsed = time.perf_counter() - start
    result = score_outputs(eval_examples, raw_outputs)
    result.sec_per_example = elapsed / len(eval_examples) if eval_examples else 0.0
    avg_cot_tokens = sum(cot_lens) / len(cot_lens) if cot_lens else 0.0

    record = RunRecord(
        run_id=run_id,
        mechanism=mechanism_name,
        stage=args.stage,
        model=ModelInfo(
            backbone=MODEL_ID,
            checkpoint=str(ckpt_dir),
            n_params=sum(p.numel() for p in model.parameters()),
        ),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=result.n, seed=args.eval_seed),
        metrics={
            "final_answer_accuracy": result.final_answer_accuracy,
            "unparseable_rate": result.unparseable_rate,
            "train_loss": train_result.training_loss,
            "sec_per_example": result.sec_per_example,
            "extra": {"cot_tokens": avg_cot_tokens},
        },
        hyperparams={
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_n": train_n if train_n is not None else len(train_examples),
            "max_new_tokens": args.max_new_tokens,
        },
        seed=seed,
        hardware=args.hardware,
        notes=f"explicit_cot pilot fine-tune on {len(train_examples)} train examples",
    )
    record.save(predictions=result.records)
    (out_dir / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=2))

    print(
        f"run_id={run_id} final_answer_accuracy={result.final_answer_accuracy:.3f} "
        f"unparseable_rate={result.unparseable_rate:.3f} train_loss={train_result.training_loss:.4f} "
        f"avg_cot_tokens={avg_cot_tokens:.1f}"
    )
    print(f"wrote results/{run_id}/ -> fill in notes.md, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
