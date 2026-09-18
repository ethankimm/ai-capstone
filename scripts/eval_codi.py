#!/usr/bin/env python3
"""Evaluate a CODI checkpoint -- the authors' released `zen-E/CODI-gpt2`, or one trained
with scripts/train_codi.sh -- under the paper's own inference protocol, and log it as
a RunRecord on the shared eval slice. See `latentreasoning/mechanisms/codi.py`.

Runs INSIDE the CODI venv, from the CODI checkout (it imports their `src.model`):

  cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/eval_codi.py \\
      --ckpt_dir /workspace/codi_released --checkpoint_label hf:zen-E/CODI-gpt2 \\
      --slug released-weights --stage full_run --hardware "RunPod RTX A5000 (secure)" \\
      --output_dir /tmp/codi_eval_unused \\
      --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \\
      --lora_r 128 --lora_alpha 32 --lora_init --batch_size 128 --greedy True \\
      --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \\
      --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True

The flags after --output_dir are `scripts/test_gpt2.sh` from the reference repo,
verbatim. Model construction / checkpoint loading and the batched greedy decode below
are copied from their `test.py` (it has no importable entry point -- `evaluation()`
loads data, generates and scores in one function), so the outputs are exactly what
their script would produce. What's ours: the questions come from the shared loader's
GSM8k-Aug test split in file order (same 1319 problems as `gsm8k/main` test, which is
what test.py loads -- the script checks and records any text mismatch), and scoring is
`latentreasoning.eval.harness.score_outputs` on the fixed `--eval_n/--eval_seed`
slice, with the paper's own last-number metric reported alongside under
`metrics.extra`. One generation pass over the full test set feeds both numbers.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers
from torch.nn import functional as F
from peft import LoraConfig, TaskType

sys.path.insert(0, os.getcwd())  # the CODI checkout: `src.model`
from src.model import CODI, DataArguments, ModelArguments, TrainingArguments  # noqa: E402

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.eval.harness import score_outputs  # noqa: E402
from latentreasoning.mechanisms.codi import name as mechanism_name  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402

# Paper script (scripts/train_gpt2_gsm8k-aug.sh @ 2c23146) -- used as `hyperparams` when
# the checkpoint dir ships no training_args.bin (the released weights don't).
PAPER_TRAIN_HYPERPARAMS = {
    "lr": 3e-3, "epochs": 40, "per_device_train_batch_size": 64, "gradient_accumulation_steps": 2,
    "effective_batch_size": 128, "lora_r": 128, "lora_alpha": 32, "lora_dropout": 0.1,
    "lora_target_modules": ["c_attn", "c_proj", "c_fc"], "num_latent": 6, "use_prj": True, "prj_dim": 768,
    "prj_layernorm": True, "distill_loss_type": "smooth_l1", "distill_loss_div_std": True,
    "distill_loss_factor": 1.0, "ref_loss_factor": 1.0, "weight_decay": 0.1, "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine", "max_grad_norm": 2.0, "bf16": True, "remove_eos": True,
    "train_seed": 11, "train_data": "zen-E/GSM8k-Aug train (385,620)",
}


@dataclass
class EvalArguments:
    slug: str = field(default="eval")
    stage: str = field(default="full_run")
    eval_n: int = field(default=200)
    eval_seed: int = field(default=0)
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: Optional[str] = field(default=None, metadata={"help": "ModelInfo.checkpoint; defaults to --ckpt_dir"})
    train_log_from: Optional[str] = field(default=None, metadata={"help": "trainer_state.json to copy log_history from"})
    run_notes: str = field(default="")


# ---- copied from the reference test.py -------------------------------------------------
def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    # use the last number as the answer
    pred_answer = float(pred[-1])
    return pred_answer


def compute_accuracy(gold: list, pred: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if p == g:
            acc += 1
    return acc / len(gold)


def build_model(model_args, training_args):
    target_modules = ["c_attn", "c_proj", 'c_fc']  # the gpt2 branch of test.py
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights=True,
    )
    model = CODI(model_args, training_args, lora_config)
    try:
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(model_args.ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin"))
    load_result = model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:  # error handling
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
    model = model.to('cuda')
    model.to(torch.bfloat16)
    return model, tokenizer, load_result


@torch.no_grad()
def generate_batches(model, tokenizer, question: list[str], data_args, training_args):
    """test.py's tokenization + generation loop, verbatim (minus per-question printing).
    Returns the decoded output per question, in order."""
    device = "cuda"
    eval_step = math.ceil(len(question) / data_args.batch_size)
    question_data = []
    for i in range(eval_step):
        if i < eval_step - 1:
            batch = tokenizer(question[i * data_args.batch_size: (i + 1) * data_args.batch_size], return_tensors="pt", padding="longest")
        else:
            batch = tokenizer(question[i * data_args.batch_size:], return_tensors="pt", padding="longest")
        if training_args.remove_eos:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 2)
        batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
        batch['input_len'] = len(batch['input_ids'][0])
        question_data.append(batch.to(device))

    model.eval()
    gen_kwargs = {"max_new_tokens": 256, "temperature": 0.1, "top_k": 40, "top_p": 0.95, "do_sample": True}
    decoded_all: list[str] = []
    for step, batch in enumerate(question_data):
        batch_size = batch["input_ids"].size(0)
        past_key_values = None
        outputs = model.codi(input_ids=batch["input_ids"], use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=batch["attention_mask"])
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

        for i in range(training_args.inf_latent_iterations):
            outputs = model.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

        if training_args.remove_eos:
            eot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)
        else:
            eot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)
        eot_emb = eot_emb.expand(batch["input_ids"].size(0), -1, -1)
        output = eot_emb

        finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
        pred_tokens = [[] for _ in range(batch_size)]
        for i in range(gen_kwargs["max_new_tokens"]):
            out = model.codi(inputs_embeds=output, output_hidden_states=False, attention_mask=None, use_cache=True, output_attentions=False, past_key_values=past_key_values)
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :model.codi.config.vocab_size - 1]

            if training_args.greedy:
                next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)
            else:
                logits /= gen_kwargs["temperature"]
                if gen_kwargs["top_k"] > 1:
                    top_k_values, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
                    min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                    logits[logits < min_top_k_value] = -float("inf")
                if gen_kwargs["top_p"] < 1.0:
                    sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > gen_kwargs["top_p"]
                    if sorted_indices_to_remove.any():
                        sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                        sorted_indices_to_remove[:, 0] = False
                    for b in range(logits.size(0)):
                        logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

            for b in range(batch_size):
                if not finished[b]:
                    pred_tokens[b].append(next_token_ids[b].item())
                    if next_token_ids[b] == tokenizer.eos_token_id:
                        finished[b] = True
            if finished.all():
                break
            output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1).to(device)

        for pred_token in pred_tokens:
            decoded_all.append(tokenizer.decode(pred_token, skip_special_tokens=True))
        print(f"batch {step + 1}/{eval_step} done ({len(decoded_all)}/{len(question)})", flush=True)
    return decoded_all
# ---- end of copied section --------------------------------------------------------------


def hyperparams_from_ckpt(ckpt_dir: str) -> dict | None:
    """What `scripts/train_codi.sh` actually ran with (Trainer writes training_args.bin
    next to the weights). None for checkpoints that don't ship it."""
    path = Path(ckpt_dir) / "training_args.bin"
    if not path.exists():
        return None
    ta = torch.load(path, weights_only=False)
    keys = ["learning_rate", "num_train_epochs", "per_device_train_batch_size", "gradient_accumulation_steps",
            "num_latent", "use_prj", "prj_dim", "prj_no_ln", "distill_loss_type", "distill_loss_div_std",
            "distill_loss_factor", "ref_loss_factor", "weight_decay", "warmup_ratio", "lr_scheduler_type",
            "max_grad_norm", "bf16", "remove_eos", "seed", "exp_mode", "exp_data_num", "max_steps", "max_token_num"]
    hp = {k: getattr(ta, k) for k in keys if hasattr(ta, k)}
    hp["lr_scheduler_type"] = str(hp.get("lr_scheduler_type"))
    hp["effective_batch_size"] = hp["per_device_train_batch_size"] * hp["gradient_accumulation_steps"]
    hp["source"] = "training_args.bin"
    return hp


def gsm8k_main_mismatches(questions: list[str], golds: list[str]) -> dict | None:
    """test.py evaluates on `gsm8k/main` test; we generate from GSM8k-Aug's test file.
    Count question-text / answer mismatches so the paper-comparison claim is checkable."""
    try:
        from datasets import load_dataset
        ref = load_dataset("gsm8k", "main")["test"]
    except Exception as e:  # network / hub id changes -- not fatal
        return {"error": repr(e)[:200]}
    ref_q = [ex["question"].strip().replace('  ', ' ') for ex in ref]
    ref_a = [ex["answer"].split('####')[-1].replace(',', '').strip() for ex in ref]
    n = min(len(ref_q), len(questions))
    q_mis = sum(a != b for a, b in zip(ref_q[:n], questions[:n]))
    a_mis = sum(float(a) != float(b.replace(',', '')) for a, b in zip(ref_a[:n], golds[:n]))
    return {"gsm8k_main_test_n": len(ref_q), "question_text_mismatches": q_mis, "answer_mismatches": a_mis}


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, EvalArguments))
    model_args, data_args, training_args, eval_args = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)

    model, tokenizer, load_result = build_model(model_args, training_args)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"loaded {model_args.ckpt_dir}: missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)} "
          f"params={n_params} trainable={n_trainable}")
    if load_result.unexpected_keys:
        print("  unexpected:", load_result.unexpected_keys[:10])
    if load_result.missing_keys:
        print("  missing:", load_result.missing_keys[:10])

    # Full test split in file order (seed=None), test.py's question formatting.
    full = load_gsm8k_aug(split="test", n=None, seed=None)
    questions = [ex.question.strip().replace('  ', ' ') for ex in full]
    golds = [ex.answer for ex in full]

    start = time.perf_counter()
    outputs = generate_batches(model, tokenizer, questions, data_args, training_args)
    elapsed = time.perf_counter() - start

    # Paper's metric on the full split (what test.py would print).
    paper_pred = [extract_answer_number(o) for o in outputs]
    paper_gold = [float(g.replace(',', '')) for g in golds]
    full_paper_acc = compute_accuracy(paper_gold, paper_pred)
    full_result = score_outputs(full, outputs)
    by_idx = {ex.idx: (out, pp) for ex, out, pp in zip(full, outputs, paper_pred)}

    # Shared slice: the same fixed eval_n/eval_seed subset every mechanism reports on.
    slice_examples = load_gsm8k_aug(split="test", n=eval_args.eval_n, seed=eval_args.eval_seed)
    result = score_outputs(slice_examples, [by_idx[ex.idx][0] for ex in slice_examples])
    result.sec_per_example = elapsed / len(full)
    for rec, ex in zip(result.records, slice_examples):
        rec["paper_metric_pred"] = by_idx[ex.idx][1]
        rec["paper_metric_correct"] = by_idx[ex.idx][1] == float(ex.answer.replace(',', ''))
    slice_paper_acc = sum(r["paper_metric_correct"] for r in result.records) / len(result.records)
    avg_out_tokens = sum(len(tokenizer.encode(o)) for o in outputs) / len(outputs)

    hp = hyperparams_from_ckpt(model_args.ckpt_dir)
    train_seed = hp["seed"] if hp else None
    if hp is None:
        hp = {**PAPER_TRAIN_HYPERPARAMS, "source": "paper script scripts/train_gpt2_gsm8k-aug.sh (no training_args.bin in ckpt_dir)"}
    hp["inference"] = {
        "inf_latent_iterations": training_args.inf_latent_iterations, "greedy": training_args.greedy,
        "batch_size": data_args.batch_size, "max_new_tokens": 256, "num_latent": training_args.num_latent,
        "use_prj": training_args.use_prj, "prj_dim": training_args.prj_dim, "remove_eos": training_args.remove_eos,
    }

    train_loss = None
    log_history = None
    if eval_args.train_log_from:
        state = json.loads(Path(eval_args.train_log_from).read_text())
        log_history = state.get("log_history", [])
        losses = [e["train_loss"] for e in log_history if "train_loss" in e] or [e["loss"] for e in log_history if "loss" in e]
        train_loss = losses[-1] if losses else None

    metrics = {
        "final_answer_accuracy": result.final_answer_accuracy,
        "unparseable_rate": result.unparseable_rate,
        "compute_steps": training_args.inf_latent_iterations,
        "sec_per_example": result.sec_per_example,
        "extra": {
            "paper_metric_accuracy_slice": slice_paper_acc,
            "full_test_n": full_result.n,
            "full_test_accuracy": full_result.final_answer_accuracy,
            "full_test_unparseable_rate": full_result.unparseable_rate,
            "full_test_paper_metric_accuracy": full_paper_acc,
            "avg_output_tokens": avg_out_tokens,
            "num_latent_train": training_args.num_latent,
            "n_trainable_params": n_trainable,
            "state_dict_missing_keys": len(load_result.missing_keys),
            "state_dict_unexpected_keys": len(load_result.unexpected_keys),
            "gsm8k_main_check": gsm8k_main_mismatches(questions, golds),
        },
    }
    if train_loss is not None:
        metrics["train_loss"] = train_loss

    record = RunRecord(
        run_id=new_run_id(mechanism_name, eval_args.slug),
        mechanism=mechanism_name,
        stage=eval_args.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=eval_args.checkpoint_label or model_args.ckpt_dir, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=result.n, seed=eval_args.eval_seed),
        metrics=metrics,
        hyperparams=hp,
        seed=train_seed,
        hardware=f"{eval_args.hardware} / {torch.cuda.get_device_name(0)}",
        notes=eval_args.run_notes or f"CODI {eval_args.slug}: paper inference protocol ({training_args.inf_latent_iterations} latents, greedy)",
    )
    manifest = record.save(predictions=result.records)
    out_dir = manifest.parent
    with (out_dir / "predictions_full_test.jsonl").open("w") as f:  # gitignored; sync with the run dir
        for r in full_result.records:
            f.write(json.dumps(r) + "\n")
    if log_history is not None:
        (out_dir / "train_log.json").write_text(json.dumps(log_history, indent=2))
    (out_dir / "eval_command.txt").write_text(argv + "\n")

    print(json.dumps({k: v for k, v in metrics.items() if k != "extra"}, indent=2))
    print(json.dumps(metrics["extra"], indent=2, default=str))
    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
