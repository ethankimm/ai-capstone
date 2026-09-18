#!/usr/bin/env python3
"""Recurrent depth (Geiping et al. 2025, arXiv:2502.05171) fine-tune + eval on GPT-2 /
GSM8K-Aug. See `latentreasoning/mechanisms/recurrent_depth.py` for the mechanism
definition (layer split, adapter, r distribution, what `compute_steps` means) and
`recurrent_depth_model.py` for the model -- this script is the training loop, the
test-time r sweep, and the run record, owned independently per CLAUDE.md ("How we work").

Plain PyTorch loop rather than HF Trainer because r is re-sampled per batch and the
forward takes it as an argument. Optimiser/schedule match the other mechanisms'
Trainer defaults (AdamW, linear decay to 0, no warmup, grad-clip 1.0, bf16 autocast).

Needs the `train` extra (`uv sync --extra train`) and a GPU in practice -- run on RunPod.

Usage:
  uv run python scripts/train_recurrent_depth.py --train-n 20000                       # pilot
  uv run python scripts/train_recurrent_depth.py --train-n -1 --stage full_run         # full
  uv run python scripts/train_recurrent_depth.py --local-sample --train-n 8 --eval-n 4 \
      --epochs 1 --mean-recurrence 2 --eval-recurrences 1,2 --stage smoke_test          # CPU smoke
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2LMHeadModel

from latentreasoning.data.gsm8k_aug import Example, load_gsm8k_aug, load_local_sample
from latentreasoning.eval.harness import score_outputs
from latentreasoning.mechanisms.recurrent_depth import (
    DEFAULT_BACKPROP_LAST_K,
    DEFAULT_INIT_STATE_STD,
    DEFAULT_LOGNORMAL_SIGMA,
    DEFAULT_MEAN_RECURRENCE,
    build_eval_prompt_ids,
    build_example_ids,
    sample_num_recurrences,
)
from latentreasoning.mechanisms.recurrent_depth import name as mechanism_name
from latentreasoning.mechanisms.recurrent_depth_model import LoopedGPT2, unrolled_depth
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id
from latentreasoning.utils import set_seed

MODEL_ID = "gpt2"
LOG_EVERY = 50


def collate_right(rows: list[tuple[list[int], list[int]]], pad_id: int, device: str) -> dict:
    max_len = max(len(ids) for ids, _ in rows)
    input_ids = torch.full((len(rows), max_len), pad_id)
    labels = torch.full((len(rows), max_len), -100)
    attn = torch.zeros((len(rows), max_len), dtype=torch.long)
    for i, (ids, lab) in enumerate(rows):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        labels[i, : len(lab)] = torch.tensor(lab)
        attn[i, : len(ids)] = 1
    return {"input_ids": input_ids.to(device), "labels": labels.to(device), "attention_mask": attn.to(device)}


def collate_left(prompts: list[list[int]], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(p) for p in prompts)
    input_ids = torch.full((len(prompts), max_len), pad_id)
    attn = torch.zeros((len(prompts), max_len), dtype=torch.long)
    for i, p in enumerate(prompts):
        input_ids[i, max_len - len(p) :] = torch.tensor(p)
        attn[i, max_len - len(p) :] = 1
    return input_ids.to(device), attn.to(device)


def train(model: LoopedGPT2, rows, args, pad_id: int, device: str) -> tuple[float, list[dict]]:
    """Returns (mean loss over all steps, log_history) -- same shape as Trainer's."""
    steps_per_epoch = math.ceil(len(rows) / args.batch_size)
    total_steps = int(math.ceil(args.epochs * steps_per_epoch))
    core_lr = args.core_lr if args.core_lr is not None else args.lr
    core_params = set(id(p) for p in model.core.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if id(p) not in core_params], "lr": args.lr},
            {"params": list(model.core.parameters()), "lr": core_lr},
        ],
        weight_decay=0.0,
    )
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: max(0.0, 1 - s / total_steps))
    r_rng = np.random.default_rng(args.seed)
    order_gen = torch.Generator().manual_seed(args.seed)
    use_amp = device == "cuda"

    model.train()
    log_history: list[dict] = []
    loss_sum, window_sum, window_n, window_r = 0.0, 0.0, 0, []
    step = 0
    t0 = time.perf_counter()
    while step < total_steps:
        perm = torch.randperm(len(rows), generator=order_gen).tolist()
        for b in range(0, len(perm), args.batch_size):
            if step >= total_steps:
                break
            batch = collate_right([rows[i] for i in perm[b : b + args.batch_size]], pad_id, device)
            r = sample_num_recurrences(r_rng, args.mean_recurrence, args.lognormal_sigma)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, _ = model(
                    batch["input_ids"], batch["attention_mask"],
                    num_recurrences=r, backprop_last_k=args.backprop_last_k,
                    checkpoint_iterations=args.checkpoint_iterations,
                )
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                batch["labels"][:, 1:].reshape(-1),
                ignore_index=-100,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            loss_sum += loss.item()
            window_sum += loss.item()
            window_n += 1
            window_r.append(r)
            if step % LOG_EVERY == 0 or step == total_steps:
                entry = {
                    "loss": round(window_sum / window_n, 4),
                    "learning_rate": schedule.get_last_lr()[0],
                    "epoch": round(step / steps_per_epoch, 3),
                    "step": step,
                    "mean_r_in_window": round(float(np.mean(window_r)), 1),
                    "elapsed_s": round(time.perf_counter() - t0, 1),
                }
                log_history.append(entry)
                print(f"step {step}/{total_steps} loss={entry['loss']} r~{entry['mean_r_in_window']} "
                      f"lr={entry['learning_rate']:.2e} {entry['elapsed_s']:.0f}s", flush=True)
                window_sum, window_n, window_r = 0.0, 0, []
    return loss_sum / max(step, 1), log_history


@torch.no_grad()
def evaluate(model: LoopedGPT2, tokenizer, examples: list[Example], r: int, args, device: str):
    """Greedy answers for every example at `r` loop iterations; returns (raw_outputs, seconds)."""
    model.eval()
    prompts = [build_eval_prompt_ids(tokenizer, ex.question) for ex in examples]
    gen = torch.Generator(device=device).manual_seed(args.seed)
    raw: list[str] = []
    start = time.perf_counter()
    for b in range(0, len(prompts), args.eval_batch_size):
        input_ids, attn = collate_left(prompts[b : b + args.eval_batch_size], tokenizer.pad_token_id, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model.generate(
                input_ids, attn, num_recurrences=r, max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id, generator=gen,
            )
        for row in out:
            toks = row.tolist()
            if tokenizer.eos_token_id in toks:
                toks = toks[: toks.index(tokenizer.eos_token_id)]
            raw.append(tokenizer.decode(toks, skip_special_tokens=True))
    return raw, time.perf_counter() - start


@torch.no_grad()
def diagnostics(model: LoopedGPT2, tokenizer, examples: list[Example], r: int, args, device: str) -> dict:
    model.eval()
    prompts = [build_eval_prompt_ids(tokenizer, ex.question) for ex in examples]
    gen = torch.Generator(device=device).manual_seed(args.seed)
    acc: dict[str, list[list[float]]] = {}
    for b in range(0, len(prompts), args.eval_batch_size):
        input_ids, attn = collate_left(prompts[b : b + args.eval_batch_size], tokenizer.pad_token_id, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            d = model.iteration_diagnostics(input_ids, attn, r, generator=gen)
        for k, v in d.items():
            acc.setdefault(k, []).append(v)
    # batch-size-weighted mean per iteration
    weights = [min(args.eval_batch_size, len(prompts) - b) for b in range(0, len(prompts), args.eval_batch_size)]
    return {k: [round(float(np.average([v[i] for v in vs], weights=weights)), 5) for i in range(r)] for k, vs in acc.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-prelude", type=int, default=4)
    parser.add_argument("--n-core", type=int, default=4)
    parser.add_argument("--n-coda", type=int, default=4)
    parser.add_argument("--mean-recurrence", type=float, default=DEFAULT_MEAN_RECURRENCE, help="r_bar of the training log-normal Poisson")
    parser.add_argument("--lognormal-sigma", type=float, default=DEFAULT_LOGNORMAL_SIGMA)
    parser.add_argument("--backprop-last-k", type=int, default=DEFAULT_BACKPROP_LAST_K)
    parser.add_argument("--init-state-std", type=float, default=DEFAULT_INIT_STATE_STD)
    parser.add_argument("--adapter-init", default="identity", choices=LoopedGPT2.ADAPTER_INITS,
                        help="identity=[I,0] (GPT-2 at init, collapsed from step 0); random_state=[I,N(0,std^2/h)]; random=fresh adapter")
    parser.add_argument("--core-init", default="pretrained", choices=LoopedGPT2.CORE_INITS,
                        help="random re-draws the looped blocks with GPT-2's init (no pretrained 12-layer shortcut)")
    parser.add_argument("--core-lr", type=float, default=None,
                        help="separate lr for the looped blocks (default = --lr); use with --core-init random")
    parser.add_argument("--checkpoint-iterations", action="store_true",
                        help="activation-checkpoint each with-grad loop iteration (needed for large --backprop-last-k)")
    parser.add_argument("--eval-recurrences", default="1,2,4,8,16,32,64", help="comma-separated test-time r sweep")
    parser.add_argument("--primary-recurrence", type=int, default=None, help="r reported as the run's compute_steps; default int(mean_recurrence)")
    parser.add_argument("--train-n", type=int, default=20000, help="pilot subset of train split; -1 for full")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--eval-n", type=int, default=200)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", default="pilot", choices=["smoke_test", "pilot", "full_run"])
    parser.add_argument("--hardware", default="RunPod GPU")
    parser.add_argument("--slug", default=None, help="run_id slug suffix; default derived from the layer split and r_bar")
    parser.add_argument("--local-sample", action="store_true", help="smoke test: bundled 30-example sample for train AND eval, no network")
    parser.add_argument("--no-save", action="store_true", help="skip the run record (smoke tests)")
    args = parser.parse_args()
    train_n = None if args.train_n < 0 else args.train_n
    eval_rs = [int(x) for x in args.eval_recurrences.split(",")]
    primary_r = args.primary_recurrence or int(args.mean_recurrence)
    if primary_r not in eval_rs:
        eval_rs.append(primary_r)
    eval_rs = sorted(set(eval_rs))

    seed = set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = LoopedGPT2(
        GPT2LMHeadModel.from_pretrained(MODEL_ID), args.n_prelude, args.n_core, args.n_coda, args.init_state_std,
        adapter_init=args.adapter_init, core_init=args.core_init,
    ).to(device)

    if args.local_sample:
        sample = load_local_sample()
        train_examples = sample[: train_n] if train_n else sample
        eval_examples = sample[: args.eval_n]
    else:
        train_examples = load_gsm8k_aug(split="train", n=train_n, seed=args.seed)
        eval_examples = load_gsm8k_aug(split="test", n=args.eval_n, seed=args.eval_seed)
    rows = [build_example_ids(tokenizer, ex.question, ex.answer) for ex in train_examples]

    split = f"{args.n_prelude}-{args.n_core}-{args.n_coda}"
    slug = args.slug or f"split{split}-rbar{int(args.mean_recurrence)}"
    run_id = new_run_id(mechanism_name, slug)
    out_dir = Path("results") / run_id
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_id={run_id} device={device} train_n={len(rows)} eval_n={len(eval_examples)} "
          f"split={split} r_bar={args.mean_recurrence} k={args.backprop_last_k} eval_r={eval_rs} "
          f"adapter_init={args.adapter_init} core_init={args.core_init} core_lr={args.core_lr or args.lr}", flush=True)

    train_loss, log_history = train(model, rows, args, tokenizer.pad_token_id, device)
    train_seconds = log_history[-1]["elapsed_s"] if log_history else 0.0

    ckpt_dir = out_dir / "ckpt"
    if not args.no_save:
        model.save(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

    # ---- test-time r sweep on the one checkpoint -------------------------------------
    sweep: dict[int, dict] = {}
    results = {}
    for r in eval_rs:
        raw, elapsed = evaluate(model, tokenizer, eval_examples, r, args, device)
        res = score_outputs(eval_examples, raw)
        res.sec_per_example = elapsed / max(len(eval_examples), 1)
        for rec in res.records:
            rec["compute_steps"] = r
        results[r] = res
        sweep[r] = {
            "final_answer_accuracy": res.final_answer_accuracy,
            "unparseable_rate": res.unparseable_rate,
            "sec_per_example": round(res.sec_per_example, 4),
            "unrolled_depth": unrolled_depth(args.n_prelude, args.n_core, args.n_coda, r),
        }
        print(f"eval r={r:3d}: acc={res.final_answer_accuracy:.3f} unparseable={res.unparseable_rate:.3f} "
              f"({res.sec_per_example:.3f}s/ex)", flush=True)
        if not args.no_save:
            with (out_dir / f"predictions_r{r}.jsonl").open("w") as f:
                for rec in res.records:
                    f.write(json.dumps(rec) + "\n")

    diag = diagnostics(model, tokenizer, eval_examples, max(eval_rs), args, device)
    show = [round(x, 5) for x in diag["state_rel_delta"][:3]] + [round(diag["state_rel_delta"][-1], 5)]
    print(f"convergence at r={max(eval_rs)}: state_rel_delta[0,1,2,-1]={show} "
          f"next_token_kl[-1]={diag['next_token_kl'][-1]:.4g}")

    if args.no_save:
        print("--no-save: not recording a run")
        return

    primary = results[primary_r]
    record = RunRecord(
        run_id=run_id,
        mechanism=mechanism_name,
        stage=args.stage,
        model=ModelInfo(
            backbone=MODEL_ID,
            checkpoint=str(ckpt_dir),
            n_params=sum(p.numel() for p in model.parameters()),
        ),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=primary.n, seed=args.eval_seed),
        metrics={
            "final_answer_accuracy": primary.final_answer_accuracy,
            "unparseable_rate": primary.unparseable_rate,
            "compute_steps": primary_r,
            "train_loss": train_loss,
            "sec_per_example": primary.sec_per_example,
            "extra": {
                "sweep_by_r": sweep,
                "train_mean_recurrence": args.mean_recurrence,
                "train_seconds": train_seconds,
                "diagnostics_r": max(eval_rs),
                **diag,
            },
        },
        hyperparams={
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_n": len(rows),
            "n_prelude": args.n_prelude,
            "n_core": args.n_core,
            "n_coda": args.n_coda,
            "mean_recurrence": args.mean_recurrence,
            "lognormal_sigma": args.lognormal_sigma,
            "backprop_last_k": args.backprop_last_k,
            "checkpoint_iterations": args.checkpoint_iterations,
            "init_state_std": args.init_state_std,
            "adapter_init": args.adapter_init,
            "core_init": args.core_init,
            "core_lr": args.core_lr if args.core_lr is not None else args.lr,
            "eval_recurrences": eval_rs,
            "max_new_tokens": args.max_new_tokens,
            "prompt_format": "direct answer (no rationale), same as filler_tokens compute_steps=0",
        },
        seed=seed,
        hardware=args.hardware,
        notes=(
            f"recurrent_depth {args.stage}: gpt2 split {split} (core looped; adapter_init={args.adapter_init}, "
            f"core_init={args.core_init}), trained with "
            f"r~lognormal-Poisson(r_bar={args.mean_recurrence}, sigma={args.lognormal_sigma}), "
            f"k={args.backprop_last_k}, on {len(rows)} train examples; eval sweep r={eval_rs}, "
            f"compute_steps reports r={primary_r}"
        ),
    )
    record.save(predictions=primary.records)
    (out_dir / "train_log.json").write_text(json.dumps(log_history, indent=2))

    print(f"run_id={run_id} final_answer_accuracy@r{primary_r}={primary.final_answer_accuracy:.3f} "
          f"train_loss={train_loss:.4f}")
    print("sweep:", json.dumps({r: v["final_answer_accuracy"] for r, v in sweep.items()}))
    print(f"wrote results/{run_id}/ -> fill in notes.md, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
