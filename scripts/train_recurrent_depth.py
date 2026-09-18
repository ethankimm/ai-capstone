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
  uv run python scripts/train_recurrent_depth.py --step-supervised --train-n 20000       # plan item (e)

`--step-supervised` swaps the objective (not the architecture): r = the example's number of
rationale steps, iteration i's read-out at the number position is trained on step i's value,
iteration n on the answer. See `build_step_supervised_row` and the mechanism module's plan.
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
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, GPT2LMHeadModel

from latentreasoning.data.gsm8k_aug import Example, load_gsm8k_aug, load_local_sample
from latentreasoning.eval.harness import score_outputs
from latentreasoning.mechanisms.recurrent_depth import (
    ANSWER_PREFIX,
    DEFAULT_BACKPROP_LAST_K,
    DEFAULT_INIT_STATE_STD,
    DEFAULT_LOGNORMAL_SIGMA,
    DEFAULT_MEAN_RECURRENCE,
    build_eval_prompt_ids,
    build_example_ids,
    build_step_supervised_row,
    first_value_token,
    is_single_token_value,
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


def make_optimizer(model: LoopedGPT2, args, total_steps: int):
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
    return optimizer, schedule


def train(model: LoopedGPT2, rows, args, pad_id: int, device: str, ckpt_dir: Path | None = None) -> tuple[float, list[dict]]:
    """Returns (mean loss over all steps, log_history) -- same shape as Trainer's."""
    steps_per_epoch = math.ceil(len(rows) / args.batch_size)
    total_steps = int(math.ceil(args.epochs * steps_per_epoch))
    optimizer, schedule = make_optimizer(model, args, total_steps)
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
        if ckpt_dir is not None:  # epoch-end safety copy for long runs (overwritten at the end)
            model.save(ckpt_dir)
    return loss_sum / max(step, 1), log_history


# ---- step-supervised objective (plan item (e) in the mechanism module) ----------------
def collate_stepsup(rows: list[dict], pad_id: int, device: str) -> dict:
    bsz = len(rows)
    max_len = max(len(r["input_ids"]) for r in rows)
    max_mid = max(1, max(len(r["step_targets"]) for r in rows))
    input_ids = torch.full((bsz, max_len), pad_id)
    labels = torch.full((bsz, max_len), -100)
    attn = torch.zeros((bsz, max_len), dtype=torch.long)
    step_targets = torch.full((bsz, max_mid), -100)
    for i, r in enumerate(rows):
        n = len(r["input_ids"])
        input_ids[i, :n] = torch.tensor(r["input_ids"])
        labels[i, :n] = torch.tensor(r["labels"])
        attn[i, :n] = 1
        if r["step_targets"]:
            step_targets[i, : len(r["step_targets"])] = torch.tensor(r["step_targets"])
    return {
        "input_ids": input_ids.to(device), "labels": labels.to(device), "attention_mask": attn.to(device),
        "hash_pos": torch.tensor([r["hash_pos"] for r in rows], device=device),
        "n_steps": torch.tensor([r["n_steps"] for r in rows], device=device),
        "step_targets": step_targets.to(device),
    }


def stepsup_iteration(model: LoopedGPT2, e, s, causal_mask, position_ids, labels, hash_pos, targets_i, is_mid, is_fin):
    """One loop iteration plus its supervised read-out. Returns `(s_new, sum CE over this
    iteration's step tokens, sum CE over this iteration's answer-span tokens)`. Only the
    supervised positions go through the LM head, so this is cheap to activation-checkpoint."""
    s_new = model.iterate(e, s, causal_mask, position_ids)
    h = model.ln_f(model._run(model.coda, s_new, causal_mask, position_ids))
    zero = h.new_zeros((), dtype=torch.float32)
    loss_mid, loss_fin = zero, zero
    if bool(is_mid.any()):
        rows = is_mid.nonzero(as_tuple=True)[0]
        logits = model.lm_head(h[rows, hash_pos[rows]]).float()
        loss_mid = F.cross_entropy(logits, targets_i[rows], reduction="sum")
    if bool(is_fin.any()):
        hf, lf = h[is_fin][:, :-1], labels[is_fin][:, 1:]
        sel = lf != -100
        logits = model.lm_head(hf[sel]).float()
        loss_fin = F.cross_entropy(logits, lf[sel], reduction="sum")
    return s_new, loss_mid, loss_fin


def train_step_supervised(model: LoopedGPT2, rows: list[dict], args, pad_id: int, device: str, ckpt_dir: Path | None = None) -> tuple[float, list[dict]]:
    """r = each example's n_steps; iteration i < n is supervised on step i's first token at the
    number position, iteration n on the answer span. Full backprop through every iteration
    (r <= --stepsup-max-steps). Loss = token mean with intermediate tokens weighted by
    --step-loss-weight. Returns (mean loss, log_history) like `train`."""
    steps_per_epoch = math.ceil(len(rows) / args.batch_size)
    total_steps = int(math.ceil(args.epochs * steps_per_epoch))
    optimizer, schedule = make_optimizer(model, args, total_steps)
    order_gen = torch.Generator().manual_seed(args.seed)
    use_amp = device == "cuda"
    w = args.step_loss_weight

    model.train()
    log_history: list[dict] = []
    loss_sum, step, t0 = 0.0, 0, time.perf_counter()
    win = {"loss": 0.0, "n": 0, "mid": 0.0, "mid_n": 0, "fin": 0.0, "fin_n": 0, "r": []}
    while step < total_steps:
        perm = torch.randperm(len(rows), generator=order_gen).tolist()
        for b in range(0, len(perm), args.batch_size):
            if step >= total_steps:
                break
            batch = collate_stepsup([rows[i] for i in perm[b : b + args.batch_size]], pad_id, device)
            r_max = int(batch["n_steps"].max())
            mid_sum = fin_sum = None
            mid_n = fin_n = 0
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                x, cm, pos = model.embed(batch["input_ids"], batch["attention_mask"])
                e = model._run(model.prelude, x, cm, pos)
                s = model.initial_state(e)
                for i in range(1, r_max + 1):
                    if i - 1 < batch["step_targets"].shape[1]:
                        tgt = batch["step_targets"][:, i - 1]
                    else:
                        tgt = torch.full_like(batch["n_steps"], -100)
                    is_mid = (batch["n_steps"] > i) & (tgt != -100)
                    is_fin = batch["n_steps"] == i
                    fn_args = (model, e, s, cm, pos, batch["labels"], batch["hash_pos"], tgt, is_mid, is_fin)
                    if args.checkpoint_iterations:
                        s, lm, lf = checkpoint(stepsup_iteration, *fn_args, use_reentrant=False)
                    else:
                        s, lm, lf = stepsup_iteration(*fn_args)
                    mid_sum = lm if mid_sum is None else mid_sum + lm
                    fin_sum = lf if fin_sum is None else fin_sum + lf
                    mid_n += int(is_mid.sum())
                    fin_n += int((batch["labels"][is_fin][:, 1:] != -100).sum())
            loss = (w * mid_sum + fin_sum) / max(w * mid_n + fin_n, 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            loss_sum += loss.item()
            win["loss"] += loss.item(); win["n"] += 1
            win["mid"] += mid_sum.item(); win["mid_n"] += mid_n
            win["fin"] += fin_sum.item(); win["fin_n"] += fin_n
            win["r"].append(r_max)
            if step % LOG_EVERY == 0 or step == total_steps:
                entry = {
                    "loss": round(win["loss"] / win["n"], 4),
                    "step_token_loss": round(win["mid"] / max(win["mid_n"], 1), 4),
                    "answer_token_loss": round(win["fin"] / max(win["fin_n"], 1), 4),
                    "learning_rate": schedule.get_last_lr()[0],
                    "epoch": round(step / steps_per_epoch, 3),
                    "step": step,
                    "mean_r_in_window": round(float(np.mean(win["r"])), 1),  # batch r_max, not per-example r
                    "elapsed_s": round(time.perf_counter() - t0, 1),
                }
                log_history.append(entry)
                print(f"step {step}/{total_steps} loss={entry['loss']} step_tok={entry['step_token_loss']} "
                      f"ans_tok={entry['answer_token_loss']} r_max~{entry['mean_r_in_window']} "
                      f"lr={entry['learning_rate']:.2e} {entry['elapsed_s']:.0f}s", flush=True)
                win = {"loss": 0.0, "n": 0, "mid": 0.0, "mid_n": 0, "fin": 0.0, "fin_n": 0, "r": []}
        if ckpt_dir is not None:  # epoch-end safety copy for long runs (overwritten at the end)
            model.save(ckpt_dir)
    return loss_sum / max(step, 1), log_history


def _decode_new_tokens(tokenizer, out: torch.Tensor) -> list[str]:
    raw = []
    for row in out:
        toks = row.tolist()
        if tokenizer.eos_token_id in toks:
            toks = toks[: toks.index(tokenizer.eos_token_id)]
        raw.append(tokenizer.decode(toks, skip_special_tokens=True))
    return raw


@torch.no_grad()
def evaluate_oracle_r(model: LoopedGPT2, tokenizer, examples: list[Example], offset: int, args, device: str):
    """Greedy answers with r = n_steps(example) + offset, n_steps read off the *test* example's
    own rationale (an oracle step count -- say so when quoting). Returns (raw, seconds, rs)."""
    model.eval()
    rs = [max(1, len(ex.intermediate_values)) + offset for ex in examples]
    raw: list[str | None] = [None] * len(examples)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    start = time.perf_counter()
    for r in sorted(set(rs)):
        idx = [i for i, rr in enumerate(rs) if rr == r]
        prompts = [build_eval_prompt_ids(tokenizer, examples[i].question) for i in idx]
        for b in range(0, len(idx), args.eval_batch_size):
            input_ids, attn = collate_left(prompts[b : b + args.eval_batch_size], tokenizer.pad_token_id, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = model.generate(input_ids, attn, num_recurrences=r, max_new_tokens=args.max_new_tokens,
                                     eos_token_id=tokenizer.eos_token_id, generator=gen)
            for j, text in zip(idx[b : b + args.eval_batch_size], _decode_new_tokens(tokenizer, out)):
                raw[j] = text
    return raw, time.perf_counter() - start, rs


@torch.no_grad()
def step_readout(model: LoopedGPT2, tokenizer, examples: list[Example], args, device: str, max_steps: int = 8):
    """Does iteration i's read-out at the number position give step i's value? For every eval
    example with >= 2 rationale steps, run prompt + ANSWER_PREFIX for n_steps-1 iterations in
    fp32 and take the top-1 token after each. Returns (summary, per-example records); the
    summary's `matrix_iteration_x_step[i][j]` is P(top-1 at iteration i+1 == first token of
    step j+1) over examples with n_steps > max(i, j)+1 -- the off-diagonal says whether
    earlier steps stay readable after later iterations (overwrite vs accumulate)."""
    model.eval()
    prefix = tokenizer(ANSWER_PREFIX, add_special_tokens=False)["input_ids"]
    items = [(k, ex) for k, ex in enumerate(examples) if len(ex.intermediate_values) >= 2]
    if not items:
        return {"n_examples": 0}, []
    K = min(max_steps, max(len(ex.intermediate_values) for _, ex in items) - 1)
    hits, tot = np.zeros((K, K)), np.zeros((K, K))
    hits_single = tot_single = 0
    records = []
    gen = torch.Generator(device=device).manual_seed(args.seed)
    for b in range(0, len(items), args.eval_batch_size):
        chunk = items[b : b + args.eval_batch_size]
        prompts = [build_eval_prompt_ids(tokenizer, ex.question) + prefix for _, ex in chunk]
        input_ids, attn = collate_left(prompts, tokenizer.pad_token_id, device)
        n_mid = [min(len(ex.intermediate_values) - 1, K) for _, ex in chunk]
        x, cm, pos = model.embed(input_ids, attn)
        e = model._run(model.prelude, x, cm, pos)
        s = model.initial_state(e, gen)
        tops = []
        for _ in range(max(n_mid)):
            s = model.iterate(e, s, cm, pos)
            tops.append(model.readout(s, cm, pos)[:, -1].argmax(-1).tolist())
        for bi, (k, ex) in enumerate(chunk):
            vals = ex.intermediate_values[: n_mid[bi]]
            tgts = [first_value_token(tokenizer, v) for v in vals]
            hit = [tops[i][bi] == tgts[i] for i in range(n_mid[bi])]
            for i in range(n_mid[bi]):
                for j in range(n_mid[bi]):
                    tot[i, j] += 1
                    hits[i, j] += tops[i][bi] == tgts[j]
                if is_single_token_value(tokenizer, vals[i]):
                    tot_single += 1
                    hits_single += hit[i]
            records.append({
                "idx": k, "n_steps": len(ex.intermediate_values), "step_values": ex.intermediate_values,
                "answer": ex.answer, "readout_by_iteration": [tokenizer.decode([tops[i][bi]]) for i in range(n_mid[bi])],
                "hit_by_iteration": hit,
            })
    diag_hits, diag_tot = hits.diagonal().sum(), tot.diagonal().sum()
    ratio = lambda h, t: (round(float(h / t), 4) if t > 0 else None)
    summary = {
        "n_examples": len(items),
        "n_step_readouts": int(diag_tot),
        "step_readout_accuracy": ratio(diag_hits, diag_tot),
        "step_readout_accuracy_single_token_values": ratio(hits_single, tot_single),
        "n_single_token_values": int(tot_single),
        "by_iteration": [ratio(hits[i, i], tot[i, i]) for i in range(K)],
        "matrix_iteration_x_step": [[ratio(hits[i, j], tot[i, j]) for j in range(K)] for i in range(K)],
        "matrix_support": tot.astype(int).tolist(),
    }
    return summary, records


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
        raw.extend(_decode_new_tokens(tokenizer, out))
    return raw, time.perf_counter() - start


@torch.no_grad()
def diagnostics(model: LoopedGPT2, tokenizer, examples: list[Example], r: int, args, device: str) -> dict:
    """Per-iteration convergence, fp32, measured at the position that predicts the *first
    answer-number token* (prompt + " ####"). Measuring at the bare prompt's last position is
    uninformative: the fine-tuned model emits " ####" there with p~1, so the KL between
    iterations is ~0 whatever the state does (the first two pilots' `next_token_kl` was
    measured that way -- see their notes). fp32 because bf16 can't resolve the sub-1%
    state changes once the loop has converged."""
    model.eval()
    suffix = tokenizer(ANSWER_PREFIX, add_special_tokens=False)["input_ids"]  # " ####"
    prompts = [build_eval_prompt_ids(tokenizer, ex.question) + suffix for ex in examples]
    gen = torch.Generator(device=device).manual_seed(args.seed)
    acc: dict[str, list[list[float]]] = {}
    for b in range(0, len(prompts), args.eval_batch_size):
        input_ids, attn = collate_left(prompts[b : b + args.eval_batch_size], tokenizer.pad_token_id, device)
        d = model.iteration_diagnostics(input_ids, attn, r, generator=gen)  # no autocast: fp32
        for k, v in d.items():
            acc.setdefault(k, []).append(v)
    # batch-size-weighted mean per iteration
    weights = [min(args.eval_batch_size, len(prompts) - b) for b in range(0, len(prompts), args.eval_batch_size)]
    out = {k: [float(np.average([v[i] for v in vs], weights=weights)) for i in range(r)] for k, vs in acc.items()}
    out["diagnostics_position"] = "first answer-number token (prompt + ' ####'), fp32"
    return out


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
    parser.add_argument("--step-supervised", action="store_true",
                        help="plan item (e): r = example's rationale step count, iteration i supervised on step i (see module docstring)")
    parser.add_argument("--step-loss-weight", type=float, default=1.0, help="weight of intermediate-step tokens in the token-mean loss")
    parser.add_argument("--stepsup-max-steps", type=int, default=8, help="drop training examples with more rationale steps than this")
    parser.add_argument("--eval-recurrences", default=None,
                        help="comma-separated test-time r sweep; default 1,2,4,8,16,32,64 (1,2,3,4,6,8 with --step-supervised)")
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
    if args.eval_recurrences is None:
        args.eval_recurrences = "1,2,3,4,6,8" if args.step_supervised else "1,2,4,8,16,32,64"
    eval_rs = [int(x) for x in args.eval_recurrences.split(",")]
    if args.step_supervised:
        primary_r = "oracle"  # r = each test example's own step count; see evaluate_oracle_r
    else:
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
    n_dropped = 0
    if args.step_supervised:
        rows = [build_step_supervised_row(tokenizer, ex.question, ex.answer, ex.intermediate_values) for ex in train_examples]
        kept = [r for r in rows if r["n_steps"] <= args.stepsup_max_steps]
        n_dropped = len(rows) - len(kept)
        rows = kept
    else:
        rows = [build_example_ids(tokenizer, ex.question, ex.answer) for ex in train_examples]

    split = f"{args.n_prelude}-{args.n_core}-{args.n_coda}"
    slug = args.slug or (f"stepsup-split{split}" if args.step_supervised else f"split{split}-rbar{int(args.mean_recurrence)}")
    run_id = new_run_id(mechanism_name, slug)
    out_dir = Path("results") / run_id
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)
    objective = "step_supervised" if args.step_supervised else "answer_only"
    print(f"run_id={run_id} device={device} objective={objective} train_n={len(rows)} (dropped {n_dropped} with "
          f">{args.stepsup_max_steps} steps) eval_n={len(eval_examples)} split={split} r_bar={args.mean_recurrence} "
          f"k={args.backprop_last_k} eval_r={eval_rs} primary={primary_r} adapter_init={args.adapter_init} "
          f"core_init={args.core_init} core_lr={args.core_lr or args.lr}", flush=True)

    ckpt_dir = out_dir / "ckpt"
    epoch_ckpt = None if args.no_save else ckpt_dir
    if args.step_supervised:
        train_loss, log_history = train_step_supervised(model, rows, args, tokenizer.pad_token_id, device, epoch_ckpt)
    else:
        train_loss, log_history = train(model, rows, args, tokenizer.pad_token_id, device, epoch_ckpt)
    train_seconds = log_history[-1]["elapsed_s"] if log_history else 0.0

    if not args.no_save:
        model.save(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

    # ---- test-time r sweep on the one checkpoint -------------------------------------
    # Fixed r for every example, plus (step-supervised only) r = the example's own step count
    # +0/+1/+2 ("oracle", "oracle+1", "oracle+2").
    sweep: dict = {}
    results = {}
    evals: list[tuple] = [(r, f"r{r}") for r in eval_rs]
    if args.step_supervised:
        evals += [("oracle", "oracle"), ("oracle+1", "oracle_plus1"), ("oracle+2", "oracle_plus2")]
    for key, fname in evals:
        if isinstance(key, int):
            raw, elapsed = evaluate(model, tokenizer, eval_examples, key, args, device)
            rs = [key] * len(eval_examples)
        else:
            raw, elapsed, rs = evaluate_oracle_r(model, tokenizer, eval_examples, int(key[6:] or 0), args, device)
        res = score_outputs(eval_examples, raw)
        res.sec_per_example = elapsed / max(len(eval_examples), 1)
        for rec, r in zip(res.records, rs):
            rec["compute_steps"] = r
        results[key] = res
        mean_r = float(np.mean(rs))
        sweep[key] = {
            "final_answer_accuracy": res.final_answer_accuracy,
            "unparseable_rate": res.unparseable_rate,
            "sec_per_example": round(res.sec_per_example, 4),
            "mean_r": round(mean_r, 3),
            "unrolled_depth": round(unrolled_depth(args.n_prelude, args.n_core, args.n_coda, mean_r), 1),
        }
        print(f"eval r={key!s:>8}: acc={res.final_answer_accuracy:.3f} unparseable={res.unparseable_rate:.3f} "
              f"mean_r={mean_r:.2f} ({res.sec_per_example:.3f}s/ex)", flush=True)
        if not args.no_save:
            with (out_dir / f"predictions_{fname}.jsonl").open("w") as f:
                for rec in res.records:
                    f.write(json.dumps(rec) + "\n")

    readout_summary: dict = {}
    if args.step_supervised:
        readout_summary, readout_records = step_readout(model, tokenizer, eval_examples, args, device, args.stepsup_max_steps)
        print(f"step read-out: acc={readout_summary.get('step_readout_accuracy')} "
              f"(single-token values {readout_summary.get('step_readout_accuracy_single_token_values')}) "
              f"by iteration={readout_summary.get('by_iteration')} on {readout_summary.get('n_step_readouts')} step read-outs", flush=True)
        if not args.no_save:
            with (out_dir / "step_readouts.jsonl").open("w") as f:
                for rec in readout_records:
                    f.write(json.dumps(rec) + "\n")

    diag = diagnostics(model, tokenizer, eval_examples, max(eval_rs), args, device)
    fmt = lambda xs: [f"{x:.2e}" for x in xs]
    print(f"convergence at r={max(eval_rs)} ({diag['diagnostics_position']}): "
          f"state_rel_delta[1..4,-1]={fmt(diag['state_rel_delta'][:4] + diag['state_rel_delta'][-1:])} "
          f"next_token_kl[2..4,-1]={fmt(diag['next_token_kl'][1:4] + diag['next_token_kl'][-1:])}")

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
            # step-supervised: the mean per-example r of the oracle eval, a float -- not
            # comparable to a fixed-r run's integer (see notes).
            "compute_steps": primary_r if isinstance(primary_r, int) else sweep[primary_r]["mean_r"],
            "train_loss": train_loss,
            "sec_per_example": primary.sec_per_example,
            "extra": {
                "primary_eval": str(primary_r),
                "sweep_by_r": sweep,
                "train_mean_recurrence": None if args.step_supervised else args.mean_recurrence,
                "train_seconds": train_seconds,
                "diagnostics_r": max(eval_rs),
                **diag,
                **({"step_readout": readout_summary} if args.step_supervised else {}),
            },
        },
        hyperparams={
            "objective": objective,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_n": len(rows),
            "train_n_dropped_over_max_steps": n_dropped,
            "n_prelude": args.n_prelude,
            "n_core": args.n_core,
            "n_coda": args.n_coda,
            "mean_recurrence": None if args.step_supervised else args.mean_recurrence,
            "lognormal_sigma": None if args.step_supervised else args.lognormal_sigma,
            "backprop_last_k": None if args.step_supervised else args.backprop_last_k,
            "step_loss_weight": args.step_loss_weight if args.step_supervised else None,
            "stepsup_max_steps": args.stepsup_max_steps if args.step_supervised else None,
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
            f"recurrent_depth {args.stage} [{objective}]: gpt2 split {split} (core looped; adapter_init={args.adapter_init}, "
            f"core_init={args.core_init}), trained with "
            + (f"r = example's rationale step count, iteration i supervised on step i (step_loss_weight={args.step_loss_weight})"
               if args.step_supervised else
               f"r~lognormal-Poisson(r_bar={args.mean_recurrence}, sigma={args.lognormal_sigma}), k={args.backprop_last_k}")
            + f", on {len(rows)} train examples; eval sweep r={eval_rs}"
            + (", plus oracle r = test example's step count (+0/+1/+2)" if args.step_supervised else "")
            + f"; compute_steps reports {primary_r}"
        ),
    )
    record.save(predictions=primary.records)
    (out_dir / "train_log.json").write_text(json.dumps(log_history, indent=2))

    print(f"run_id={run_id} final_answer_accuracy@{primary_r}={primary.final_answer_accuracy:.3f} "
          f"train_loss={train_loss:.4f}")
    print("sweep:", json.dumps({str(r): v["final_answer_accuracy"] for r, v in sweep.items()}))
    print(f"wrote results/{run_id}/ -> fill in notes.md, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
