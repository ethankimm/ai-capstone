#!/usr/bin/env python3
"""Linear (ridge) and 2-layer MLP probes on CODI's continuous thoughts z0..z5, to test
the "basis drift" reading of the interchange-patching null: maybe the intermediate
values ARE linearly (or shallowly non-linearly) recoverable from z0/z3, just rotated
out of the unembedding basis W_U that the logit lens reads through -- so the model
offloads legible read-outs to z2/z4 for the distillation loss while doing the real
arithmetic in a differently-oriented subspace at z0/z3.

Fits probes on `split="train"` (never seen at eval time by anyone else's numbers
either), evaluates on `split="validation"` (the fixed 1000-example held-out set from
`latentreasoning/data/gsm8k_aug.py` -- same examples for everyone, `test` stays
untouched). Compares, for every (iteration, step) pair: probe accuracy (prediction
within `--tol` relative error of the gold value) vs the logit-lens top-1/top-5 hit
rate computed on the SAME examples with the SAME decode call this script already makes
(no need to reuse another run's numbers).

No sklearn/scipy dependency (consistent with the rest of the CODI venv): ridge
regression is closed-form numpy; the MLP is a ~50k-param torch module trained with Adam.
Target transform: signed-log1p (`sign(v) * log1p(|v|)`) to compress GSM8K's value range
before regressing; predictions are inverted back to raw values for the tolerance check.

Reuses `run_thoughts` / `topk_token_strings` / `num_match` / `wilson_ci` from
`decode_patch_codi.py`.

Run inside the CODI venv, from the CODI checkout:

  cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/probe_codi.py \\
      --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \\
      --slug probe-pilot --stage pilot --hardware "RunPod RTX A5000 (secure)" \\
      --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \\
      --lora_r 128 --lora_alpha 32 --lora_init --greedy True \\
      --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \\
      --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \\
      --train_n 3000 --eval_n 300
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import transformers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())  # the CODI checkout: `src.model`

from src.model import DataArguments, ModelArguments, TrainingArguments  # noqa: E402
from eval_codi import build_model  # noqa: E402
from decode_patch_codi import run_thoughts, topk_token_strings, num_match, wilson_ci  # noqa: E402

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402


@dataclass
class ProbeArguments:
    slug: str = field(default="probe")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: Optional[str] = field(default=None)
    train_n: int = field(default=3000, metadata={"help": "train-split examples for fitting probes"})
    eval_n: int = field(default=300, metadata={"help": "validation-split examples for reporting (<=1000)"})
    train_seed: int = field(default=0)
    max_step: int = field(default=3, metadata={"help": "probe/report steps 1..max_step (most GSM8K-Aug problems have <=3)"})
    ridge_lambda: float = field(default=1.0)
    mlp_hidden: int = field(default=64)
    mlp_epochs: int = field(default=200)
    mlp_lr: float = field(default=1e-3)
    tol: float = field(default=0.01, metadata={"help": "relative-error tolerance counted as a probe hit"})


def parse_value(v: str) -> Optional[float]:
    v = v.strip().lstrip("+")
    try:
        return float(v.replace(",", ""))
    except ValueError:
        return None


def signed_log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def inv_signed_log1p(y: torch.Tensor) -> torch.Tensor:
    return torch.sign(y) * torch.expm1(y.abs())


def within_tol(pred: float, gold: float, tol: float) -> bool:
    denom = max(1.0, abs(gold))
    return abs(pred - gold) / denom <= tol


@torch.no_grad()
def extract_features(model, tokenizer, examples, device, n_latents, max_step: int):
    """For each example, returns per-iteration z vectors (float32, cpu) and, for each
    step 1..max_step present, the gold value and the top-1/top-5 logit-lens strings at
    every iteration (so logit-lens accuracy is computed on the identical example set)."""
    rows = []
    for ex in examples:
        q = ex.question.strip().replace("  ", " ")
        pkv, thoughts, latent = run_thoughts(model, tokenizer, q, device, n_latents)
        steps = ex.intermediate_values
        z = {it: rec["post"].detach().float().cpu().squeeze(0).squeeze(0) for it, rec in enumerate(thoughts, start=1)}
        per_iter_topk = {}
        for it, rec in enumerate(thoughts, start=1):
            logits = rec["logits"][0]
            per_iter_topk[it] = {
                "top1": topk_token_strings(tokenizer, logits, 1)[0],
                "top5": topk_token_strings(tokenizer, logits, 5),
            }
        rows.append({"idx": ex.idx, "steps": steps[:max_step], "z": z, "per_iter_topk": per_iter_topk})
    return rows


def fit_ridge(X: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    """Closed-form ridge: w = (X^T X + lam*I)^-1 X^T y, X already has a bias column."""
    d = X.shape[1]
    XtX = X.T @ X + lam * torch.eye(d, dtype=X.dtype)
    Xty = X.T @ y
    return torch.linalg.solve(XtX, Xty)


class TinyMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X: torch.Tensor, y: torch.Tensor, hidden: int, epochs: int, lr: float) -> TinyMLP:
    mlp = TinyMLP(X.shape[1], hidden)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        pred = mlp(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
    return mlp


def probe_one(it: int, s: int, train_rows, eval_rows, ridge_lambda: float, mlp_hidden: int,
              mlp_epochs: int, mlp_lr: float, tol: float) -> Optional[dict]:
    def collect(rows):
        xs, ys = [], []
        for r in rows:
            if s > len(r["steps"]):
                continue
            v = parse_value(r["steps"][s - 1])
            if v is None:
                continue
            xs.append(r["z"][it])
            ys.append(v)
        return xs, ys

    x_tr, y_tr = collect(train_rows)
    x_ev, y_ev = collect(eval_rows)
    if len(x_tr) < 20 or len(x_ev) < 10:
        return None

    X_tr = torch.stack(x_tr)
    X_ev = torch.stack(x_ev)
    y_tr_t = signed_log1p(torch.tensor(y_tr, dtype=torch.float32))
    y_ev_raw = torch.tensor(y_ev, dtype=torch.float32)

    mu, sigma = X_tr.mean(0, keepdim=True), X_tr.std(0, keepdim=True).clamp_min(1e-6)
    X_tr_n = (X_tr - mu) / sigma
    X_ev_n = (X_ev - mu) / sigma
    X_tr_b = torch.cat([X_tr_n, torch.ones(X_tr_n.shape[0], 1)], dim=1)
    X_ev_b = torch.cat([X_ev_n, torch.ones(X_ev_n.shape[0], 1)], dim=1)

    w = fit_ridge(X_tr_b, y_tr_t, ridge_lambda)
    ridge_pred = inv_signed_log1p(X_ev_b @ w)
    ridge_hits = sum(within_tol(p.item(), g.item(), tol) for p, g in zip(ridge_pred, y_ev_raw))
    ridge_acc = ridge_hits / len(y_ev)

    mlp = train_mlp(X_tr_n, y_tr_t, mlp_hidden, mlp_epochs, mlp_lr)
    with torch.no_grad():
        mlp_pred = inv_signed_log1p(mlp(X_ev_n))
    mlp_hits = sum(within_tol(p.item(), g.item(), tol) for p, g in zip(mlp_pred, y_ev_raw))
    mlp_acc = mlp_hits / len(y_ev)

    # logit-lens accuracy on the SAME eval rows / same step
    ll_hits1 = ll_hits5 = 0
    n_ll = 0
    for r in eval_rows:
        if s > len(r["steps"]):
            continue
        gold = r["steps"][s - 1]
        if parse_value(gold) is None:
            continue
        n_ll += 1
        topk = r["per_iter_topk"][it]
        if num_match([topk["top1"]], gold):
            ll_hits1 += 1
        if num_match(topk["top5"], gold):
            ll_hits5 += 1
    ll_acc1 = ll_hits1 / n_ll if n_ll else 0.0
    ll_acc5 = ll_hits5 / n_ll if n_ll else 0.0

    return {
        "iter": it, "step": s, "n_train": len(x_tr), "n_eval": len(x_ev),
        "ridge_acc_tol": ridge_acc, "ridge_acc_wilson_ci": list(wilson_ci(ridge_hits, len(y_ev))),
        "mlp_acc_tol": mlp_acc, "mlp_acc_wilson_ci": list(wilson_ci(mlp_hits, len(y_ev))),
        "logit_lens_top1": ll_acc1, "logit_lens_top5": ll_acc5,
    }


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ProbeArguments))
    model_args, data_args, training_args, pa = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = "cuda"
    torch.manual_seed(0)

    model, tokenizer, load_result = build_model(model_args, training_args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {model_args.ckpt_dir}: missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}")

    n_latents = training_args.inf_latent_iterations
    train_examples = load_gsm8k_aug(split="train", n=pa.train_n, seed=pa.train_seed)
    eval_examples = load_gsm8k_aug(split="validation", n=pa.eval_n, seed=0)
    print(f"train_n={len(train_examples)} eval_n(validation)={len(eval_examples)}")

    t0 = time.perf_counter()
    train_rows = extract_features(model, tokenizer, train_examples, device, n_latents, pa.max_step)
    print(f"train features extracted in {time.perf_counter() - t0:.0f}s")
    t1 = time.perf_counter()
    eval_rows = extract_features(model, tokenizer, eval_examples, device, n_latents, pa.max_step)
    print(f"eval features extracted in {time.perf_counter() - t1:.0f}s")

    results = []
    for it in range(1, n_latents + 1):
        for s in range(1, pa.max_step + 1):
            r = probe_one(it, s, train_rows, eval_rows, pa.ridge_lambda, pa.mlp_hidden, pa.mlp_epochs, pa.mlp_lr, pa.tol)
            if r is None:
                continue
            results.append(r)
            print(f"iter={it} step={s}: ridge={r['ridge_acc_tol']:.3f} mlp={r['mlp_acc_tol']:.3f} "
                  f"logit_lens_top1={r['logit_lens_top1']:.3f} (n_eval={r['n_eval']})", flush=True)

    nondecodable_iters = (1, 4)
    decodable_iters = (3, 5)
    nondec_rows = [r for r in results if r["iter"] in nondecodable_iters]
    dec_rows = [r for r in results if r["iter"] in decodable_iters]

    def avg(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else None

    print(f"NONDECODABLE iters {nondecodable_iters}: avg ridge={avg(nondec_rows, 'ridge_acc_tol')} "
          f"avg mlp={avg(nondec_rows, 'mlp_acc_tol')} avg logit_lens_top1={avg(nondec_rows, 'logit_lens_top1')}")
    print(f"DECODABLE iters {decodable_iters}: avg ridge={avg(dec_rows, 'ridge_acc_tol')} "
          f"avg mlp={avg(dec_rows, 'mlp_acc_tol')} avg logit_lens_top1={avg(dec_rows, 'logit_lens_top1')}")

    metrics = {
        "compute_steps": n_latents,
        "decoding_accuracy": avg(nondec_rows, "ridge_acc_tol"),  # headline: probe recoverability at the placeholder slots
        "extra": {
            "per_iter_step": results,
            "nondecodable_iters": list(nondecodable_iters),
            "decodable_iters": list(decodable_iters),
            "nondecodable_avg_ridge_acc": avg(nondec_rows, "ridge_acc_tol"),
            "nondecodable_avg_mlp_acc": avg(nondec_rows, "mlp_acc_tol"),
            "nondecodable_avg_logit_lens_top1": avg(nondec_rows, "logit_lens_top1"),
            "decodable_avg_ridge_acc": avg(dec_rows, "ridge_acc_tol"),
            "decodable_avg_mlp_acc": avg(dec_rows, "mlp_acc_tol"),
            "decodable_avg_logit_lens_top1": avg(dec_rows, "logit_lens_top1"),
            "tol": pa.tol, "ridge_lambda": pa.ridge_lambda,
            "mlp_hidden": pa.mlp_hidden, "mlp_epochs": pa.mlp_epochs, "mlp_lr": pa.mlp_lr,
            "n_trainable_params_backbone": n_params,
            "das_note": "Distributed Alignment Search on probe-extracted directions is a "
                        "logical next step (verify a probe-aligned subspace has causal "
                        "control where raw donor-interchange patching at z0/z3 fails) but "
                        "is NOT implemented in this script -- it needs a differentiable "
                        "rotation/subspace-intervention pass, not just a fitted readout.",
        },
    }

    record = RunRecord(
        run_id=new_run_id("codi", pa.slug),
        mechanism="codi",
        stage=pa.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=pa.checkpoint_label or model_args.ckpt_dir, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="validation", n_examples=len(eval_examples), seed=0),
        metrics=metrics,
        hyperparams={
            "inf_latent_iterations": n_latents, "num_latent": training_args.num_latent,
            "train_n": pa.train_n, "eval_n": pa.eval_n, "max_step": pa.max_step,
            "ridge_lambda": pa.ridge_lambda, "mlp_hidden": pa.mlp_hidden,
            "mlp_epochs": pa.mlp_epochs, "mlp_lr": pa.mlp_lr, "tol": pa.tol,
        },
        seed=pa.train_seed,
        hardware=f"{pa.hardware} / {torch.cuda.get_device_name(0)}",
        notes="Linear ridge + 2-layer MLP probes on z0..z5, fit on train / reported on "
              "the fixed validation split, compared position-by-position to logit-lens "
              "accuracy on the same eval examples: tests whether z0/z3 carry recoverable "
              "intermediate values under a rotated basis even though the logit lens null.",
    )
    manifest = record.save(predictions=[{"iter": r["iter"], "step": r["step"], **r} for r in results])
    out_dir = manifest.parent
    (out_dir / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
