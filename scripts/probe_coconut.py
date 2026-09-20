#!/usr/bin/env python3
"""Linear (ridge) and 2-layer MLP probes on Coconut's continuous latent passes 0..5 --
the Coconut counterpart to `probe_codi.py` (see that script's docstring for the full
basis-drift motivation). Same question, same method, pointed at the other width-based
LRM this project studies: does `connordilgren/gpt2-gsm8k-coconut` encode intermediate
calculator values recoverably in the raw 768-d live hidden vector at each latent pass,
compared against the paper's own vocabulary-projection logit lens on the same vectors?

Uses `scripts/coconut_common.py` for model loading and the latent-filling forward pass
(`run_passes`), and the same `z_i = pass i` <-> `iter = i+1` indexing convention as
`probe_codi.py` / `probe_metrics_continuous.py` (NONDECODABLE_ITERS=(1,4) i.e. z0/z3,
DECODABLE_ITERS=(3,5) i.e. z2/z4), so `scripts/probe_metrics_continuous.py` can be
pointed at either mechanism's `raw_predictions.jsonl` unmodified.

Data: `gsm_valid-gold-reasoning-trace_test.json` (1194 examples, gold step values) --
the same file `decode_patch_coconut.py` uses, since it's the only local Coconut GSM8K
data with verified step values (see that script's docstring for provenance). No
separate held-out `train`/`validation` split exists for Coconut the way `gsm8k_aug`
gives CODI one, so this script splits that single file in half by index parity
(`idx % 2`) for probe-fit vs probe-report, mirroring `decode_patch_coconut.py`'s own
`mapping_split` convention -- NOT the same split as CODI's `train`/`validation`, so
probe accuracy between the two mechanisms is not literally apples-to-apples on split
provenance (flag this in notes.md).

Run inside the pinned Coconut venv (torch==2.5.1, transformers==4.46.2 -- see
`coconut_common.py` / `decode_patch_coconut.py` docstrings for why), from this repo's
root (not the Coconut checkout -- unlike CODI, `coconut_common.py` has no reference-repo
import dependency):

  cd /workspace/ai-capstone && .venv_coconut/bin/python scripts/probe_coconut.py \\
      --checkpoint_path /workspace/coconut_checkpoints/gsm-coconut/checkpoint_33 \\
      --data_dir /workspace/coconut_data \\
      --slug probe-pilot --stage pilot --hardware "RunPod RTX A5000 (secure)" \\
      --num_latents 6 --train_n 500 --eval_n 300
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coconut_common import (  # noqa: E402
    load_coconut, encode_question, run_passes, topk_token_strings, parse_step_value,
    num_match, wilson_ci,
)
from probe_common import (  # noqa: E402
    parse_value, signed_log1p, inv_signed_log1p, within_tol, fit_ridge, train_mlp,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from latentreasoning.mechanisms.coconut import name as mechanism_name  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402


@dataclass
class ProbeArguments:
    checkpoint_path: str = field(metadata={"help": "path to the downloaded checkpoint_33 file"})
    data_dir: str = field(metadata={"help": "dir with gsm_valid-gold-reasoning-trace_test.json"})
    model_id: str = field(default="openai-community/gpt2")
    slug: str = field(default="probe")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: str = field(default="hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33")
    num_latents: int = field(default=6)
    train_n: int = field(default=500, metadata={"help": "probe-fit examples (half-A split, capped)"})
    eval_n: int = field(default=300, metadata={"help": "probe-report examples (half-B split, capped)"})
    split_seed: int = field(default=0)
    max_step: int = field(default=3)
    ridge_lambda: float = field(default=1.0)
    mlp_hidden: int = field(default=64)
    mlp_epochs: int = field(default=200)
    mlp_lr: float = field(default=1e-3)
    tol: float = field(default=0.01)
    device: str = field(default="cuda")


class Example:
    __slots__ = ("idx", "question", "answer", "steps")

    def __init__(self, idx, question, answer, steps):
        self.idx, self.question, self.answer, self.steps = idx, question, answer, steps


def load_examples(data_dir: str) -> list[Example]:
    rows = json.loads((Path(data_dir) / "gsm_valid-gold-reasoning-trace_test.json").read_text())
    return [Example(i, r["question"], r["answer"].strip(), r["steps"]) for i, r in enumerate(rows)]


def split_train_eval(examples: list[Example], train_n: int, eval_n: int, seed: int):
    """Index-parity split (mirrors `decode_patch_coconut.py`'s `mapping_split`), then
    capped/shuffled to the requested sizes."""
    half_a = [e for e in examples if e.idx % 2 == 0]
    half_b = [e for e in examples if e.idx % 2 == 1]
    rng = random.Random(seed)
    rng.shuffle(half_a)
    rng.shuffle(half_b)
    return half_a[:train_n], half_b[:eval_n]


@torch.no_grad()
def extract_features(base_model, embedding, tokenizer, special_ids, examples, device,
                      num_latents: int, max_step: int):
    """Per example: pass-indexed live hidden vectors (`z`, 1-indexed pass->iter same as
    `probe_codi.py`) and, for each step 1..max_step present, the gold value + top-1/top-5
    vocabulary-projection strings at every pass (paper's own logit-lens method, already
    computed inside `run_passes`, no extra forward pass)."""
    rows = []
    for ex in examples:
        input_ids, attn = encode_question(tokenizer, special_ids, ex.question, num_latents, device)
        pass_records, _, _, _ = run_passes(base_model, embedding, input_ids, attn, device,
                                            num_latents, latent_token_id=special_ids["latent"])
        steps = [parse_step_value(s) for s in ex.steps]
        z = {pr["pass"] + 1: pr["live_hidden"].detach().float().cpu() for pr in pass_records}
        per_iter_topk = {
            pr["pass"] + 1: {
                "top1": topk_token_strings(tokenizer, pr["logits"], 1)[0],
                "top5": topk_token_strings(tokenizer, pr["logits"], 5),
            }
            for pr in pass_records
        }
        rows.append({"idx": ex.idx, "steps": steps[:max_step], "z": z, "per_iter_topk": per_iter_topk})
    return rows


def probe_one(it: int, s: int, train_rows, eval_rows, ridge_lambda: float, mlp_hidden: int,
              mlp_epochs: int, mlp_lr: float, tol: float) -> Optional[dict]:
    def collect(rows):
        xs, ys = [], []
        for r in rows:
            if s > len(r["steps"]):
                continue
            v = r["steps"][s - 1]
            if v is None:
                continue
            v = parse_value(v)
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

    eval_idx = [r["idx"] for r in eval_rows if s <= len(r["steps"]) and r["steps"][s - 1] is not None
                and parse_value(r["steps"][s - 1]) is not None]
    raw = [
        {"idx": idx, "y_gold": g.item(), "ridge_pred": p_r.item(), "mlp_pred": p_m.item()}
        for idx, g, p_r, p_m in zip(eval_idx, y_ev_raw, ridge_pred, mlp_pred)
    ]

    ll_hits1 = ll_hits5 = 0
    n_ll = 0
    for r in eval_rows:
        if s > len(r["steps"]) or r["steps"][s - 1] is None:
            continue
        gold = r["steps"][s - 1]
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
        "raw": raw,
    }


def main() -> None:
    parser = transformers.HfArgumentParser((ProbeArguments,))
    (pa,) = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = pa.device
    torch.manual_seed(0)

    tokenizer, base_model, embedding, special_ids = load_coconut(pa.model_id, pa.checkpoint_path, device)
    n_params = sum(p.numel() for p in base_model.parameters())

    examples = load_examples(pa.data_dir)
    train_examples, eval_examples = split_train_eval(examples, pa.train_n, pa.eval_n, pa.split_seed)
    print(f"train_n={len(train_examples)} eval_n={len(eval_examples)}")

    t0 = time.perf_counter()
    train_rows = extract_features(base_model, embedding, tokenizer, special_ids, train_examples,
                                   device, pa.num_latents, pa.max_step)
    print(f"train features extracted in {time.perf_counter() - t0:.0f}s")
    t1 = time.perf_counter()
    eval_rows = extract_features(base_model, embedding, tokenizer, special_ids, eval_examples,
                                  device, pa.num_latents, pa.max_step)
    print(f"eval features extracted in {time.perf_counter() - t1:.0f}s")

    results = []
    for it in range(1, pa.num_latents + 1):
        for s in range(1, pa.max_step + 1):
            r = probe_one(it, s, train_rows, eval_rows, pa.ridge_lambda, pa.mlp_hidden,
                           pa.mlp_epochs, pa.mlp_lr, pa.tol)
            if r is None:
                continue
            results.append(r)
            print(f"iter={it} step={s}: ridge={r['ridge_acc_tol']:.3f} mlp={r['mlp_acc_tol']:.3f} "
                  f"logit_lens_top1={r['logit_lens_top1']:.3f} (n_eval={r['n_eval']})", flush=True)

    nondecodable_iters = (1, 4)  # z0, z3 -- same fixed naming convention as probe_codi.py
    decodable_iters = (3, 5)  # z2, z4
    nondec_rows = [r for r in results if r["iter"] in nondecodable_iters]
    dec_rows = [r for r in results if r["iter"] in decodable_iters]

    def avg(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else None

    print(f"NONDECODABLE iters {nondecodable_iters}: avg ridge={avg(nondec_rows, 'ridge_acc_tol')} "
          f"avg mlp={avg(nondec_rows, 'mlp_acc_tol')} avg logit_lens_top1={avg(nondec_rows, 'logit_lens_top1')}")
    print(f"DECODABLE iters {decodable_iters}: avg ridge={avg(dec_rows, 'ridge_acc_tol')} "
          f"avg mlp={avg(dec_rows, 'mlp_acc_tol')} avg logit_lens_top1={avg(dec_rows, 'logit_lens_top1')}")

    metrics = {
        "compute_steps": pa.num_latents,
        "decoding_accuracy": avg(nondec_rows, "ridge_acc_tol"),
        "extra": {
            "per_iter_step": [{k: v for k, v in r.items() if k != "raw"} for r in results],
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
            "n_trainable_params": n_params,
            "split_note": "train/eval split is index-parity on gsm_valid-gold-reasoning-trace_test.json "
                           "(decode_patch_coconut.py's own mapping_split convention), NOT the shared "
                           "gsm8k_aug train/validation split probe_codi.py uses -- not directly comparable "
                           "on split provenance across mechanisms.",
        },
    }

    record = RunRecord(
        run_id=new_run_id("coconut", pa.slug),
        mechanism=mechanism_name,
        stage=pa.stage,
        model=ModelInfo(backbone=pa.model_id, checkpoint=pa.checkpoint_label, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(eval_examples), seed=pa.split_seed),
        metrics=metrics,
        hyperparams={
            "num_latents": pa.num_latents, "train_n": pa.train_n, "eval_n": pa.eval_n,
            "max_step": pa.max_step, "ridge_lambda": pa.ridge_lambda, "mlp_hidden": pa.mlp_hidden,
            "mlp_epochs": pa.mlp_epochs, "mlp_lr": pa.mlp_lr, "tol": pa.tol,
        },
        seed=pa.split_seed,
        hardware=f"{pa.hardware} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        notes="Linear ridge + 2-layer MLP probes on Coconut's z0..z5 live latent-pass "
              "hidden vectors, fit on an index-parity half-split of the gold-trace test "
              "set / reported on the other half, compared position-by-position to the "
              "paper's own vocabulary-projection logit lens on the same examples -- "
              "Coconut counterpart to probe_codi.py's basis-drift check.",
    )
    manifest = record.save(predictions=[
        {"iter": r["iter"], "step": r["step"], **{k: v for k, v in r.items() if k != "raw"}}
        for r in results
    ])
    out_dir = manifest.parent
    (out_dir / "eval_command.txt").write_text(argv + "\n")

    with (out_dir / "raw_predictions.jsonl").open("w") as f:
        for r in results:
            for row in r["raw"]:
                f.write(json.dumps({"iter": r["iter"], "step": r["step"], **row}) + "\n")

    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
