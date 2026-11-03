#!/usr/bin/env python3
"""Layerwise branch-vs-value probes on a RuleChain (depth2) CODI-0 / CODI checkpoint.

Question: inside the single forward pass, does the model compute each rule's branch decision (b_j: did the condition
hold?) and its result value (s_j), and at which layer? If branching is the bottleneck, b_j should become decodable
late or weakly while values that do not depend on it stay readable; if rules are computed while the question is read,
the layer where s_j appears should climb with j (a staircase whose slope = layers per rule).

Positions (one forward through question + <bot> [+ k latents] + <eot> + "The answer is:", exactly as eval):
  * ``rule_end``: the "." token closing rule j's clause. The model has read x0 and rules 1..j here, so s_j is
    computable in principle (causal reading order).
  * ``ans``: the ":" of "The answer is:" (CODI's distillation position, where the answer token is read out).

Targets per item and rule j (only items with level >= j):
  * ``in``  = s_{j-1} (the value entering rule j; s_0 = x0),
  * ``br``  = b_j (1 = the first branch: even / at least t / below t),
  * ``out`` = s_j.
Values are probed as two 10-way digit heads (tens, ones); "exact" = both right.

Probe: per (position, j, layer, target) a linear softmax probe on per-layer-standardised hidden states, full-batch
L-BFGS with L2; lambda chosen from LAMBDAS on a 20% validation slice of the probe-train split. Items are split 70/30
by a hash of the item id (same split for every cell). Reported on the 30% probe-test split: branch = balanced
accuracy; values = exact accuracy. Baselines on the same split: majority class, and a "surface" probe (same fitting)
on question features that need no rule execution: x0, x0 mod 2, and kind one-hot + constants of rules 1..j. For j = 1
the surface probe can solve the branch (x0 vs t is linear); for j >= 2 the branch depends on s_{j-1}, which needs
execution.

Runs inside the CODI venv from the CODI checkout, like diagnose_codi_latents.py:

  cd ../codi && PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 .venv/bin/python ../ai-capstone/scripts/probe_rulechain_branch_codi.py \\
      --ckpt_dir ../ai-capstone/rulechain-checkpoints/gpt2_depth2_codi0/checkpoint-29716 \\
      --test_json ../synthetic_data_v2/depth2/test.json --max_level 8 --slug v2-depth2-codi0-branch-probes \\
      --num_latent 0 --inf_latent_iterations 0 <the model flags from diagnose_codi_latents.py>
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers
from torch.nn import functional as F

sys.path.insert(0, os.getcwd())  # the CODI checkout: `src.model`
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.model import DataArguments, ModelArguments, TrainingArguments  # noqa: E402

from diagnose_codi_latents import encode_and_think_lean, free, pick_device  # noqa: E402
from eval_codi import build_model, tokenize_questions  # noqa: E402
from eval_synthetic_codi import answer_token, check_flags  # noqa: E402
from synthetic_common import load_synthetic  # noqa: E402

from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402
from latentreasoning.tasks.synthetic_v2 import parse_chain_question  # noqa: E402

LAMBDAS = (1e-4, 1e-3, 1e-2)
TARGETS = ("in", "br", "out")


@dataclass
class ProbeArguments:
    test_json: str = field(default="")
    max_level: int = field(default=8)
    device: str = field(default="auto")
    slug: str = field(default="v2-depth2-codi0-branch-probes")
    stage: str = field(default="full_run")
    hardware: str = field(default="local")
    checkpoint_label: Optional[str] = field(default=None)
    smoke_n: int = field(default=0, metadata={"help": "items per level (smoke test, not logged)"})
    probe_iters: int = field(default=150)


# ---------------------------------------------------------------- task side

def first_branch(rule, s: int) -> bool:
    return s % 2 == 0 if rule.kind == "P" else (s >= rule.t if rule.kind == "T" else s < rule.t)


def walk_states(chain) -> tuple[list[int], list[int]]:
    s, states, branches = chain.x0, [chain.x0], []
    for r in chain.rules:
        b = first_branch(r, s)
        branches.append(int(b))
        st = r.step(s)
        assert st is not None
        s = st[3]
        states.append(s)
    return states, branches


def rule_end_tokens(tokenizer, question: str, n_rules: int) -> list[int]:
    """Token index (within the question's own token ids) of the '.' that closes each rule's clause."""
    full = tokenizer.encode(question)
    out, start = [], 0
    for j in range(1, n_rules + 1):
        nxt = f". Rule {j + 1}:" if j < n_rules else ". What is"
        c = question.index(nxt, question.index(f"Rule {j}:", start))
        prefix = tokenizer.encode(question[:c + 1])
        assert full[:len(prefix)] == prefix and tokenizer.decode(prefix[-1:]) == ".", (question, j)
        out.append(len(prefix) - 1)
        start = c
    return out


def surface_features(chain, j: int, max_rules: int) -> list[float]:
    f = [chain.x0 / 100, chain.x0 % 2]
    for i in range(max_rules):
        if i < j:
            r = chain.rules[i]
            f += [r.kind == "P", r.kind == "T", r.kind == "D", r.t / 100, r.a / 100, r.b / 100]
        else:
            f += [0.0] * 6
    return [float(x) for x in f]


def in_probe_train(item_id: str) -> bool:
    return int(hashlib.sha256(item_id.encode()).hexdigest(), 16) % 10 < 7


# ---------------------------------------------------------------- model side

@torch.no_grad()
def collect(model, tokenizer, rows, data_args, training_args, max_level):
    """Per item: hidden states [n_rules, L, d] at the rule-end tokens and [L, d] at the answer position (float16),
    plus the argmax answer token."""
    device = next(model.parameters()).device
    body = model.codi.get_base_model().transformer
    dec = [model.eot_id] + tokenizer.encode("The answer is:")
    embed = model.get_embd(model.codi, model.model_name)
    questions = [r["question"].strip() + "\n" for r in rows]
    q_ids = [tokenizer.encode(q) for q in questions]
    ends = [rule_end_tokens(tokenizer, q, int(r["level"])) for q, r in zip(questions, rows)]
    rule_states, ans_states, preds = [], [], []
    bs = data_args.batch_size
    for bi, batch in enumerate(tokenize_questions(model, tokenizer, questions, bs, training_args, device)):
        width = batch["input_ids"].size(1)  # left-padded question + <bot>
        # question-pass hidden states (all layers) -- same call as encode_and_think_lean's first step
        from eval_codi import question_position_ids
        q_pos = question_position_ids(batch["attention_mask"]) if training_args.pad_invariant else None
        qout = body(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], position_ids=q_pos,
                    output_hidden_states=True)
        hs = torch.stack(qout.hidden_states, 1)  # [B, L, T, d]
        for b in range(batch["input_ids"].size(0)):
            i = bi * bs + b
            off = width - 1 - len(q_ids[i])
            idx = torch.tensor([off + t for t in ends[i]], device=device)
            assert batch["input_ids"][b, idx].tolist() == [q_ids[i][t] for t in ends[i]]
            rule_states.append(hs[b][:, idx].permute(1, 0, 2).half().cpu())  # [n, L, d]
        del qout, hs
        # latents (k = inf_latent_iterations) + decoder prefix: identical to diagnose_codi_latents.student_states
        past, pi = encode_and_think_lean(model, batch, training_args)
        emb = embed(torch.tensor(dec, dtype=torch.long, device=device))[None].expand(batch["input_ids"].size(0), -1, -1)
        out = model.codi(inputs_embeds=emb, use_cache=True, past_key_values=past, output_hidden_states=True,
                         **(pi.next(len(dec)) if pi else {"attention_mask": None}))
        ans_states.append(torch.stack([h[:, -1] for h in out.hidden_states], 1).half().cpu())
        preds += out.logits[:, -1, :model.codi.config.vocab_size - 1].argmax(-1).tolist()
        del out, past
        free(device.type)
    return rule_states, torch.cat(ans_states), preds


# ---------------------------------------------------------------- probes

def fit_probe(Xtr, ytr, Xte, n_cls: int, iters: int):
    """Linear softmax probes for all layers and lambdas at once.
    X: [N, L, d] float32 (standardised), y: [N] long. Returns test logits per layer for the val-selected lambda,
    and the selected lambda index per layer."""
    N, L, d = Xtr.shape
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    nv = max(1, N // 5)
    va, tr = perm[:nv], perm[nv:]
    K = len(LAMBDAS)
    W = torch.zeros(K, L, d, n_cls, requires_grad=True)
    B = torch.zeros(K, L, n_cls, requires_grad=True)
    lam = torch.tensor(LAMBDAS)[:, None]
    opt = torch.optim.LBFGS([W, B], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")
    Xt, yt = Xtr[tr], ytr[tr]
    y_flat = yt[None, :, None].expand(K, len(tr), L).reshape(-1)  # same (k, n, l) order as the logits

    def closure():
        opt.zero_grad()
        logits = torch.einsum("nld,kldc->knlc", Xt, W) + B[:, None]
        ce = F.cross_entropy(logits.reshape(-1, n_cls), y_flat, reduction="none").view(K, -1, L).mean(1)
        loss = (ce + lam * W.pow(2).sum((2, 3))).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        vlog = torch.einsum("nld,kldc->knlc", Xtr[va], W) + B[:, None]
        vacc = (vlog.argmax(-1) == ytr[va][None, :, None]).float().mean(1)  # [K, L]
        best = vacc.argmax(0)  # [L]
        tlog = torch.einsum("nld,kldc->knlc", Xte, W) + B[:, None]
        tlog = tlog[best, :, torch.arange(L)].permute(1, 0, 2)  # [Nte, L, C]
    return tlog, best


def standardise(Xtr, Xte):
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp(min=1e-4)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def balanced_acc(pred: torch.Tensor, y: torch.Tensor) -> float:
    accs = [(pred[y == c] == c).float().mean().item() for c in y.unique()]
    return sum(accs) / len(accs)


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ProbeArguments))
    model_args, data_args, training_args, pa = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    flags = check_flags(model_args.ckpt_dir, training_args)
    device = pick_device(pa.device)
    model, tokenizer, load_result = build_model(model_args, training_args, device=device, dtype=torch.float32)
    if any(not k.startswith("codi.base_model.model.lm_head") for k in load_result.missing_keys):
        raise SystemExit(f"checkpoint is missing weights: {load_result.missing_keys[:5]}")
    torch.manual_seed(0)

    rows = [r for r in load_synthetic(pa.test_json).rows if int(r["level"]) <= pa.max_level]
    if pa.smoke_n:
        by = {}
        for r in rows:
            by.setdefault(int(r["level"]), []).append(r)
        rows = [r for L in sorted(by) for r in by[L][:pa.smoke_n]]
    chains = [parse_chain_question(r["question"]) for r in rows]
    walks = [walk_states(c) for c in chains]
    for r, (st, _) in zip(rows, walks):
        assert str(st[-1]) == r["answer"]

    t0 = time.perf_counter()
    rule_states, ans_states, preds = collect(model, tokenizer, rows, data_args, training_args, pa.max_level)
    gold = [answer_token(tokenizer, r["answer"]) for r in rows]
    correct = [None if g is None else p == g for p, g in zip(preds, gold)]
    t_fwd = time.perf_counter() - t0
    acc_lv = {}
    for L in sorted({int(r["level"]) for r in rows}):
        c = [correct[i] for i, r in enumerate(rows) if int(r["level"]) == L and correct[i] is not None]
        acc_lv[L] = round(sum(c) / len(c), 4)
    print(f"forward done ({t_fwd:.0f}s); teacher-forced acc per level {acc_lv}", flush=True)

    is_tr = torch.tensor([in_probe_train(r["id"]) for r in rows])
    n_layers = ans_states.size(1)
    results: dict = {}
    per_item: dict = {i: {} for i in range(len(rows))}
    for pos in ("rule_end", "ans"):
        for j in range(1, pa.max_level + 1):
            items = [i for i, r in enumerate(rows) if int(r["level"]) >= j]
            if len(items) < 50:
                continue
            if pos == "rule_end":
                X = torch.stack([rule_states[i][j - 1] for i in items]).float()
            else:
                X = ans_states[items].float()
            tr_mask = is_tr[items]
            Xtr, Xte = standardise(X[tr_mask], X[~tr_mask])
            te_items = [i for i, m in zip(items, tr_mask.tolist()) if not m]
            S = torch.tensor([surface_features(chains[i], j, pa.max_level) for i in items])
            Str, Ste = standardise(S[tr_mask, None], S[~tr_mask, None])
            cell: dict = {"n_train": int(tr_mask.sum()), "n_test": len(te_items)}
            tgt_vals = {
                "in": torch.tensor([walks[i][0][j - 1] for i in items]),
                "br": torch.tensor([walks[i][1][j - 1] for i in items]),
                "out": torch.tensor([walks[i][0][j] for i in items]),
            }
            for tname in TARGETS:
                y = tgt_vals[tname]
                heads = {"br": y} if tname == "br" else {"tens": y // 10, "ones": y % 10}
                pred_layers, pred_surf = {}, {}
                for hname, yh in heads.items():
                    nc = 2 if hname == "br" else 10
                    tl, best = fit_probe(Xtr, yh[tr_mask], Xte, nc, pa.probe_iters)
                    pred_layers[hname] = tl.argmax(-1)  # [Nte, L]
                    sl, _ = fit_probe(Str, yh[tr_mask], Ste, nc, pa.probe_iters)
                    pred_surf[hname] = sl.argmax(-1)[:, 0]
                yte = y[~tr_mask]
                if tname == "br":
                    ok = pred_layers["br"] == yte[:, None]  # [Nte, L]
                    layer_score = [balanced_acc(pred_layers["br"][:, l], yte) for l in range(n_layers)]
                    surf = balanced_acc(pred_surf["br"], yte)
                    maj = 0.5
                else:
                    ok = (pred_layers["tens"] == (yte // 10)[:, None]) & (pred_layers["ones"] == (yte % 10)[:, None])
                    layer_score = ok.float().mean(0).tolist()
                    surf = ((pred_surf["tens"] == yte // 10) & (pred_surf["ones"] == yte % 10)).float().mean().item()
                    maj = (yte == yte.mode().values).float().mean().item()
                cor = torch.tensor([bool(correct[i]) for i in te_items])
                ent = {"by_layer": [round(x, 4) for x in layer_score], "surface": round(surf, 4),
                       "majority": round(maj, 4),
                       "raw_acc_model_correct": [round(x, 4) for x in ok[cor].float().mean(0).tolist()] if cor.any() else None,
                       "raw_acc_model_wrong": [round(x, 4) for x in ok[~cor].float().mean(0).tolist()] if (~cor).any() else None}
                if tname == "br":  # split by rule kind: parity vs comparison
                    kinds = [chains[i].rules[j - 1].kind for i in te_items]
                    for kd in "PTD":
                        m = torch.tensor([k == kd for k in kinds])
                        if m.sum() >= 20 and yte[m].unique().numel() == 2:
                            ent[f"by_layer_kind_{kd}"] = [round(balanced_acc(pred_layers["br"][m, l], yte[m]), 4)
                                                          for l in range(n_layers)]
                cell[tname] = ent
                for row_i, i in enumerate(te_items):
                    per_item[i][f"{pos}_r{j}_{tname}"] = "".join("1" if v else "0" for v in ok[row_i].tolist())
            results.setdefault(pos, {})[j] = cell
            print(f"{pos} j={j} n_te={len(te_items)} | " + " | ".join(
                f"{t}: max {max(cell[t]['by_layer']):.3f}@L{cell[t]['by_layer'].index(max(cell[t]['by_layer']))} "
                f"surf {cell[t]['surface']:.3f}" for t in TARGETS), flush=True)

    elapsed = time.perf_counter() - t0
    records = []
    for i, r in enumerate(rows):
        rec = {"idx": i, "id": r["id"], "level": int(r["level"]), "question": r["question"],
               "gold_answer": r["answer"], "predicted_answer": tokenizer.decode([preds[i]]).strip(),
               "correct": correct[i], "probe_split": "train" if bool(is_tr[i]) else "test",
               "branches": walks[i][1], "states": walks[i][0]}
        if per_item[i]:
            rec["probe_ok_by_layer"] = per_item[i]
        records.append(rec)
    if pa.smoke_n:
        print(json.dumps({p: {j: {t: c[t]["by_layer"] for t in TARGETS} for j, c in v.items()} for p, v in results.items()})[:3000])
        print("not logging (smoke_n)")
        return
    ok_all = [c for c in correct if c is not None]
    metrics = {
        "final_answer_accuracy": round(sum(ok_all) / len(ok_all), 4),
        "unparseable_rate": 0.0,
        "compute_steps": training_args.inf_latent_iterations,
        "sec_per_example": elapsed / len(rows),
        "extra": {"accuracy_kind": "teacher-forced answer token at the distillation position (no generation)",
                  "accuracy_per_level": acc_lv,
                  "data_sha256": hashlib.sha256(Path(pa.test_json).read_bytes()).hexdigest(),
                  "matched_flags": flags, "probe": {"lambdas": LAMBDAS, "iters": pa.probe_iters,
                                                    "split": "70/30 by sha256(id)", "val": "20% of probe-train"},
                  "probes": results},
    }
    record = RunRecord(
        run_id=new_run_id("codi", pa.slug),
        mechanism="codi",
        stage=pa.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=pa.checkpoint_label or model_args.ckpt_dir,
                        n_params=sum(p.numel() for p in model.parameters())),
        dataset=DatasetInfo(name="synthetic-depth2", split="test", n_examples=len(rows), seed=None),
        metrics=metrics,
        hyperparams={"max_level": pa.max_level, "batch_size": data_args.batch_size, "device": device,
                     "test_json": pa.test_json, "latents": training_args.inf_latent_iterations},
        seed=0,
        hardware=f"{pa.hardware} / {device}",
        notes="RuleChain layerwise probes: branch bit b_j vs values s_{j-1}, s_j at rule-end tokens and the answer position.",
    )
    manifest = record.save(predictions=records)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
