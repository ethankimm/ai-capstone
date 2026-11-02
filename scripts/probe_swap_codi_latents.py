#!/usr/bin/env python3
"""Necessity (latent swap / ablation) and intermediate-value probes for a RuleChain (depth2) CODI or CODI-0 checkpoint.
Follow-up to scripts/diagnose_codi_latents.py; same model loading, pad_invariant passes and teacher-forced readout.

Latent inputs: z_1 = prj(h_<bot>) (last hidden state of the question pass), z_{j+1} = prj(h of latent pass j).
Latent pass j consumes z_j. Teacher-forced answer token at ":" of "<eot> The answer is:".

1. Necessity (CODI only, trained latent count k):
   - swap_all:  every z_j taken from a donor (same level, different gold answer), fixed.
   - swap_z1:   z_1 from the donor, then the model runs its own passes from there.
   - swap_rest: own z_1, then z_2..z_k from the donor, fixed.
   - mean_all / zero_all: every z_j replaced by its train-sample mean / by zeros.
   Scored per level as accuracy, follows-donor (prediction = donor's answer), keeps-own (= own answer), other.
2. Probes: linear softmax probes (100 classes, values 0..99) for v_1..v_6 (result of rule j) and the answer, from the
   hidden state at every other layer (0, 2, ..., 12) at positions <bot>, latent pass 1..k (CODI), and ":" (answer
   position). Trained on --train_per_level items per level of train.json, tested on test.json. Target v_j only on items
   with at least j rules. Baseline: majority class of the training targets.

Run from the CODI checkout in its venv, flags as for diagnose_codi_latents.py (see that docstring).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
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

from eval_codi import PadInvariant, build_model, question_position_ids, tokenize_questions  # noqa: E402
from eval_synthetic_codi import answer_token, check_flags  # noqa: E402
from synthetic_common import load_synthetic  # noqa: E402

from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402

if not torch.cuda.is_available():  # checkpoints were saved on CUDA
    _torch_load = torch.load
    torch.load = lambda *a, **kw: _torch_load(*a, **{"map_location": "cpu", **kw})

PROBE_LAYERS = (0, 2, 4, 6, 8, 10, 12)


@dataclass
class Args:
    test_json: str = field(default="")
    train_json: str = field(default="")
    train_per_level: int = field(default=500)
    smoke_n: int = field(default=0)
    probe_steps: int = field(default=300)
    slug: str = field(default="v2-depth2-codi-swap-probes")
    stage: str = field(default="full_run")
    hardware: str = field(default="local")
    checkpoint_label: Optional[str] = field(default=None)
    no_log: bool = field(default=False)


def per_level(rows, n, seed=0):
    by: dict[int, list] = {}
    for r in rows:
        by.setdefault(int(r["level"]), []).append(r)
    out = []
    for lvl in sorted(by):
        items = by[lvl]
        if n and len(items) > n:
            items = random.Random(seed + lvl).sample(items, n)
        out += items
    return out


def step_values(row) -> list[int]:
    return [int(re.search(r"=(-?\d+)>>", s).group(1)) for s in row["steps"]]


@torch.no_grad()
def run(model, tokenizer, rows, k, training_args, bs, override=None, collect=False):
    """Question pass, k latent passes, decoder "<eot> The answer is:". override(i_global, j) -> tensor [d] or None
    replaces z_j (0-based j) for item i; "free" positions use the model's own z. Returns predictions (token ids), the
    fed z's [N, k, d], and if collect, hidden states {pos: [N, len(PROBE_LAYERS), d]} (fp16, CPU)."""
    device = next(model.parameters()).device
    body = model.codi.get_base_model().transformer
    embed = model.get_embd(model.codi, model.model_name)
    dec = torch.tensor([model.eot_id] + tokenizer.encode("The answer is:"), device=device)
    preds, zs, states = [], [], {}
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        batch = tokenize_questions(model, tokenizer, [r["question"].strip() + "\n" for r in chunk], len(chunk),
                                   training_args, device)[0]
        q_pos = question_position_ids(batch["attention_mask"])
        out = body(input_ids=batch["input_ids"], use_cache=True, attention_mask=batch["attention_mask"],
                   position_ids=q_pos, output_hidden_states=collect)
        past = out.past_key_values
        if collect:
            states.setdefault("bot", []).append(torch.stack([out.hidden_states[l][:, -1] for l in PROBE_LAYERS], 1).half().cpu())
        z = model.prj(out.last_hidden_state[:, -1:, :])
        pi = PadInvariant(batch["attention_mask"])
        fed = []
        for j in range(k):
            if override is not None:
                rep = [override(s + b, j) for b in range(len(chunk))]
                if any(r is not None for r in rep):
                    z = torch.stack([rep[b].to(device) if rep[b] is not None else z[b, 0] for b in range(len(chunk))])[:, None]
            fed.append(z[:, 0].float().cpu())
            out = body(inputs_embeds=z, use_cache=True, past_key_values=past, output_hidden_states=collect, **pi.next(1))
            past = out.past_key_values
            if collect:
                states.setdefault(f"z{j + 1}", []).append(torch.stack([out.hidden_states[l][:, -1] for l in PROBE_LAYERS], 1).half().cpu())
            z = model.prj(out.last_hidden_state[:, -1:, :])
        out = model.codi(inputs_embeds=embed(dec)[None].expand(len(chunk), -1, -1), use_cache=True, past_key_values=past,
                         output_hidden_states=collect, **pi.next(len(dec)))
        if collect:
            states.setdefault("ans", []).append(torch.stack([out.hidden_states[l][:, -1] for l in PROBE_LAYERS], 1).half().cpu())
        preds += out.logits[:, -1, :model.codi.config.vocab_size - 1].argmax(-1).tolist()
        zs.append(torch.stack(fed, 1) if fed else torch.zeros(len(chunk), 0, model.codi.config.n_embd))
        del out, past
        if device.type == "mps":
            torch.mps.empty_cache()
    return preds, torch.cat(zs), {p: torch.cat(v) for p, v in states.items()}


def score_swap(rows, preds, gold_tok, donor_tok) -> dict:
    by: dict[int, list] = {}
    for i, r in enumerate(rows):
        g, d = gold_tok[i], donor_tok[i] if donor_tok else None
        if g is None:
            continue
        by.setdefault(int(r["level"]), []).append((preds[i] == g, d is not None and preds[i] == d))
    out = {}
    for L, v in sorted(by.items()):
        n = len(v)
        own = sum(a for a, _ in v) / n
        don = sum(b for _, b in v) / n
        out[L] = {"keeps_own": round(own, 4), "follows_donor": round(don, 4), "other": round(1 - own - don, 4), "n": n}
    return out


def fit_probe(xtr, ytr, xte, yte, steps):
    """Multinomial logistic probe (100 classes), standardised features, full-batch Adam, small L2. CPU."""
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-5
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    lin = torch.nn.Linear(xtr.size(1), 100)
    opt = torch.optim.Adam(lin.parameters(), lr=0.01, weight_decay=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(lin(xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return (lin(xte).argmax(-1) == yte).float().mean().item(), (lin(xtr).argmax(-1) == ytr).float().mean().item()


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, Args))
    model_args, data_args, training_args, a = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    flags = check_flags(model_args.ckpt_dir, training_args)
    assert training_args.pad_invariant, "written for pad_invariant checkpoints"
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, load_result = build_model(model_args, training_args, device=device, dtype=torch.float32)
    if any(not k.startswith("codi.base_model.model.lm_head") for k in load_result.missing_keys):
        raise SystemExit(f"checkpoint is missing weights: {load_result.missing_keys[:5]}")
    k = training_args.num_latent
    bs = data_args.batch_size
    test = per_level(load_synthetic(a.test_json).rows, a.smoke_n)
    train = per_level(load_synthetic(a.train_json).rows, a.smoke_n or a.train_per_level)
    gold_te = [answer_token(tokenizer, r["answer"]) for r in test]
    t0 = time.perf_counter()
    results: dict = {}

    # Clean runs (+ hidden states for probes, + z's for swaps / mean ablation).
    p_te, z_te, h_te = run(model, tokenizer, test, k, training_args, bs, collect=True)
    p_tr, z_tr, h_tr = run(model, tokenizer, train, k, training_args, bs, collect=True)
    results["clean"] = score_swap(test, p_te, gold_te, None)
    print(f"clean done ({time.perf_counter() - t0:.0f}s)", flush=True)
    records = [{"idx": i, "id": r["id"], "level": int(r["level"]), "gold_answer": r["answer"],
                "predicted_answer": tokenizer.decode([p]).strip(),
                "correct": None if gold_te[i] is None else p == gold_te[i]} for i, (r, p) in enumerate(zip(test, p_te))]

    # 1. Necessity (only with latents).
    if k > 0:
        rng = random.Random(5980)
        by_level: dict[int, list[int]] = {}
        for i, r in enumerate(test):
            by_level.setdefault(int(r["level"]), []).append(i)
        donor = []
        for i, r in enumerate(test):
            cands = [j for j in by_level[int(r["level"])] if j != i and test[j]["answer"] != r["answer"]]
            donor.append(rng.choice(cands))
        donor_tok = [gold_te[d] for d in donor]
        z_mean = z_tr.mean(0)
        conds = {
            "swap_all": lambda i, j: z_te[donor[i], j],
            "swap_z1": lambda i, j: z_te[donor[i], 0] if j == 0 else None,
            "swap_rest": lambda i, j: z_te[donor[i], j] if j > 0 else None,
            "mean_all": lambda i, j: z_mean[j],
            "zero_all": lambda i, j: torch.zeros_like(z_mean[j]),
        }
        for name, fn in conds.items():
            p, _, _ = run(model, tokenizer, test, k, training_args, bs, override=fn)
            results[name] = score_swap(test, p, gold_te, donor_tok if name.startswith("swap") else None)
            for rec, pi_ in zip(records, p):
                rec[f"pred_{name}"] = tokenizer.decode([pi_]).strip()
            tot = sum(v["keeps_own"] * v["n"] for v in results[name].values()) / sum(v["n"] for v in results[name].values())
            fol = sum(v["follows_donor"] * v["n"] for v in results[name].values()) / sum(v["n"] for v in results[name].values())
            print(f"{name}: keeps_own {tot:.3f} follows_donor {fol:.3f} | per level "
                  + " ".join(f"L{L}:{v['keeps_own']:.2f}/{v['follows_donor']:.2f}" for L, v in results[name].items())
                  + f" ({time.perf_counter() - t0:.0f}s)", flush=True)
        for rec, d in zip(records, donor):
            rec["donor_id"] = test[d]["id"]
            rec["donor_answer"] = test[d]["answer"]

    # 2. Probes.
    vals_tr = [step_values(r) for r in train]
    vals_te = [step_values(r) for r in test]
    probes: dict = {}
    for target in [f"v{j}" for j in range(1, 7)] + ["answer"]:
        if target == "answer":
            itr, ite = list(range(len(train))), list(range(len(test)))
            ytr = torch.tensor([v[-1] for v in vals_tr])
            yte = torch.tensor([v[-1] for v in vals_te])
        else:
            j = int(target[1:])
            itr = [i for i, v in enumerate(vals_tr) if len(v) >= j]
            ite = [i for i, v in enumerate(vals_te) if len(v) >= j]
            ytr = torch.tensor([vals_tr[i][j - 1] for i in itr])
            yte = torch.tensor([vals_te[i][j - 1] for i in ite])
        if not (ytr.min() >= 0 and ytr.max() < 100 and yte.min() >= 0 and yte.max() < 100):
            raise SystemExit(f"{target}: values outside 0..99")
        maj = torch.bincount(ytr, minlength=100).argmax()
        probes[target] = {"n_train": len(itr), "n_test": len(ite), "majority_baseline": round((yte == maj).float().mean().item(), 4)}
        for pos in h_te:
            accs = {}
            for li, layer in enumerate(PROBE_LAYERS):
                te_acc, tr_acc = fit_probe(h_tr[pos][itr, li].float(), ytr, h_te[pos][ite, li].float(), yte, a.probe_steps)
                accs[layer] = {"test": round(te_acc, 4), "train": round(tr_acc, 4)}
            probes[target][pos] = accs
        best = {pos: max(v["test"] for v in probes[target][pos].values()) for pos in h_te}
        print(f"probe {target} (baseline {probes[target]['majority_baseline']}): best test acc by position "
              + json.dumps(best) + f" ({time.perf_counter() - t0:.0f}s)", flush=True)
    results["probes"] = probes
    results["probe_layers"] = list(PROBE_LAYERS)

    if a.smoke_n or a.no_log:
        print("not logging (smoke_n / no_log)")
        return
    acc = sum(r["correct"] for r in records if r["correct"] is not None) / sum(r["correct"] is not None for r in records)
    metrics = {"final_answer_accuracy": round(acc, 4), "unparseable_rate": 0.0, "compute_steps": k,
               "sec_per_example": (time.perf_counter() - t0) / len(test),
               "extra": {"accuracy_kind": "teacher-forced answer token at the distillation position",
                         "data_sha256": hashlib.sha256(Path(a.test_json).read_bytes()).hexdigest(),
                         "matched_flags": flags, **results}}
    if k > 0:
        n = len(records)
        metrics["intervention_accuracy"] = round(sum(
            v["follows_donor"] * v["n"] for v in results["swap_all"].values()) / sum(v["n"] for v in results["swap_all"].values()), 4)
    record = RunRecord(
        run_id=new_run_id("codi", a.slug), mechanism="codi", stage=a.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=a.checkpoint_label or model_args.ckpt_dir,
                        n_params=sum(p.numel() for p in model.parameters())),
        dataset=DatasetInfo(name="synthetic-depth2", split="test", n_examples=len(test), seed=None),
        metrics=metrics,
        hyperparams={"latent_count": k, "batch_size": bs, "train_per_level": a.train_per_level, "probe_steps": a.probe_steps,
                     "probe_layers": list(PROBE_LAYERS), "donor_seed": 5980, "device": device, "dtype": "float32"},
        seed=None, hardware=f"{a.hardware} / {device}",
        notes="RuleChain CODI latent swap/ablation and intermediate-value probes.",
    )
    manifest = record.save(predictions=records)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}")


if __name__ == "__main__":
    main()
