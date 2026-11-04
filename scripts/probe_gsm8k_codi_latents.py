#!/usr/bin/env python3
"""What do CODI's latents z_0..z_5 encode on GSM8K-Aug? Linear class probes per site, plus the logit lens.

Sites follow ablate_latents_rulechain_codi.py: z_s = the projected vector fed INTO latent pass s+1, z_0 = prj(hidden at
<bot>). The latent loop is that script's `run()` (pad-invariant, prompt = question.strip() + "\n" + <bot>), extended
here only to also read the LM head; `--check_n` items are compared against `run()` itself before anything is fitted.

Targets per item, parsed from the "<<a op b = c>>" rationale: v_1..v_4 (result of step j; only items with >= j steps)
and the final answer. Each target is a class over values seen in the probe-training data (values occurring at least
--min_count times; rarer training items are dropped from fitting). Test items whose value is not a class count as
wrong; `coverage` = share of test items whose value is a class (the probe's ceiling).

Probe: multinomial logistic regression on standardized z_s (statistics from the fitting set), weight decay picked per
(site, target) on gsm8k_aug/valid.json from --wds, then evaluated once on test. Fitting set = the first --train_n items
of gsm8k_aug/train.json (never test). Majority baseline = accuracy on test of always predicting the most frequent
training class.

Logit lens per site, on test: top-1 token of the LM head applied to (a) the final-layer hidden state whose projection
is z_s (`lens_hidden`; for z_0 the hidden at <bot>) and (b) z_s itself (`lens_z`; tied weights, so nearest input
embedding). Reports how often the top token equals v_j or the answer, and the most frequent top tokens.

Run inside the CODI venv from the CODI checkout, with the same model flags as ablate_latents_rulechain_codi.py plus
  --train_json .../gsm8k_aug/train.json --valid_json .../gsm8k_aug/valid.json --test_json .../gsm8k_aug/test.json
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers
from torch.nn import functional as F

sys.path.insert(0, os.getcwd())  # the CODI checkout: `src.model`
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.model import DataArguments, ModelArguments, TrainingArguments  # noqa: E402

from ablate_latents_rulechain_codi import run  # noqa: E402
from diagnose_codi_latents import pick_device  # noqa: E402  (also patches torch.load for CPU)
from eval_codi import PadInvariant, build_model, question_position_ids, tokenize_questions  # noqa: E402
from eval_synthetic_codi import check_flags  # noqa: E402
from synthetic_common import load_synthetic  # noqa: E402

from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402

N_STEPS = 4
_STEP = re.compile(r"<<[^<>=]*=([^<>]*)>>")


@dataclass
class ProbeArguments:
    train_json: str = field(default="")
    valid_json: str = field(default="")
    test_json: str = field(default="")
    train_n: int = field(default=20000)
    min_count: int = field(default=3)
    wds: str = field(default="1e-4,1e-3,1e-2,1e-1")
    epochs: int = field(default=300)
    check_n: int = field(default=128)
    device: str = field(default="cuda")
    slug: str = field(default="v2-gsm8k-aug-codi-latent-probes")
    stage: str = field(default="full_run")
    hardware: str = field(default="local")
    checkpoint_label: Optional[str] = field(default=None)
    smoke_n: int = field(default=0, metadata={"help": "first n items of every split, not logged"})


def norm_value(s: str) -> Optional[str]:
    try:
        v = float(s.strip().replace(",", "").lstrip("$"))
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else f"{v:g}"


def targets(row: dict) -> dict[str, Optional[str]]:
    vals = [norm_value(m) for m in _STEP.findall(" ".join(row["steps"]))]
    out = {f"v{j + 1}": (vals[j] if j < len(vals) else None) for j in range(N_STEPS)}
    out["answer"] = norm_value(str(row["answer"]))
    return out


@torch.no_grad()
def run_with_lens(model, tokenizer, rows, data_args, training_args):
    """`run()` without overrides, also returning top-1 LM-head tokens per site: (z [N, K, d], lens_hidden [N, K],
    lens_z [N, K])."""
    device = next(model.parameters()).device
    body = model.codi.get_base_model().transformer
    head = model.codi.get_base_model().lm_head
    V = model.codi.config.vocab_size - 1
    K = training_args.inf_latent_iterations
    top = lambda x: head(x)[..., :V].argmax(-1)  # noqa: E731
    questions = [r["question"].strip() + "\n" for r in rows]
    zs, lh, lz = [], [], []
    for batch in tokenize_questions(model, tokenizer, questions, data_args.batch_size, training_args, device):
        q_pos = question_position_ids(batch["attention_mask"]) if training_args.pad_invariant else None
        out = body(input_ids=batch["input_ids"], use_cache=True, attention_mask=batch["attention_mask"], position_ids=q_pos)
        past, h = out.past_key_values, out.last_hidden_state[:, -1:, :]
        pi = PadInvariant(batch["attention_mask"]) if training_args.pad_invariant else None
        used, th, tz = [], [], []
        for _ in range(K):
            z = model.prj(h)
            used.append(z[:, 0].float().cpu()); th.append(top(h)[:, 0].cpu()); tz.append(top(z)[:, 0].cpu())
            out = body(inputs_embeds=z, use_cache=True, past_key_values=past, **(pi.next(1) if pi else {}))
            past, h = out.past_key_values, out.last_hidden_state[:, -1:, :]
        zs.append(torch.stack(used, 1)); lh.append(torch.stack(th, 1)); lz.append(torch.stack(tz, 1))
    return torch.cat(zs), torch.cat(lh), torch.cat(lz)


def fit_logreg(X, y, n_cls, wd, epochs, device):
    W = torch.zeros(X.size(1), n_cls, device=device, requires_grad=True)
    b = torch.zeros(n_cls, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=1e-2, weight_decay=0)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(X @ W + b, y) + wd * W.pow(2).sum()
        loss.backward()
        opt.step()
    return W.detach(), b.detach()


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ProbeArguments))
    model_args, data_args, training_args, pa = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    flags = check_flags(model_args.ckpt_dir, training_args)
    device = pick_device(pa.device)
    model, tokenizer, load_result = build_model(model_args, training_args, device=device, dtype=torch.float32)
    if any(not k.startswith("codi.base_model.model.lm_head") for k in load_result.missing_keys):
        raise SystemExit(f"checkpoint is missing weights: {load_result.missing_keys[:5]}")
    model.eval()
    K = training_args.inf_latent_iterations
    cut = (lambda r: r[:pa.smoke_n]) if pa.smoke_n else (lambda r: r)
    fit_rows = cut(load_synthetic(pa.train_json).rows[:pa.train_n])
    valid_rows, test_rows = cut(load_synthetic(pa.valid_json).rows), cut(load_synthetic(pa.test_json).rows)

    t0 = time.perf_counter()
    z_chk, _ = run(model, tokenizer, test_rows[:pa.check_n], data_args, training_args)
    sets = {}
    for name, rows in (("fit", fit_rows), ("valid", valid_rows), ("test", test_rows)):
        sets[name] = run_with_lens(model, tokenizer, rows, data_args, training_args)
        print(f"{name}: {len(rows)} items ({time.perf_counter() - t0:.0f}s)", flush=True)
    max_dz = (sets["test"][0][:pa.check_n] - z_chk).abs().max().item()
    print(f"check vs run(): max |dz| = {max_dz:.2e}")
    if max_dz > 1e-3:
        raise SystemExit("run_with_lens disagrees with ablate_latents_rulechain_codi.run")

    tg = {n: [targets(r) for r in rows] for n, rows in (("fit", fit_rows), ("valid", valid_rows), ("test", test_rows))}
    names = [f"v{j + 1}" for j in range(N_STEPS)] + ["answer"]
    wds = [float(w) for w in pa.wds.split(",")]
    probes: dict = {}
    for t in names:
        counts = Counter(x[t] for x in tg["fit"] if x[t] is not None)
        classes = sorted(v for v, c in counts.items() if c >= pa.min_count)
        cid = {v: i for i, v in enumerate(classes)}
        ix = {n: [i for i, x in enumerate(tg[n]) if x[t] is not None] for n in tg}
        lab = {n: torch.tensor([cid.get(tg[n][i][t], -1) for i in ix[n]]) for n in tg}
        major = counts.most_common(1)[0][0]
        probes[t] = {"n_fit": len(ix["fit"]), "n_test": len(ix["test"]), "n_classes": len(classes),
                     "coverage_test": round((lab["test"] >= 0).float().mean().item(), 4),
                     "majority_class": major,
                     "majority_acc_test": round(sum(tg["test"][i][t] == major for i in ix["test"]) / len(ix["test"]), 4),
                     "sites": {}}
        for s in range(K):
            X = {n: sets[n][0][ix[n], s] for n in tg}
            keep = lab["fit"] >= 0
            mu, sd = X["fit"][keep].mean(0), X["fit"][keep].std(0) + 1e-6
            Xd = {n: ((X[n] - mu) / sd).to(device) for n in tg}
            best = None
            for wd in wds:
                W, b = fit_logreg(Xd["fit"][keep.to(device)], lab["fit"][keep].to(device), len(classes), wd, pa.epochs, device)
                acc = {n: ((Xd[n] @ W + b).argmax(-1).cpu() == lab[n]).float().mean().item() for n in ("fit", "valid", "test")}
                if best is None or acc["valid"] > best[1]["valid"]:
                    best = (wd, acc)
            probes[t]["sites"][s] = {"wd": best[0], **{f"acc_{n}": round(v, 4) for n, v in best[1].items()}}
        print(t, {s: v["acc_test"] for s, v in probes[t]["sites"].items()}, "majority", probes[t]["majority_acc_test"],
              "coverage", probes[t]["coverage_test"], flush=True)

    lens = {}
    for kind, li in (("lens_hidden", 1), ("lens_z", 2)):
        toks = sets["test"][li]
        lens[kind] = {}
        for s in range(K):
            dec = [tokenizer.decode([int(x)]).strip() for x in toks[:, s]]
            hit = {t: round(sum(dec[i] == tg["test"][i][t] for i in range(len(dec)) if tg["test"][i][t] is not None)
                            / max(1, sum(x[t] is not None for x in tg["test"])), 4) for t in names}
            any_v = round(sum(dec[i] in {tg["test"][i][f"v{j + 1}"] for j in range(N_STEPS)} - {None}
                              for i in range(len(dec))) / len(dec), 4)
            lens[kind][s] = {"hit": hit, "hit_any_step": any_v, "top_tokens": Counter(dec).most_common(5)}
        print(kind, {s: v["hit"] for s, v in lens[kind].items()}, flush=True)
    if pa.smoke_n:
        print("not logging (smoke_n)")
        return

    elapsed = time.perf_counter() - t0
    records = []
    for i, r in enumerate(test_rows):
        records.append({"idx": i, "id": r["id"], "level": int(r["level"]), "gold_answer": r["answer"],
                        "targets": tg["test"][i],
                        "lens_hidden": [tokenizer.decode([int(x)]) for x in sets["test"][1][i]],
                        "lens_z": [tokenizer.decode([int(x)]) for x in sets["test"][2][i]]})
    metrics = {"final_answer_accuracy": None, "unparseable_rate": None, "compute_steps": K,
               "sec_per_example": elapsed / (len(fit_rows) + len(valid_rows) + len(test_rows)),
               "extra": {"probe_kind": "multinomial logistic regression on standardized z_s, wd picked on valid",
                         "probes": probes, "lens": lens, "check_max_abs_dz_vs_run": max_dz,
                         "fit_json": pa.train_json, "fit_n": len(fit_rows), "valid_json": pa.valid_json,
                         "min_count": pa.min_count, "matched_flags": flags,
                         "data_sha256": hashlib.sha256(Path(pa.test_json).read_bytes()).hexdigest()}}
    record = RunRecord(
        run_id=new_run_id("codi", pa.slug), mechanism="codi", stage=pa.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=pa.checkpoint_label or model_args.ckpt_dir,
                        n_params=sum(p.numel() for p in model.parameters())),
        dataset=DatasetInfo(name="synthetic-gsm8k_aug", split="test", n_examples=len(test_rows), seed=None),
        metrics=metrics,
        hyperparams={"train_n": pa.train_n, "min_count": pa.min_count, "wds": wds, "epochs": pa.epochs,
                     "batch_size": data_args.batch_size, "dtype": "float32", "device": device},
        seed=None, hardware=f"{pa.hardware} / {device}",
        notes="GSM8K CODI: linear class probes per latent site for step results v_1..v_4 and the answer, plus logit lens.")
    manifest = record.save(predictions=records)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
