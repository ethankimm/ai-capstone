"""Depth2 layer-count test on T5-efficient (Tay et al. 2021, "Scale Efficiently"): google/t5-efficient-base-nl{k}
share width (d_model 768, d_ff 3072, 12 heads), tokenizer, pretraining data and recipe (C4 span corruption, 524k
steps), and differ only in the number of layers k (encoder and decoder each). Each was pretrained at its own depth,
so unlike truncating Qwen nothing is cut away. Does the number of rules a model can chain without CoT grow with k?

Per depth: full fine-tune (no LoRA; the 2021 pretraining is weak) on the same nested depth2 answer-only subset
(encoder `Question: ...`, decoder target the bare answer), same recipe for every k, then greedy eval on the first
--per-level test items of each level. Read 2+ rule accuracy only at depths whose 1-rule accuracy clears the gate
(~0.9): below it the model has not learned the task at all and says nothing about composition.
One RunRecord per depth (mechanism baseline_no_cot); depths with a saved run are skipped (resumable).

--digits writes every number digit by digit ("65" -> "6 5") in question and target, since the T5 vocabulary has
atomic, inconsistently split 2-digit tokens; --optim adafactor is T5's own fine-tuning optimizer (constant-scale,
lr ~1e-3). Variant runs get --tag in their slug so they do not collide with (or skip on) the default recipe.

Usage (pod):  python scripts/t5_depth2.py --train-json /root/depth2/train.json --test-json /root/depth2/test.json
Smoke (local CPU, no logging):  ... --depths 2 --train-n 64 --epochs 1 --per-level 2 --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_cosine_schedule_with_warmup
from transformers.optimization import Adafactor

from latentreasoning.eval.metrics import extract_final_number, is_correct
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id
from latentreasoning.utils import set_seed

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load(path: str, levels: set[int]) -> list[dict]:
    return [r for r in json.loads(Path(path).read_text()) if r["level"] in levels]


def balanced_order(rows: list[dict], seed: int) -> list[dict]:
    """One fixed shuffle that interleaves levels (as in lora_depth2.py), so every prefix is level-balanced."""
    by = defaultdict(list)
    for r in rows:
        by[r["level"]].append(r)
    rng = random.Random(seed)
    for v in by.values():
        rng.shuffle(v)
    out, i = [], 0
    while any(i < len(v) for v in by.values()):
        out += [v[i] for v in by.values() if i < len(v)]
        i += 1
    return out


DIGITS = False  # set from --digits


def spell(text: str) -> str:
    return re.sub(r"\d+", lambda m: " ".join(m.group()), text) if DIGITS else text


def unspell(text: str) -> str:
    return re.sub(r"(?<=\d) (?=\d)", "", text) if DIGITS else text


def prompt(q: dict) -> str:
    return f"Question: {spell(q['question'])}"


def batches(src, tgt, bs: int, pad: int, rng: random.Random, device):
    idx = list(range(len(src)))
    rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = idx[i:i + bs]
        L, T = max(len(src[j]) for j in chunk), max(len(tgt[j]) for j in chunk)
        ids = torch.tensor([src[j] + [pad] * (L - len(src[j])) for j in chunk], device=device)
        att = torch.tensor([[1] * len(src[j]) + [0] * (L - len(src[j])) for j in chunk], device=device)
        lab = torch.tensor([tgt[j] + [-100] * (T - len(tgt[j])) for j in chunk], device=device)
        yield ids, att, lab


def train(model, tok, rows, a, device):
    src = [tok(prompt(r))["input_ids"] for r in rows]
    tgt = [tok(spell(r["answer"]))["input_ids"] for r in rows]
    steps = a.epochs * math.ceil(len(rows) / a.bs)
    if a.optim == "adafactor":
        opt = Adafactor(model.parameters(), lr=a.lr, scale_parameter=False, relative_step=False, warmup_init=False)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * steps)), steps)
    rng, log, step, t0 = random.Random(0), [], 0, time.perf_counter()
    model.train()
    for ep in range(a.epochs):
        for ids, att, lab in batches(src, tgt, a.bs, tok.pad_token_id, rng, device):
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % max(1, steps // 40) == 0 or step == steps:
                log.append({"step": step, "epoch": ep + 1, "loss": loss.item(), "lr": sched.get_last_lr()[0],
                            "seconds": time.perf_counter() - t0})
                print(f"  step {step}/{steps} ep {ep + 1}/{a.epochs} loss {loss.item():.4f}", flush=True)
    model.eval()
    return log, steps, time.perf_counter() - t0


@torch.no_grad()
def evaluate(model, tok, items, bs: int, device) -> list[str]:
    outs = []
    for i in range(0, len(items), bs):
        enc = tok([prompt(q) for q in items[i:i + bs]], return_tensors="pt", padding=True).to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            g = model.generate(**enc, max_new_tokens=6, do_sample=False)
        outs += tok.batch_decode(g, skip_special_tokens=True)
    return outs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--family", default="google/t5-efficient-base-nl{k}")
    p.add_argument("--depths", default="2,4,8,12,16,24,32,48")
    p.add_argument("--train-json", required=True)
    p.add_argument("--test-json", required=True)
    p.add_argument("--levels", default="1,2,3,4,5,6")
    p.add_argument("--train-n", type=int, default=30_000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per-level", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--eval-bs", type=int, default=128)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--optim", choices=["adamw", "adafactor"], default="adamw")
    p.add_argument("--digits", action="store_true", help="spell numbers digit by digit")
    p.add_argument("--tag", default="", help="slug suffix for variant recipes")
    p.add_argument("--smoke", action="store_true", help="no logging")
    p.add_argument("--stage", default="pilot")
    p.add_argument("--hardware", default="RunPod")
    a = p.parse_args()
    global DIGITS
    DIGITS = a.digits
    device = "cuda" if torch.cuda.is_available() else "cpu"
    levels = {int(x) for x in a.levels.split(",")}

    test_by = defaultdict(list)
    for r in sorted(load(a.test_json, levels), key=lambda r: r["id"]):
        test_by[r["level"]].append(r)
    items = [q for L in sorted(test_by) for q in test_by[L][:a.per_level]]
    rows_train = balanced_order(load(a.train_json, levels), a.data_seed)[:a.train_n]

    for k in [int(x) for x in a.depths.split(",")]:
        model_id = a.family.format(k=k) if k != 12 else a.family.split("-nl")[0]  # nl12 is the plain t5-efficient-base
        slug = f"depth2-t5eff-{model_id.split('/')[-1].split('-')[2]}-nl{k}" + (f"-{a.tag}" if a.tag else "")
        if not a.smoke and any(RESULTS.glob(f"*_baseline_no_cot_{slug}/manifest.json")):
            print(f"skip {slug}: already saved", flush=True)
            continue
        seed = set_seed(0)
        tok = AutoTokenizer.from_pretrained(model_id)
        model = T5ForConditionalGeneration.from_pretrained(model_id).to(device)
        n_params = sum(q.numel() for q in model.parameters())
        print(f"loaded {model_id}: {model.config.num_layers}+{model.config.num_decoder_layers} layers, "
              f"{n_params / 1e6:.0f}M params", flush=True)
        log, steps, train_secs = train(model, tok, rows_train, a, device)
        t0 = time.perf_counter()
        outs = evaluate(model, tok, items, a.eval_bs, device)
        eval_secs = time.perf_counter() - t0
        hit, cnt, rows, unparse = defaultdict(int), defaultdict(int), [], 0
        for q, o in zip(items, outs):
            pred = extract_final_number(unspell(o.strip()))
            ok = is_correct(pred, q["answer"])
            unparse += pred is None
            hit[q["level"]] += ok
            cnt[q["level"]] += 1
            rows.append({"id": q["id"], "level": q["level"], "gold": q["answer"], "raw": o, "pred": pred, "correct": ok})
        acc = {str(L): hit[L] / cnt[L] for L in sorted(cnt)}
        final_loss = sum(e["loss"] for e in log[-4:]) / len(log[-4:])
        print(f"nl{k}: per level " + " ".join(f"{L}:{v:.2f}" for L, v in acc.items())
              + f" | final loss {final_loss:.3f} | train {train_secs:.0f}s ({steps} steps) eval {eval_secs:.0f}s",
              flush=True)
        if not a.smoke:
            record = RunRecord(
                run_id=new_run_id("baseline_no_cot", slug), mechanism="baseline_no_cot", stage=a.stage,
                model=ModelInfo(backbone=model_id, checkpoint=f"{model_id} + full fine-tune (not saved)",
                                n_params=n_params),
                dataset=DatasetInfo(name="synthetic-depth2", split="test", n_examples=len(items), seed=None),
                metrics={"final_answer_accuracy": sum(hit.values()) / len(items),
                         "unparseable_rate": unparse / len(items), "compute_steps": 0,
                         "extra": {"task": "depth2", "design": "T5-efficient layer-count scan: same width and "
                                   "pretraining, k encoder + k decoder layers, full answer-only fine-tune, eval on "
                                   "levels " + a.levels,
                                   "n_layers": k, "train_n": len(rows_train), "per_level_accuracy": acc,
                                   "per_level_n": {str(L): cnt[L] for L in sorted(cnt)},
                                   "final_train_loss": final_loss, "optimizer_steps": steps,
                                   "train_seconds": train_secs, "eval_seconds": eval_secs}},
                hyperparams={"train_n": len(rows_train), "train_levels": sorted(levels),
                             "train_subset": "nested prefix of one level-interleaved shuffle",
                             "data_seed": a.data_seed, "lr": a.lr, "batch_size": a.bs, "epochs": a.epochs,
                             "optimizer": ("Adafactor (no param scaling, no relative step)" if a.optim == "adafactor"
                                           else "AdamW, wd 0") + ", full fine-tune, bf16 autocast",
                             "digit_split_numbers": a.digits,
                             "schedule": "cosine, 3% warmup", "per_level": a.per_level,
                             "decoding": "greedy, 6 tokens", "prompt": "encoder 'Question: ...', target = answer"},
                seed=seed, hardware=a.hardware,
                notes="depth2 layer-count scan on T5-efficient-base: does no-CoT rule-chaining reach grow with depth?")
            path = record.save(predictions=rows)
            (path.parent / "train_log.json").write_text(json.dumps(log, indent=1) + "\n")
            print(f"saved {path}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
