"""Depth2 data-efficiency test: is the pretrained models' one-rule no-CoT ceiling (depth2 layer scan + prompt
check) a task-specification problem that a few hundred examples fix, or a missing skill that needs lots of data?

LoRA-fine-tune one pretrained base model (default Qwen/Qwen3-1.7B-Base) on depth2 answer-only data
(`Question: ...\\nAnswer: N`, loss on the answer tokens only) at several training-set sizes, fresh adapter per size,
then score zero-shot on the first --per-level test items of each level in --levels with the same prompt, greedy.
Size 0 = the untrained base model with the same zero-shot prompt (reference). Training subsets are nested (first n of
one fixed shuffle, balanced over --levels). Small sets get more epochs: passes = max(--min-passes, --min-epochs * n).
One RunRecord per size (mechanism baseline_no_cot).

Usage (pod):  python scripts/lora_depth2.py --train-json /root/depth2/train.json --test-json /root/depth2/test.json
Smoke (local CPU, no logging):  ... --model Qwen/Qwen3-0.6B-Base --sizes 0,16 --per-level 2 --min-passes 32 --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from latentreasoning.eval.metrics import extract_final_number, is_correct
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id
from latentreasoning.utils import set_seed


def prompt(q: dict) -> str:
    return f"Question: {q['question']}\nAnswer:"


def load(path: str, levels: set[int]) -> list[dict]:
    return [r for r in json.loads(Path(path).read_text()) if r["level"] in levels]


def balanced_order(rows: list[dict], seed: int) -> list[dict]:
    """One fixed shuffle that interleaves levels, so every prefix is (nearly) level-balanced."""
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


def encode(tok, rows: list[dict]) -> list[tuple[list[int], list[int]]]:
    out = []
    for r in rows:
        p = tok(prompt(r), add_special_tokens=False)["input_ids"]
        a = tok(f" {r['answer']}\n", add_special_tokens=False)["input_ids"]
        out.append((p + a, [-100] * len(p) + a))
    return out


def batches(enc, bs: int, pad: int, rng: random.Random, device):
    idx = list(range(len(enc)))
    rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [enc[j] for j in idx[i:i + bs]]
        L = max(len(x) for x, _ in chunk)
        ids = torch.tensor([x + [pad] * (L - len(x)) for x, _ in chunk], device=device)
        lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in chunk], device=device)
        att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in chunk], device=device)
        yield ids, att, lab


def train(model, tok, rows, a, device) -> list[dict]:
    enc = encode(tok, rows)
    epochs = max(a.min_epochs, math.ceil(a.min_passes / len(rows)))
    steps = epochs * math.ceil(len(enc) / a.bs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * steps)), steps)
    rng, log, step, t0 = random.Random(0), [], 0, time.perf_counter()
    model.train()
    for ep in range(epochs):
        for ids, att, lab in batches(enc, a.bs, tok.pad_token_id, rng, device):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % max(1, steps // 40) == 0 or step == steps:
                log.append({"step": step, "epoch": ep + 1, "loss": loss.item(), "lr": sched.get_last_lr()[0],
                            "seconds": time.perf_counter() - t0})
                print(f"  step {step}/{steps} ep {ep + 1}/{epochs} loss {loss.item():.4f}", flush=True)
    model.eval()
    return log, epochs, steps, time.perf_counter() - t0


@torch.no_grad()
def evaluate(model, tok, items, bs: int, device) -> list[str]:
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(items), bs):
        enc = tok([prompt(q) for q in items[i:i + bs]], return_tensors="pt", padding=True).to(device)
        g = model.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--train-json", required=True)
    p.add_argument("--test-json", required=True)
    p.add_argument("--sizes", default="0,200,1000,5000,20000")
    p.add_argument("--levels", default="1,2,3,4")
    p.add_argument("--per-level", type=int, default=200)
    p.add_argument("--min-passes", type=int, default=10_000, help="minimum training examples seen per size")
    p.add_argument("--min-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--eval-bs", type=int, default=64)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="no logging")
    p.add_argument("--stage", default="pilot")
    p.add_argument("--hardware", default="RunPod")
    a = p.parse_args()
    seed = set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    levels = {int(x) for x in a.levels.split(",")}

    test_by = defaultdict(list)
    for r in sorted(load(a.test_json, levels), key=lambda r: r["id"]):
        test_by[r["level"]].append(r)
    items = [q for L in sorted(test_by) for q in test_by[L][:a.per_level]]
    pool = balanced_order(load(a.train_json, levels), a.data_seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for n in [int(x) for x in a.sizes.split(",")]:
        set_seed(0)
        base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(device)
        n_layers, n_params = base.config.num_hidden_layers, sum(q.numel() for q in base.parameters())
        log, epochs, steps, train_secs = [], 0, 0, 0.0
        if n:
            model = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                                                    target_modules="all-linear", task_type="CAUSAL_LM"))
            print(f"n={n}: training", flush=True)
            log, epochs, steps, train_secs = train(model, tok, pool[:n], a, device)
        else:
            model = base.eval()
        t0 = time.perf_counter()
        outs = evaluate(model, tok, items, a.eval_bs, device)
        eval_secs = time.perf_counter() - t0
        k, cnt, rows, unparse = defaultdict(int), defaultdict(int), [], 0
        for q, o in zip(items, outs):
            pred = extract_final_number(o.strip().split("\n")[0])
            ok = is_correct(pred, q["answer"])
            unparse += pred is None
            k[q["level"]] += ok
            cnt[q["level"]] += 1
            rows.append({"id": q["id"], "level": q["level"], "gold": q["answer"], "raw": o, "pred": pred, "correct": ok})
        acc = {str(L): k[L] / cnt[L] for L in sorted(cnt)}
        print(f"n={n}: per level " + " ".join(f"{L}:{v:.2f}" for L, v in acc.items())
              + f" | train {train_secs:.0f}s ({epochs} ep, {steps} steps) eval {eval_secs:.0f}s", flush=True)
        if not a.smoke:
            name = a.model.split("/")[-1].lower()
            record = RunRecord(
                run_id=new_run_id("baseline_no_cot", f"depth2-lora-{name}-n{n}"), mechanism="baseline_no_cot",
                stage=a.stage,
                model=ModelInfo(backbone=a.model, checkpoint=f"{a.model} + LoRA (not saved)" if n else a.model,
                                n_params=n_params),
                dataset=DatasetInfo(name="synthetic-depth2", split="test", n_examples=len(items), seed=None),
                metrics={"final_answer_accuracy": sum(k.values()) / len(items),
                         "unparseable_rate": unparse / len(items), "compute_steps": 0,
                         "extra": {"task": "depth2", "design": "LoRA data-efficiency: answer-only fine-tune at one "
                                   "training-set size, zero-shot eval on levels " + a.levels,
                                   "train_n": n, "n_layers": n_layers, "per_level_accuracy": acc,
                                   "per_level_n": {str(L): cnt[L] for L in sorted(cnt)},
                                   "epochs": epochs, "optimizer_steps": steps, "train_seconds": train_secs,
                                   "eval_seconds": eval_secs}},
                hyperparams={"train_n": n, "train_levels": sorted(levels), "train_subset": "nested prefix of one "
                             "level-interleaved shuffle", "data_seed": a.data_seed, "lr": a.lr, "batch_size": a.bs,
                             "epochs": epochs, "min_passes": a.min_passes, "min_epochs": a.min_epochs,
                             "lora_rank": a.rank, "lora_alpha": 2 * a.rank, "lora_targets": "all-linear",
                             "schedule": "cosine, 3% warmup", "per_level": a.per_level, "decoding": "greedy, 6 tokens",
                             "prompt": "zero-shot 'Question: ...\\nAnswer:'"},
                seed=seed, hardware=a.hardware,
                notes="depth2 LoRA data-efficiency: does a small answer-only fine-tune unlock 2+ rule no-CoT chaining?")
            path = record.save(predictions=rows)
            if log:
                (path.parent / "train_log.json").write_text(json.dumps(log, indent=1) + "\n")
            print(f"saved {path}", flush=True)
        del model, base
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
