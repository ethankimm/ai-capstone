"""Depth2 loop-count test on Ouro (ByteDance, arXiv 2510.25741): one LoRA fine-tune, then score the SAME weights at
1..4 loops. Ouro repeats its whole 24-layer stack `total_ut_steps` times, so the loop count varies effective depth
(24, 48, 72, 96 layer passes) with parameters, tokenizer, pretraining and fine-tuning data held fixed -- the clean
version of the T5-efficient / truncated-Qwen layer-count comparisons. Does the number of rules chained without CoT
grow with loops?

Recipe = the scripts/lora_depth2.py 20k point (Qwen3-1.7B: L2 0.88): answer-only `Question: ...\\nAnswer: N`, LoRA
r16 on all attention/MLP projections, AdamW 2e-4, cosine, effective batch 32, 2 epochs, run at 4 loops. The loss is
the mean answer-token cross-entropy of the read-outs after EVERY loop (1..4), so each loop count is a trained exit
(Ouro's own pretraining also supervises every exit; its built-in `labels` loss mixes exits by the early-exit gate,
which we bypass). Eval: merge the adapter, pin exactly k loops and read loop k's output (same pinning as
scripts/depth_scaling_fewshot.py), greedy, HF batch size 1 (Ouro's remote code breaks on left-padded batches).
One RunRecord per loop count (mechanism recurrent_depth, compute_steps = loops).

Usage (pod):  python scripts/ouro_lora_depth2.py --train-json /root/depth2/train.json --test-json /root/depth2/test.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from latentreasoning.eval.metrics import extract_final_number, is_correct
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id
from latentreasoning.utils import set_seed
from lora_depth2 import balanced_order, batches, encode, load, prompt


def set_loops(model, k: int) -> None:
    model.config.total_ut_steps = k
    for m in (model, getattr(model, "model", None)):
        if m is not None and hasattr(m, "total_ut_steps"):
            m.total_ut_steps = k
    model.early_exit_step = k - 1
    model.early_exit_threshold = None


def exit_losses(causal_lm, ids, att, lab) -> list[torch.Tensor]:
    """Answer-token CE of each loop's read-out (final norm + lm_head on that loop's hidden states)."""
    _, hidden_list, _ = causal_lm.model(input_ids=ids, attention_mask=att, use_cache=False)
    tgt = lab[:, 1:]
    keep = tgt != -100
    return [F.cross_entropy(causal_lm.lm_head(h[:, :-1][keep]).float(), tgt[keep]) for h in hidden_list]


def train(model, causal_lm, tok, rows, a, device):
    enc = encode(tok, rows)
    steps = a.epochs * math.ceil(len(enc) / a.bs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * steps)), steps)
    rng, log, step, t0 = random.Random(0), [], 0, time.perf_counter()
    model.train()
    for ep in range(a.epochs):
        for ids, att, lab in batches(enc, a.bs, tok.pad_token_id, rng, device):  # right-padded
            per_exit = [0.0] * a.train_loops
            for s in range(0, ids.shape[0], a.micro_bs):
                sl = slice(s, s + a.micro_bs)
                losses = exit_losses(causal_lm, ids[sl], att[sl], lab[sl])
                frac = ids[sl].shape[0] / ids.shape[0]
                (sum(losses) / len(losses) * frac).backward()
                per_exit = [x + l.item() * frac for x, l in zip(per_exit, losses)]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % max(1, steps // 40) == 0 or step == steps:
                log.append({"step": step, "epoch": ep + 1, "loss": sum(per_exit) / len(per_exit),
                            "loss_per_loop": per_exit, "lr": sched.get_last_lr()[0],
                            "seconds": time.perf_counter() - t0})
                print(f"  step {step}/{steps} ep {ep + 1}/{a.epochs} loss per loop "
                      + " ".join(f"{x:.3f}" for x in per_exit), flush=True)
    model.eval()
    return log, steps, time.perf_counter() - t0


@torch.no_grad()
def evaluate(model, tok, items) -> list[str]:
    outs = []
    for q in items:
        enc = tok([prompt(q)], return_tensors="pt").to(model.device)
        g = model.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
        outs.append(tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ByteDance/Ouro-1.4B")
    p.add_argument("--train-json", required=True)
    p.add_argument("--test-json", required=True)
    p.add_argument("--train-n", type=int, default=20_000)
    p.add_argument("--levels", default="1,2,3,4")
    p.add_argument("--loops", default="1,2,3,4", help="loop counts to score")
    p.add_argument("--train-loops", type=int, default=4)
    p.add_argument("--per-level", type=int, default=200)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--micro-bs", type=int, default=8)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="no logging")
    p.add_argument("--stage", default="pilot")
    p.add_argument("--hardware", default="RunPod")
    a = p.parse_args()
    seed = set_seed(0)
    device = "cuda"
    levels = {int(x) for x in a.levels.split(",")}

    test_by = defaultdict(list)
    for r in sorted(load(a.test_json, levels), key=lambda r: r["id"]):
        test_by[r["level"]].append(r)
    items = [q for L in sorted(test_by) for q in test_by[L][:a.per_level]]
    rows_train = balanced_order(load(a.train_json, levels), a.data_seed)[:a.train_n]

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to(device)
    n_layers, n_params = base.config.num_hidden_layers, sum(q.numel() for q in base.parameters())
    set_loops(base, a.train_loops)
    model = get_peft_model(base, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    log, steps, train_secs = train(model, base, tok, rows_train, a, device)
    merged = model.merge_and_unload().eval()

    for k in [int(x) for x in a.loops.split(",")]:
        set_loops(merged, k)
        t0 = time.perf_counter()
        outs = evaluate(merged, tok, items)
        eval_secs = time.perf_counter() - t0
        hit, cnt, rows, unparse = defaultdict(int), defaultdict(int), [], 0
        for q, o in zip(items, outs):
            pred = extract_final_number(o.strip().split("\n")[0])
            ok = is_correct(pred, q["answer"])
            unparse += pred is None
            hit[q["level"]] += ok
            cnt[q["level"]] += 1
            rows.append({"id": q["id"], "level": q["level"], "gold": q["answer"], "raw": o, "pred": pred, "correct": ok})
        acc = {str(L): hit[L] / cnt[L] for L in sorted(cnt)}
        print(f"loops {k} ({k * n_layers} layer passes): per level "
              + " ".join(f"{L}:{v:.2f}" for L, v in acc.items()) + f" | eval {eval_secs:.0f}s", flush=True)
        if a.smoke:
            continue
        name = a.model.split("/")[-1].lower()
        record = RunRecord(
            run_id=new_run_id("recurrent_depth", f"depth2-lora-{name}-n{len(rows_train)}-l{k}"),
            mechanism="recurrent_depth", stage=a.stage,
            model=ModelInfo(backbone=a.model, checkpoint=f"{a.model} + LoRA trained at {a.train_loops} loops "
                            "(not saved)", n_params=n_params),
            dataset=DatasetInfo(name="synthetic-depth2", split="test", n_examples=len(items), seed=None),
            metrics={"final_answer_accuracy": sum(hit.values()) / len(items),
                     "unparseable_rate": unparse / len(items), "compute_steps": k,
                     "extra": {"task": "depth2", "design": "Ouro loop-count test: one LoRA fine-tune (loss on every "
                               "loop's exit), same merged weights scored at each loop count, eval on levels "
                               + a.levels,
                               "loops": k, "n_layers": n_layers, "layer_passes": k * n_layers,
                               "train_n": len(rows_train), "per_level_accuracy": acc,
                               "per_level_n": {str(L): cnt[L] for L in sorted(cnt)},
                               "final_train_loss_per_loop": [sum(e["loss_per_loop"][i] for e in log[-4:]) / len(log[-4:])
                                                             for i in range(a.train_loops)],
                               "optimizer_steps": steps, "train_seconds": train_secs, "eval_seconds": eval_secs}},
            hyperparams={"train_n": len(rows_train), "train_levels": sorted(levels), "train_loops": a.train_loops,
                         "loss": "mean over loops 1..train_loops of answer-token CE at that loop's exit",
                         "train_subset": "nested prefix of one level-interleaved shuffle", "data_seed": a.data_seed,
                         "lr": a.lr, "batch_size": a.bs, "micro_batch_size": a.micro_bs, "epochs": a.epochs,
                         "lora_rank": a.rank, "lora_alpha": 2 * a.rank,
                         "lora_targets": "q/k/v/o/gate/up/down proj", "schedule": "cosine, 3% warmup",
                         "per_level": a.per_level, "decoding": "greedy, 6 tokens, HF bs=1, adapter merged",
                         "prompt": "zero-shot 'Question: ...\\nAnswer:'"},
            seed=seed, hardware=a.hardware,
            notes="depth2 Ouro loop-count test: does no-CoT rule-chaining reach grow with loops at fixed weights?")
        path = record.save(predictions=rows)
        (path.parent / "train_log.json").write_text(json.dumps(log, indent=1) + "\n")
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
