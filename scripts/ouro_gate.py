"""Ouro step 0 (Q4 extension to a looped model): can a released Ouro base model answer the
xmech eval questions WITHOUT chain of thought, and at which loop count?

Ouro (ByteDance, arXiv 2510.25741) repeats its whole layer stack `total_ut_steps` times per
token (trained at 4). No GSM8K fine-tune exists, so this is few-shot prompted:
`Question: ...\\nAnswer: <number>` shots drawn from the xmech `fit` split, greedy decoding, a
short answer budget so the model cannot write a rationale. Scored on the xmech `eval` split
(GSM8K test, 1319) -- the same questions the CODI/Coconut Q4 dumps cover -- at each loop
count in --loops. Also reports how many eval questions are correct in Ouro AND CODI AND
Coconut, since Q4's ladder pairs need base-correct recipients/donors in both mechanisms.

A small CoT reference (--cot-n, at the trained loop count) separates "cannot do the task
format" from "cannot do it without CoT".

Gate (fixed before running): continue to step 1 (localization) iff at some loop count
<= 4 no-CoT accuracy >= 0.15 AND the Ouro & CODI & Coconut correct set has >= 100 eval
questions. Otherwise log the null and stop.

Usage (pod, Ouro venv):  python scripts/ouro_gate.py --model ByteDance/Ouro-1.4B \\
    --questions xmech_questions.jsonl --codi codi_correct.json --coconut coconut_correct.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latentreasoning.eval.metrics import extract_final_number, is_correct
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id
from latentreasoning.utils import set_seed


def read_questions(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def build_prefix(fit: list[dict], n_shots: int, seed: int, cot: bool) -> str:
    shots = random.Random(seed).sample(fit, n_shots)
    parts = []
    for q in shots:
        if cot:
            steps = q["rationale"].replace("<<", "").replace(">>", "").split()
            parts.append(f"Question: {q['question']}\nAnswer: {' '.join(steps)}. The answer is {q['answer']}.")
        else:
            parts.append(f"Question: {q['question']}\nAnswer: {q['answer']}")
    return "\n\n".join(parts) + "\n\n"


def parse(text: str, cot: bool) -> str | None:
    if cot:
        head = text.split("Question:")[0]
        if "The answer is" in head:
            head = head.split("The answer is")[1]
        return extract_final_number(head)
    # no-CoT: the first line only -- the model should emit just the number
    return extract_final_number(text.split("\n")[0])


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new: int, bs: int, stop: list[str]) -> list[str]:
    """bs must stay 1: Ouro's remote code mishandles left padding (padded rows decode to
    garbage, only the longest row in a batch is right; checked on the pod 2026-10-14)."""
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id, stop_strings=stop, tokenizer=tok)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def set_loops(model, k: int) -> None:
    """Run exactly k loops and read logits from the last one. Ouro's default exit rule
    (early_exit_threshold=1.0 on the cumulative exit-gate pdf) can exit early when the
    bf16 cumsum rounds to 1.0, so pin the exit step instead."""
    model.config.total_ut_steps = k
    for m in (model, getattr(model, "model", None)):
        if m is not None and hasattr(m, "total_ut_steps"):
            m.total_ut_steps = k
    model.early_exit_step = k - 1
    model.early_exit_threshold = None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ByteDance/Ouro-1.4B")
    p.add_argument("--questions", required=True)
    p.add_argument("--codi", help="json {key: bool} CODI eval correctness (from codi_latents.pt)")
    p.add_argument("--coconut", help="json {key: bool} Coconut eval correctness")
    p.add_argument("--loops", default="1,2,3,4,5,6")
    p.add_argument("--n-shots", type=int, default=8)
    p.add_argument("--shot-seed", type=int, default=0)
    p.add_argument("--max-new", type=int, default=8)
    p.add_argument("--cot-n", type=int, default=200)
    p.add_argument("--cot-max-new", type=int, default=160)
    p.add_argument("--bs", type=int, default=1, help="keep 1, see generate()")
    p.add_argument("--attn", default="eager", help="sdpa crashes on padded batches at loops > 1")
    p.add_argument("--smoke-n", type=int, default=0)
    p.add_argument("--stage", default="pilot")
    p.add_argument("--hardware", default="RunPod")
    a = p.parse_args()
    seed = set_seed(0)

    qs = read_questions(a.questions)
    fit = [q for q in qs if q["split"] == "fit"]
    ev = [q for q in qs if q["split"] == "eval"]
    if a.smoke_n:
        ev = ev[:a.smoke_n]
    loops = [int(x) for x in a.loops.split(",")]

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation=a.attn).eval()
    trained_loops = int(getattr(model.config, "total_ut_steps", 4))
    print(f"loaded {a.model}; config total_ut_steps={trained_loops}; "
          f"early_exit_threshold={getattr(model.config, 'early_exit_threshold', None)}")

    prefix = build_prefix(fit, a.n_shots, a.shot_seed, cot=False)
    prompts = [prefix + f"Question: {q['question']}\nAnswer:" for q in ev]
    rows = {q["key"]: {"key": q["key"], "gold": q["answer"]} for q in ev}
    acc, timing = {}, {}
    for k in loops:
        set_loops(model, k)
        t0 = time.perf_counter()
        outs = generate(model, tok, prompts, a.max_new, a.bs, ["\n"])
        timing[k] = time.perf_counter() - t0
        n_ok = 0
        for q, o in zip(ev, outs):
            pred = parse(o, cot=False)
            ok = is_correct(pred, q["answer"])
            n_ok += ok
            rows[q["key"]].update({f"raw_l{k}": o, f"pred_l{k}": pred, f"correct_l{k}": ok})
        acc[k] = n_ok / len(ev)
        print(f"no-CoT loops={k}: acc {acc[k]:.3f} ({n_ok}/{len(ev)}) in {timing[k]:.0f}s")

    cot_acc = None
    if a.cot_n:
        set_loops(model, trained_loops)
        cot_prefix = build_prefix(fit, a.n_shots, a.shot_seed, cot=True)
        sub = ev[:a.cot_n]
        outs = generate(model, tok, [cot_prefix + f"Question: {q['question']}\nAnswer:" for q in sub],
                        a.cot_max_new, a.bs, ["\n\nQuestion"])
        n_ok = 0
        for q, o in zip(sub, outs):
            pred = parse(o, cot=True)
            ok = is_correct(pred, q["answer"])
            n_ok += ok
            rows[q["key"]].update({"raw_cot": o, "pred_cot": pred, "correct_cot": ok})
        cot_acc = n_ok / len(sub)
        print(f"CoT reference loops={trained_loops}, n={len(sub)}: acc {cot_acc:.3f}")

    overlap = {}
    if a.codi and a.coconut:
        codi, coco = json.loads(Path(a.codi).read_text()), json.loads(Path(a.coconut).read_text())
        both = {q["key"] for q in ev if codi.get(q["key"]) and coco.get(q["key"])}
        for k in loops:
            ok = {key for key, r in rows.items() if r[f"correct_l{k}"]}
            overlap[k] = {"ouro_and_codi": len(ok & {x for x in codi if codi[x]}),
                          "ouro_and_coconut": len(ok & {x for x in coco if coco[x]}),
                          "ouro_and_both": len(ok & both), "codi_and_coconut": len(both)}
            print(f"loops={k}: correct overlap {overlap[k]}")

    passed = any(acc[k] >= 0.15 and overlap.get(k, {}).get("ouro_and_both", 0) >= 100
                 for k in loops if k <= 4)
    print(f"GATE {'PASS' if passed else 'FAIL'}")
    if a.smoke_n:
        print("smoke run -- not logging")
        return

    best = max(acc, key=acc.get)
    slug = "gate-" + a.model.split("/")[-1].lower()
    record = RunRecord(
        run_id=new_run_id("recurrent_depth", slug), mechanism="recurrent_depth", stage=a.stage,
        model=ModelInfo(backbone=a.model, checkpoint=a.model,
                        n_params=sum(p.numel() for p in model.parameters())),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(ev), seed=None),
        metrics={
            "final_answer_accuracy": acc[trained_loops] if trained_loops in acc else acc[best],
            "compute_steps": trained_loops,
            "extra": {
                "design": "Ouro step 0 gate: few-shot no-CoT accuracy on the xmech eval split by loop count, "
                          "CoT reference at the trained loop count, overlap with CODI/Coconut base-correct sets",
                "acc_by_loops": {str(k): v for k, v in acc.items()},
                "seconds_by_loops": {str(k): v for k, v in timing.items()},
                "cot_reference_acc": cot_acc, "cot_n": a.cot_n if a.cot_n else 0,
                "correct_overlap_by_loops": {str(k): v for k, v in overlap.items()},
                "gate": {"rule": "some loops<=4 with no-CoT acc>=0.15 and ouro&codi&coconut correct>=100",
                         "passed": passed},
            },
        },
        hyperparams={"n_shots": a.n_shots, "shot_seed": a.shot_seed, "max_new_tokens": a.max_new,
                     "loops": loops, "cot_max_new_tokens": a.cot_max_new, "decoding": "greedy", "attn_implementation": a.attn,
                     "prompt": "Question: ...\\nAnswer: <number> (shots from xmech fit split)"},
        seed=seed,
        hardware=f"{a.hardware} / {torch.cuda.get_device_name(0)}",
        notes="Ouro step 0: does the released looped model answer GSM8K without CoT, and at which loop count?",
    )
    manifest = record.save(predictions=list(rows.values()))
    (manifest.parent / "eval_command.txt").write_text(" ".join(sys.argv) + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}")


if __name__ == "__main__":
    main()
