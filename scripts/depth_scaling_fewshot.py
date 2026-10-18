"""Depth-vs-layers scan: how many depth2 rules can a *pretrained, not fine-tuned* model chain
without CoT, and does that track its layer count?

Few-shot prompted (`Question: ...\\nAnswer: N`, shots drawn from depth2 train at the same level
as the query, fixed per level), greedy, answer budget of a few tokens so no rationale fits.
Scored per level on the first --per-level test items of each level. A small CoT reference
(--cot-per-level, shots show the applied arithmetic) separates "cannot execute a rule" from
"cannot chain rules without CoT": a no-CoT curve only reads as a depth limit where CoT is high.

Backends: vLLM for ordinary models; HF bs=1 for Ouro (--loops), whose remote code breaks on
left-padded batches (see scripts/ouro_gate.py). For Ouro each loop count is scored on the same
weights, so loops vary effective depth at fixed parameters.

Shortcut floors on this test set (computed 2026-10-17): majority answer 0.03-0.04 per level;
"apply only the first / last rule" <= 0.05 for levels >= 2. Level 1 is solvable by one rule.

Usage (pod):  python scripts/depth_scaling_fewshot.py --model Qwen/Qwen3-4B-Base \\
    --test-json /workspace/synthetic_data_v2/depth2/test.json \\
    --train-json /workspace/synthetic_data_v2/depth2/train.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

from latentreasoning.eval.metrics import extract_final_number, is_correct
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id
from latentreasoning.utils import set_seed


def by_level(rows: list[dict]) -> dict[int, list[dict]]:
    out = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["id"]):
        out[r["level"]].append(r)
    return dict(sorted(out.items()))


def cot_text(steps: list[str]) -> str:
    return ". ".join(s.strip("<>") for s in steps)


def build_prefix(shots: list[dict], cot: bool) -> str:
    parts = []
    for s in shots:
        ans = f"{cot_text(s['steps'])}. The answer is {s['answer']}." if cot else s["answer"]
        parts.append(f"Question: {s['question']}\nAnswer: {ans}")
    return "\n\n".join(parts) + "\n\n"


def parse(text: str, cot: bool) -> str | None:
    if cot:
        head = text.split("Question:")[0]
        if "The answer is" in head:
            head = head.split("The answer is")[1]
        return extract_final_number(head)
    return extract_final_number(text.split("\n")[0])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def crossing(acc: dict[int, float], thr: float) -> float | None:
    """First level where the curve falls below thr, linearly interpolated; None if it never
    reaches thr at level 1, max level + 1 if it never falls below."""
    levels = sorted(acc)
    if acc[levels[0]] < thr:
        return None
    for a, b in zip(levels, levels[1:]):
        if acc[b] < thr:
            return a + (acc[a] - thr) / (acc[a] - acc[b]) * (b - a)
    return levels[-1] + 1.0


def reach(acc: dict[int, int], n: dict[int, int], floor: float) -> int:
    """Largest L such that every level <= L has Wilson lower bound above `floor`."""
    r = 0
    for L in sorted(acc):
        if wilson(acc[L], n[L])[0] <= floor:
            break
        r = L
    return r


class VLLMBackend:
    def __init__(self, model: str, max_len: int, tp: int):
        from vllm import LLM
        self.llm = LLM(model=model, dtype="bfloat16", max_model_len=max_len, tensor_parallel_size=tp,
                       trust_remote_code=True, seed=0, enable_prefix_caching=True)
        cfg = self.llm.llm_engine.model_config.hf_config
        self.n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
        self.n_params = None

    def generate(self, prompts: list[str], max_new: int, stop: list[str]) -> list[str]:
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=max_new, stop=stop)
        return [o.outputs[0].text for o in self.llm.generate(prompts, sp)]


class OuroBackend:
    def __init__(self, model: str, attn: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                          device_map="cuda", attn_implementation=attn).eval()
        self.n_layers = self.model.config.num_hidden_layers
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def set_loops(self, k: int) -> None:
        # same pinning as scripts/ouro_gate.py: run exactly k loops, read the last one
        self.model.config.total_ut_steps = k
        for m in (self.model, getattr(self.model, "model", None)):
            if m is not None and hasattr(m, "total_ut_steps"):
                m.total_ut_steps = k
        self.model.early_exit_step = k - 1
        self.model.early_exit_threshold = None

    def generate(self, prompts: list[str], max_new: int, stop: list[str]) -> list[str]:
        outs = []
        with self.torch.no_grad():
            for p in prompts:
                enc = self.tok([p], return_tensors="pt").to(self.model.device)
                g = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                        pad_token_id=self.tok.pad_token_id, stop_strings=stop, tokenizer=self.tok)
                outs.append(self.tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        return outs


def score(backend, items, prefixes, cot: bool, max_new: int, stop: list[str], tag: str, rows: dict):
    prompts = [prefixes[q["level"]] + f"Question: {q['question']}\nAnswer:" for q in items]
    t0 = time.perf_counter()
    outs = backend.generate(prompts, max_new, stop)
    secs = time.perf_counter() - t0
    k, n = defaultdict(int), defaultdict(int)
    for q, o in zip(items, outs):
        pred = parse(o, cot)
        ok = is_correct(pred, q["answer"])
        k[q["level"]] += ok
        n[q["level"]] += 1
        rows[q["id"]].update({f"raw_{tag}": o, f"pred_{tag}": pred, f"correct_{tag}": ok})
    return dict(k), dict(n), secs


def summarize(k: dict, n: dict, floor: float) -> dict:
    acc = {L: k[L] / n[L] for L in sorted(n)}
    return {"per_level_accuracy": {str(L): a for L, a in acc.items()},
            "per_level_ci95": {str(L): wilson(k[L], n[L]) for L in sorted(n)},
            "overall": sum(k.values()) / sum(n.values()),
            "crossing_50": crossing(acc, 0.5), "crossing_25": crossing(acc, 0.25),
            "reach_above_floor": reach(k, n, floor)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--test-json", required=True)
    p.add_argument("--train-json", required=True)
    p.add_argument("--per-level", type=int, default=200)
    p.add_argument("--levels", default="", help="comma list of levels to score (default all); Ouro runs use 1-4")
    p.add_argument("--cot-per-level", type=int, default=50)
    p.add_argument("--n-shots", type=int, default=6)
    p.add_argument("--shot-seed", type=int, default=0)
    p.add_argument("--max-new", type=int, default=6)
    p.add_argument("--cot-max-new", type=int, default=200)
    p.add_argument("--floor", type=float, default=0.08, help="reach threshold: ~2x the shortcut floors")
    p.add_argument("--loops", default="", help="Ouro only: comma list of loop counts; uses the HF bs=1 backend")
    p.add_argument("--cot-loops", type=int, default=4)
    p.add_argument("--attn", default="eager")
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--smoke-n", type=int, default=0, help="items per level; not logged")
    p.add_argument("--stage", default="pilot")
    p.add_argument("--hardware", default="RunPod")
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()
    seed = set_seed(0)

    test = by_level(json.loads(Path(a.test_json).read_text()))
    if a.levels:
        keep = {int(x) for x in a.levels.split(",")}
        test = {L: v for L, v in test.items() if L in keep}
    train = by_level(json.loads(Path(a.train_json).read_text()))
    per = a.smoke_n or a.per_level
    items = [q for L in test for q in test[L][:per]]
    cot_items = [q for L in test for q in test[L][:min(per, a.cot_per_level)]] if a.cot_per_level else []
    rng = random.Random(a.shot_seed)
    shots = {L: rng.sample(train[L], a.n_shots) for L in test}
    prefix = {L: build_prefix(s, cot=False) for L, s in shots.items()}
    cot_prefix = {L: build_prefix(s, cot=True) for L, s in shots.items()}
    rows = {q["id"]: {"id": q["id"], "level": q["level"], "gold": q["answer"]} for q in items}

    loops = [int(x) for x in a.loops.split(",") if x]
    backend = OuroBackend(a.model, a.attn) if loops else VLLMBackend(a.model, a.max_len, a.tp)
    print(f"loaded {a.model}: {backend.n_layers} layers")

    res, secs = {}, {}
    for tag, k_loop in ([(f"l{k}", k) for k in loops] or [("nocot", None)]):
        if k_loop:
            backend.set_loops(k_loop)
        k, n, secs[tag] = score(backend, items, prefix, False, a.max_new, ["\n"], tag, rows)
        res[tag] = summarize(k, n, a.floor)
        pl = " ".join(f"{L}:{v:.2f}" for L, v in res[tag]["per_level_accuracy"].items())
        print(f"{tag} no-CoT overall {res[tag]['overall']:.3f} reach {res[tag]['reach_above_floor']} | {pl}")

    cot = None
    if cot_items:
        if loops:
            backend.set_loops(a.cot_loops)
        k, n, secs["cot"] = score(backend, cot_items, cot_prefix, True, a.cot_max_new, ["\n\nQuestion"], "cot", rows)
        cot = summarize(k, n, a.floor)
        pl = " ".join(f"{L}:{v:.2f}" for L, v in cot["per_level_accuracy"].items())
        print(f"CoT reference overall {cot['overall']:.3f} | {pl}")

    if a.smoke_n:
        print("smoke run -- not logging")
        return

    main_tag = f"l{max(loops)}" if loops else "nocot"
    eff = {t: backend.n_layers * (int(t[1:]) if loops else 1) for t in res}
    slug = "depth2-fewshot-" + a.model.split("/")[-1].lower()
    record = RunRecord(
        run_id=new_run_id("baseline_fewshot_nocot", slug), mechanism="baseline_fewshot_nocot", stage=a.stage,
        model=ModelInfo(backbone=a.model, checkpoint=a.model, n_params=backend.n_params),
        dataset=DatasetInfo(name="synthetic-depth2", split="test", n_examples=len(items), seed=None),
        metrics={
            "final_answer_accuracy": res[main_tag]["overall"],
            "compute_steps": 0,
            "extra": {
                "task": "depth2",
                "design": "depth-vs-layers scan: pretrained model, few-shot same-level shots, no CoT, "
                          "per-level accuracy on the first per_level test items of each level; CoT reference",
                "n_layers": backend.n_layers,
                "effective_layers": eff,
                "nocot": res, "cot_reference": cot,
                "seconds": secs,
                "main_condition": main_tag,
            },
        },
        hyperparams={"n_shots": a.n_shots, "shot_seed": a.shot_seed, "shot_source": "train, same level",
                     "per_level": a.per_level, "levels": sorted(test), "cot_per_level": a.cot_per_level, "max_new": a.max_new,
                     "cot_max_new": a.cot_max_new, "reach_floor": a.floor, "loops": loops or None,
                     "cot_loops": a.cot_loops if loops else None, "decoding": "greedy",
                     "backend": "hf-bs1" if loops else "vllm"},
        seed=seed, hardware=a.hardware,
        notes="depth2 few-shot no-CoT scan of a pretrained model: accuracy by rule count vs layer count",
    )
    path = record.save(out_dir=Path(a.out_dir) / record.run_id if a.out_dir else None,
                       predictions=list(rows.values()))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
