#!/usr/bin/env python3
"""Positive control for the activation-patching methodology, on the explicit-CoT model.

Every causal result on the latent mechanisms so far is a null (patch ~ control). This
run asks whether a single-position intervention *can* move this backbone's answer when
the intermediate value is visible in the CoT: take a recipient's own greedy CoT, and at
the token span of its step-s result either (a) swap the text for a donor's value, or
(b) overwrite the residual stream at layer L at those positions with the donor's
activations at ITS step-s result span, then continue greedy generation from there.
Controls at the same span: a random unrelated donor activation, zero, and the mean
result-span activation. Metrics mirror the CODI/recurrent_depth patch runs: does the
next generated step use the injected value (tracking), does the answer change, and does
it change to the counterfactual-consistent value.

  uv run python scripts/patch_explicit_cot.py \
      --ckpt-dir results/20260918-021435_explicit_cot_baseline-full/ckpt \
      --n-pairs 200 --layers 0,3,6,9,11 --ablate-layers 0,6 \
      --slug patch-positive-control --stage full_run --hardware "RunPod RTX A5000 (secure)"
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latentreasoning.data.gsm8k_aug import Example, load_gsm8k_aug
from latentreasoning.eval.metrics import extract_final_number, is_correct
from latentreasoning.mechanisms.explicit_cot import build_eval_prompt_ids
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id
from latentreasoning.utils import set_seed

STEP_RE = re.compile(r"<<([^<>=]*)=([^<>]*)>>")
# Unsigned on purpose: inside a calculator expression "16-4" the minus is the operator, and
# a signed pattern would swallow it into "-4" and miss the operand. (Signed results such as
# "<<5-8=-3>>" are rare in GSM8K-Aug and are simply not matched.)
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
SAFE_EXPR_RE = re.compile(r"^[\d\s.+\-*/()]+$")


# ---- stats helpers (copied from scripts/decode_patch_codi.py, which can't be imported
# outside the CODI checkout because it pulls in `src.model` at import time) ----------------
def wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = x / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n))


# ---- arithmetic-chain helpers --------------------------------------------------------------
def num_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    a, b = a.strip().replace(",", ""), b.strip().replace(",", "")
    try:
        return float(a) == float(b)
    except ValueError:
        return a == b


def fmt_num(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def safe_eval(expr: str) -> float | None:
    expr = expr.replace(",", "").strip()
    if not expr or not SAFE_EXPR_RE.match(expr):
        return None
    try:
        v = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 -- whitelisted charset above
    except Exception:  # noqa: BLE001 -- ZeroDivisionError, SyntaxError, ...
        return None
    return float(v) if isinstance(v, (int, float)) else None


def operand_in(val: str, expr: str) -> bool:
    return any(num_equal(m, val) for m in NUM_RE.findall(expr))


def substitute(expr: str, mapping: dict[str, str]) -> str:
    pieces = re.split(r"(\d[\d,]*(?:\.\d+)?)", expr)
    out = []
    for p in pieces:
        rep = p
        if p and NUM_RE.fullmatch(p):
            for old, new in mapping.items():
                if num_equal(p, old):
                    rep = new
                    break
        out.append(rep)
    return "".join(out)


def parse_steps(text: str) -> list[dict]:
    """`<<expr=val>>` steps in order, with the value's character span in `text`."""
    steps = []
    for m in STEP_RE.finditer(text):
        raw_val = m.group(2)
        lead = len(raw_val) - len(raw_val.lstrip())
        val = raw_val.strip()
        vs = m.start(2) + lead
        steps.append({"expr": m.group(1).strip(), "val": val, "val_span": (vs, vs + len(val))})
    return steps


def reeval_chain(steps: list[dict], k: int, new_val: str) -> dict[int, str] | None:
    """Substitute `new_val` for step k's result in the downstream expressions of a chain and
    re-evaluate it forward, propagating any changed results. Returns {j: new result} for
    j > k, or None if any downstream step can't be evaluated."""
    mapping = {steps[k]["val"]: new_val}
    results: dict[int, str] = {}
    for j in range(k + 1, len(steps)):
        r = safe_eval(substitute(steps[j]["expr"], mapping))
        if r is None:
            return None
        r_str = fmt_num(r)
        results[j] = r_str
        if not num_equal(r_str, steps[j]["val"]) and steps[j]["val"] not in mapping:
            mapping[steps[j]["val"]] = r_str
    return results


def counterfactual_answer(steps: list[dict], k: int, new_val: str, base_answer: str | None) -> str | None:
    """The answer a faithful continuation would give if step k's result were `new_val`:
    re-evaluate the chain and return the re-evaluated version of whichever downstream step
    the baseline answer came from. None if undefined (answer isn't a downstream result, or
    the chain can't be evaluated)."""
    if base_answer is None:
        return None
    results = reeval_chain(steps, k, new_val)
    if results is None:
        return None
    for j in range(len(steps) - 1, k, -1):
        if num_equal(steps[j]["val"], base_answer):
            return results[j]
    return None


# ---- tokenization bookkeeping ---------------------------------------------------------------
def token_char_spans(tokenizer, ids: list[int]) -> list[tuple[int, int]]:
    """Character span of each token in `tokenizer.decode(ids)`, via cumulative decoding
    (robust to the generated ids not being the canonical BPE of the decoded text)."""
    spans, prev = [], 0
    for i in range(len(ids)):
        end = len(tokenizer.decode(ids[: i + 1], skip_special_tokens=False))
        spans.append((prev, end))
        prev = end
    return spans


def span_tokens(spans: list[tuple[int, int]], cs: int, ce: int) -> tuple[int, int] | None:
    idx = [i for i, (s, e) in enumerate(spans) if s < ce and e > cs]
    if not idx or idx != list(range(idx[0], idx[-1] + 1)):
        return None
    return idx[0], idx[-1] + 1


# ---- model plumbing --------------------------------------------------------------------------
def hook_point(model, layer: int):
    """Residual-stream hook site: block `layer`'s output, or the embedding output (-1)."""
    return model.transformer.drop if layer < 0 else model.transformer.h[layer]


def _hs(output):
    return output[0] if isinstance(output, tuple) else output


@torch.no_grad()
def capture_activations(model, ids: list[int], layers: list[int], positions: list[int], device) -> dict[int, torch.Tensor]:
    """One forward over `ids`; returns {layer: [len(positions), d]} of residual-stream states."""
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in layers:
        def _cap(module, args, output, layer=layer):
            captured[layer] = _hs(output)[0, positions, :].detach().clone()
        handles.append(hook_point(model, layer).register_forward_hook(_cap))
    try:
        model(input_ids=torch.tensor([ids], device=device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return captured


@torch.no_grad()
def continue_greedy(model, prefix_ids: list[int], max_new_tokens: int, eos_id: int, device,
                    patch: tuple[int, list[int], torch.Tensor] | None = None) -> list[int]:
    """Run `prefix_ids` (optionally overwriting the residual stream at `patch=(layer,
    positions, [n, d])` during that pass -- the KV cache then carries the patch), then
    greedy-decode until EOS. Returns the generated ids (EOS excluded)."""
    handle = None
    if patch is not None:
        layer, positions, vec = patch

        def _patch(module, args, output):
            hs = _hs(output)
            hs[:, positions, :] = vec.to(hs.dtype)
            return output
        handle = hook_point(model, layer).register_forward_hook(_patch)
    try:
        out = model(input_ids=torch.tensor([prefix_ids], device=device), use_cache=True)
    finally:
        if handle is not None:
            handle.remove()
    pkv = out.past_key_values
    next_id = int(out.logits[0, -1].argmax())
    gen: list[int] = []
    for _ in range(max_new_tokens):
        if next_id == eos_id:
            break
        gen.append(next_id)
        out = model(input_ids=torch.tensor([[next_id]], device=device), past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        next_id = int(out.logits[0, -1].argmax())
    return gen


# ---- main ------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--checkpoint-label", default=None)
    ap.add_argument("--n-examples", type=int, default=None, help="test examples to generate for (None = all 1319)")
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--max-pairs-per-recipient", type=int, default=2)
    ap.add_argument("--layers", default="0,3,6,9,11", help="residual-stream layers for real/random-donor patches (-1 = embeddings)")
    ap.add_argument("--ablate-layers", default="0,6", help="layers for zero/mean ablation")
    ap.add_argument("--mean-sample-n", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache", default=None, help="jsonl of baseline generations to reuse")
    ap.add_argument("--slug", default="patch-positive-control")
    ap.add_argument("--stage", default="full_run", choices=["smoke_test", "pilot", "full_run"])
    ap.add_argument("--hardware", default="local")
    args = ap.parse_args()

    seed = set_seed(args.seed)
    rng = random.Random(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    layers = [int(x) for x in args.layers.split(",") if x != ""]
    ablate_layers = [int(x) for x in args.ablate_layers.split(",") if x != ""]
    all_layers = sorted(set(layers) | set(ablate_layers))

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt_dir, attn_implementation="eager").to(device)
    model.eval()  # LoRA-less here, but the CODI runs were bitten by exactly this omission
    eos_id = tokenizer.eos_token_id
    n_params = sum(p.numel() for p in model.parameters())
    hw = f"{args.hardware} / {torch.cuda.get_device_name(0)}" if device == "cuda" else f"{args.hardware} / {device}"

    examples = load_gsm8k_aug(split="test", n=None, seed=None)
    if args.n_examples is not None:
        examples = examples[: args.n_examples]
    by_idx = {ex.idx: ex for ex in examples}

    # ---- Phase 1: baseline greedy CoT for every example (cached) ---------------------------
    t0 = time.perf_counter()
    gens: dict[int, dict] = {}
    if args.cache and Path(args.cache).exists():
        for line in open(args.cache):
            r = json.loads(line)
            gens[r["idx"]] = r
        print(f"loaded {len(gens)} cached generations from {args.cache}")
    todo = [ex for ex in examples if ex.idx not in gens]
    for i, ex in enumerate(todo):
        prompt = build_eval_prompt_ids(tokenizer, ex.question)
        gen = continue_greedy(model, prompt, args.max_new_tokens, eos_id, device)
        text = tokenizer.decode(gen, skip_special_tokens=True)
        pred = extract_final_number(text)
        gens[ex.idx] = {"idx": ex.idx, "question": ex.question, "gold_answer": ex.answer, "raw_output": text,
                        "predicted_answer": pred, "correct": is_correct(pred, ex.answer),
                        "prompt_ids": prompt, "gen_ids": gen}
        if (i + 1) % 100 == 0:
            print(f"generated {i + 1}/{len(todo)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    if args.cache:
        with open(args.cache, "w") as f:
            for r in gens.values():
                f.write(json.dumps(r) + "\n")
    gen_elapsed = time.perf_counter() - t0
    n_gen = len(gens)
    base_acc = sum(r["correct"] for r in gens.values()) / n_gen
    cot_tokens = sum(len(r["gen_ids"]) for r in gens.values()) / n_gen
    print(f"baseline: n={n_gen} accuracy={base_acc:.3f} mean_gen_tokens={cot_tokens:.1f} ({gen_elapsed:.0f}s)")

    # ---- Phase 2: locate result spans; build qualifying (recipient, step) units ------------
    info: dict[int, dict] = {}  # idx -> {"steps", "spans", "clean": {k: (a, b)}}
    for idx, r in gens.items():
        steps = parse_steps(r["raw_output"])
        spans = token_char_spans(tokenizer, r["gen_ids"])
        clean = {}
        for k, st in enumerate(steps):
            ab = span_tokens(spans, *st["val_span"])
            if ab is None:
                continue
            a, b = ab
            if tokenizer.decode(r["gen_ids"][a:b]).strip() == st["val"]:
                clean[k] = (a, b)
        info[idx] = {"steps": steps, "clean": clean}

    units = []  # (recipient idx, k)
    for idx, r in gens.items():
        ex, st, clean = by_idx[idx], info[idx]["steps"], info[idx]["clean"]
        gold = ex.intermediate_values
        for k, (a, b) in clean.items():
            if k >= len(gold) or not num_equal(st[k]["val"], gold[k]):
                continue
            if k + 1 >= len(st) or not operand_in(st[k]["val"], st[k + 1]["expr"]):
                continue
            units.append((idx, k))
    # donor pool: (idx, k, n_span_tokens) with a gold-correct, cleanly-tokenized step-k result
    donor_pool: dict[int, list[tuple[int, int]]] = {}  # k -> [(idx, ntok)]
    any_pool: list[tuple[int, int, int]] = []  # (idx, k, ntok), any clean result span
    for idx, r in gens.items():
        gold = by_idx[idx].intermediate_values
        for k, (a, b) in info[idx]["clean"].items():
            any_pool.append((idx, k, b - a))
            if k < len(gold) and num_equal(info[idx]["steps"][k]["val"], gold[k]):
                donor_pool.setdefault(k, []).append((idx, b - a))
    print(f"qualifying (recipient, step) units: {len(units)} from {len({u[0] for u in units})} recipients; "
          f"any-span pool: {len(any_pool)}")

    rng.shuffle(units)
    per_recipient: dict[int, int] = {}
    pairs = []
    for idx, k in units:
        if per_recipient.get(idx, 0) >= args.max_pairs_per_recipient:
            continue
        a, b = info[idx]["clean"][k]
        ntok, val = b - a, info[idx]["steps"][k]["val"]
        cands = [d for d, nt in donor_pool.get(k, []) if d != idx and nt == ntok
                 and not num_equal(info[d]["steps"][k]["val"], val)]
        rcands = [(d, kk) for d, kk, nt in any_pool if d != idx and nt == ntok
                  and not num_equal(info[d]["steps"][kk]["val"], val)]
        if not cands or not rcands:
            continue
        donor = rng.choice(cands)
        rdonor, rk = rng.choice(rcands)
        pairs.append({"recipient_idx": idx, "step": k + 1, "donor_idx": donor, "random_donor_idx": rdonor,
                      "random_donor_step": rk + 1})
        per_recipient[idx] = per_recipient.get(idx, 0) + 1
        if len(pairs) >= args.n_pairs:
            break
    print(f"pairs: {len(pairs)} ({len(per_recipient)} recipients)")
    if not pairs:
        sys.exit("no pairs -- increase --n-examples")

    # ---- Phase 3: mean result-span activation per layer (content-free, on-manifold control) --
    t1 = time.perf_counter()
    sample = [idx for idx in gens if info[idx]["clean"]]
    rng.shuffle(sample)
    sample = sample[: args.mean_sample_n]
    acc = {L: None for L in all_layers}
    n_acc = 0
    for idx in sample:
        r = gens[idx]
        ids = r["prompt_ids"] + r["gen_ids"]
        pos = [len(r["prompt_ids"]) + p for (a, b) in info[idx]["clean"].values() for p in range(a, b)]
        caps = capture_activations(model, ids, all_layers, pos, device)
        for L in all_layers:
            s = caps[L].float().sum(0)
            acc[L] = s if acc[L] is None else acc[L] + s
        n_acc += len(pos)
    mean_vec = {L: (acc[L] / n_acc) for L in all_layers}
    print(f"mean vectors from {len(sample)} examples / {n_acc} result-span tokens ({time.perf_counter() - t1:.0f}s)")

    # ---- Phase 4: interventions ------------------------------------------------------------
    def cond_record(cont_ids: list[int], pre_text_ids: list[int], base_answer, ex: Example, steps, k,
                    injected_val: str | None, cf_ans: str | None, cf_ans_gold: str | None) -> dict:
        cont_text = tokenizer.decode(cont_ids, skip_special_tokens=True)
        full_text = tokenizer.decode(pre_text_ids + cont_ids, skip_special_tokens=True)
        ans = extract_final_number(full_text)
        nxt = parse_steps(cont_text)
        rec = {
            "answer": ans, "answer_changed": not num_equal(ans, base_answer) if (ans or base_answer) else False,
            "correct": is_correct(ans, ex.answer), "n_cont_tokens": len(cont_ids),
            "next_expr": nxt[0]["expr"] if nxt else None,
            "tracking_next": (operand_in(injected_val, nxt[0]["expr"]) if (nxt and injected_val is not None) else None),
            "tracking_any": (any(operand_in(injected_val, s["expr"]) for s in nxt) if (nxt and injected_val is not None) else None),
            "matches_cf": (num_equal(ans, cf_ans) if cf_ans is not None else None),
            "matches_cf_gold": (num_equal(ans, cf_ans_gold) if cf_ans_gold is not None else None),
            "cont_text": cont_text,
        }
        return rec

    t2 = time.perf_counter()
    pair_records = []
    n_base_repro = 0
    for pi, pr in enumerate(pairs):
        ex, k = by_idx[pr["recipient_idx"]], pr["step"] - 1
        r, steps = gens[ex.idx], info[ex.idx]["steps"]
        a, b = info[ex.idx]["clean"][k]
        P, G = r["prompt_ids"], r["gen_ids"]
        abs_pos = list(range(len(P) + a, len(P) + b))
        prefix = P + G[:b]
        own_val = steps[k]["val"]

        # baseline, re-derived through the same teacher-forced path (should reproduce G[b:])
        base_cont = continue_greedy(model, prefix, args.max_new_tokens, eos_id, device)
        base_repro = base_cont == G[b:]
        n_base_repro += base_repro
        base_answer = extract_final_number(tokenizer.decode(G[:b] + base_cont, skip_special_tokens=True))

        # donors
        d, rd, rk = pr["donor_idx"], pr["random_donor_idx"], pr["random_donor_step"] - 1
        rd_r, dr = gens[d], gens[rd]
        da, db = info[d]["clean"][k]
        ra, rb = info[rd]["clean"][rk]
        d_prefix = rd_r["prompt_ids"] + rd_r["gen_ids"][:db]
        rd_prefix = dr["prompt_ids"] + dr["gen_ids"][:rb]
        d_pos = list(range(len(rd_r["prompt_ids"]) + da, len(rd_r["prompt_ids"]) + db))
        rd_pos = list(range(len(dr["prompt_ids"]) + ra, len(dr["prompt_ids"]) + rb))
        d_caps = capture_activations(model, d_prefix, layers, d_pos, device)
        rd_caps = capture_activations(model, rd_prefix, layers, rd_pos, device)
        d_val, rd_val = info[d]["steps"][k]["val"], info[rd]["steps"][rk]["val"]
        d_val_ids = rd_r["gen_ids"][da:db]

        # counterfactual-consistent answers (recipient's own generated chain; gold chain)
        cf_ans = counterfactual_answer(steps, k, d_val, base_answer)
        gold_steps = parse_steps(ex.rationale)
        cf_ans_gold = counterfactual_answer(gold_steps, k, d_val, ex.answer) if k < len(gold_steps) else None
        cf_ans_r = counterfactual_answer(steps, k, rd_val, base_answer)
        cf_ans_r_gold = counterfactual_answer(gold_steps, k, rd_val, ex.answer) if k < len(gold_steps) else None

        conds = {}
        conds["baseline"] = cond_record(base_cont, G[:b], base_answer, ex, steps, k, own_val, None, None)
        # (a) text-level counterfactual
        text_prefix = P + G[:a] + d_val_ids
        cont = continue_greedy(model, text_prefix, args.max_new_tokens, eos_id, device)
        conds["text"] = cond_record(cont, G[:a] + d_val_ids, base_answer, ex, steps, k, d_val, cf_ans, cf_ans_gold)
        # (b) real-donor activation patch per layer; (c) random-donor control per layer
        for L in layers:
            cont = continue_greedy(model, prefix, args.max_new_tokens, eos_id, device, patch=(L, abs_pos, d_caps[L]))
            conds[f"real_L{L}"] = cond_record(cont, G[:b], base_answer, ex, steps, k, d_val, cf_ans, cf_ans_gold)
            cont = continue_greedy(model, prefix, args.max_new_tokens, eos_id, device, patch=(L, abs_pos, rd_caps[L]))
            conds[f"random_L{L}"] = cond_record(cont, G[:b], base_answer, ex, steps, k, rd_val, cf_ans_r, cf_ans_r_gold)
        # (c) content-free ablations
        for L in ablate_layers:
            zero = torch.zeros(len(abs_pos), mean_vec[L].shape[-1], device=device)
            cont = continue_greedy(model, prefix, args.max_new_tokens, eos_id, device, patch=(L, abs_pos, zero))
            conds[f"zero_L{L}"] = cond_record(cont, G[:b], base_answer, ex, steps, k, None, None, None)
            mv = mean_vec[L].unsqueeze(0).expand(len(abs_pos), -1).contiguous()
            cont = continue_greedy(model, prefix, args.max_new_tokens, eos_id, device, patch=(L, abs_pos, mv))
            conds[f"mean_L{L}"] = cond_record(cont, G[:b], base_answer, ex, steps, k, None, None, None)

        pair_records.append({
            **pr, "recipient_gold": ex.answer, "recipient_value": own_val, "donor_value": d_val,
            "random_donor_value": rd_val, "n_span_tokens": b - a, "span_abs_start": abs_pos[0],
            "donor_span_abs_start": d_pos[0], "delta_pos_donor": abs(abs_pos[0] - d_pos[0]),
            "baseline_reproduced": base_repro, "baseline_answer": base_answer,
            "baseline_correct": is_correct(base_answer, ex.answer),
            "cf_answer": cf_ans, "cf_answer_gold": cf_ans_gold, "conditions": conds,
        })
        if (pi + 1) % 25 == 0:
            print(f"patched {pi + 1}/{len(pairs)} ({time.perf_counter() - t2:.0f}s)", flush=True)
    patch_elapsed = time.perf_counter() - t2
    n = len(pair_records)

    # ---- aggregate ---------------------------------------------------------------------------
    cond_names = list(pair_records[0]["conditions"].keys())

    def agg(cond: str, key: str):
        vals = [p["conditions"][cond][key] for p in pair_records]
        vals = [v for v in vals if v is not None]
        c = sum(bool(v) for v in vals)
        lo, hi = wilson_ci(c, len(vals))
        return {"rate": (c / len(vals) if vals else None), "n": len(vals), "count": c, "ci": [lo, hi]}

    def mcnemar(cond_a: str, cond_b: str, key: str):
        b = c = 0
        for p in pair_records:
            va, vb = p["conditions"][cond_a][key], p["conditions"][cond_b][key]
            if va is None or vb is None:
                continue
            b += bool(va) and not vb
            c += bool(vb) and not va
        return {"b": b, "c": c, "p_exact": mcnemar_exact_p(b, c)}

    table = {cond: {key: agg(cond, key) for key in ("tracking_next", "tracking_any", "answer_changed",
                                                     "matches_cf", "matches_cf_gold", "correct")}
             for cond in cond_names}
    tests = {}
    for L in layers:
        for key in ("tracking_next", "answer_changed", "matches_cf"):
            tests[f"real_vs_random_L{L}:{key}"] = mcnemar(f"real_L{L}", f"random_L{L}", key)
    for L in ablate_layers:
        if L in layers:
            for key in ("answer_changed", "correct"):
                tests[f"real_vs_zero_L{L}:{key}"] = mcnemar(f"real_L{L}", f"zero_L{L}", key)
                tests[f"real_vs_mean_L{L}:{key}"] = mcnemar(f"real_L{L}", f"mean_L{L}", key)
    for cond in cond_names:
        if cond != "baseline":
            tests[f"baseline_vs_{cond}:correct"] = mcnemar("baseline", cond, "correct")

    best_L = max(layers, key=lambda L: table[f"real_L{L}"]["matches_cf"]["rate"] or 0.0)
    print(f"\n{'condition':14s} {'track_next':>11s} {'ans_changed':>11s} {'matches_cf':>11s} {'cf_gold':>8s} {'acc':>6s}")
    for cond in cond_names:
        t = table[cond]
        f = lambda k: (f"{t[k]['rate']:.3f}" if t[k]["rate"] is not None else "  -  ")  # noqa: E731
        print(f"{cond:14s} {f('tracking_next'):>11s} {f('answer_changed'):>11s} {f('matches_cf'):>11s} {f('matches_cf_gold'):>8s} {f('correct'):>6s}")
    for name, tst in tests.items():
        if name.startswith("real_vs_random"):
            print(f"{name}: b={tst['b']} c={tst['c']} p={tst['p_exact']:.4f}")
    print(f"baseline reproduced through the teacher-forced path: {n_base_repro}/{n}")

    metrics = {
        "final_answer_accuracy": None, "unparseable_rate": None, "compute_steps": None, "sec_per_example": None,
        "extra": {
            "cot_tokens": cot_tokens, "baseline_accuracy_all_generated": base_acc, "n_generated": n_gen,
            "n_pairs": n, "n_recipients": len(per_recipient), "n_qualifying_units": len(units),
            "layers": layers, "ablate_layers": ablate_layers, "mean_sample_n": len(sample),
            "baseline_reproduced_frac": n_base_repro / n,
            "recipient_baseline_accuracy": sum(p["baseline_correct"] for p in pair_records) / n,
            "n_pairs_with_cf_answer": sum(p["cf_answer"] is not None for p in pair_records),
            "n_pairs_with_cf_answer_gold": sum(p["cf_answer_gold"] is not None for p in pair_records),
            "best_layer_by_matches_cf": best_L,
            "intervention_accuracy": table[f"real_L{best_L}"]["matches_cf"]["rate"],
            "intervention_accuracy_text_level": table["text"]["matches_cf"]["rate"],
            "table": table, "tests": tests,
            "gen_elapsed_sec": gen_elapsed, "patch_elapsed_sec": patch_elapsed,
        },
    }
    record = RunRecord(
        run_id=new_run_id("explicit_cot", args.slug), mechanism="explicit_cot", stage=args.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=args.checkpoint_label or args.ckpt_dir, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=n, seed=None),
        metrics=metrics,
        hyperparams={"n_pairs": args.n_pairs, "max_pairs_per_recipient": args.max_pairs_per_recipient,
                     "layers": layers, "ablate_layers": ablate_layers, "mean_sample_n": args.mean_sample_n,
                     "max_new_tokens": args.max_new_tokens, "greedy": True, "n_examples_generated": n_gen},
        seed=seed, hardware=hw,
        notes="Positive control for activation patching: text-level and per-layer residual patches of a "
              "step result in the explicit-CoT rollout, vs. random-donor / zero / mean controls.",
    )
    predictions = [{k: v for k, v in r.items() if k not in ("prompt_ids", "gen_ids")} for r in gens.values()]
    manifest_path = record.save(predictions=predictions)
    out_dir = manifest_path.parent
    with (out_dir / "patch_pairs.jsonl").open("w") as f:
        for p in pair_records:
            f.write(json.dumps(p) + "\n")
    (out_dir / "eval_command.txt").write_text(" ".join(sys.argv) + "\n")
    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
