"""Ouro step 2 (Q4 extension to a looped model): is the intermediate step value *present*
(linearly decodable) in Ouro's residual stream at the end of the question / at the answer
suffix, per (loop, layer)? Step 1 (`ouro_twin_patch.py`) showed the suffix only *causally*
carries the answer's leading digit; this asks whether the value is there at all.

One forward pass per question at the trained 4 loops: cached few-shot prefix (the gate's
8-shot no-CoT prompt) + `Question: ...` + the `\\nAnswer: ` suffix. Captured: decoder-layer
outputs after layer l in loop u at positions
  q_last   last question token
  sfx0-3   the 4 suffix tokens ('\\n', 'Answer', ':', ' ')
Targets (from the `<<a op b=c>>` rationale, chains of >= 2 steps, integer values only):
  step1        first intermediate result
  penult       last intermediate result before the answer (= step1 for 2-step chains)
  answer       final answer
  qlast_num    last number literally in the question (surface control: the probe can read numbers)
  step1_resid  slog(step1) with its linear dependence on slog(answer), slog(qlast_num)
               removed (fit on the probe-fit split): step1 information beyond the answer
Probes: ridge on z-scored features. Regression on signed-log1p values (Pearson r, R^2 in log
space, hit@5% relative error in raw space; the CODI/Coconut probe runs used the same
transform), and ridge-classifier on the value's ones digit (10-way accuracy vs majority).
lambda per (site, target) on a validation split of the fit pool; fit on xmech fit questions
(GSM8K train), reported on xmech eval (GSM8K test), also split by whether Ouro answered the
question correctly at 4 loops in the step-0 gate.

Usage (pod, Ouro venv, from the repo):
  python scripts/ouro_probe.py --questions xmech_questions.jsonl \\
      --gate-predictions results/<1.4B gate run>/predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ouro_gate import build_prefix, read_questions, set_loops  # noqa: E402
from ouro_twin_patch import SUFFIX, _expanded, _inputs, prefill  # noqa: E402

from latentreasoning.eval.counterfactual import parse_steps  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402
from latentreasoning.utils import set_seed  # noqa: E402

POSITIONS = ("q_last", "sfx0", "sfx1", "sfx2", "sfx3")
TARGETS = ("step1", "penult", "answer", "qlast_num")
REG_TARGETS = TARGETS + ("step1_resid",)
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
LAMBDA_FRACS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)  # x n_fit, on z-scored features


def as_int(v: str) -> int | None:
    try:
        f = float(str(v).replace(",", ""))
    except ValueError:
        return None
    return int(f) if f == int(f) else None


def slog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def targets_for(q: dict) -> dict | None:
    steps = parse_steps(q["rationale"])
    if len(steps) < 2:
        return None
    nums = NUM_RE.findall(q["question"])
    vals = {"step1": as_int(steps[0]["val"]), "penult": as_int(steps[-2]["val"]),
            "answer": as_int(q["answer"]), "qlast_num": as_int(nums[-1]) if nums else None}
    if any(v is None for v in vals.values()):
        return None
    qnums = {as_int(n) for n in nums}
    vals["step1_in_question"] = vals["step1"] in qnums
    vals["chain_len"] = len(steps)
    return vals


class Capture:
    """Forward hooks storing row 0's states at `positions` for every (loop, layer) into `buf`
    [n_pos, loops, layers, d]."""

    def __init__(self, layers, n_pos, loops, d, device):
        self.buf = torch.zeros(n_pos, loops, len(layers), d, dtype=torch.bfloat16, device=device)
        self.positions: list[int] | None = None
        for li, layer in enumerate(layers):
            layer.register_forward_hook(self._hook(li), with_kwargs=True)

    def _hook(self, li):
        def fn(module, args, kwargs, output):
            if self.positions is not None:
                hs = output[0] if isinstance(output, tuple) else output
                self.buf[:, kwargs.get("current_ut"), li] = hs[0, self.positions]
            return None
        return fn


@torch.no_grad()
def ridge_site(X: torch.Tensor, fit: torch.Tensor, Y: torch.Tensor):
    """Ridge for all target columns at once. X [N, d] (bf16), fit index tensor, Y [N, k]
    (fp32). Returns predictions [N, k] for every lambda: [n_lam, N, k]."""
    Xf = X.float()
    mu, sd = Xf[fit].mean(0), Xf[fit].std(0) + 1e-4
    Z = (Xf - mu) / sd
    Zf = Z[fit]
    ymu = Y[fit].mean(0)
    evals, V = torch.linalg.eigh(Zf.T @ Zf)
    A = V.T @ (Zf.T @ (Y[fit] - ymu))
    ZV = Z @ V
    n = len(fit)
    return torch.stack([ZV @ (A / (evals.clamp_min(0) + lam * n)[:, None]) + ymu for lam in LAMBDA_FRACS])


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def r2(pred: torch.Tensor, gold: torch.Tensor) -> float:
    return float(1 - ((pred - gold) ** 2).sum() / (((gold - gold.mean()) ** 2).sum() + 1e-12))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ByteDance/Ouro-1.4B")
    p.add_argument("--questions", required=True)
    p.add_argument("--gate-predictions", required=True, help="step-0 predictions.jsonl for this model")
    p.add_argument("--loops", type=int, default=4)
    p.add_argument("--n-fit", type=int, default=4000, help="fit-split questions (80/20 probe fit / lambda val)")
    p.add_argument("--n-shots", type=int, default=8)
    p.add_argument("--shot-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attn", default="eager")
    p.add_argument("--smoke-n", type=int, default=0, help=">0: this many fit + eval questions, don't log")
    p.add_argument("--slug", default="probe-ouro-1.4b")
    p.add_argument("--stage", default="pilot")
    p.add_argument("--hardware", default="RunPod")
    a = p.parse_args()
    seed = set_seed(a.seed)
    rng = random.Random(a.seed)

    qs = read_questions(a.questions)
    fit_all = [q for q in qs if q["split"] == "fit"]
    prefix = build_prefix(fit_all, a.n_shots, a.shot_seed, cot=False)
    shots = {q["key"] for q in random.Random(a.shot_seed).sample(fit_all, a.n_shots)}
    gate = {r["key"]: r for r in (json.loads(x) for x in Path(a.gate_predictions).read_text().splitlines() if x.strip())}

    pool = [q for q in fit_all if q["key"] not in shots]
    rng.shuffle(pool)
    fit_q = [(q, t) for q in pool if (t := targets_for(q)) is not None][: a.smoke_n or a.n_fit]
    ev_q = [(q, t) for q in qs if q["split"] == "eval" and (t := targets_for(q)) is not None]
    if a.smoke_n:
        ev_q = ev_q[: a.smoke_n]
    items = fit_q + ev_q
    n_fit_all, N = len(fit_q), len(items)
    print(f"{n_fit_all} fit + {len(ev_q)} eval questions (chains >= 2 steps, integer values)")

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16,
                                                 device_map="cuda", attn_implementation=a.attn).eval()
    set_loops(model, a.loops)
    assert not getattr(model.config, "use_sliding_window", False), "full causal mask assumed for all layers"
    layers = model.model.layers[: model.config.num_hidden_layers]
    n_layers, d = len(layers), model.config.hidden_size
    C = Capture(layers, len(POSITIONS), a.loops, d, model.device)
    feats = torch.zeros(N, len(POSITIONS), a.loops, n_layers, d, dtype=torch.bfloat16, device=model.device)
    print(f"features {tuple(feats.shape)} = {feats.numel() * 2 / 1e9:.1f} GB on GPU")

    prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tok(SUFFIX, add_special_tokens=False)["input_ids"]
    pre = prefill(model, prefix_ids)
    n_mismatch = 0
    t0 = time.perf_counter()
    for i, (q, _) in enumerate(items):
        ids = tok(prefix + f"Question: {q['question']}", add_special_tokens=False)["input_ids"]
        if ids[: len(prefix_ids)] == prefix_ids:
            new, past, cache = ids[len(prefix_ids):] + suffix_ids, len(prefix_ids), _expanded(pre[0], 1)
        else:  # tokenization merged across the prefix boundary: no cache for this one
            n_mismatch += 1
            new, past, cache = ids + suffix_ids, 0, None
        nq = len(new) - len(suffix_ids)
        C.positions = [nq - 1] + [nq + k for k in range(len(suffix_ids))]
        with torch.no_grad():
            model(**_inputs(model, torch.tensor([new], device=model.device), past),
                  past_key_values=cache, use_cache=cache is not None)
        C.positions = None
        feats[i] = C.buf
        if (i + 1) % 500 == 0:
            print(f"extracted {i + 1}/{N} ({time.perf_counter() - t0:.0f}s)", flush=True)
    extract_elapsed = time.perf_counter() - t0
    print(f"extraction {extract_elapsed:.0f}s; prefix-boundary tokenization mismatches: {n_mismatch}")

    # ---- targets ---------------------------------------------------------------------------
    dev = model.device
    raw = torch.tensor([[float(t[k]) for k in TARGETS] for _, t in items], device=dev)
    logv = slog(raw)
    n_val = n_fit_all // 5
    fit_idx = torch.arange(0, n_fit_all - n_val, device=dev)
    val_idx = torch.arange(n_fit_all - n_val, n_fit_all, device=dev)
    ev_idx = torch.arange(n_fit_all, N, device=dev)
    # step1 residualized on answer + qlast_num (linear in log space, fit on fit_idx)
    B = torch.cat([logv[:, [2, 3]], torch.ones(N, 1, device=dev)], 1)
    coef = torch.linalg.lstsq(B[fit_idx], logv[fit_idx, :1]).solution
    resid = logv[:, :1] - B @ coef
    reg_Y = torch.cat([logv, resid], 1)  # [N, 5]
    ones = (raw.abs().long() % 10)  # [N, 4]
    cls_Y = torch.nn.functional.one_hot(ones, 10).float().reshape(N, -1)  # [N, 40]
    Y = torch.cat([reg_Y, cls_Y], 1)
    nr = reg_Y.shape[1]
    majority = {k: float((ones[ev_idx, j] == ones[fit_idx, j].mode().values).float().mean())
                for j, k in enumerate(TARGETS)}
    target_corr = {f"{a_}~{b_}": pearson(logv[fit_idx, i], logv[fit_idx, j])
                   for i, a_ in enumerate(TARGETS) for j, b_ in enumerate(TARGETS) if i < j}
    print("majority ones-digit acc (eval):", {k: round(v, 3) for k, v in majority.items()})
    print("target log-corr (fit):", {k: round(v, 2) for k, v in target_corr.items()})

    ouro_ok = torch.tensor([bool(gate.get(q["key"], {}).get(f"correct_l{a.loops}", False)) for q, _ in ev_q], device=dev)
    novel = torch.tensor([not t["step1_in_question"] for _, t in ev_q], device=dev)
    subsets = {"all": torch.ones(len(ev_q), dtype=torch.bool, device=dev), "ouro_correct": ouro_ok,
               "ouro_wrong": ~ouro_ok, "step1_novel": novel}

    # ---- probes per (position, loop, layer) -----------------------------------------------
    def metrics(pred_reg, pred_cls, idx):
        out = {}
        for j, k in enumerate(REG_TARGETS):
            pr, gd = pred_reg[idx, j], reg_Y[idx, j]
            out[k] = {"r": pearson(pr, gd), "r2": r2(pr, gd)}
            if k != "step1_resid":
                rv = torch.sign(pr) * torch.expm1(pr.abs())
                out[k]["hit05"] = float(((rv - raw[idx, j]).abs() / raw[idx, j].abs().clamp_min(1) <= 0.05).float().mean())
                out[k]["digit_acc"] = float((pred_cls[idx, j].argmax(-1) == ones[idx, j]).float().mean())
        return out

    sites = [(pi, u, l) for pi in range(len(POSITIONS)) for u in range(a.loops) for l in range(n_layers)]
    results, best_pred = {}, {}
    t1 = time.perf_counter()
    for si, (pi, u, l) in enumerate(sites):
        P = ridge_site(feats[:, pi, u, l], fit_idx, Y)  # [n_lam, N, k]
        # lambda per column on the val split: R^2 for regression columns, digit acc for each 10-block
        lam_reg = [max(range(len(LAMBDA_FRACS)), key=lambda li: r2(P[li, val_idx, j], reg_Y[val_idx, j]))
                   for j in range(nr)]
        lam_cls = [max(range(len(LAMBDA_FRACS)), key=lambda li: float(
            (P[li, val_idx, nr + 10 * j: nr + 10 * j + 10].argmax(-1) == ones[val_idx, j]).float().mean()))
            for j in range(len(TARGETS))]
        pred_reg = torch.stack([P[lam_reg[j], :, j] for j in range(nr)], 1)
        pred_cls = torch.stack([P[lam_cls[j], :, nr + 10 * j: nr + 10 * j + 10] for j in range(len(TARGETS))], 1)
        key = f"{POSITIONS[pi]}/{u}_{l}"
        results[key] = {"val": metrics(pred_reg, pred_cls, val_idx),
                        **{s: metrics(pred_reg, pred_cls, ev_idx[m]) for s, m in subsets.items()},
                        "lambda_reg": [LAMBDA_FRACS[i] for i in lam_reg],
                        "lambda_cls": [LAMBDA_FRACS[i] for i in lam_cls]}
        best_pred[key] = (pred_reg[ev_idx].cpu(), pred_cls[ev_idx].argmax(-1).cpu())
        if (si + 1) % 96 == 0:
            print(f"probed {si + 1}/{len(sites)} sites ({time.perf_counter() - t1:.0f}s)", flush=True)
    probe_elapsed = time.perf_counter() - t1

    # ---- summaries: grids [loop][layer] per position, best site chosen on val ------------
    def grid(subset, tgt, met, pos):
        return [[results[f"{pos}/{u}_{l}"][subset][tgt][met] for l in range(n_layers)] for u in range(a.loops)]

    grids = {f"{pos}/{tgt}/{met}": grid("all", tgt, met, pos)
             for pos in POSITIONS for tgt in REG_TARGETS for met in (("r", "r2") if tgt == "step1_resid" else ("r", "r2", "hit05", "digit_acc"))}
    best = {}
    for tgt in REG_TARGETS:
        for met in ("r", "digit_acc"):
            if tgt == "step1_resid" and met == "digit_acc":
                continue
            k = max(results, key=lambda s: results[s]["val"][tgt][met])
            best[f"{tgt}/{met}"] = {"site": k, "val": results[k]["val"][tgt][met],
                                    **{s: results[k][s][tgt][met] for s in subsets},
                                    "loop1_layer0_same_pos": results[f"{k.split('/')[0]}/0_0"]["all"][tgt][met]}
    for pos in POSITIONS:
        for tgt, met in (("step1", "r"), ("step1_resid", "r"), ("answer", "r"), ("step1", "digit_acc"),
                         ("answer", "digit_acc"), ("qlast_num", "r")):
            print(f"--- {pos} {tgt} {met} (eval n={len(ev_q)}; rows = loop, cols = layer 0,2,..,{n_layers - 2})")
            for u, line in enumerate(grids[f"{pos}/{tgt}/{met}"]):
                print(f"loop{u + 1} " + " ".join(f"{v:5.2f}" for v in line[::2]))
    print("best sites (chosen on val):")
    for k, v in best.items():
        print(f"  {k:22s} {v['site']:10s} all {v['all']:.3f}  correct {v['ouro_correct']:.3f}  wrong {v['ouro_wrong']:.3f}"
              f"  novel {v['step1_novel']:.3f}  (loop1 layer0 {v['loop1_layer0_same_pos']:.3f})")
    if a.smoke_n:
        print(f"smoke run -- not logging ({extract_elapsed:.0f}s extract, {probe_elapsed:.0f}s probes)")
        return

    keep = sorted({v["site"] for v in best.values()} | {f"{pos}/{a.loops - 1}_{n_layers - 1}" for pos in POSITIONS})
    rows = []
    for i, (q, t) in enumerate(ev_q):
        row = {"key": q["key"], "ouro_correct": bool(ouro_ok[i]), **{k: t[k] for k in TARGETS},
               "step1_in_question": t["step1_in_question"], "chain_len": t["chain_len"], "pred": {}}
        for s in keep:
            pr, pc = best_pred[s]
            row["pred"][s] = {**{f"{k}_slog": round(float(pr[i, j]), 4) for j, k in enumerate(REG_TARGETS)},
                              **{f"{k}_digit": int(pc[i, j]) for j, k in enumerate(TARGETS)}}
        rows.append(row)

    record = RunRecord(
        run_id=new_run_id("recurrent_depth", a.slug), mechanism="recurrent_depth", stage=a.stage,
        model=ModelInfo(backbone=a.model, checkpoint=a.model, n_params=sum(x.numel() for x in model.parameters())),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(ev_q), seed=a.seed),
        metrics={
            "compute_steps": a.loops,
            "extra": {
                "design": "linear (ridge) probes for intermediate step values on Ouro decoder-layer outputs at the last "
                          "question token and the '\\nAnswer: ' suffix tokens, per (position, loop, layer); fit on xmech "
                          "fit questions, lambda on a val split, reported on xmech eval (GSM8K test)",
                "site_definition": "position/u_l = output of decoder layer l in loop u (0-indexed)",
                "positions": list(POSITIONS), "targets": list(REG_TARGETS), "lambda_fracs": list(LAMBDA_FRACS),
                "n_fit": len(fit_idx), "n_val": len(val_idx), "n_eval": len(ev_q),
                "n_eval_subsets": {s: int(m.sum()) for s, m in subsets.items()},
                "majority_digit_acc_eval": majority, "target_log_corr_fit": target_corr,
                "step1_resid_coef": coef.flatten().tolist(),
                "best_sites_chosen_on_val": best, "grids_eval_all": grids,
                "per_site_file": "per_site.json (all subsets, val, lambdas)", "prefix_tokenization_mismatches": n_mismatch,
                "extract_elapsed_sec": extract_elapsed, "probe_elapsed_sec": probe_elapsed,
                "related_runs": ["20261014-100712_recurrent_depth_gate-ouro-1.4b",
                                 "20261014-163349_recurrent_depth_twin-patch-ouro-1.4b"],
            },
        },
        hyperparams={"n_fit_pool": n_fit_all, "n_shots": a.n_shots, "shot_seed": a.shot_seed, "suffix": SUFFIX,
                     "attn": a.attn, "probe": "ridge, z-scored features, slog targets; ridge classifier on ones digit"},
        seed=seed,
        hardware=f"{a.hardware} / {torch.cuda.get_device_name(0)}",
        notes="Ouro step 2: is the intermediate step value linearly present at the question end / answer suffix?",
    )
    manifest = record.save(predictions=rows)
    (manifest.parent / "per_site.json").write_text(json.dumps(results))
    (manifest.parent / "eval_command.txt").write_text(" ".join(sys.argv) + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}")


if __name__ == "__main__":
    main()
