#!/usr/bin/env python3
"""Does the learned DAS value subspace linearly decode the step values? Model-free (torch only):
reads the P4 latent dumps (`xmech_codi.py --mode dump`, `xmech_coconut.py`) and the DAS rotations
(`das_minimal_pair_*.py --save_rotations`), fits ridge probes on the `fit` split (GSM8K-Aug train
questions) and reports them on the `eval` split (the test set). Examples are restricted to ones
the mechanism answers correctly, so the latents should hold the gold step values.

Per mechanism, per rationale step j (0-indexed; target = signed-log1p of step j's value), features
at the DAS group's sites, concatenated:
  das          the k learned DAS coordinates per site
  random       k coordinates of a random k-dim subspace per site (mean over `--n_random` draws)
  complement   the 768-k coordinates orthogonal to the DAS subspace per site
  full         the whole 768-dim vectors
  das@s        the DAS coordinates at a single site s
  random@s     k random coordinates at site s (one draw, seed 100) -- the matched control for das@s
  full@s       the whole 768-dim vector at site s
Metrics: held-out R^2 (log space), tol5 = |pred - gold| / max(1, |gold|) <= 0.05, exact = rounded
prediction equals the gold value -- all from the ridge probe, i.e. a LINEAR read of log-magnitude.
Two readouts that don't assume a number line: knn_exact = the nearest fit example (standardized
features, Euclidean) has the same step value; last_digit = linear softmax classifier on the value's
last integer digit (chance ~ majority_last_digit). Logs one run record per mechanism.

  uv run python scripts/das_subspace_probe.py --questions .../xmech_questions.jsonl \\
      --codi_latents .../codi_latents.pt --coconut_latents .../coconut_latents.pt \\
      --codi_rotation .../rot_0+2+4_k16.pt --coconut_rotation .../rot_1+4_k16.pt --stage full_run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xmech_common as xc  # noqa: E402
from probe_common import inv_signed_log1p, parse_value, signed_log1p, within_tol  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from latentreasoning.eval.counterfactual import parse_steps  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402

MODELS = {
    "codi": ModelInfo(backbone="gpt2", checkpoint="hf:zen-E/CODI-gpt2@fd641b3", n_params=144499200),
    "coconut": ModelInfo(backbone="openai-community/gpt2", checkpoint="hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33",
                         n_params=124442112),
}
N_STEPS = 3


def step_targets(questions: list[dict], dump: dict, split: str, j: int) -> tuple[list[str], torch.Tensor]:
    keys, ys = [], []
    for q in questions:
        if q["split"] != split or not dump["rows"][q["key"]]["correct"]:
            continue
        steps = parse_steps(q["rationale"])
        v = parse_value(steps[j]["val"]) if len(steps) > j else None
        if v is not None:
            keys.append(q["key"])
            ys.append(v)
    return keys, torch.tensor(ys, dtype=torch.float64)


def probe(X_fit, y_fit, X_ev, y_ev) -> dict:
    m, info = xc.fit_map(X_fit, signed_log1p(y_fit)[:, None])
    pred_log = m(X_ev)[:, 0]
    pred = inv_signed_log1p(pred_log)
    n = len(y_ev)
    return {"r2": xc.r2(pred_log[:, None], signed_log1p(y_ev)[:, None]),
            "tol5": sum(within_tol(float(p), float(g), 0.05) for p, g in zip(pred, y_ev)) / n,
            "exact": sum(round(float(p)) == float(g) for p, g in zip(pred, y_ev)) / n,
            "lambda": info["lambda"], "d_in": X_fit.shape[1]}


def knn_exact(X_fit, y_fit, X_ev, y_ev) -> float:
    mu, sd = X_fit.mean(0), X_fit.std(0).clamp_min(1e-6)
    nn = torch.cdist((X_ev - mu) / sd, (X_fit - mu) / sd).argmin(1)
    return float((y_fit[nn] == y_ev).double().mean())


def last_digit_acc(X_fit, y_fit, X_ev, y_ev, epochs: int = 300) -> float:
    torch.manual_seed(0)
    lf, le = y_fit.round().long().abs() % 10, y_ev.round().long().abs() % 10
    mu, sd = X_fit.mean(0), X_fit.std(0).clamp_min(1e-6)
    Xf, Xe = ((X_fit - mu) / sd).float(), ((X_ev - mu) / sd).float()
    lin = torch.nn.Linear(Xf.shape[1], 10)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(lin(Xf), lf).backward()
        opt.step()
    return float((lin(Xe).argmax(1) == le).float().mean())


def run_mechanism(mech: str, questions, dump, rot_path: str, n_random: int) -> dict:
    Wk, group, k = xc.load_subspace(rot_path)
    Wc = xc.complement_basis(Wk)
    Wr = xc.random_subspace(k, seed=100)
    feats = {"das": lambda z: xc.coords(z, Wk, group),
             "complement": lambda z: xc.coords(z, Wc, group),
             "full": lambda z: z[:, group, :].double().reshape(z.shape[0], -1),
             **{f"das@{s}": (lambda z, s=s: xc.coords(z, Wk, [s])) for s in group},
             **{f"random@{s}": (lambda z, s=s: xc.coords(z, Wr, [s])) for s in group},
             **{f"full@{s}": (lambda z, s=s: z[:, s, :].double()) for s in group}}
    out = {"group": group, "k": k, "rotation": rot_path, "steps": {}}
    for j in range(N_STEPS):
        kf, yf = step_targets(questions, dump, "fit", j)
        ke, ye = step_targets(questions, dump, "eval", j)
        zf, ze = xc.stack(dump, kf), xc.stack(dump, ke)
        def readouts(Xf, Xe) -> dict:
            return {**probe(Xf, yf, Xe, ye), "knn_exact": knn_exact(Xf, yf, Xe, ye),
                    "last_digit": last_digit_acc(Xf, yf, Xe, ye)}
        res = {name: readouts(f(zf), f(ze)) for name, f in feats.items()}
        rand = [readouts(xc.coords(zf, W, group), xc.coords(ze, W, group))
                for W in (xc.random_subspace(k, seed=100 + i) for i in range(n_random))]
        res["random"] = {m: sum(r[m] for r in rand) / n_random for m in ("r2", "tol5", "exact", "knn_exact", "last_digit")}
        res["random"]["max_tol5"] = max(r["tol5"] for r in rand)
        res["random"]["max_knn_exact"] = max(r["knn_exact"] for r in rand)
        lf = yf.round().long().abs() % 10
        res["majority_last_digit"] = float(((ye.round().long().abs() % 10) == torch.bincount(lf).argmax()).float().mean())
        res["n_fit"], res["n_eval"] = len(yf), len(ye)
        out["steps"][str(j)] = res
        print(f"[{mech}] step {j} (n_fit={len(yf)}, n_eval={len(ye)}):")
        for name in ("das", "random", "complement", "full") + tuple(f"{f}@{s}" for s in group
                                                                     for f in ("das", "random", "full")):
            r = res[name]
            print(f"  {name:12s} R2={r['r2']:.3f} tol5={r['tol5']:.3f} exact={r['exact']:.3f} "
                  f"knn_exact={r['knn_exact']:.3f} last_digit={r['last_digit']:.3f}")
        print(f"  majority last digit {res['majority_last_digit']:.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--codi_latents", required=True)
    ap.add_argument("--coconut_latents", required=True)
    ap.add_argument("--codi_rotation", required=True)
    ap.add_argument("--coconut_rotation", required=True)
    ap.add_argument("--n_random", type=int, default=5)
    ap.add_argument("--stage", default="pilot")
    ap.add_argument("--slug", default="das-subspace-probe")
    ap.add_argument("--no_log", action="store_true")
    a = ap.parse_args()
    questions = xc.read_questions(a.questions)
    for mech, lat, rot in (("codi", a.codi_latents, a.codi_rotation), ("coconut", a.coconut_latents, a.coconut_rotation)):
        dump = torch.load(lat)
        res = run_mechanism(mech, questions, dump, rot, a.n_random)
        if a.no_log:
            continue
        eval_keys = xc.split_keys(questions, "eval")
        record = RunRecord(
            run_id=new_run_id(mech, a.slug), mechanism=mech, stage=a.stage, model=MODELS[mech],
            dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=res["steps"]["0"]["n_eval"], seed=None),
            metrics={
                "final_answer_accuracy": sum(dump["rows"][k]["correct"] for k in eval_keys) / len(eval_keys),
                "decoding_accuracy": res["steps"]["0"]["das"]["knn_exact"],
                "extra": {"design": "ridge probes (signed-log1p step value) on the learned DAS value subspace vs "
                                    "random k-dim subspaces, its complement and the full vectors; fit = GSM8K-Aug "
                                    "train questions, eval = test; model-correct examples only",
                          "decoding_accuracy_definition": "step-0 knn_exact on the DAS-subspace coordinates",
                          **res,
                          "related_runs": ["20260927-094215_codi_das-minimal-pair-fixed",
                                           "20260927-084449_coconut_das-minimal-pair-fixed"]},
            },
            hyperparams={"n_random": a.n_random, "lambdas": list(xc.LAMBDAS), "n_steps": N_STEPS},
            seed=0, hardware="local-cpu",
            notes=f"Linear decodability of step values from the learned {mech} DAS subspace.",
        )
        manifest = record.save(predictions=[])
        (manifest.parent / "eval_command.txt").write_text(" ".join(sys.argv) + "\n")
        print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}")


if __name__ == "__main__":
    main()
