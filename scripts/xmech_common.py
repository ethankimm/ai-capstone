"""P4 (RESEARCH_PLAN §4): cross-mechanism latent maps, shared by `xmech_codi.py` (CODI venv) and
`xmech_coconut.py` (Coconut venv). torch + stdlib only, so it imports in both venvs.

A latent dump is `{"meta": {...}, "rows": {key: {"z": FloatTensor[6, 768], "pred": str|None,
"correct": bool}}}` saved with `torch.save`; keys come from the shared question file
(`xmech_questions.jsonl`, built by `xmech_codi.py questions`), so both mechanisms saw the
identical question text.

Maps: ridge regression from ALL six source sites concatenated (6*768) to one target site
(768), per target site, fit on the `fit` split with lambda chosen on a held-out 20% of it
(best mean R^2). `shuffled` = the same fit with source rows permuted against target rows
(a map that has seen the right marginals but no pairing).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch

LAMBDAS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)


def read_questions(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def stack(dump: dict, keys: list[str]) -> torch.Tensor:
    """[n, 6, 768] float64."""
    return torch.stack([dump["rows"][k]["z"].double() for k in keys])


class RidgeMap:
    def __init__(self, W: torch.Tensor, x_mu: torch.Tensor, x_sd: torch.Tensor, y_mu: torch.Tensor, lam: float):
        self.W, self.x_mu, self.x_sd, self.y_mu, self.lam = W, x_mu, x_sd, y_mu, lam

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d_in] -> [..., d_out] (float64)."""
        return ((x.double() - self.x_mu) / self.x_sd) @ self.W + self.y_mu


def _fit(X: torch.Tensor, Y: torch.Tensor, lam: float) -> RidgeMap:
    x_mu, y_mu = X.mean(0), Y.mean(0)
    x_sd = X.std(0).clamp_min(1e-6)
    Xs = (X - x_mu) / x_sd
    A = Xs.T @ Xs + lam * torch.eye(Xs.shape[1], dtype=Xs.dtype)
    W = torch.linalg.solve(A, Xs.T @ (Y - y_mu))
    return RidgeMap(W, x_mu, x_sd, y_mu, lam)


def r2(pred: torch.Tensor, Y: torch.Tensor) -> float:
    ss_res = ((Y - pred) ** 2).sum()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def fit_map(X: torch.Tensor, Y: torch.Tensor, seed: int = 0) -> tuple[RidgeMap, dict]:
    """Choose lambda on a held-out 20% of (X, Y), refit on all of it."""
    n = X.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = max(1, n // 5)
    va, tr = perm[:n_val], perm[n_val:]
    scores = {lam: r2(_fit(X[tr], Y[tr], lam)(X[va]), Y[va]) for lam in LAMBDAS}
    best = max(scores, key=scores.get)
    return _fit(X, Y, best), {"lambda": best, "val_r2_by_lambda": {str(k): v for k, v in scores.items()}}


def fit_site_maps(src: torch.Tensor, tgt: torch.Tensor, target_sites: list[int], shuffle: bool = False,
                  seed: int = 0) -> tuple[dict[int, RidgeMap], dict]:
    """src, tgt: [n, 6, 768]. One map per target site from all source sites concatenated."""
    X = src.reshape(src.shape[0], -1)
    if shuffle:
        X = X[torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(seed + 7))]
    maps, info = {}, {}
    for p in target_sites:
        maps[p], info[p] = fit_map(X, tgt[:, p, :], seed)
    return maps, info


def eval_r2(maps: dict[int, RidgeMap], src: torch.Tensor, tgt: torch.Tensor) -> dict[int, float]:
    X = src.reshape(src.shape[0], -1)
    return {p: r2(m(X), tgt[:, p, :]) for p, m in maps.items()}


def linear_cka(A: torch.Tensor, B: torch.Tensor) -> float:
    A = A - A.mean(0)
    B = B - B.mean(0)
    hsic = (A.T @ B).norm() ** 2
    return float(hsic / ((A.T @ A).norm() * (B.T @ B).norm()))


def cka_matrix(src: torch.Tensor, tgt: torch.Tensor) -> list[list[float]]:
    """[source site][target site] linear CKA over examples."""
    return [[linear_cka(src[:, s, :], tgt[:, p, :]) for p in range(tgt.shape[1])] for s in range(src.shape[1])]


def split_keys(questions: list[dict], split: str) -> list[str]:
    return [q["key"] for q in questions if q["split"] == split]


def shuffled_copy(xs: list, seed: int) -> list:
    ys = list(xs)
    random.Random(seed).shuffle(ys)
    return ys


def run_transplant(questions: list[dict], tgt: dict, src: dict, maps: dict, shuf_maps: dict, carriers: list[int],
                   answer_with, n_recipients: int, pair_seed: int, reps: int, log_every: int = 25):
    """Ladder-style transplant into the TARGET mechanism on the `eval` split.

    tgt / src: latent dumps. maps / shuf_maps: {target site: RidgeMap} for every target site.
    answer_with(question, {target site: FloatTensor[768]}) -> pred string (target model).
    Recipients and donors must be base-correct in BOTH mechanisms (the source latents should
    encode correct values; the target's own donor latents are the ceiling). Conditions per level:
      own_carriers      target's own donor latents at `carriers`
      mapped_carriers   map(source donor latents) at `carriers`
      shuffled_carriers shuffled-pair map at `carriers`
      mapped_all        map(source donor latents) at every target site
    plus, once per recipient, `self_mapped_carriers`: map(source RECIPIENT latents) -- a
    reconstruction check (a faithful map should leave the answer unchanged)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ladder_common import LEVELS, build_ladder_pairs, summarize, targets  # noqa: E402
    from latentreasoning.data.gsm8k_aug import Example  # noqa: E402

    evq = [q for q in questions if q["split"] == "eval"]
    examples = [Example(question=q["question"], rationale=q["rationale"], answer=q["answer"], idx=i)
                for i, q in enumerate(evq)]
    key_of = {i: q["key"] for i, q in enumerate(evq)}
    correct = {i: tgt["rows"][key_of[i]]["correct"] and src["rows"][key_of[i]]["correct"] for i in key_of}
    pairs = build_ladder_pairs(examples, correct, random.Random(pair_seed), n_recipients)
    n_sites = len(maps)

    def mapped(m: dict, key: str, sites) -> dict:
        x = src["rows"][key]["z"].double().reshape(1, -1)
        return {p: m[p](x)[0].float() for p in sites}

    rows = []
    for pi, p in enumerate(pairs):
        r = p["recipient"]
        rk = key_of[r.idx]
        row = {"recipient_key": rk, "recipient_answer": r.answer, "base": tgt["rows"][rk]["pred"],
               "self_mapped_carriers": answer_with(r.question, mapped(maps, rk, carriers))}
        for level in LEVELS:
            d = p[level]
            dk = key_of[d.idx]
            own = {s: tgt["rows"][dk]["z"][s].float() for s in carriers}
            preds = {"own_carriers": answer_with(r.question, own),
                     "mapped_carriers": answer_with(r.question, mapped(maps, dk, carriers)),
                     "shuffled_carriers": answer_with(r.question, mapped(shuf_maps, dk, carriers)),
                     "mapped_all": answer_with(r.question, mapped(maps, dk, range(n_sites)))}
            row[level] = {"donor_key": dk, "donor_answer": d.answer, "targets": targets(r, d), "preds": preds}
        rows.append(row)
        if (pi + 1) % log_every == 0:
            print(f"transplanted {pi + 1}/{len(pairs)} recipients", flush=True)

    rng = random.Random(pair_seed + 1)
    conds = ("own_carriers", "mapped_carriers", "shuffled_carriers", "mapped_all")
    summary = {level: {c: summarize(rows, c, level, reps, rng) for c in conds} for level in LEVELS}
    n = len(rows)
    self_unchanged = sum(r["self_mapped_carriers"] is not None and r["base"] is not None
                         and float(r["self_mapped_carriers"].replace(",", "")) == float(r["base"].replace(",", ""))
                         for r in rows if _is_num(r["self_mapped_carriers"]) and _is_num(r["base"]))
    summary["self_mapped_carriers_unchanged_rate"] = self_unchanged / n if n else None
    return rows, summary


def _is_num(s) -> bool:
    try:
        float(str(s).replace(",", ""))
        return True
    except (TypeError, ValueError):
        return False


def build_questions(out: str, n_fit: int, fit_seed: int) -> None:
    """Local, no model: fit = `n_fit` GSM8K-Aug train questions, eval = the full test split."""
    from latentreasoning.data.gsm8k_aug import load_gsm8k_aug
    rows = []
    for split, exs in (("fit", load_gsm8k_aug(split="train", n=n_fit, seed=fit_seed)),
                       ("eval", load_gsm8k_aug(split="test", n=None, seed=None))):
        rows += [{"key": f"{split}-{ex.idx}", "split": split, "question": ex.question,
                  "rationale": ex.rationale, "answer": ex.answer} for ex in exs]
    Path(out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} questions to {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="xmech_questions.jsonl")
    ap.add_argument("--n_fit", type=int, default=3000)
    ap.add_argument("--fit_seed", type=int, default=0)
    a = ap.parse_args()
    build_questions(a.out, a.n_fit, a.fit_seed)
