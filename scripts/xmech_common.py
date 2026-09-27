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
    from ladder_common import LEVELS, targets  # noqa: E402
    pairs, key_of = _ladder_pairs(questions, tgt, src, n_recipients, pair_seed)
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

    conds = ("own_carriers", "mapped_carriers", "shuffled_carriers", "mapped_all")
    return rows, _summarize(rows, conds, "self_mapped_carriers", reps, pair_seed)


def _ladder_pairs(questions: list[dict], tgt: dict, src: dict, n_recipients: int, pair_seed: int):
    """P1 ladder pairs on the eval split; recipients and donors base-correct in BOTH mechanisms."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ladder_common import build_ladder_pairs  # noqa: E402
    from latentreasoning.data.gsm8k_aug import Example  # noqa: E402

    evq = [q for q in questions if q["split"] == "eval"]
    examples = [Example(question=q["question"], rationale=q["rationale"], answer=q["answer"], idx=i)
                for i, q in enumerate(evq)]
    key_of = {i: q["key"] for i, q in enumerate(evq)}
    correct = {i: tgt["rows"][key_of[i]]["correct"] and src["rows"][key_of[i]]["correct"] for i in key_of}
    return build_ladder_pairs(examples, correct, random.Random(pair_seed), n_recipients), key_of


def _summarize(rows: list[dict], conds, self_cond: str, reps: int, pair_seed: int) -> dict:
    from ladder_common import LEVELS, summarize  # noqa: E402
    rng = random.Random(pair_seed + 1)
    summary = {level: {c: summarize(rows, c, level, reps, rng) for c in conds} for level in LEVELS}
    n = len(rows)
    self_unchanged = sum(float(r[self_cond].replace(",", "")) == float(r["base"].replace(",", ""))
                         for r in rows if _is_num(r[self_cond]) and _is_num(r["base"]))
    summary[f"{self_cond}_unchanged_rate"] = self_unchanged / n if n else None
    return summary


# ---- value-subspace restriction (DAS rotations from das_minimal_pair_*.py --save_rotations) ----
# A saved rotation W [768, 768] maps z -> W z (nn.Linear); the value subspace is its first k rows.

def load_subspace(path: str | Path) -> tuple[torch.Tensor, list[int], int]:
    """(Wk [k, 768] float64, DAS site group, k)."""
    r = torch.load(path)
    return r["W"][: r["k"]].double(), list(r["group"]), r["k"]


def random_subspace(k: int, d: int = 768, seed: int = 0) -> torch.Tensor:
    """[k, d] orthonormal rows, uniformly random (QR of a Gaussian)."""
    q, _ = torch.linalg.qr(torch.randn(d, k, generator=torch.Generator().manual_seed(seed), dtype=torch.float64))
    return q.T.contiguous()


def complement_basis(Wk: torch.Tensor) -> torch.Tensor:
    """[d-k, d] orthonormal basis of the orthogonal complement of Wk's row space."""
    q, _ = torch.linalg.qr(torch.cat([Wk.T, torch.randn(Wk.shape[1], Wk.shape[1] - Wk.shape[0],
                                      generator=torch.Generator().manual_seed(0), dtype=Wk.dtype)], dim=1))
    return q[:, Wk.shape[0]:].T.contiguous()


def coords(z: torch.Tensor, B: torch.Tensor, sites: list[int]) -> torch.Tensor:
    """z [n, 6, 768] -> [n, len(sites) * rows(B)]: B-coordinates at `sites`, concatenated."""
    return torch.cat([z[:, s, :].double() @ B.T for s in sites], dim=1)


def subspace_patcher(Wk: torch.Tensor, c_new: torch.Tensor):
    """f(live vector) -> the live vector with its Wk-coordinates set to c_new, the rest (the
    orthogonal complement) left as the recipient's own live value -- the DAS interchange."""
    def f(live: torch.Tensor) -> torch.Tensor:
        W = Wk.to(device=live.device, dtype=torch.float32)
        x = live.float().reshape(-1)
        out = x + (c_new.to(device=live.device, dtype=torch.float32) - W @ x) @ W
        return out.reshape(live.shape).to(live.dtype)
    return f


def fit_coord_maps(src_c: torch.Tensor, tgt_z: torch.Tensor, Wk_tgt: torch.Tensor, tgt_sites: list[int],
                   shuffle: bool = False, seed: int = 0) -> tuple[dict[int, RidgeMap], dict]:
    """Ridge from source features src_c [n, d_in] to the target's Wk-coordinates at each target site."""
    X = src_c
    if shuffle:
        X = X[torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(seed + 7))]
    maps, info = {}, {}
    for p in tgt_sites:
        maps[p], info[p] = fit_map(X, tgt_z[:, p, :].double() @ Wk_tgt.T, seed)
    return maps, info


def run_subspace_transplant(questions: list[dict], tgt: dict, src: dict, Wk_tgt: torch.Tensor, Wk_src: torch.Tensor,
                            src_sites: list[int], tgt_sites: list[int], answer_with, n_recipients: int,
                            pair_seed: int, reps: int, log_every: int = 25):
    """P4 restricted to the DAS value subspace, on the same ladder recipients/donors as `run_transplant`.

    Every condition writes k coordinates into the TARGET's value subspace (Wk_tgt) at `tgt_sites`
    and leaves the recipient's own live complement alone (subspace_patcher):
      own_sub         the target's own donor coordinates (cross-problem DAS interchange; ceiling)
      mapped_sub      ridge(source donor's Wk_src coordinates at src_sites) -> target coordinates
      shuffled_sub    the same ridge fit on permuted pairs (control)
      complement_sub  ridge(source donor's COMPLEMENT coordinates at src_sites) -> target coordinates
                      (is the value also readable from outside the source's value subspace?)
      random_sub      the target's own donor coordinates in a random k-dim subspace (floor)
      mapped_full     the unrestricted P4 condition (map of all 6 source sites -> whole target vectors
                      at tgt_sites), re-run here for a same-recipient comparison
    plus once per recipient `self_mapped_sub`: mapped_sub from the RECIPIENT's own source latents."""
    from ladder_common import LEVELS, targets  # noqa: E402
    fit_keys, eval_keys = split_keys(questions, "fit"), split_keys(questions, "eval")
    k = Wk_tgt.shape[0]
    Wc_src = complement_basis(Wk_src)
    Wr_tgt = random_subspace(k, Wk_tgt.shape[1], seed=pair_seed)
    src_fit, tgt_fit = stack(src, fit_keys), stack(tgt, fit_keys)
    src_ev, tgt_ev = stack(src, eval_keys), stack(tgt, eval_keys)

    feats = {"sub": lambda z: coords(z, Wk_src, src_sites), "comp": lambda z: coords(z, Wc_src, src_sites)}
    maps = {"mapped_sub": fit_coord_maps(feats["sub"](src_fit), tgt_fit, Wk_tgt, tgt_sites),
            "shuffled_sub": fit_coord_maps(feats["sub"](src_fit), tgt_fit, Wk_tgt, tgt_sites, shuffle=True),
            "complement_sub": fit_coord_maps(feats["comp"](src_fit), tgt_fit, Wk_tgt, tgt_sites)}
    feat_of = {"mapped_sub": "sub", "shuffled_sub": "sub", "complement_sub": "comp"}
    full_maps, _ = fit_site_maps(src_fit, tgt_fit, tgt_sites)
    fit_info = {}
    for name, (m, info) in maps.items():
        Xe = feats[feat_of[name]](src_ev)
        fit_info[name] = {str(p): {"lambda": info[p]["lambda"],
                                   "eval_r2": r2(m[p](Xe), tgt_ev[:, p, :].double() @ Wk_tgt.T)} for p in tgt_sites}

    def mapped_coords(name: str, key: str) -> dict:
        x = feats[feat_of[name]](src["rows"][key]["z"].unsqueeze(0))
        return {p: maps[name][0][p](x)[0] for p in tgt_sites}

    def patch(W: torch.Tensor, cs: dict) -> dict:
        return {p: subspace_patcher(W, c) for p, c in cs.items()}

    def own(W: torch.Tensor, key: str) -> dict:
        return {p: W @ tgt["rows"][key]["z"][p].double() for p in tgt_sites}

    pairs, key_of = _ladder_pairs(questions, tgt, src, n_recipients, pair_seed)
    rows = []
    for pi, pr in enumerate(pairs):
        r = pr["recipient"]
        rk = key_of[r.idx]
        row = {"recipient_key": rk, "recipient_answer": r.answer, "base": tgt["rows"][rk]["pred"],
               "self_mapped_sub": answer_with(r.question, patch(Wk_tgt, mapped_coords("mapped_sub", rk)))}
        for level in LEVELS:
            d = pr[level]
            dk = key_of[d.idx]
            x_full = src["rows"][dk]["z"].double().reshape(1, -1)
            preds = {"own_sub": answer_with(r.question, patch(Wk_tgt, own(Wk_tgt, dk))),
                     **{name: answer_with(r.question, patch(Wk_tgt, mapped_coords(name, dk))) for name in maps},
                     "random_sub": answer_with(r.question, patch(Wr_tgt, own(Wr_tgt, dk))),
                     "mapped_full": answer_with(r.question, {p: full_maps[p](x_full)[0].float() for p in tgt_sites})}
            row[level] = {"donor_key": dk, "donor_answer": d.answer, "targets": targets(r, d), "preds": preds}
        rows.append(row)
        if (pi + 1) % log_every == 0:
            print(f"transplanted {pi + 1}/{len(pairs)} recipients", flush=True)
    conds = ("own_sub", "mapped_sub", "shuffled_sub", "complement_sub", "random_sub", "mapped_full")
    summary = _summarize(rows, conds, "self_mapped_sub", reps, pair_seed)
    summary["coord_map_fit"] = fit_info
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
