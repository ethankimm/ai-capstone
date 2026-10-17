"""Shared pieces of the answer-phase KV-eviction runs (evict_kv_codi.py, evict_kv_coconut.py):
the attention-mask hook, the condition grid, and the paired summary.

Eviction = an additive -inf on chosen key columns (the thought slots) at chosen layers, applied
only to queries at or after `ans_start` (the first answer-phase position: CODI's eot, Coconut's
<|end-latent|>). Thoughts themselves are computed unmasked. Slot i (1-indexed) is the i-th thought
position in the sequence; in both methods slot 1 holds the vector derived from the token just
before the thoughts (CODI bot / Coconut <|start-latent|>).
"""
from __future__ import annotations

import torch

from necessity_common import mcnemar_exact_p, wilson_ci

N_LAYERS = 12
BANDS = {"L0-3": range(0, 4), "L4-7": range(4, 8), "L8-11": range(8, 12)}


def conditions(n_slots: int, keep_sets: dict[str, set[int]]) -> dict[str, dict[int, set[int]] | None]:
    """name -> {layer: set of slots evicted at that layer}; None = merge_mean (special)."""
    all_slots = set(range(1, n_slots + 1))
    every = lambda slots: {l: set(slots) for l in range(N_LAYERS)}  # noqa: E731
    c: dict[str, dict[int, set[int]] | None] = {"full": {}, "evict_all": every(all_slots)}
    for i in all_slots:
        c[f"keep_only_{i}"] = every(all_slots - {i})
    for i in all_slots:
        c[f"evict_only_{i}"] = every({i})
    for name, keep in keep_sets.items():
        c[name] = every(all_slots - keep)
    for b, layers in BANDS.items():
        c[f"evict_band_{b}"] = {l: set(all_slots) for l in layers}
    for b, layers in BANDS.items():
        c[f"keep_band_{b}"] = {l: set(all_slots) for l in range(N_LAYERS) if l not in layers}
    c["merge_mean"] = None
    return c


class AttnMasker:
    """Forward pre-hooks on every GPT2Attention: while `cols` is set ({layer: absolute key
    positions}), add -inf to those key columns for every query at absolute position >= ans_start."""

    def __init__(self, model):
        self.cols: dict[int, list[int]] = {}
        self.ans_start = 0
        self.n_hooked = 0
        for mod in model.modules():
            if type(mod).__name__ == "GPT2Attention":
                mod.register_forward_pre_hook(self._hook, with_kwargs=True)
                self.n_hooked += 1

    def _hook(self, mod, args, kwargs):
        cols = self.cols.get(mod.layer_idx)
        if not cols:
            return None
        hs = args[0] if args else kwargs["hidden_states"]
        q_len = hs.shape[1]
        past = kwargs.get("past_key_value")
        k_len = (past.get_seq_length(mod.layer_idx) if past is not None else 0) + q_len
        neg = torch.finfo(hs.dtype).min
        q_pos = torch.arange(k_len - q_len, k_len, device=hs.device)
        mask = torch.zeros(1, 1, q_len, k_len, dtype=hs.dtype, device=hs.device)
        mask[0, 0, (q_pos >= self.ans_start).nonzero(as_tuple=True)[0].unsqueeze(1), torch.tensor(cols, device=hs.device)] = neg
        prev = kwargs.get("attention_mask")
        if prev is not None:
            mask = torch.clamp(prev[..., :k_len] + mask, min=neg)
        elif q_len > 1:  # the model left causality to SDPA's is_causal; an explicit mask turns that off
            causal = torch.ones(q_len, k_len, dtype=torch.bool, device=hs.device).tril(k_len - q_len)
            mask = mask.masked_fill(~causal, neg)
        kwargs["attention_mask"] = mask
        return args, kwargs


def merge_slots(pkv, positions: list[int]) -> None:
    """In place: per layer, average the K/V at `positions` into positions[0] (legacy tuple cache)."""
    for k, v in pkv:
        for t in (k, v):
            t[:, :, positions[0]] = t[:, :, positions].mean(dim=2)


def summarize(preds: list[dict], names, pred_key) -> dict:
    """Per condition: accuracy + Wilson CI, agreement with full, paired flips + McNemar vs full.
    `pred_key(output_str)` -> extracted answer used for agreement."""
    n = len(preds)
    summary = {}
    for name in names:
        k = sum(r["correct"][name] for r in preds)
        same = sum(pred_key(r["outputs"][name]) == pred_key(r["outputs"]["full"]) for r in preds)
        b = sum(r["correct"]["full"] and not r["correct"][name] for r in preds)
        c = sum(r["correct"][name] and not r["correct"]["full"] for r in preds)
        lo, hi = wilson_ci(k, n)
        summary[name] = {"acc": k / n, "ci95": [lo, hi], "agree_with_full": same / n,
                         "lost": b, "gained": c, "mcnemar_p_vs_full": mcnemar_exact_p(b, c) if name != "full" else 1.0}
    print(f"\n{'condition':<20}{'acc':>7}{'95% CI':>17}{'agree':>8}{'lost':>6}{'gain':>6}{'p':>10}")
    for name, s in summary.items():
        print(f"{name:<20}{s['acc']:>7.3f}   [{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}]{s['agree_with_full']:>8.3f}"
              f"{s['lost']:>6}{s['gained']:>6}{s['mcnemar_p_vs_full']:>10.2g}")
    return summary
