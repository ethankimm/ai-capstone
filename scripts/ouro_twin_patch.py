"""Ouro step 1 (Q4 extension to a looped model): WHERE does Ouro carry the value a
same-problem twin changes? Q1-style minimal-pair interchange patching over (loop, layer).

Ouro has no latent slots: all computation lives in the residual stream of the prompt
tokens. Sites are the decoder-layer outputs at the answer-prompt suffix tokens
(`\\nAnswer: ` -- 4 tokens incl. the space Ouro always emits before an answer's digits;
tokenized separately and appended after the question, so it is identical for recipient
and donor), after layer l in loop u, u = 0..3 at the trained 4 loops. Everything before the
suffix is the recipient's own few-shot prefix + question, computed once and cached, so the
donor's number can only enter through the patched suffix positions.

Scoring is teacher-forced, not greedy: Ouro's tokenizer splits numbers into single digits,
so a multi-digit answer is produced over several positions that each re-read the
recipient's question; the greedy answer barely moves under a suffix patch (smoke test
2026-10-14: 0 changes at all 96 sites; the first generated token is always " ", so the
suffix's last state only ever picked that). Per patched row we score the twin answer
(digits + "\\n", after the suffix's space) and the recipient's own answer under the patch:
  ld = logp(twin answer) - logp(recipient answer)
  restored = (ld_patched - ld_recipient) / (ld_twin_own - ld_recipient)   0 = no effect, 1 = full twin
  flips = ld_patched > 0 (the twin answer now beats the recipient's)
The cross-problem control does the same with another recipient's answer as the target.

Donors, per recipient (base-correct at 4 loops under this script's tokenization, greedy):
  twin    the recipient's own question with one chain-operand number perturbed and the
          gold chain re-executed (`latentreasoning.data.minimal_pairs`, as in Q1/E3); twin
          answered correctly too (responsive pair).
  cross   another qualified recipient's question with a different gold answer.
Conditions per site (u, l): twin_all (all suffix positions), twin_last (last position),
cross_all.

Patched rows run batched: every row has the same tokens (suffix + one candidate answer), so
there is no padding (Ouro's remote code breaks under left padding), and each batch shares
the recipient's cached prefix via expanded views (the cache's update() concatenates into new
tensors, so the prefill cache is never mutated).

Usage (pod, Ouro venv, from the repo):
  python scripts/ouro_twin_patch.py --questions xmech_questions.jsonl \\
      --gate-predictions results/<1.4B gate run>/predictions.jsonl --n-pairs 150
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ouro_gate import build_prefix, read_questions, set_loops  # noqa: E402

from latentreasoning.data.gsm8k_aug import Example  # noqa: E402
from latentreasoning.data.minimal_pairs import generate_minimal_pair  # noqa: E402
from latentreasoning.eval.counterfactual import parse_steps, qualifying_steps  # noqa: E402
from latentreasoning.eval.metrics import extract_final_number, is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402
from latentreasoning.utils import set_seed  # noqa: E402

DELTAS = (1, -1, 2, -2, 3, -3)
SUFFIX = "\nAnswer: "
CONDS = ("twin_all", "twin_last", "cross_all")


class SitePatcher:
    """Forward hooks on every decoder layer. Ouro calls each layer once per loop with a
    `current_ut` kwarg, so site = (current_ut, layer index). `rows[b]` is row b's patch
    {(u, l): (positions, tensor[len(positions), d])} or None; positions index the current
    input chunk. `capture` stores row 0's states at positions [:n_capture]."""

    def __init__(self, layers):
        self.rows: list | None = None
        self.capture: dict | None = None
        self.n_capture = 0
        for li, layer in enumerate(layers):
            layer.register_forward_hook(self._hook(li), with_kwargs=True)

    def _hook(self, li):
        def fn(module, args, kwargs, output):
            if self.rows is None and self.capture is None:
                return None
            u = kwargs.get("current_ut")
            hs = output[0] if isinstance(output, tuple) else output
            if self.capture is not None:
                self.capture[(u, li)] = hs[0, : self.n_capture].detach().clone()
            if self.rows is not None:
                hit = [(b, pd[(u, li)]) for b, pd in enumerate(self.rows) if pd is not None and (u, li) in pd]
                if hit:
                    hs = hs.clone()
                    for b, (pos, vec) in hit:
                        hs[b, pos] = vec.to(hs.dtype)
                    return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
            return None
        return fn


def _inputs(model, ids: torch.Tensor, past: int) -> dict:
    """Explicit positions and a 4-D additive causal mask over [cached + new] keys, passed as
    the per-layer-type dict: the remote code's create_causal_mask path can't size Ouro's
    UniversalTransformerCache and builds a mask without the cached keys."""
    b, n = ids.shape
    pos = torch.arange(past, past + n, device=model.device)
    keys = torch.arange(past + n, device=model.device)
    mask = torch.zeros(b, 1, n, past + n, dtype=model.dtype, device=model.device)
    mask.masked_fill_(keys[None, :] > pos[:, None], torch.finfo(model.dtype).min)
    return {"input_ids": ids, "attention_mask": {"full_attention": mask, "sliding_attention": mask},
            "position_ids": pos[None].expand(b, -1), "cache_position": pos}


@torch.no_grad()
def prefill(model, ids: list[int]):
    out = model(**_inputs(model, torch.tensor([ids], device=model.device), 0), use_cache=True)
    return out.past_key_values, len(ids)


def _expanded(cache, b: int):
    """A fresh cache object whose entries are batch-expanded views of `cache`'s."""
    new = type(cache)(cache.max_cache_size)
    new.key_cache = [None if k is None else k.expand(b, -1, -1, -1) for k in cache.key_cache]
    new.value_cache = [None if v is None else v.expand(b, -1, -1, -1) for v in cache.value_cache]
    new._seen_tokens = cache._seen_tokens
    return new


@torch.no_grad()
def greedy(model, tok, P: SitePatcher, pre, suffix_ids: list[int], capture: bool = False, max_new: int = 8):
    """Unpatched greedy answer from a prefilled (cache, past). Returns (pred, raw, captured
    suffix states {(u, l): [n_suffix, d]} or None)."""
    cache, past = _expanded(pre[0], 1), pre[1]
    ids = torch.tensor([suffix_ids], device=model.device)
    out_tokens: list[int] = []
    captured = None
    for t in range(max_new):
        if t == 0 and capture:
            P.capture, P.n_capture = {}, len(suffix_ids)
        o = model(**_inputs(model, ids, past), past_key_values=cache, use_cache=True)
        if t == 0 and capture:
            captured, P.capture = P.capture, None
        past += ids.shape[1]
        cache = o.past_key_values
        nxt = int(o.logits[0, -1].argmax())
        out_tokens.append(nxt)
        if "\n" in tok.decode(out_tokens) or nxt == tok.eos_token_id:
            break
        ids = torch.tensor([[nxt]], device=model.device)
    raw = tok.decode(out_tokens)
    return extract_final_number(raw.split("\n")[0]), raw, captured


@torch.no_grad()
def score(model, P: SitePatcher, pre, suffix_ids: list[int], cont_ids: list[int], patches: list,
          bs: int) -> torch.Tensor:
    """Sum log-prob of `cont_ids` after `suffix_ids`, one row per entry of `patches`."""
    seq = suffix_ids + cont_ids
    tgt = torch.tensor(cont_ids, device=model.device)
    first = len(suffix_ids) - 1  # logits at this position predict cont_ids[0]
    out = []
    for i in range(0, len(patches), bs):
        chunk = patches[i:i + bs]
        b = len(chunk)
        ids = torch.tensor([seq] * b, device=model.device)
        P.rows = chunk
        o = model(**_inputs(model, ids, pre[1]), past_key_values=_expanded(pre[0], b), use_cache=True)
        P.rows = None
        lp = torch.log_softmax(o.logits[:, first:first + len(cont_ids)].float(), dim=-1)
        out.append(lp.gather(-1, tgt[None, :, None].expand(b, -1, 1)).squeeze(-1).sum(-1))
    return torch.cat(out).cpu()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ByteDance/Ouro-1.4B")
    p.add_argument("--questions", required=True)
    p.add_argument("--gate-predictions", required=True, help="step-0 predictions.jsonl for this model")
    p.add_argument("--loops", type=int, default=4)
    p.add_argument("--n-pairs", type=int, default=150)
    p.add_argument("--max-candidates", type=int, default=600)
    p.add_argument("--n-shots", type=int, default=8)
    p.add_argument("--shot-seed", type=int, default=0)
    p.add_argument("--pair-seed", type=int, default=0)
    p.add_argument("--bs", type=int, default=48)
    p.add_argument("--attn", default="eager")
    p.add_argument("--smoke-n", type=int, default=0, help=">0: patch this many pairs, don't log")
    p.add_argument("--slug", default="twin-patch-ouro-1.4b")
    p.add_argument("--stage", default="pilot")
    p.add_argument("--hardware", default="RunPod")
    a = p.parse_args()
    seed = set_seed(a.pair_seed)
    rng = random.Random(a.pair_seed)

    qs = read_questions(a.questions)
    fit = [q for q in qs if q["split"] == "fit"]
    ev = {q["key"]: q for q in qs if q["split"] == "eval"}
    gate = [json.loads(line) for line in Path(a.gate_predictions).read_text().splitlines() if line.strip()]
    freq = collections.Counter(r[f"pred_l{a.loops}"] for r in gate)
    frequent = {v for v, _ in freq.most_common(10)}
    gate_ok = [r["key"] for r in gate if r[f"correct_l{a.loops}"]]

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16,
                                                 device_map="cuda", attn_implementation=a.attn).eval()
    set_loops(model, a.loops)
    assert not getattr(model.config, "use_sliding_window", False), "full causal mask assumed for all layers"
    layers = model.model.layers[: model.config.num_hidden_layers]
    n_layers = len(layers)
    P = SitePatcher(layers)

    prefix = build_prefix(fit, a.n_shots, a.shot_seed, cot=False)
    suffix_ids = tok(SUFFIX, add_special_tokens=False)["input_ids"]
    n_suf = len(suffix_ids)
    print(f"suffix {SUFFIX!r} -> {[tok.decode([i]) for i in suffix_ids]}; {n_layers} layers x {a.loops} loops")

    def ans_ids(ans: str) -> list[int]:
        """Answer digits + newline, as Ouro emits them after the suffix's space."""
        return tok(f"{ans}\n", add_special_tokens=False)["input_ids"]

    def pre_for(question: str):
        return prefill(model, tok(prefix + f"Question: {question}", add_special_tokens=False)["input_ids"])

    # ---- twins for gate-correct recipients; keep pairs where both are correct here ----------
    t0 = time.perf_counter()
    keys = list(gate_ok)
    rng.shuffle(keys)
    pairs, n_cand, n_rec_ok, cand_log = [], 0, 0, []
    for i, key in enumerate(keys):
        if len(pairs) >= a.n_pairs or n_cand >= a.max_candidates:
            break
        q = ev[key]
        ex = Example(question=q["question"], rationale=q["rationale"], answer=q["answer"], idx=i)
        steps = parse_steps(ex.rationale)
        mp = None
        for k in qualifying_steps(steps):
            for d in DELTAS:
                mp = generate_minimal_pair(ex, k, d, rng=rng)
                if mp is not None:
                    break
            if mp is not None:
                break
        if mp is None:
            continue
        n_cand += 1
        r_pred, _, r_cap = greedy(model, tok, P, pre_for(ex.question), suffix_ids, capture=True)
        r_ok = is_correct(r_pred, ex.answer)
        n_rec_ok += r_ok
        t_pre = pre_for(mp.twin.question) if r_ok else None
        t_pred, _, t_cap = greedy(model, tok, P, t_pre, suffix_ids, capture=True) if r_ok else (None, "", None)
        t_ok = r_ok and is_correct(t_pred, mp.twin.answer)
        cand_log.append({"key": key, "step": mp.step, "delta": mp.delta, "perturbed_number": mp.perturbed_number,
                         "chain_len": len(steps), "recipient_pred": r_pred, "recipient_correct": r_ok,
                         "twin_answer": mp.twin.answer, "twin_pred": t_pred, "twin_correct": t_ok})
        if t_ok and not is_correct(r_pred, mp.twin.answer):
            # twin's own preference: logp(twin) - logp(recipient answer) under the twin's prompt
            tw = ans_ids(mp.twin.answer)
            ra = ans_ids(ex.answer)
            ld_twin = float(score(model, P, t_pre, suffix_ids, tw, [None], 1)[0]
                            - score(model, P, t_pre, suffix_ids, ra, [None], 1)[0])
            pairs.append({"key": key, "answer": ex.answer, "twin_answer": mp.twin.answer, "step": mp.step,
                          "chain_len": len(steps), "r_cap": r_cap, "t_cap": t_cap, "ld_twin": ld_twin,
                          "clean": r_pred not in frequent and t_pred not in frequent})
        if n_cand % 50 == 0:
            print(f"candidates {n_cand}: recipient ok {n_rec_ok}, pairs {len(pairs)} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    twin_elapsed = time.perf_counter() - t0
    cfr = len(pairs) / max(1, n_rec_ok)
    print(f"{len(pairs)} responsive pairs from {n_cand} candidates ({n_rec_ok} recipients correct here); "
          f"CFR {cfr:.3f}; clean {sum(p['clean'] for p in pairs)}")
    if a.smoke_n:
        pairs = pairs[: a.smoke_n]

    # ---- patch grid ---------------------------------------------------------------------------
    sites = [(u, l) for u in range(a.loops) for l in range(n_layers)]
    allpos, last = list(range(n_suf)), [n_suf - 1]
    rows = []
    t1 = time.perf_counter()
    for pi, pr in enumerate(pairs):
        pre = pre_for(ev[pr["key"]]["question"])
        others = [d for d in pairs if d["key"] != pr["key"] and not is_correct(d["answer"], pr["answer"])]
        donor = rng.choice(others)
        ra, tw, da = ans_ids(pr["answer"]), ans_ids(pr["twin_answer"]), ans_ids(donor["answer"])

        # sanity on the first pair: own states patched back in at every site is a no-op
        if pi == 0:
            own = {s: (allpos, pr["r_cap"][s]) for s in sites}
            s0 = score(model, P, pre, suffix_ids, ra, [None, own], 2)
            assert abs(float(s0[0] - s0[1])) < 0.05, f"self-patch moved logp: {s0.tolist()}"
            print(f"self-patch sanity ok ({s0.tolist()})")

        twin_rows = ([{s: (allpos, pr["t_cap"][s])} for s in sites]
                     + [{s: (last, pr["t_cap"][s][last])} for s in sites])
        cross_rows = [{s: (allpos, donor["r_cap"][s])} for s in sites]
        base_rows = [None] + twin_rows + cross_rows
        lp_r = score(model, P, pre, suffix_ids, ra, base_rows, a.bs)       # recipient answer, all rows
        lp_t = score(model, P, pre, suffix_ids, tw, [None] + twin_rows, a.bs)
        lp_d = score(model, P, pre, suffix_ids, da, [None] + cross_rows, a.bs)
        ld0_t = float(lp_t[0] - lp_r[0])  # recipient's own preference twin vs own answer (negative)
        ld0_d = float(lp_d[0] - lp_r[0])
        n = len(sites)
        ld_t = (lp_t[1:] - lp_r[1:1 + 2 * n]).tolist()
        ld_d = (lp_d[1:] - lp_r[1 + 2 * n:]).tolist()
        denom = pr["ld_twin"] - ld0_t
        row = {"key": pr["key"], "answer": pr["answer"], "twin_answer": pr["twin_answer"],
               "cross_key": donor["key"], "cross_answer": donor["answer"], "step": pr["step"],
               "chain_len": pr["chain_len"], "clean": pr["clean"], "ld_recipient_twin": ld0_t,
               "ld_twin_own": pr["ld_twin"], "ld_recipient_cross": ld0_d}
        for ci, cond in enumerate(("twin_all", "twin_last")):
            vals = ld_t[ci * n:(ci + 1) * n]
            row[cond] = {f"{u}_{l}": {"ld": v, "flip": v > 0, "restored": (v - ld0_t) / denom if denom > 0 else None}
                         for (u, l), v in zip(sites, vals)}
        row["cross_all"] = {f"{u}_{l}": {"ld": v, "flip": v > 0, "restored": None} for (u, l), v in zip(sites, ld_d)}
        rows.append(row)
        if (pi + 1) % 10 == 0 or pi == 0:
            print(f"patched {pi + 1}/{len(pairs)} ({time.perf_counter() - t1:.0f}s)", flush=True)
    patch_elapsed = time.perf_counter() - t1

    def grid(rs, cond, key):
        out = []
        for u in range(a.loops):
            line = []
            for l in range(n_layers):
                vs = [r[cond][f"{u}_{l}"][key] for r in rs if r[cond][f"{u}_{l}"][key] is not None]
                line.append(sum(vs) / len(vs) if vs else None)
            out.append(line)
        return out

    summary = {}
    for subset, rs in (("all", rows), ("clean", [r for r in rows if r["clean"]])):
        for cond in CONDS:
            summary[f"{cond}/{subset}"] = {"n": len(rs), "flip": grid(rs, cond, "flip")}
            if cond != "cross_all":
                summary[f"{cond}/{subset}"]["restored"] = grid(rs, cond, "restored")
    for name, g in (("twin_all restored", summary["twin_all/all"]["restored"]),
                    ("twin_last restored", summary["twin_last/all"]["restored"]),
                    ("twin_all flip", summary["twin_all/all"]["flip"]),
                    ("cross_all flip (-> donor answer)", summary["cross_all/all"]["flip"])):
        print(f"--- {name} (n={len(rows)}; rows = loop, cols = layer 0,2,..,{n_layers - 2})")
        for u, line in enumerate(g):
            print(f"loop{u + 1} " + " ".join("  -  " if v is None else f"{v:5.2f}" for v in line[::2]))
    if a.smoke_n:
        print(f"smoke run -- not logging ({patch_elapsed:.0f}s for {len(rows)} pairs)")
        return

    record = RunRecord(
        run_id=new_run_id("recurrent_depth", a.slug), mechanism="recurrent_depth", stage=a.stage,
        model=ModelInfo(backbone=a.model, checkpoint=a.model, n_params=sum(p.numel() for p in model.parameters())),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(rows), seed=None),
        metrics={
            "compute_steps": a.loops,
            "intervention_accuracy": max(v for line in summary["twin_all/all"]["flip"] for v in line),
            "extra": {
                "design": "Q1-style same-problem twin interchange patching on Ouro; sites = decoder-layer outputs at "
                          "the '\\nAnswer: ' suffix positions per (loop, layer); teacher-forced scoring of twin vs "
                          "recipient answer; conditions twin_all / twin_last / cross_all (cross-problem donor)",
                "intervention_accuracy_definition": "max over sites of the twin_all flip rate",
                "site_definition": "site (u, l) = output of decoder layer l in loop u (0-indexed) at the suffix tokens",
                "suffix_ids": suffix_ids, "n_layers": n_layers, "loops": a.loops,
                "n_candidates": n_cand, "n_recipients_correct": n_rec_ok, "n_pairs": len(rows),
                "n_clean_pairs": sum(r["clean"] for r in rows),
                "cfr_twin_correct_given_recipient_correct": cfr,
                "frequent_outputs_excluded_in_clean": sorted(frequent),
                "grids": summary, "twin_elapsed_sec": twin_elapsed, "patch_elapsed_sec": patch_elapsed,
                "related_runs": ["20261014-100712_recurrent_depth_gate-ouro-1.4b"],
            },
        },
        hyperparams={"n_pairs_cap": a.n_pairs, "max_candidates": a.max_candidates, "deltas": list(DELTAS),
                     "n_shots": a.n_shots, "shot_seed": a.shot_seed, "suffix": SUFFIX, "attn": a.attn,
                     "scoring": "teacher-forced sum logp of answer digits + newline", "bs": a.bs},
        seed=seed,
        hardware=f"{a.hardware} / {torch.cuda.get_device_name(0)}",
        notes="Ouro step 1: where (loop, layer) does the suffix stream carry the twin-changed answer?",
    )
    manifest = record.save(predictions=rows)
    with (manifest.parent / "candidates.jsonl").open("w") as f:
        for c in cand_log:
            f.write(json.dumps(c, default=str) + "\n")
    (manifest.parent / "eval_command.txt").write_text(" ".join(sys.argv) + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}")


if __name__ == "__main__":
    main()
