#!/usr/bin/env python3
"""Inference-only ablation / swap battery for a Coconut checkpoint on synthetic depth2 (COCONUT_CODI_PLAN.md H1/H3/H4/H6).

Runs in the Coconut venv (like eval_synthetic_coconut.py). One checkpoint, one split (valid.json for development / selection,
test.json once for the final numbers). Conditions per item (all greedy, batch 1, using coconut_common.run_passes' splice hook
`override_at_pass`, so the SAME code path as intact inference):

  intact                  no intervention (== eval_synthetic_coconut.py --method coconut)
  trunc_k                 only k latent tokens (k = 0..N-1; the start/end markers are kept) -- truncation, secondary
  ablate_all_{zero,mean,noise}   replace every latent: zero / TRAIN-split mean per pass / norm-matched random direction
  swap_same_level         PRIMARY H1 intervention: every latent replaced by a same-level donor's (fixed derangement)
  swap_other_level        donor from a different level (portability tier)
  swap_slot_s             (--per_slot) only pass s replaced by the same-level donor's
  ablate_single_{zero,mean}_s   (--per_slot)
  pair_<tier>_k<k>        (--pairs) H4: latents 2k-2 and 2k-1 (pair k <-> step k, c_thought 2) taken from the donor of a
                          twins_v2 pair; twin donors are new items run through the model. Scored follows_donor /
                          carries_state / keeps_recipient / other.

Scoring per condition: correct vs gold; for swaps also follows_donor (pred == donor's gold, only when donor gold != recipient
gold), keeps_recipient, other. Summaries (per level and pooled over --window): accuracy, drop vs intact, rel_drop =
(intact - cond)/(intact - floor) with floor from --floor, PWC, flips, exact McNemar vs intact (scripts/necessity_common.py).
Everything is written to one RunRecord (extra.conditions) + predictions.jsonl, unless --no_log.

  .venv_coconut/bin/python scripts/ablate_synthetic_coconut.py --checkpoint_path /workspace/coconut_runs/NAME/checkpoint_7 \\
      --config_path /workspace/logs/configs/NAME.yaml --test_json /workspace/synthetic_data_v2/depth2/valid.json \\
      --mean_json /workspace/synthetic_data_v2/depth2/train.json --pairs --slug v2-depth2-coconutA-ablate --no_log
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coconut_common import (encode_question, extract_answer_after_delimiter, finish_and_decode,  # noqa: E402
                            load_coconut, run_passes)
from necessity_common import summarize, wilson_ci  # noqa: E402
from synthetic_common import assert_known_hash, load_synthetic  # noqa: E402

from latentreasoning.eval.metrics import is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402
from latentreasoning.tasks import twins_v2  # noqa: E402


@dataclass
class AblateArguments:
    checkpoint_path: str = field(metadata={"help": "Coconut run.py checkpoint_N"})
    test_json: str = field(metadata={"help": "synthetic depth2 split to run (valid.json or test.json)"})
    config_path: Optional[str] = field(default=None, metadata={"help": "training yaml (num_latents = c_thought x max_latent_stage)"})
    num_latents: int = field(default=-1)
    model_id: str = field(default="openai-community/gpt2")
    mean_json: Optional[str] = field(default=None, metadata={"help": "TRAIN split json for the mean latents (never the eval split)"})
    mean_n: int = field(default=500)
    levels: str = field(default="1-8", metadata={"help": "levels to run, e.g. 1-8"})
    window: str = field(default="1-5", metadata={"help": "levels pooled for the headline numbers"})
    floor: float = field(default=0.03, metadata={"help": "floor for rel_drop (per-level majority rate is better; sensitivity 0.05)"})
    per_level_n: int = field(default=0, metadata={"help": "0 = all items per level; else first n per level"})
    per_slot: bool = field(default=False)
    pairs: bool = field(default=False, metadata={"help": "H4/portability donor pairs from twins_v2 (built from this split)"})
    pairs_seed: int = field(default=0)
    perm_seed: int = field(default=0)
    max_new_tokens: int = field(default=16)
    slug: Optional[str] = field(default=None)
    stage: str = field(default="full_run")
    hardware: str = field(default="RunPod GPU")
    smoke_n: int = field(default=0, metadata={"help": ">0: first n items only, no logging"})
    no_log: bool = field(default=False)
    out_json: Optional[str] = field(default=None, metadata={"help": "also dump the full records here"})
    device: str = field(default="cuda")


def parse_range(s: str) -> list[int]:
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


class Runner:
    def __init__(self, ea: AblateArguments, tokenizer, model, embedding, special_ids, num_latents: int):
        self.ea, self.tok, self.model, self.emb, self.ids, self.N = ea, tokenizer, model, embedding, special_ids, num_latents
        self.eos = tokenizer.eos_token_id

    @torch.no_grad()
    def run(self, question: str, n_lat: int | None = None, override: dict | None = None):
        n = self.N if n_lat is None else n_lat
        ids, attn = encode_question(self.tok, self.ids, question, n, self.ea.device)
        if n == 0:  # no latent tokens: plain forward over question + markers, then decode
            recs, embeds, kv, ncr = [], self.emb(ids), None, (0, ids.shape[1])
        else:
            recs, embeds, kv, ncr = run_passes(self.model, self.emb, ids, attn, self.ea.device, n,
                                               override_at_pass=override, latent_token_id=self.ids["latent"])
        text = finish_and_decode(self.model, self.emb, self.tok, embeds, kv, ncr, attn, self.ea.device,
                                 self.ea.max_new_tokens, self.eos)
        return extract_answer_after_delimiter(text), [r["live_hidden"] for r in recs]


def main() -> None:
    (ea,) = transformers.HfArgumentParser((AblateArguments,)).parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    import yaml
    cfg = yaml.safe_load(Path(ea.config_path).read_text()) if ea.config_path else None
    N = ea.num_latents if ea.num_latents >= 0 else cfg["c_thought"] * cfg["max_latent_stage"]
    c_thought = cfg["c_thought"] if cfg else 2
    tok, model, emb, ids = load_coconut(ea.model_id, ea.checkpoint_path, ea.device)
    R = Runner(ea, tok, model, emb, ids, N)

    ds = load_synthetic(ea.test_json)
    split = "validation" if Path(ea.test_json).name.startswith("valid") else "test"
    levels = set(parse_range(ea.levels))
    keep, seen = [], defaultdict(int)
    for i, r in enumerate(ds.rows):
        lv = int(r["level"])
        if lv in levels and (not ea.per_level_n or seen[lv] < ea.per_level_n):
            seen[lv] += 1
            keep.append(i)
    if ea.smoke_n:
        keep = keep[:ea.smoke_n]
    rows = {i: ds.rows[i] for i in keep}
    gold = {i: str(rows[i]["answer"]).strip() for i in keep}
    level = {i: int(rows[i]["level"]) for i in keep}
    print(f"items={len(keep)} levels={sorted(set(level.values()))} num_latents={N} split={split}", flush=True)

    t0 = time.perf_counter()
    # 1) intact run + own latents
    own: dict[int, list[torch.Tensor]] = {}
    records: dict[str, dict[int, dict]] = defaultdict(dict)
    for i in keep:
        pred, lat = R.run(rows[i]["question"])
        own[i] = lat
        records["intact"][i] = {"pred": pred, "correct": is_correct(pred, gold[i])}
    print(f"intact acc={sum(r['correct'] for r in records['intact'].values()) / len(keep):.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)

    # 2) mean latents from the TRAIN split (never the eval split)
    mean_vec = None
    if ea.mean_json:
        tr = load_synthetic(ea.mean_json)
        sub = random.Random(0).sample(range(len(tr.rows)), min(ea.mean_n, len(tr.rows)))
        acc = [torch.zeros_like(own[keep[0]][p]) for p in range(N)]
        for j in sub:
            _, lat = R.run(tr.rows[j]["question"])
            for p in range(N):
                acc[p] += lat[p]
        mean_vec = [a / len(sub) for a in acc]
        print(f"mean latents from {len(sub)} train items; norms {[round(float(m.norm()), 1) for m in mean_vec]}", flush=True)

    dev = ea.device
    gen = torch.Generator().manual_seed(ea.perm_seed)

    def cond_run(name, i, override=None, n_lat=None, extra=None):
        pred, _ = R.run(rows[i]["question"], n_lat=n_lat, override=override)
        rec = {"pred": pred, "correct": is_correct(pred, gold[i])}
        if extra:
            rec.update(extra)
        records[name][i] = rec

    # fixed derangements within each level (same_level) and random other-level donors
    rng = random.Random(ea.perm_seed)
    by_level = defaultdict(list)
    for i in keep:
        by_level[level[i]].append(i)
    same_donor, other_donor = {}, {}
    for lv, items in by_level.items():
        if len(items) < 2:
            continue
        for _ in range(1000):
            perm = items[:]
            rng.shuffle(perm)
            if all(a != b for a, b in zip(items, perm)):
                same_donor.update(dict(zip(items, perm)))
                break
    for i in keep:
        others = [j for j in keep if level[j] != level[i]]
        if others:
            other_donor[i] = rng.choice(others)

    def swap_extra(i, j):
        return {"donor_idx": j, "donor_gold": gold[j], "donor_level": level[j]}

    # 3) truncation
    for k in range(N):
        for i in keep:
            cond_run(f"trunc_{k}", i, n_lat=k)
    # 4) ablate all
    for i in keep:
        cond_run("ablate_all_zero", i, override={p: torch.zeros_like(own[i][p]) for p in range(N)})
        if mean_vec is not None:
            cond_run("ablate_all_mean", i, override={p: mean_vec[p] for p in range(N)})
        noise = {}
        for p in range(N):
            v = torch.randn(own[i][p].shape, generator=gen)
            noise[p] = v / v.norm() * own[i][p].norm()
        cond_run("ablate_all_noise", i, override=noise)
    # 5) swaps
    for name, donors in (("swap_same_level", same_donor), ("swap_other_level", other_donor)):
        for i in keep:
            if i in donors:
                j = donors[i]
                cond_run(name, i, override={p: own[j][p] for p in range(N)}, extra=swap_extra(i, j))
    if ea.per_slot:
        for s in range(N):
            for i in keep:
                if i in same_donor:
                    j = same_donor[i]
                    cond_run(f"swap_slot_{s}", i, override={s: own[j][s]}, extra=swap_extra(i, j))
                cond_run(f"ablate_single_zero_{s}", i, override={s: torch.zeros_like(own[i][s])})
                if mean_vec is not None:
                    cond_run(f"ablate_single_mean_{s}", i, override={s: mean_vec[s]})
    # 6) H4 / portability pairs (twins_v2): donor may be a new item -> run it through the model
    pair_meta = []
    if ea.pairs:
        id2idx = {rows[i]["id"]: i for i in keep}
        built = twins_v2.build_pairs([rows[i] for i in keep], seed=ea.pairs_seed)
        donor_cache: dict[str, list[torch.Tensor]] = {}
        for p in built:
            i = id2idx[p["recipient_id"]]
            did = p["donor_id"]
            if did not in donor_cache:
                donor_cache[did] = (own[id2idx[did]] if did in id2idx else R.run(p["donor"]["question"])[1])
            dl = donor_cache[did]
            k = p["k"]
            slots = [c_thought * (k - 1) + t for t in range(c_thought)]
            name = f"pair_{p['tier']}_k{k}"
            cond_run(name, i, override={s: dl[s] for s in slots},
                     extra={"donor_gold": p["donor_answer"], "predicted_answer": p["predicted_answer"],
                            "separable": p["separable"], "donor_id": did, "donor_level": int(p["donor"]["level"])})
            pair_meta.append({"recipient_id": p["recipient_id"], "donor_id": did, "tier": p["tier"], "k": k})
    print(f"all conditions done in {time.perf_counter() - t0:.0f}s: {sorted(records)}", flush=True)

    # ---------------- summaries
    window = set(parse_range(ea.window))
    base = records["intact"]

    def sub(recs, keep_idx):
        return {i: r for i, r in recs.items() if i in keep_idx}

    def swap_breakdown(recs):
        f = k_ = o = n = 0
        for i, r in recs.items():
            if "donor_gold" not in r or r["donor_gold"] == gold[i]:
                continue
            n += 1
            if r["pred"] is not None and r["pred"].strip() == r["donor_gold"]:
                f += 1
            elif r["correct"]:
                k_ += 1
            else:
                o += 1
        return {"n_scored": n, "follows_donor": f / n if n else None, "keeps_recipient": k_ / n if n else None, "other": o / n if n else None}

    def carries(recs):
        rows_ = [r for r in recs.values() if r.get("separable")]
        if not rows_:
            return None
        return {"n_separable": len(rows_), "carries_state": sum(r["pred"] is not None and r["pred"].strip() == r["predicted_answer"] for r in rows_) / len(rows_)}

    out = {"pooled_window": {}, "per_level": {}}
    win_idx = {i for i in keep if level[i] in window}
    base_w = sub(base, win_idx)
    acc_i = sum(r["correct"] for r in base_w.values()) / max(1, len(base_w))
    for name, recs in records.items():
        rw = sub(recs, win_idx)
        if not rw:
            continue
        s = summarize({"intact": {i: base[i] for i in rw}, name: rw}, {}, "intact", 1)[name]
        acc_c = s["accuracy"]
        acc_i_c = sum(base[i]["correct"] for i in rw) / len(rw)
        s["intact_accuracy_same_items"] = acc_i_c
        s["drop"] = acc_i_c - acc_c
        s["rel_drop"] = (acc_i_c - acc_c) / (acc_i_c - ea.floor) if acc_i_c > ea.floor else None
        s["swap"] = swap_breakdown(rw)
        s["carry"] = carries(rw)
        out["pooled_window"][name] = s
    for lv in sorted(by_level):
        idx = set(by_level[lv])
        out["per_level"][lv] = {name: {"n": len(sub(recs, idx)), "accuracy": sum(r["correct"] for r in sub(recs, idx).values()) / len(sub(recs, idx))}
                                 for name, recs in records.items() if sub(recs, idx)}

    print(f"\nPOOLED over window {ea.window} (n={len(base_w)}, intact acc {acc_i:.3f}); floor={ea.floor}")
    print(f"{'condition':26s}    n    acc   drop rel_drop    PWC  McNemar p | follows keeps  other")
    for name, s in out["pooled_window"].items():
        sw = s["swap"]
        tail = (f"| {sw['follows_donor']:.3f}  {sw['keeps_recipient']:.3f}  {sw['other']:.3f}" if sw["n_scored"] else "|")
        rd = f"{s['rel_drop']:.3f}" if s["rel_drop"] is not None else "  -  "
        pwc = f"{s['pwc']:.3f}" if s["pwc"] is not None else "  -  "
        print(f"{name:26s} {s['n']:5d}  {s['accuracy']:.3f}  {s['drop']:+.3f}   {rd}  {pwc}  {s['mcnemar_p_exact']:9.3g} {tail}")

    if ea.out_json:
        Path(ea.out_json).write_text(json.dumps({"summary": out, "records": {c: {str(i): r for i, r in rs.items()} for c, rs in records.items()}}, default=str))
    if ea.smoke_n or ea.no_log:
        print("smoke / --no_log run -- not logging")
        return

    n_params = sum(p.numel() for p in model.parameters())
    record = RunRecord(
        run_id=new_run_id("coconut", ea.slug or "v2-depth2-ablate"), mechanism="coconut", stage=ea.stage,
        model=ModelInfo(backbone=ea.model_id, checkpoint=ea.checkpoint_path, n_params=n_params),
        dataset=DatasetInfo(name="synthetic-depth2", split=split, n_examples=len(keep), seed=None),
        metrics={"final_answer_accuracy": sum(r["correct"] for r in base.values()) / len(base), "compute_steps": N,
                 "extra": {"window": ea.window, "floor": ea.floor, "data_sha256": assert_known_hash(ea.test_json),
                           "summary": out, "pairs_n": len(pair_meta)}},
        hyperparams={"script": "scripts/ablate_synthetic_coconut.py", "argv": argv, "perm_seed": ea.perm_seed, "pairs_seed": ea.pairs_seed,
                     "mean_json": ea.mean_json, "mean_n": ea.mean_n, "levels": ea.levels, "per_slot": ea.per_slot, "decoding": "greedy"},
        seed=cfg.get("seed") if cfg else None,
        hardware=f"{ea.hardware} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        notes=f"Inference-only ablation/swap battery on a Coconut checkpoint, synthetic depth2 {split}.",
    )
    preds = [{"idx": i, "id": rows[i]["id"], "level": level[i], "gold_answer": gold[i],
              "conditions": {c: records[c][i] for c in records if i in records[c]}} for i in keep]
    manifest = record.save(predictions=preds)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
