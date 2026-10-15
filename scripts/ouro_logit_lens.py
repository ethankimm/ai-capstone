"""Ouro step 2b: logit lens (and a supervised leading-digit probe) for the intermediate value
per (loop, layer). Follow-up to `ouro_probe.py`, which probed log-magnitude and the *ones* digit;
Ouro emits numbers digit by digit after a " " token, so what its unembedding can show at the
suffix's last position is a number's *leading* digit. Does the first intermediate's leading
digit show up there in early loops before the answer's takes over?

Lens: lm_head(model.norm(h)) on the decoder-layer output h after layer l in loop u. Ouro applies
model.norm at the end of every loop (and feeds the normed state into the next loop), so at the
last layer of each loop this is exactly that loop's own exit readout. Read at
  q_last   last question token
  sfx3     the suffix's final ' ' (its next token is the answer's leading digit)
and restricted to the 10 digit tokens (renormalized), plus the full-vocab digit mass.

Per eval item, leading digits of step1 (first intermediate), penult (last intermediate) and the
answer. Metrics per site:
  top1_<t>        argmax over digits == leading digit of t
  beats_donor_<t> P(digit of t) > P(digit of a donor), donor = another item's step1 leading
                  digit, drawn so it differs from this item's step1 and answer leading digits.
                  Chance 0.5 whatever the digit prior (the donor comes from the same marginal),
                  so this is the Benford-robust number; reported on items where step1's and the
                  answer's leading digits differ ("differ") so the two can't be confused.
Probe: ridge classifier on the same features for each target's leading digit (10-way), fit on
xmech fit questions, lambda on a val split, majority baseline (as in ouro_probe.py).

Usage (pod, Ouro venv, from the repo):
  python scripts/ouro_logit_lens.py --questions xmech_questions.jsonl \\
      --gate-predictions results/<1.4B gate run>/predictions.jsonl
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ouro_gate import build_prefix, read_questions, set_loops  # noqa: E402
from ouro_probe import LAMBDA_FRACS, Capture, ridge_site, targets_for  # noqa: E402
from ouro_twin_patch import SUFFIX, _inputs  # noqa: E402

from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402
from latentreasoning.utils import set_seed  # noqa: E402

POSITIONS = ("q_last", "sfx3")
TARGETS = ("step1", "penult", "answer")


def lead(v: int) -> int:
    return int(str(abs(int(v)))[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ByteDance/Ouro-1.4B")
    p.add_argument("--questions", required=True)
    p.add_argument("--gate-predictions", required=True)
    p.add_argument("--loops", type=int, default=4)
    p.add_argument("--n-fit", type=int, default=4000)
    p.add_argument("--n-shots", type=int, default=8)
    p.add_argument("--shot-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attn", default="eager")
    p.add_argument("--smoke-n", type=int, default=0)
    p.add_argument("--slug", default="logit-lens-ouro-1.4b")
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
    n_fit_all, N, n_ev = len(fit_q), len(fit_q) + len(ev_q), len(ev_q)
    print(f"{n_fit_all} fit + {n_ev} eval questions")

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16,
                                                 device_map="cuda", attn_implementation=a.attn).eval()
    set_loops(model, a.loops)
    layers = model.model.layers[: model.config.num_hidden_layers]
    n_layers, d, dev = len(layers), model.config.hidden_size, model.device
    digit_ids = [tok(str(k), add_special_tokens=False)["input_ids"] for k in range(10)]
    assert all(len(x) == 1 for x in digit_ids), digit_ids
    digit_ids = torch.tensor([x[0] for x in digit_ids], device=dev)
    suffix_ids = tok(SUFFIX, add_special_tokens=False)["input_ids"]
    C = Capture(layers, len(POSITIONS), a.loops, d, dev)
    feats = torch.zeros(N, len(POSITIONS), a.loops, n_layers, d, dtype=torch.bfloat16, device=dev)
    lens = torch.zeros(n_ev, len(POSITIONS), a.loops, n_layers, 10, device=dev)  # full-vocab log-probs of digits
    top_is_digit = torch.zeros(n_ev, len(POSITIONS), a.loops, n_layers, dtype=torch.bool, device=dev)
    last_logits_digit = torch.zeros(n_ev, dtype=torch.long, device=dev)

    t0 = time.perf_counter()
    for i, (q, _) in enumerate(items):
        ids = tok(prefix + f"Question: {q['question']}", add_special_tokens=False)["input_ids"] + suffix_ids
        nq = len(ids) - len(suffix_ids)
        C.positions = [nq - 1, len(ids) - 1]
        with torch.no_grad():
            out = model(**_inputs(model, torch.tensor([ids], device=dev), 0), use_cache=False)
            C.positions = None
            feats[i] = C.buf
            if i >= n_fit_all:
                j = i - n_fit_all
                lp = torch.log_softmax(model.lm_head(model.model.norm(C.buf)).float(), -1)  # [P, U, L, V]
                lens[j] = lp[..., digit_ids]
                top_is_digit[j] = torch.isin(lp.argmax(-1), digit_ids)
                last_logits_digit[j] = out.logits[0, -1, digit_ids].argmax()
        if (i + 1) % 500 == 0:
            print(f"extracted {i + 1}/{N} ({time.perf_counter() - t0:.0f}s)", flush=True)
    extract_elapsed = time.perf_counter() - t0

    # sanity: the lens at (sfx3, last loop, last layer) is the model's own output distribution
    own = lens[:, 1, -1, -1].argmax(-1)
    agree = float((own == last_logits_digit).float().mean())
    print(f"lens(sfx3, last loop, last layer) digit argmax == model output digit argmax: {agree:.3f}")
    assert agree > 0.98, agree

    # ---- targets, donors, subsets --------------------------------------------------------
    L = torch.tensor([[lead(t[k]) for k in TARGETS] for _, t in items], device=dev)  # [N, 3]
    Le = L[n_fit_all:]
    donor = []
    for j in range(n_ev):
        s1, an = int(Le[j, 0]), int(Le[j, 2])
        cands = [int(Le[k, 0]) for k in rng.sample(range(n_ev), min(n_ev, 50)) if int(Le[k, 0]) not in (s1, an)]
        donor.append(cands[0] if cands else (s1 + 1) % 10)
    donor = torch.tensor(donor, device=dev)
    ok = torch.tensor([bool(gate.get(q["key"], {}).get(f"correct_l{a.loops}", False)) for q, _ in ev_q], device=dev)
    differ = Le[:, 0] != Le[:, 2]
    subsets = {"all": torch.ones(n_ev, dtype=torch.bool, device=dev), "differ": differ,
               "differ_correct": differ & ok, "ouro_correct": ok}
    print({k: int(v.sum()) for k, v in subsets.items()})

    dl = torch.log_softmax(lens, -1)  # renormalized over digits
    pd = dl.gather(-1, donor[:, None, None, None, None].expand(-1, *dl.shape[1:4], 1)).squeeze(-1)
    lens_res = {}
    for pi, pos in enumerate(POSITIONS):
        for s, m in subsets.items():
            r = {"digit_mass": lens[m, pi].logsumexp(-1).exp().mean(0).tolist(),
                 "top_is_digit": top_is_digit[m, pi].float().mean(0).tolist()}
            for ti, t in enumerate(TARGETS):
                pt = dl[m, pi].gather(-1, Le[m, ti][:, None, None, None].expand(-1, a.loops, n_layers, 1)).squeeze(-1)
                r[f"top1_{t}"] = (dl[m, pi].argmax(-1) == Le[m, ti][:, None, None]).float().mean(0).tolist()
                r[f"logp_{t}"] = pt.mean(0).tolist()
                r[f"beats_donor_{t}"] = (pt > pd[m, pi]).float().mean(0).tolist()
            lens_res[f"{pos}/{s}"] = r

    # ---- supervised leading-digit probe ------------------------------------------------------
    n_val = n_fit_all // 5
    fit_idx = torch.arange(0, n_fit_all - n_val, device=dev)
    val_idx = torch.arange(n_fit_all - n_val, n_fit_all, device=dev)
    ev_idx = torch.arange(n_fit_all, N, device=dev)
    Y = torch.nn.functional.one_hot(L, 10).float().reshape(N, -1)
    majority = {t: float((Le[:, ti] == L[fit_idx, ti].mode().values).float().mean()) for ti, t in enumerate(TARGETS)}
    majority_differ = {t: float((Le[differ, ti] == L[fit_idx, ti].mode().values).float().mean()) for ti, t in enumerate(TARGETS)}
    probe = {f"{pos}/{s}/{t}": [[None] * n_layers for _ in range(a.loops)]
             for pos in POSITIONS for s in subsets for t in TARGETS}
    t1 = time.perf_counter()
    for pi, pos in enumerate(POSITIONS):
        for u in range(a.loops):
            for l in range(n_layers):
                P = ridge_site(feats[:, pi, u, l], fit_idx, Y)
                for ti, t in enumerate(TARGETS):
                    blk = P[:, :, 10 * ti: 10 * ti + 10].argmax(-1)  # [n_lam, N]
                    li = max(range(len(LAMBDA_FRACS)), key=lambda k: float((blk[k, val_idx] == L[val_idx, ti]).float().mean()))
                    hit = (blk[li, ev_idx] == Le[:, ti]).float()
                    for s, m in subsets.items():
                        probe[f"{pos}/{s}/{t}"][u][l] = float(hit[m].mean())
    probe_elapsed = time.perf_counter() - t1

    # ---- print ---------------------------------------------------------------------------
    def show(name, g, step=2):
        print(f"--- {name} (rows = loop, cols = layer 0,{step},..)")
        for u, line in enumerate(g):
            print(f"loop{u + 1} " + " ".join(f"{v:5.2f}" for v in line[::step]))
    print("majority leading digit acc (eval / differ):", majority, majority_differ)
    for pos in POSITIONS:
        show(f"{pos} lens digit mass (all)", lens_res[f"{pos}/all"]["digit_mass"])
        for t in TARGETS:
            show(f"{pos} lens top1 {t} (all)", lens_res[f"{pos}/all"][f"top1_{t}"])
        for t in ("step1", "answer"):
            show(f"{pos} lens beats_donor {t} (differ)", lens_res[f"{pos}/differ"][f"beats_donor_{t}"])
            show(f"{pos} lens top1 {t} (differ)", lens_res[f"{pos}/differ"][f"top1_{t}"])
            show(f"{pos} probe {t} (differ)", probe[f"{pos}/differ/{t}"])
        show(f"{pos} probe answer (all)", probe[f"{pos}/all/answer"])
    print(f"extract {extract_elapsed:.0f}s, probes {probe_elapsed:.0f}s")
    if a.smoke_n:
        print("smoke run -- not logging")
        return

    rows = []
    for j, (q, t) in enumerate(ev_q):
        rows.append({"key": q["key"], "ouro_correct": bool(ok[j]), **{k: t[k] for k in TARGETS},
                     **{f"lead_{k}": int(Le[j, ti]) for ti, k in enumerate(TARGETS)}, "donor_digit": int(donor[j]),
                     "lens_digit_logp_sfx3_end_of_loop": [[round(float(x), 3) for x in lens[j, 1, u, -1]] for u in range(a.loops)],
                     "lens_digit_logp_qlast_end_of_loop": [[round(float(x), 3) for x in lens[j, 0, u, -1]] for u in range(a.loops)]})
    record = RunRecord(
        run_id=new_run_id("recurrent_depth", a.slug), mechanism="recurrent_depth", stage=a.stage,
        model=ModelInfo(backbone=a.model, checkpoint=a.model, n_params=sum(x.numel() for x in model.parameters())),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=n_ev, seed=a.seed),
        metrics={
            "compute_steps": a.loops,
            "extra": {
                "design": "logit lens lm_head(norm(h)) per (loop, layer) at the last question token and the suffix's "
                          "final ' ', restricted to digit tokens; leading digits of step1 / penult / answer; "
                          "donor-digit control; plus a ridge leading-digit probe on the same features",
                "site_definition": "grids are [loop][layer], 0-indexed; h = output of decoder layer l in loop u",
                "positions": list(POSITIONS), "n_fit": len(fit_idx), "n_val": len(val_idx), "n_eval": n_ev,
                "n_subsets": {k: int(v.sum()) for k, v in subsets.items()},
                "lens_own_output_agreement": agree, "majority_lead_digit_eval": majority,
                "majority_lead_digit_differ": majority_differ, "lens": lens_res, "probe_lead_digit": probe,
                "extract_elapsed_sec": extract_elapsed, "probe_elapsed_sec": probe_elapsed,
                "related_runs": ["20261014-100712_recurrent_depth_gate-ouro-1.4b",
                                 "20261014-163349_recurrent_depth_twin-patch-ouro-1.4b",
                                 "20261015-022002_recurrent_depth_probe-ouro-1.4b"],
            },
        },
        hyperparams={"n_fit_pool": n_fit_all, "n_shots": a.n_shots, "shot_seed": a.shot_seed, "suffix": SUFFIX,
                     "attn": a.attn, "lens": "lm_head(model.norm(h)), digits renormalized"},
        seed=seed, hardware=f"{a.hardware} / {torch.cuda.get_device_name(0)}",
        notes="Ouro step 2b: does the intermediate's leading digit appear in the logit lens before the answer's?",
    )
    manifest = record.save(predictions=rows)
    (manifest.parent / "eval_command.txt").write_text(" ".join(sys.argv) + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}")


if __name__ == "__main__":
    main()
