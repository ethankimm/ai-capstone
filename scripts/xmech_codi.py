#!/usr/bin/env python3
"""P4 (RESEARCH_PLAN §4), CODI side. Two modes (`--mode`):
  dump        run CODI on the shared question set, save z_0..z_5 (aligned sites: z_s feeds
              iteration s+1) + greedy answer + correctness -> codi_latents.pt
  transplant  fit Coconut -> CODI ridge maps on the fit split (needs coconut_latents.pt from
              `xmech_coconut.py`) and transplant mapped Coconut donor latents into CODI recipients
              at z0/z2/z4 on the eval split; logs a run record
  subspace    the same transplant restricted to the DAS value subspaces (`--codi_rotation`,
              `--coconut_rotation` from das_minimal_pair_*.py --save_rotations):
              `xmech_common.run_subspace_transplant`; logs a run record

Run inside the CODI venv, from the CODI checkout:

  cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/xmech_codi.py $CODI_FLAGS \\
      --output_dir /tmp/o --mode dump --questions /workspace/xmech_questions.jsonl \\
      --codi_latents /workspace/codi_latents.pt
  ... --mode transplant --coconut_latents /workspace/coconut_latents.pt --stage full_run --hardware "..."
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())  # the CODI checkout: `src.model`

from src.model import DataArguments, ModelArguments, TrainingArguments  # noqa: E402
from eval_codi import build_model  # noqa: E402
from decode_patch_codi import decode_answer, run_thoughts  # noqa: E402
import xmech_common as xc  # noqa: E402

from latentreasoning.eval.metrics import extract_final_number, is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402


@dataclass
class XArgs:
    mode: str = field(default="dump", metadata={"help": "dump | transplant | subspace | ctrl"})
    codi_rotation: str = field(default="", metadata={"help": "subspace mode: CODI DAS rotation .pt (target)"})
    coconut_rotation: str = field(default="", metadata={"help": "subspace mode: Coconut DAS rotation .pt (source)"})
    ctrl_kind: str = field(default="", metadata={"help": "ctrl mode: site0 | sites1to5 | gpt2 -- Experiment A "
                                                          "question-only control (see xmech_common.run_transplant)"})
    gpt2_latents: str = field(default="/workspace/gpt2_latents.pt",
                              metadata={"help": "ctrl mode, ctrl_kind=gpt2: dump from gpt2_plain_dump.py"})
    questions: str = field(default="/workspace/xmech_questions.jsonl")
    codi_latents: str = field(default="/workspace/codi_latents.pt")
    coconut_latents: str = field(default="/workspace/coconut_latents.pt")
    checkpoint_label: Optional[str] = field(default=None)
    slug: str = field(default="xmech-coconut-to-codi")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    carriers: str = field(default="0,2,4")
    n_recipients: int = field(default=1000)
    pair_seed: int = field(default=0)
    reps: int = field(default=200)
    max_new_tokens: int = field(default=64)
    smoke_n: int = field(default=0, metadata={"help": ">0: only this many fit and eval questions, don't log"})


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, XArgs))
    model_args, data_args, training_args, a = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = "cuda"
    model, tokenizer, _ = build_model(model_args, training_args)
    assert not model.training, "model must be in eval mode"
    n_params = sum(p.numel() for p in model.parameters())
    n_latents = training_args.inf_latent_iterations
    sites = list(range(n_latents))
    carriers = [int(s) for s in a.carriers.split(",")]

    questions = xc.read_questions(a.questions)
    if a.smoke_n:
        questions = ([q for q in questions if q["split"] == "fit"][:a.smoke_n]
                     + [q for q in questions if q["split"] == "eval"][:a.smoke_n])
    fit_keys, eval_keys = xc.split_keys(questions, "fit"), xc.split_keys(questions, "eval")

    def prep(q: str) -> str:
        return q.strip().replace("  ", " ")

    def answer_with(question: str, override: Optional[dict] = None):
        feed = {s + 1: v if callable(v) else v.to(device=device, dtype=torch.bfloat16).view(1, 1, -1)
                for s, v in (override or {}).items()}
        pkv, thoughts, latent = run_thoughts(model, tokenizer, prep(question), device, n_latents,
                                             override_input_at=feed or None, include_latent0=True)
        return extract_final_number(decode_answer(model, tokenizer, pkv, latent, device, a.max_new_tokens)), thoughts

    if a.mode == "dump":
        t0 = time.perf_counter()
        rows = {}
        for q in questions:
            pred, thoughts = answer_with(q["question"])
            z = {rec["iter"]: rec["post"] for rec in thoughts}
            rows[q["key"]] = {"z": torch.stack([z[s].float().cpu().reshape(-1) for s in sites]),
                              "pred": pred, "correct": is_correct(pred, q["answer"])}
        torch.save({"meta": {"mechanism": "codi", "site_definition": "z_s feeds iteration s+1; z_0 = latent-0"},
                    "rows": rows}, a.codi_latents)
        acc = {s: sum(rows[k]["correct"] for k in ks) / len(ks) for s, ks in (("fit", fit_keys), ("eval", eval_keys))}
        print(f"dumped {len(rows)} in {time.perf_counter() - t0:.0f}s -> {a.codi_latents}; accuracy {acc}")
        return

    codi, coconut = torch.load(a.codi_latents), torch.load(a.coconut_latents)
    if a.mode == "subspace":
        run_subspace(a, model_args, questions, codi, coconut, answer_with, n_params, n_latents, argv)
        return
    if a.mode == "ctrl":
        run_ctrl(a, model_args, questions, codi, coconut, answer_with, n_params, n_latents, argv)
        return
    src_fit, tgt_fit = xc.stack(coconut, fit_keys), xc.stack(codi, fit_keys)
    src_ev, tgt_ev = xc.stack(coconut, eval_keys), xc.stack(codi, eval_keys)
    maps, map_info = xc.fit_site_maps(src_fit, tgt_fit, sites)
    shuf, _ = xc.fit_site_maps(src_fit, tgt_fit, sites, shuffle=True)
    r2_eval, r2_shuf = xc.eval_r2(maps, src_ev, tgt_ev), xc.eval_r2(shuf, src_ev, tgt_ev)
    cka = xc.cka_matrix(src_ev, tgt_ev)
    for s in sites:
        print(f"codi z{s}: eval R2 {r2_eval[s]:.3f} (shuffled {r2_shuf[s]:.3f}), lambda {map_info[s]['lambda']}")

    t1 = time.perf_counter()
    rows_t, summary = xc.run_transplant(questions, codi, coconut, maps, shuf, carriers,
                                        lambda q, vecs: answer_with(q, vecs)[0],
                                        a.n_recipients, a.pair_seed, a.reps)
    patch_elapsed = time.perf_counter() - t1
    from ladder_common import format_row  # noqa: E402
    for level in ("L2", "L3", "L4"):
        for c, s in summary[level].items():
            print(format_row(f"{level} {c}", s))
    print(f"self_mapped_carriers unchanged: {summary['self_mapped_carriers_unchanged_rate']}")
    if a.smoke_n:
        print("smoke run -- not logging")
        return

    acc_eval = sum(codi["rows"][k]["correct"] for k in eval_keys) / len(eval_keys)
    record = RunRecord(
        run_id=new_run_id("codi", a.slug), mechanism="codi", stage=a.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=a.checkpoint_label or model_args.ckpt_dir, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(eval_keys), seed=None),
        metrics={
            "final_answer_accuracy": acc_eval, "compute_steps": n_latents,
            "intervention_accuracy": summary["L4"]["mapped_carriers"]["rates"]["cf_joint"],
            "extra": {
                "design": "P4 Coconut -> CODI: ridge maps from all 6 Coconut passes to each CODI site (fit split = "
                          "GSM8K-Aug train questions), transplant mapped Coconut donor latents into CODI recipients "
                          "at the ladder levels; controls own / shuffled-pair map / self-mapped reconstruction",
                "direction": "coconut_to_codi", "carriers": carriers,
                "n_fit": len(fit_keys), "n_eval": len(eval_keys),
                "map_info": {str(k): v for k, v in map_info.items()},
                "eval_r2": {str(k): v for k, v in r2_eval.items()},
                "eval_r2_shuffled": {str(k): v for k, v in r2_shuf.items()},
                "cka_coconut_pass_by_codi_site": cka,
                "transplant": summary, "n_recipients": len(rows_t), "patch_elapsed_sec": patch_elapsed,
                "related_runs": ["20260927-063622_codi_ladder-patch", "20260927-062452_coconut_ladder-patch"],
            },
        },
        hyperparams={"carriers": carriers, "n_recipients_cap": a.n_recipients, "pair_seed": a.pair_seed,
                     "reps": a.reps, "lambdas": list(xc.LAMBDAS)},
        seed=a.pair_seed,
        hardware=f"{a.hardware} / {torch.cuda.get_device_name(0)}",
        notes="P4 cross-mechanism transplant, Coconut -> CODI, with R2 / CKA alignment.",
    )
    manifest = record.save(predictions=rows_t)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}")


def run_subspace(a, model_args, questions, codi, coconut, answer_with, n_params, n_latents, argv) -> None:
    Wk_tgt, tgt_sites, k = xc.load_subspace(a.codi_rotation)
    Wk_src, src_sites, k_src = xc.load_subspace(a.coconut_rotation)
    print(f"CODI subspace k={k} at z{tgt_sites}; Coconut subspace k={k_src} at passes {src_sites}")
    t1 = time.perf_counter()
    rows_t, summary = xc.run_subspace_transplant(questions, codi, coconut, Wk_tgt, Wk_src, src_sites, tgt_sites,
                                                 lambda q, vecs: answer_with(q, vecs)[0],
                                                 a.n_recipients, a.pair_seed, a.reps)
    patch_elapsed = time.perf_counter() - t1
    from ladder_common import format_row  # noqa: E402
    for level in ("L2", "L3", "L4"):
        for c, s in summary[level].items():
            print(format_row(f"{level} {c}", s))
    print(f"self_mapped_sub unchanged: {summary['self_mapped_sub_unchanged_rate']}")
    print(f"coordinate map fit: {summary['coord_map_fit']}")
    if a.smoke_n:
        print("smoke run -- not logging")
        return
    eval_keys = xc.split_keys(questions, "eval")
    record = RunRecord(
        run_id=new_run_id("codi", a.slug), mechanism="codi", stage=a.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=a.checkpoint_label or model_args.ckpt_dir, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(eval_keys), seed=None),
        metrics={
            "final_answer_accuracy": sum(codi["rows"][k_]["correct"] for k_ in eval_keys) / len(eval_keys),
            "compute_steps": n_latents,
            "intervention_accuracy": summary["L4"]["mapped_sub"]["rates"]["cf_joint"],
            "extra": {
                "design": "P4 Coconut -> CODI restricted to the DAS value subspaces: ridge from the Coconut donor's "
                          "k DAS coordinates (passes " + ",".join(map(str, src_sites)) + ") to CODI's k DAS "
                          "coordinates at each carrier, written into the recipient's subspace only; controls own / "
                          "shuffled / complement-input / random-subspace / unrestricted map",
                "direction": "coconut_to_codi", "k": k, "k_src": k_src, "tgt_sites": tgt_sites, "src_sites": src_sites,
                "codi_rotation": a.codi_rotation, "coconut_rotation": a.coconut_rotation,
                "transplant": summary, "n_recipients": len(rows_t), "patch_elapsed_sec": patch_elapsed,
                "related_runs": ["20260927-092012_codi_xmech-coconut-to-codi", "20260927-094215_codi_das-minimal-pair-fixed",
                                 "20260927-084449_coconut_das-minimal-pair-fixed"],
            },
        },
        hyperparams={"k": k, "tgt_sites": tgt_sites, "src_sites": src_sites, "n_recipients_cap": a.n_recipients,
                     "pair_seed": a.pair_seed, "reps": a.reps, "lambdas": list(xc.LAMBDAS)},
        seed=a.pair_seed,
        hardware=f"{a.hardware} / {torch.cuda.get_device_name(0)}",
        notes="P4 cross-mechanism transplant restricted to the DAS value subspace, Coconut -> CODI.",
    )
    manifest = record.save(predictions=rows_t)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}")


def run_ctrl(a, model_args, questions, codi, coconut, answer_with, n_params, n_latents, argv) -> None:
    """Experiment A (question-only control for P4): same map + transplant as `--mode transplant`,
    but the map's INPUT is a representation that has done no latent reasoning, while the
    recipients/donors (`coconut`'s correctness) stay pinned to the unrestricted P4 pairs."""
    sites = list(range(n_latents))
    carriers = [int(s) for s in a.carriers.split(",")]
    fit_keys, eval_keys = xc.split_keys(questions, "fit"), xc.split_keys(questions, "eval")
    if a.ctrl_kind == "gpt2":
        feat_dump, feat_sites = torch.load(a.gpt2_latents), None
        feat_desc = "plain gpt2 (layer6 last-token, layer12 last-token, layer12 mean-over-question-tokens)"
    elif a.ctrl_kind == "site0":
        feat_dump, feat_sites = coconut, [0]
        feat_desc = "coconut pass 0 only (question-encode pass, no reasoning)"
    elif a.ctrl_kind == "sites1to5":
        feat_dump, feat_sites = coconut, [1, 2, 3, 4, 5]
        feat_desc = "coconut passes 1-5 (complement of pass 0)"
    else:
        raise ValueError(f"unknown --ctrl_kind {a.ctrl_kind!r}: site0 | sites1to5 | gpt2")

    feat_fit, tgt_fit = xc.stack(feat_dump, fit_keys), xc.stack(codi, fit_keys)
    feat_ev, tgt_ev = xc.stack(feat_dump, eval_keys), xc.stack(codi, eval_keys)
    maps, map_info = xc.fit_site_maps(feat_fit, tgt_fit, sites, src_sites=feat_sites)
    shuf, _ = xc.fit_site_maps(feat_fit, tgt_fit, sites, src_sites=feat_sites, shuffle=True)
    r2_eval = xc.eval_r2(maps, feat_ev, tgt_ev, feat_sites)
    r2_shuf = xc.eval_r2(shuf, feat_ev, tgt_ev, feat_sites)
    for s in sites:
        print(f"codi z{s} <- {a.ctrl_kind}: eval R2 {r2_eval[s]:.3f} (shuffled {r2_shuf[s]:.3f}), "
              f"lambda {map_info[s]['lambda']}")

    t1 = time.perf_counter()
    rows_t, summary = xc.run_transplant(questions, codi, coconut, maps, shuf, carriers,
                                        lambda q, vecs: answer_with(q, vecs)[0],
                                        a.n_recipients, a.pair_seed, a.reps,
                                        feat=feat_dump, feat_sites=feat_sites)
    patch_elapsed = time.perf_counter() - t1
    from ladder_common import format_row  # noqa: E402
    for level in ("L2", "L3", "L4"):
        for c, s in summary[level].items():
            print(format_row(f"{level} {c}", s))
    print(f"self_mapped_carriers unchanged: {summary['self_mapped_carriers_unchanged_rate']}")
    if a.smoke_n:
        print("smoke run -- not logging")
        return
    acc_eval = sum(codi["rows"][k]["correct"] for k in eval_keys) / len(eval_keys)
    record = RunRecord(
        run_id=new_run_id("codi", a.slug), mechanism="codi", stage=a.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=a.checkpoint_label or model_args.ckpt_dir, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(eval_keys), seed=None),
        metrics={
            "final_answer_accuracy": acc_eval, "compute_steps": n_latents,
            "intervention_accuracy": summary["L4"]["mapped_carriers"]["rates"]["cf_joint"],
            "extra": {
                "design": f"Experiment A (question-only control for P4, RESEARCH_PLAN pasted plan): ridge map "
                          f"from {feat_desc} to each CODI site (fit split = same 8000 GSM8K-Aug train "
                          "questions), transplanted on the SAME 259 eval-split ladder recipients/donors as "
                          "the unrestricted P4 run (pair_seed 0) -- only the map's input changes, testing "
                          "whether unrestricted transfer is explained by question information alone.",
                "direction": "coconut_to_codi", "ctrl_kind": a.ctrl_kind, "feat_desc": feat_desc,
                "carriers": carriers, "feat_sites": feat_sites,
                "n_fit": len(fit_keys), "n_eval": len(eval_keys),
                "map_info": {str(k): v for k, v in map_info.items()},
                "eval_r2": {str(k): v for k, v in r2_eval.items()},
                "eval_r2_shuffled": {str(k): v for k, v in r2_shuf.items()},
                "transplant": summary, "n_recipients": len(rows_t), "patch_elapsed_sec": patch_elapsed,
                "related_runs": ["20260927-092012_codi_xmech-coconut-to-codi"],
            },
        },
        hyperparams={"carriers": carriers, "feat_sites": feat_sites, "ctrl_kind": a.ctrl_kind,
                     "n_recipients_cap": a.n_recipients, "pair_seed": a.pair_seed, "reps": a.reps,
                     "lambdas": list(xc.LAMBDAS)},
        seed=a.pair_seed,
        hardware=f"{a.hardware} / {torch.cuda.get_device_name(0)}",
        notes=f"Experiment A control ({a.ctrl_kind}): {feat_desc} -> CODI ridge map, transplant on the same "
              "P4 ladder pairs as 20260927-092012_codi_xmech-coconut-to-codi.",
    )
    manifest = record.save(predictions=rows_t)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}")


if __name__ == "__main__":
    main()
