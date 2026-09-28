#!/usr/bin/env python3
"""P4 (RESEARCH_PLAN §4), Coconut side: dump Coconut's pass latents for the shared question set,
fit CODI -> Coconut ridge maps on the fit split, and transplant mapped CODI donor latents into
Coconut recipients on the eval split (`xmech_common.run_transplant`). Also writes
`coconut_latents.pt` for the reverse direction (`xmech_codi.py transplant`).

Order on the pod:
  1. CODI venv:    xmech_codi.py dump        -> codi_latents.pt
  2. Coconut venv: xmech_coconut.py          -> coconut_latents.pt + run record (CODI -> Coconut)
  3. CODI venv:    xmech_codi.py transplant  -> run record (Coconut -> CODI)

  cd /workspace/ai-capstone && .venv_coconut/bin/python scripts/xmech_coconut.py \\
      --checkpoint_path /workspace/coconut_checkpoints/checkpoint_33 \\
      --questions /workspace/xmech_questions.jsonl --codi_latents /workspace/codi_latents.pt \\
      --out_latents /workspace/coconut_latents.pt --stage full_run --hardware "RunPod ..."
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coconut_common import encode_question, extract_answer_after_delimiter, finish_and_decode, load_coconut, run_passes  # noqa: E402
import xmech_common as xc  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from latentreasoning.eval.metrics import is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402


@dataclass
class Args:
    checkpoint_path: str = field(metadata={"help": "path to checkpoint_33"})
    questions: str = field(metadata={"help": "xmech_questions.jsonl"})
    codi_latents: str = field(default="", metadata={"help": "codi_latents.pt from xmech_codi.py dump"})
    out_latents: str = field(default="/workspace/coconut_latents.pt")
    model_id: str = field(default="openai-community/gpt2")
    checkpoint_label: str = field(default="hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33")
    slug: str = field(default="xmech-codi-to-coconut")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    num_latents: int = field(default=6)
    carriers: str = field(default="1,4")
    n_recipients: int = field(default=1000)
    pair_seed: int = field(default=0)
    reps: int = field(default=200)
    max_new_tokens: int = field(default=48)
    smoke_n: int = field(default=0, metadata={"help": ">0: only this many fit and eval questions, don't log"})
    device: str = field(default="cuda")
    mode: str = field(default="full", metadata={"help": "full (dump + unrestricted transplant) | dump (dump only) | "
                                                          "subspace (reuse out_latents, DAS-subspace transplant) | "
                                                          "ctrl (Experiment A question-only control, reuse out_latents)"})
    codi_rotation: str = field(default="", metadata={"help": "subspace mode: CODI DAS rotation .pt (source)"})
    coconut_rotation: str = field(default="", metadata={"help": "subspace mode: Coconut DAS rotation .pt (target)"})
    ctrl_kind: str = field(default="", metadata={"help": "ctrl mode: site0 | sites1to5 | gpt2"})
    gpt2_latents: str = field(default="/workspace/gpt2_latents.pt",
                              metadata={"help": "ctrl mode, ctrl_kind=gpt2: dump from gpt2_plain_dump.py"})


def main() -> None:
    (a,) = transformers.HfArgumentParser((Args,)).parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    tokenizer, base_model, embedding, special_ids = load_coconut(a.model_id, a.checkpoint_path, a.device)
    n_params = sum(p.numel() for p in base_model.parameters())
    eos_id = tokenizer.eos_token_id
    carriers = [int(s) for s in a.carriers.split(",")]
    sites = list(range(a.num_latents))

    questions = xc.read_questions(a.questions)
    if a.smoke_n:
        questions = ([q for q in questions if q["split"] == "fit"][:a.smoke_n]
                     + [q for q in questions if q["split"] == "eval"][:a.smoke_n])
    codi = torch.load(a.codi_latents) if a.mode != "dump" else None

    def answer_with(question: str, override=None):
        input_ids, attn = encode_question(tokenizer, special_ids, question, a.num_latents, a.device)
        recs, inputs_embeds, kv_cache, ncr = run_passes(base_model, embedding, input_ids, attn, a.device, a.num_latents,
                                                        override_at_pass=override, latent_token_id=special_ids["latent"])
        raw = finish_and_decode(base_model, embedding, tokenizer, inputs_embeds, kv_cache, ncr, attn,
                                a.device, a.max_new_tokens, eos_id)
        return extract_answer_after_delimiter(raw), recs

    if a.mode == "subspace":
        run_subspace(a, questions, torch.load(a.out_latents), codi, answer_with, n_params, argv)
        return
    if a.mode == "ctrl":
        run_ctrl(a, questions, torch.load(a.out_latents), codi, answer_with, n_params, argv)
        return

    t0 = time.perf_counter()
    rows = {}
    for q in questions:
        pred, recs = answer_with(q["question"])
        rows[q["key"]] = {"z": torch.stack([recs[p]["live_hidden"].float() for p in sites]),
                          "pred": pred, "correct": is_correct(pred, q["answer"])}
    dump = {"meta": {"mechanism": "coconut", "site_definition": "pass p = vector spliced into the p-th <|latent|> slot"},
            "rows": rows}
    torch.save(dump, a.out_latents)
    dump_elapsed = time.perf_counter() - t0
    fit_keys, eval_keys = xc.split_keys(questions, "fit"), xc.split_keys(questions, "eval")
    acc = {s: sum(rows[k]["correct"] for k in ks) / len(ks) for s, ks in (("fit", fit_keys), ("eval", eval_keys))}
    if a.mode == "dump":
        print(f"dumped {len(rows)} in {dump_elapsed:.0f}s -> {a.out_latents}; accuracy {acc}")
        return
    print(f"dumped {len(rows)} in {dump_elapsed:.0f}s; coconut accuracy {acc}; "
          f"codi accuracy eval {sum(codi['rows'][k]['correct'] for k in eval_keys) / len(eval_keys):.3f}")

    src_fit, tgt_fit = xc.stack(codi, fit_keys), xc.stack(dump, fit_keys)
    src_ev, tgt_ev = xc.stack(codi, eval_keys), xc.stack(dump, eval_keys)
    maps, map_info = xc.fit_site_maps(src_fit, tgt_fit, sites)
    shuf, shuf_info = xc.fit_site_maps(src_fit, tgt_fit, sites, shuffle=True)
    r2_eval, r2_shuf = xc.eval_r2(maps, src_ev, tgt_ev), xc.eval_r2(shuf, src_ev, tgt_ev)
    cka = xc.cka_matrix(src_ev, tgt_ev)
    for p in sites:
        print(f"coconut pass {p}: eval R2 {r2_eval[p]:.3f} (shuffled {r2_shuf[p]:.3f}), lambda {map_info[p]['lambda']}")
    print("CKA [codi z_s][coconut pass p]:")
    for s, row in enumerate(cka):
        print(f"  z{s}: " + " ".join(f"{v:.2f}" for v in row))

    def transplant_answer(question: str, vecs: dict):
        return answer_with(question, override={p: v for p, v in vecs.items()})[0]

    t1 = time.perf_counter()
    rows_t, summary = xc.run_transplant(questions, dump, codi, maps, shuf, carriers, transplant_answer,
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
    record = RunRecord(
        run_id=new_run_id("coconut", a.slug), mechanism="coconut", stage=a.stage,
        model=ModelInfo(backbone=a.model_id, checkpoint=a.checkpoint_label, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(eval_keys), seed=None),
        metrics={
            "final_answer_accuracy": acc["eval"], "compute_steps": a.num_latents,
            "intervention_accuracy": summary["L4"]["mapped_carriers"]["rates"]["cf_joint"],
            "extra": {
                "design": "P4 CODI -> Coconut: ridge maps from all 6 CODI sites to each Coconut pass (fit split = "
                          "GSM8K-Aug train questions), transplant mapped CODI donor latents into Coconut recipients "
                          "at the ladder levels; controls own / shuffled-pair map / self-mapped reconstruction",
                "direction": "codi_to_coconut", "carriers": carriers,
                "n_fit": len(fit_keys), "n_eval": len(eval_keys), "accuracy": acc,
                "map_info": {str(k): v for k, v in map_info.items()},
                "eval_r2": {str(k): v for k, v in r2_eval.items()},
                "eval_r2_shuffled": {str(k): v for k, v in r2_shuf.items()},
                "cka_codi_site_by_coconut_pass": cka,
                "transplant": summary, "n_recipients": len(rows_t),
                "dump_elapsed_sec": dump_elapsed, "patch_elapsed_sec": patch_elapsed,
                "related_runs": ["20260927-062452_coconut_ladder-patch", "20260927-063622_codi_ladder-patch"],
            },
        },
        hyperparams={"carriers": carriers, "n_recipients_cap": a.n_recipients, "pair_seed": a.pair_seed,
                     "reps": a.reps, "lambdas": list(xc.LAMBDAS)},
        seed=a.pair_seed,
        hardware=f"{a.hardware} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        notes="P4 cross-mechanism transplant, CODI -> Coconut, with R2 / CKA alignment.",
    )
    manifest = record.save(predictions=rows_t)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}")


def run_subspace(a, questions, dump, codi, answer_with, n_params, argv) -> None:
    Wk_tgt, tgt_sites, k = xc.load_subspace(a.coconut_rotation)
    Wk_src, src_sites, k_src = xc.load_subspace(a.codi_rotation)
    print(f"Coconut subspace k={k} at passes {tgt_sites}; CODI subspace k={k_src} at z{src_sites}")
    t1 = time.perf_counter()
    rows_t, summary = xc.run_subspace_transplant(questions, dump, codi, Wk_tgt, Wk_src, src_sites, tgt_sites,
                                                 lambda q, vecs: answer_with(q, override=vecs)[0],
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
        run_id=new_run_id("coconut", a.slug), mechanism="coconut", stage=a.stage,
        model=ModelInfo(backbone=a.model_id, checkpoint=a.checkpoint_label, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(eval_keys), seed=None),
        metrics={
            "final_answer_accuracy": sum(dump["rows"][k_]["correct"] for k_ in eval_keys) / len(eval_keys),
            "compute_steps": a.num_latents,
            "intervention_accuracy": summary["L4"]["mapped_sub"]["rates"]["cf_joint"],
            "extra": {
                "design": "P4 CODI -> Coconut restricted to the DAS value subspaces: ridge from the CODI donor's "
                          "k DAS coordinates (z" + ",".join(map(str, src_sites)) + ") to Coconut's k DAS "
                          "coordinates at each carrier pass, written into the recipient's subspace only; controls "
                          "own / shuffled / complement-input / random-subspace / unrestricted map",
                "direction": "codi_to_coconut", "k": k, "k_src": k_src, "tgt_sites": tgt_sites, "src_sites": src_sites,
                "codi_rotation": a.codi_rotation, "coconut_rotation": a.coconut_rotation,
                "transplant": summary, "n_recipients": len(rows_t), "patch_elapsed_sec": patch_elapsed,
                "related_runs": ["20260927-085211_coconut_xmech-codi-to-coconut", "20260927-094215_codi_das-minimal-pair-fixed",
                                 "20260927-084449_coconut_das-minimal-pair-fixed"],
            },
        },
        hyperparams={"k": k, "tgt_sites": tgt_sites, "src_sites": src_sites, "n_recipients_cap": a.n_recipients,
                     "pair_seed": a.pair_seed, "reps": a.reps, "lambdas": list(xc.LAMBDAS)},
        seed=a.pair_seed,
        hardware=f"{a.hardware} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        notes="P4 cross-mechanism transplant restricted to the DAS value subspace, CODI -> Coconut.",
    )
    manifest = record.save(predictions=rows_t)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}")


def run_ctrl(a, questions, dump, codi, answer_with, n_params, argv) -> None:
    """Experiment A (question-only control for P4): same map + transplant as `--mode full`'s
    transplant, but the map's INPUT is a representation that has done no latent reasoning,
    while recipients/donors (`codi`'s correctness) stay pinned to the unrestricted P4 pairs."""
    sites = list(range(a.num_latents))
    carriers = [int(s) for s in a.carriers.split(",")]
    fit_keys, eval_keys = xc.split_keys(questions, "fit"), xc.split_keys(questions, "eval")
    if a.ctrl_kind == "gpt2":
        feat_dump, feat_sites = torch.load(a.gpt2_latents), None
        feat_desc = "plain gpt2 (layer6 last-token, layer12 last-token, layer12 mean-over-question-tokens)"
    elif a.ctrl_kind == "site0":
        feat_dump, feat_sites = codi, [0]
        feat_desc = "codi z0 only (question-encode pass, no reasoning)"
    elif a.ctrl_kind == "sites1to5":
        feat_dump, feat_sites = codi, [1, 2, 3, 4, 5]
        feat_desc = "codi z1-z5 (complement of z0)"
    else:
        raise ValueError(f"unknown --ctrl_kind {a.ctrl_kind!r}: site0 | sites1to5 | gpt2")

    feat_fit, tgt_fit = xc.stack(feat_dump, fit_keys), xc.stack(dump, fit_keys)
    feat_ev, tgt_ev = xc.stack(feat_dump, eval_keys), xc.stack(dump, eval_keys)
    maps, map_info = xc.fit_site_maps(feat_fit, tgt_fit, sites, src_sites=feat_sites)
    shuf, _ = xc.fit_site_maps(feat_fit, tgt_fit, sites, src_sites=feat_sites, shuffle=True)
    r2_eval = xc.eval_r2(maps, feat_ev, tgt_ev, feat_sites)
    r2_shuf = xc.eval_r2(shuf, feat_ev, tgt_ev, feat_sites)
    for p in sites:
        print(f"coconut pass {p} <- {a.ctrl_kind}: eval R2 {r2_eval[p]:.3f} (shuffled {r2_shuf[p]:.3f}), "
              f"lambda {map_info[p]['lambda']}")

    def transplant_answer(question: str, vecs: dict):
        return answer_with(question, override={p: v for p, v in vecs.items()})[0]

    t1 = time.perf_counter()
    rows_t, summary = xc.run_transplant(questions, dump, codi, maps, shuf, carriers, transplant_answer,
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
    record = RunRecord(
        run_id=new_run_id("coconut", a.slug), mechanism="coconut", stage=a.stage,
        model=ModelInfo(backbone=a.model_id, checkpoint=a.checkpoint_label, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(eval_keys), seed=None),
        metrics={
            "final_answer_accuracy": sum(dump["rows"][k_]["correct"] for k_ in eval_keys) / len(eval_keys),
            "compute_steps": a.num_latents,
            "intervention_accuracy": summary["L4"]["mapped_carriers"]["rates"]["cf_joint"],
            "extra": {
                "design": f"Experiment A (question-only control for P4, RESEARCH_PLAN pasted plan): ridge map "
                          f"from {feat_desc} to each Coconut pass (fit split = same 8000 GSM8K-Aug train "
                          "questions), transplanted on the SAME 259 eval-split ladder recipients/donors as "
                          "the unrestricted P4 run (pair_seed 0) -- only the map's input changes, testing "
                          "whether unrestricted transfer is explained by question information alone.",
                "direction": "codi_to_coconut", "ctrl_kind": a.ctrl_kind, "feat_desc": feat_desc,
                "carriers": carriers, "feat_sites": feat_sites,
                "n_fit": len(fit_keys), "n_eval": len(eval_keys),
                "map_info": {str(k): v for k, v in map_info.items()},
                "eval_r2": {str(k): v for k, v in r2_eval.items()},
                "eval_r2_shuffled": {str(k): v for k, v in r2_shuf.items()},
                "transplant": summary, "n_recipients": len(rows_t), "patch_elapsed_sec": patch_elapsed,
                "related_runs": ["20260927-085211_coconut_xmech-codi-to-coconut"],
            },
        },
        hyperparams={"carriers": carriers, "feat_sites": feat_sites, "ctrl_kind": a.ctrl_kind,
                     "n_recipients_cap": a.n_recipients, "pair_seed": a.pair_seed, "reps": a.reps,
                     "lambdas": list(xc.LAMBDAS)},
        seed=a.pair_seed,
        hardware=f"{a.hardware} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        notes=f"Experiment A control ({a.ctrl_kind}): {feat_desc} -> Coconut ridge map, transplant on the same "
              "P4 ladder pairs as 20260927-085211_coconut_xmech-codi-to-coconut.",
    )
    manifest = record.save(predictions=rows_t)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}")


if __name__ == "__main__":
    main()
