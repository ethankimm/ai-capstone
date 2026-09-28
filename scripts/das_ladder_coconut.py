#!/usr/bin/env python3
"""Experiment B (round-4 follow-up, `next_experiments.md` pasted plan): Coconut counterpart
of `das_ladder_codi.py` -- read that module's docstring for the full motivation (round 4's
minimal-pair-trained rotation's own_sub ceiling, `20260927-190704/-190716_coconut_xmech-
subspace-codi-to-coconut-k*`, was far below whole-vector own transfer; this tests whether
training the rotation on cross-problem ladder donors instead closes the gap).

Reuses `das_minimal_pair_coconut.py`'s DAS machinery unchanged (`OrthogonalRotation`,
`intervene`, `run_intervened`, `teacher_forced_ce`, `run_pass_range`, `full_trace`) --
only the donor source and the teacher-forcing TARGET change: donors are ladder L2/L3
cross-problem donors, not same-problem minimal-pair twins, and the target is the
recipient's own program re-run on the donor's step values (`ladder_common.targets(...)
["cf_joint"]`) formatted as `"### {cf_joint}"` (no leading space -- the round-3 bug), not
the twin's own gold answer.

  cd /workspace/ai-capstone && .venv_coconut/bin/python scripts/das_ladder_coconut.py \\
      --checkpoint_path /workspace/coconut_checkpoints/checkpoint_33 --data_dir /workspace/coconut_data \\
      --slug das-ladder --stage full_run --hardware "RunPod RTX A5000 (secure)" \\
      --num_latents 6 --site_group 1,4 --k_values 16,32,64 --train_n_recipients 500 \\
      --questions /workspace/xmech_questions.jsonl --codi_latents /workspace/codi_latents.pt \\
      --coconut_latents /workspace/coconut_latents.pt --save_rotations /workspace/rotations_ladder/coconut
"""
from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coconut_common import extract_answer_after_delimiter, finish_and_decode, load_coconut  # noqa: E402
from das_minimal_pair_coconut import (  # noqa: E402
    HIDDEN_DIM, OrthogonalRotation, load_examples, run_intervened, run_pass_range, teacher_forced_ce,
)
import xmech_common as xc  # noqa: E402
from ladder_common import LEVELS, build_ladder_pairs, format_row, summarize, targets  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from latentreasoning.eval.metrics import is_correct  # noqa: E402
from latentreasoning.mechanisms.coconut import name as mechanism_name  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402


@dataclass
class DASArguments:
    checkpoint_path: str = field(metadata={"help": "path to the downloaded checkpoint_33 file"})
    data_dir: str = field(metadata={"help": "dir with gsm_original_{train,valid}.json"})
    model_id: str = field(default="openai-community/gpt2")
    slug: str = field(default="das-ladder")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: str = field(default="hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33")
    num_latents: int = field(default=6)
    site_group: str = field(default="1,4", metadata={"help": "comma-separated Coconut passes sharing one rotation"})
    k_values: str = field(default="16,32,64")
    train_n_recipients: int = field(default=500)
    train_pool_n: int = field(default=3000, metadata={"help": "gsm_original_train.json examples to decode"})
    train_seed: int = field(default=0)
    pair_seed: int = field(default=0)
    epochs: int = field(default=5)
    lr: float = field(default=1e-3)
    max_new_tokens: int = field(default=48)
    questions: str = field(default="/workspace/xmech_questions.jsonl",
                           metadata={"help": "for eval-pair reproduction only (xmech_common._ladder_pairs)"})
    codi_latents: str = field(default="/workspace/codi_latents.pt")
    coconut_latents: str = field(default="/workspace/coconut_latents.pt")
    n_eval_recipients: int = field(default=1000)
    eval_pair_seed: int = field(default=0)
    reps: int = field(default=200)
    save_rotations: str = field(default="")
    smoke_n: int = field(default=0)
    device: str = field(default="cuda")


def group_label(group: list[int]) -> str:
    return "+".join(str(s) for s in group)


def answer_target(answer: str) -> str:
    return f"### {answer.strip()}"  # "###" (token 21017) as the model emits it, not " ###"


def build_train_tuples(examples: list, correct: dict[int, bool], z_cache: dict[int, dict],
                       n_recipients: int, pair_seed: int) -> tuple[list[tuple], list[dict]]:
    pairs = build_ladder_pairs(examples, correct, random.Random(pair_seed), n_recipients)
    tuples = []
    for p in pairs:
        r = p["recipient"]
        for level in ("L2", "L3"):
            d = p[level]
            t = targets(r, d)
            if t["cf_joint"] is None:
                continue
            tuples.append((r, d, t["cf_joint"], z_cache[d.idx]))
    return tuples, pairs


@torch.no_grad()
def decode_of(base_model, embedding, tokenizer, eos_id, special_ids, device, num_latents, max_new_tokens, ex, cache):
    if ex.idx not in cache:
        input_ids, attn = _encode(tokenizer, special_ids, ex.question, num_latents, device)
        records, state = run_pass_range(base_model, embedding, input_ids, attn, device, num_latents,
                                        special_ids["latent"], 0, num_latents, grad=False)
        inputs_embeds, kv_cache, next_compute_range, *_ = state
        raw = finish_and_decode(base_model, embedding, tokenizer, inputs_embeds, kv_cache, next_compute_range,
                                attn, device, max_new_tokens, eos_id)
        pred = extract_answer_after_delimiter(raw)
        cache[ex.idx] = {"pred": pred, "z": {r["pass"]: r["live_hidden"] for r in records}}
    return cache[ex.idx]


def _encode(tokenizer, special_ids, question, num_latents, device):
    from coconut_common import encode_question
    return encode_question(tokenizer, special_ids, question, num_latents, device)


def train_rotation(base_model, embedding, tokenizer, special_ids, device, num_latents, group: list[int],
                   k: int, train_tuples: list[tuple], lr: float, epochs: int) -> OrthogonalRotation:
    rotation = OrthogonalRotation(HIDDEN_DIM).to(device)
    total_steps = max(1, epochs * len(train_tuples))
    opt = torch.optim.AdamW(rotation.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    step = 0
    t0 = time.perf_counter()
    for _ in range(epochs):
        random.shuffle(train_tuples)
        for r, _d, target_text, donor_vecs in train_tuples:
            final_state, _input_ids, attn = run_intervened(
                base_model, embedding, tokenizer, special_ids, r.question, device, num_latents,
                rotation, group, donor_vecs, k, grad=True)
            loss = teacher_forced_ce(base_model, embedding, tokenizer, final_state, attn, device,
                                     answer_target(target_text))
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"  group={group_label(group)} k={k} step={step}/{total_steps} loss={loss.item():.4f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    rotation.eval()
    return rotation


@torch.no_grad()
def evaluate_conditions(base_model, embedding, tokenizer, eos_id, special_ids, device, num_latents,
                        group: list[int], conditions: dict, pairs: list[dict], max_new_tokens: int,
                        decode_cache: dict) -> list[dict]:
    rows = []
    for pi, p in enumerate(pairs):
        r = p["recipient"]
        base = decode_of(base_model, embedding, tokenizer, eos_id, special_ids, device, num_latents,
                         max_new_tokens, r, decode_cache)
        row = {"recipient_answer": r.answer, "base": base["pred"]}
        for level in LEVELS:
            d = p[level]
            donor = decode_of(base_model, embedding, tokenizer, eos_id, special_ids, device, num_latents,
                              max_new_tokens, d, decode_cache)
            preds = {}
            for name, (rotation, k) in conditions.items():
                final_state, _input_ids, attn = run_intervened(
                    base_model, embedding, tokenizer, special_ids, r.question, device, num_latents,
                    rotation, group, donor["z"], k, grad=False)
                inputs_embeds, kv_cache, next_compute_range, *_ = final_state
                raw = finish_and_decode(base_model, embedding, tokenizer, inputs_embeds, kv_cache,
                                        next_compute_range, attn, device, max_new_tokens, eos_id)
                preds[name] = extract_answer_after_delimiter(raw)
            row[level] = {"donor_answer": d.answer, "targets": targets(r, d), "preds": preds}
        rows.append(row)
        if (pi + 1) % 25 == 0:
            print(f"evaluated {pi + 1}/{len(pairs)} recipients", flush=True)
    return rows


def main() -> None:
    parser = transformers.HfArgumentParser((DASArguments,))
    (da,) = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = da.device

    tokenizer, base_model, embedding, special_ids = load_coconut(da.model_id, da.checkpoint_path, device)
    n_params = sum(p.numel() for p in base_model.parameters())
    for p in base_model.parameters():
        p.requires_grad_(False)
    eos_id = tokenizer.eos_token_id

    group = [int(s) for s in da.site_group.split(",")]
    k_values = [int(s) for s in da.k_values.split(",")]
    print(f"site_group={group} k_values={k_values}")

    train_pool_n = 30 if da.smoke_n else da.train_pool_n
    train_n_recipients = 5 if da.smoke_n else da.train_n_recipients
    train_examples = load_examples(da.data_dir, train_pool_n, da.train_seed, "gsm_original_train.json")

    t0 = time.perf_counter()
    base, correct, z_cache = {}, {}, {}
    decode_cache_train: dict = {}
    for ex in train_examples:
        d = decode_of(base_model, embedding, tokenizer, eos_id, special_ids, device, da.num_latents,
                     da.max_new_tokens, ex, decode_cache_train)
        base[ex.idx] = d["pred"]
        correct[ex.idx] = is_correct(d["pred"], ex.answer)
        z_cache[ex.idx] = d["z"]
    decode_elapsed = time.perf_counter() - t0
    train_accuracy = sum(correct.values()) / len(correct)
    print(f"[train pool] decoded {len(train_examples)} in {decode_elapsed:.0f}s, accuracy={train_accuracy:.3f}", flush=True)

    train_tuples, train_pairs = build_train_tuples(train_examples, correct, z_cache, train_n_recipients, da.pair_seed)
    print(f"[train pool] ladder recipients: {len(train_pairs)}, qualifying (recipient,donor,level) "
          f"tuples: {len(train_tuples)}/{2 * len(train_pairs)}", flush=True)

    questions = xc.read_questions(da.questions)
    codi_dump, coconut_dump = torch.load(da.codi_latents), torch.load(da.coconut_latents)
    n_eval = 5 if da.smoke_n else da.n_eval_recipients
    eval_pairs, _key_of = xc._ladder_pairs(questions, coconut_dump, codi_dump, n_eval, da.eval_pair_seed)
    print(f"[eval] P4-identical recipients: {len(eval_pairs)}", flush=True)

    decode_cache: dict = {}
    torch.manual_seed(da.train_seed)
    conditions = {"full": (OrthogonalRotation(HIDDEN_DIM).to(device).eval(), HIDDEN_DIM)}
    for k in k_values:
        conditions[f"untrained_k{k}"] = (OrthogonalRotation(HIDDEN_DIM).to(device).eval(), k)

    t1 = time.perf_counter()
    for k in k_values:
        print(f"--- training group={group_label(group)} k={k} (n_train_tuples={len(train_tuples)}) ---", flush=True)
        rotation = train_rotation(base_model, embedding, tokenizer, special_ids, device, da.num_latents,
                                  group, k, list(train_tuples), da.lr, da.epochs)
        conditions[f"trained_k{k}"] = (rotation, k)
        if da.save_rotations:
            import os
            os.makedirs(da.save_rotations, exist_ok=True)
            torch.save({"W": rotation.rot.weight.detach().float().cpu(), "k": k, "group": group,
                       "mechanism": "coconut", "basis": "ladder_L2L3"},
                      os.path.join(da.save_rotations, f"rot_{group_label(group)}_k{k}.pt"))
    train_elapsed = time.perf_counter() - t1

    t2 = time.perf_counter()
    rows = evaluate_conditions(base_model, embedding, tokenizer, eos_id, special_ids, device, da.num_latents,
                               group, conditions, eval_pairs, da.max_new_tokens, decode_cache)
    eval_elapsed = time.perf_counter() - t2

    rng = random.Random(da.eval_pair_seed + 1)
    summary = {level: {c: summarize(rows, c, level, da.reps, rng) for c in conditions} for level in LEVELS}
    for level in LEVELS:
        for c in conditions:
            print(format_row(f"{level} {c}", summary[level][c]))

    if da.smoke_n:
        print("smoke run -- not logging")
        return

    metrics = {
        "compute_steps": da.num_latents,
        "intervention_accuracy": summary["L4"][f"trained_k{k_values[-1]}"]["rates"]["cf_joint"],
        "extra": {
            "design": "Experiment B: DAS rotation trained on CROSS-PROBLEM ladder (L2+L3) donor pairs "
                      "instead of within-problem minimal-pair twins; teacher-forcing target is the "
                      "recipient's own program re-run on the donor's step values (cf_joint), not the "
                      "donor's own answer. Evaluated on the SAME 259 eval recipients as the unrestricted "
                      "P4 runs (own-mechanism donor, subspace-only interchange -- own_sub).",
            "site_definition": "pass p = vector spliced into the p-th <|latent|> slot",
            "site_group": group_label(group), "k_values": k_values,
            "train_n_recipients_requested": train_n_recipients, "train_n_recipients_actual": len(train_pairs),
            "train_n_tuples": len(train_tuples), "train_pool_n": train_pool_n, "train_accuracy": train_accuracy,
            "n_eval_recipients": len(eval_pairs), "eval_pair_seed": da.eval_pair_seed,
            "epochs": da.epochs, "lr": da.lr,
            "conditions": list(conditions.keys()),
            "summary": summary,
            "decode_elapsed_sec": decode_elapsed, "train_elapsed_sec": train_elapsed, "eval_elapsed_sec": eval_elapsed,
            "round4_minimal_pair_rotation_own_sub_cf_joint": {
                "note": "not re-executed here -- same 259 recipients (xmech_common._ladder_pairs, pair_seed 0), "
                        "quoted from the already-logged run for direct comparison",
                "k16": {"L2": 0.085, "L3": 0.108, "L4": 0.085, "run_id": "20260927-190704_coconut_xmech-subspace-codi-to-coconut-k16"},
                "k32": {"L2": 0.104, "L3": 0.120, "L4": 0.120, "run_id": "20260927-190716_coconut_xmech-subspace-codi-to-coconut-k32"},
            },
            "related_runs": ["20260927-190704_coconut_xmech-subspace-codi-to-coconut-k16",
                             "20260927-190716_coconut_xmech-subspace-codi-to-coconut-k32",
                             "20260927-085211_coconut_xmech-codi-to-coconut"],
        },
    }

    record = RunRecord(
        run_id=new_run_id("coconut", da.slug),
        mechanism=mechanism_name,
        stage=da.stage,
        model=ModelInfo(backbone=da.model_id, checkpoint=da.checkpoint_label, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(eval_pairs), seed=da.eval_pair_seed),
        metrics=metrics,
        hyperparams={"num_latents": da.num_latents, "site_group": group_label(group), "k_values": k_values,
                     "train_n_recipients": train_n_recipients, "epochs": da.epochs, "lr": da.lr,
                     "n_eval_recipients_cap": da.n_eval_recipients, "eval_pair_seed": da.eval_pair_seed,
                     "reps": da.reps},
        seed=da.train_seed,
        hardware=f"{da.hardware} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        notes="Experiment B: DAS on cross-problem ladder donors (Coconut) -- own-mechanism subspace-only "
              "transfer, trained rotation vs untrained vs raw whole-vector swap, on the same 259 P4 "
              "eval recipients as the minimal-pair-trained rotation.",
    )
    manifest = record.save(predictions=rows)
    out_dir = manifest.parent
    (out_dir / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
