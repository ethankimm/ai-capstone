#!/usr/bin/env python3
"""Experiment B (round-4 follow-up, `next_experiments.md` pasted plan): train a DAS
rotation on CROSS-PROBLEM (ladder L2/L3) donor pairs instead of within-problem
minimal-pair twins, and test whether a subspace-only patch trained this way reaches
whole-vector transfer across problems. Round 4's minimal-pair-trained rotation
(`20260927-191455/-191525_codi_xmech-subspace-coconut-to-codi-k*`) found own_sub (a
cross-problem DAS interchange using the rotation trained on within-problem minimal
pairs) at only 0.07-0.21 cf_joint against ~0.20-0.39 for a whole-vector cross-problem
patch, and hypothesized the training basis (within-problem, near-identical complement)
doesn't suit cross-problem transfer. This asks: does training on ladder donors directly
close that gap?

Reuses `das_minimal_pair_codi.py`'s DAS machinery unchanged (`OrthogonalRotation`,
`intervene`, `run_intervened`, `teacher_forced_ce`, `answer_target`) -- only the donor
source and the teacher-forcing TARGET change: donors are ladder L2/L3 cross-problem
donors (`ladder_common.build_ladder_pairs`), not same-problem minimal-pair twins, and
the target is the recipient's OWN program re-run on the donor's step values
(`ladder_common.targets(...)["cf_joint"]`) -- a ladder donor's own final answer belongs
to a different problem and is not a valid teacher-forcing target for the recipient's
program (unlike a minimal-pair twin, whose gold answer under the interchange IS the
counterfactual target by construction).

Two phases:
  train  decode a TRAIN-split pool with this model alone (single-mechanism
         correctness), build ladder pairs (`build_train_tuples`), train one rotation
         per k on the L2+L3 tuples (`--site_group`, default CODI's carriers 0,2,4).
  eval   reproduce the EXACT SAME 259 eval recipients as the unrestricted P4 runs
         (`xmech_common._ladder_pairs`, needs both `codi_latents.pt` and
         `coconut_latents.pt` purely to reconstruct that base-correct-in-both
         intersection -- no cross-mechanism computation happens here), for a
         same-recipient comparison to round 4's own_sub numbers. Conditions: full
         (raw whole-vector swap, k=768 -- any rotation gives this exactly), untrained
         random rotation per k, and the newly trained rotation per k. Scored with
         `ladder_common.summarize` (cf_joint bucket + permutation null), matching P4's
         own metric.

If own_sub with this basis approaches the whole-vector ceiling, the follow-up is to
rerun `xmech_codi.py --mode subspace` / `xmech_coconut.py --mode subspace` (already
implemented, round 4) pointing `--codi_rotation`/`--coconut_rotation` at rotations
saved here instead of the minimal-pair ones -- no new code needed for that step.

  cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/das_ladder_codi.py \\
      --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \\
      --slug das-ladder --stage full_run --hardware "RunPod RTX A5000 (secure)" \\
      --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \\
      --lora_r 128 --lora_alpha 32 --lora_init --greedy True \\
      --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \\
      --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \\
      --site_group 0,2,4 --k_values 16,32,64 --train_n_recipients 500 \\
      --questions /workspace/xmech_questions.jsonl --codi_latents /workspace/codi_latents.pt \\
      --coconut_latents /workspace/coconut_latents.pt --save_rotations /workspace/rotations_ladder/codi
"""
from __future__ import annotations

import os
import random
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
from das_minimal_pair_codi import (  # noqa: E402
    HIDDEN_DIM, OrthogonalRotation, answer_target, intervene, run_intervened, teacher_forced_ce,
)
import xmech_common as xc  # noqa: E402
from ladder_common import LEVELS, build_ladder_pairs, format_row, summarize, targets  # noqa: E402

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.eval.metrics import extract_final_number, is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402


@dataclass
class DASArguments:
    slug: str = field(default="das-ladder")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: Optional[str] = field(default=None)
    site_group: str = field(default="0,2,4", metadata={"help": "comma-separated CODI sites sharing one rotation"})
    k_values: str = field(default="16,32,64")
    train_n_recipients: int = field(default=500, metadata={"help": "ladder recipients on the TRAIN split "
                                                                    "(each yields an L2 and an L3 training tuple)"})
    train_pool_n: int = field(default=3000, metadata={"help": "gsm8k-aug train examples to decode looking for "
                                                               "train recipients/donors"})
    train_seed: int = field(default=0)
    pair_seed: int = field(default=0, metadata={"help": "train-pool ladder pairing seed"})
    epochs: int = field(default=5)
    lr: float = field(default=1e-3)
    max_new_tokens: int = field(default=64)
    questions: str = field(default="/workspace/xmech_questions.jsonl",
                           metadata={"help": "for eval-pair reproduction only (xmech_common._ladder_pairs)"})
    codi_latents: str = field(default="/workspace/codi_latents.pt")
    coconut_latents: str = field(default="/workspace/coconut_latents.pt")
    n_eval_recipients: int = field(default=1000, metadata={"help": "must match the P4 runs' cap (1000) to "
                                                                    "reproduce the same 259 recipients"})
    eval_pair_seed: int = field(default=0, metadata={"help": "must match the P4 runs' pair_seed (0)"})
    reps: int = field(default=200)
    save_rotations: str = field(default="", metadata={"help": "dir: save each trained R as rot_<group>_k<k>.pt"})
    smoke_n: int = field(default=0, metadata={"help": ">0: tiny train pool + eval cap, don't log"})


def group_label(group: list[int]) -> str:
    return "+".join(str(s) for s in group)


def build_train_tuples(examples: list, correct: dict[int, bool], z_cache: dict[int, dict],
                       n_recipients: int, pair_seed: int) -> tuple[list[tuple], list[dict]]:
    """Pure (no model): ladder pairs on `examples` (base-correct in THIS mechanism only),
    flattened to (recipient, donor, cf_joint_target_text, donor_vecs) training tuples --
    up to two per recipient (L2 and L3), dropped where cf_joint is undefined (the last
    step's operands aren't all substitutable from earlier steps -- see `ladder_common`)."""
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


def train_rotation(model, tokenizer, device, n_iters, group: list[int], k: int, train_tuples: list[tuple],
                   lr: float, epochs: int) -> OrthogonalRotation:
    rotation = OrthogonalRotation(HIDDEN_DIM).to(device)
    sites = set(group)
    total_steps = max(1, epochs * len(train_tuples))
    opt = torch.optim.AdamW(rotation.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    step = 0
    t0 = time.perf_counter()
    for _ in range(epochs):
        random.shuffle(train_tuples)
        for r, _d, target_text, donor_vecs in train_tuples:
            q = r.question.strip().replace("  ", " ")
            pkv, _latent = run_intervened(model, tokenizer, q, device, n_iters, rotation, sites, donor_vecs, k)
            loss = teacher_forced_ce(model, tokenizer, pkv, device, answer_target(target_text))
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
def decode_of(model, tokenizer, device, n_iters, max_new_tokens, ex, cache: dict):
    if ex.idx not in cache:
        pkv, thoughts, latent = run_thoughts(model, tokenizer, ex.question.strip().replace("  ", " "), device,
                                             n_iters, include_latent0=True)
        pred = extract_final_number(decode_answer(model, tokenizer, pkv, latent, device, max_new_tokens))
        cache[ex.idx] = {"pred": pred, "z": {rec["iter"]: rec["post"] for rec in thoughts if rec["iter"] < n_iters}}
    return cache[ex.idx]


@torch.no_grad()
def evaluate_conditions(model, tokenizer, device, n_iters, group: list[int], conditions: dict, pairs: list[dict],
                        max_new_tokens: int, decode_cache: dict) -> list[dict]:
    """conditions: {name: (rotation, k)}. One row per recipient with every condition's
    prediction at every level, for `ladder_common.summarize` to bucket per condition."""
    sites = set(group)
    rows = []
    for pi, p in enumerate(pairs):
        r = p["recipient"]
        base = decode_of(model, tokenizer, device, n_iters, max_new_tokens, r, decode_cache)
        row = {"recipient_answer": r.answer, "base": base["pred"]}
        for level in LEVELS:
            d = p[level]
            donor = decode_of(model, tokenizer, device, n_iters, max_new_tokens, d, decode_cache)
            preds = {}
            for name, (rotation, k) in conditions.items():
                pkv, latent = run_intervened(model, tokenizer, r.question.strip().replace("  ", " "), device,
                                             n_iters, rotation, sites, donor["z"], k)
                preds[name] = extract_final_number(decode_answer(model, tokenizer, pkv, latent, device, max_new_tokens))
            row[level] = {"donor_answer": d.answer, "targets": targets(r, d), "preds": preds}
        rows.append(row)
        if (pi + 1) % 25 == 0:
            print(f"evaluated {pi + 1}/{len(pairs)} recipients", flush=True)
    return rows


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, DASArguments))
    model_args, data_args, training_args, da = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = "cuda"

    model, tokenizer, load_result = build_model(model_args, training_args)
    n_params = sum(p.numel() for p in model.parameters())
    model.requires_grad_(False)
    print(f"loaded {model_args.ckpt_dir}: missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}")

    n_iters = training_args.inf_latent_iterations
    group = [int(s) for s in da.site_group.split(",")]
    k_values = [int(s) for s in da.k_values.split(",")]
    print(f"site_group={group} k_values={k_values}")

    train_pool_n = 30 if da.smoke_n else da.train_pool_n
    train_n_recipients = 5 if da.smoke_n else da.train_n_recipients
    train_examples = load_gsm8k_aug(split="train", n=train_pool_n, seed=da.train_seed)

    t0 = time.perf_counter()
    base, correct, z_cache = {}, {}, {}
    for ex in train_examples:
        pkv, thoughts, latent = run_thoughts(model, tokenizer, ex.question.strip().replace("  ", " "), device,
                                             n_iters, include_latent0=True)
        pred = extract_final_number(decode_answer(model, tokenizer, pkv, latent, device, da.max_new_tokens))
        base[ex.idx] = pred
        correct[ex.idx] = is_correct(pred, ex.answer)
        z_cache[ex.idx] = {rec["iter"]: rec["post"] for rec in thoughts if rec["iter"] < n_iters}
    decode_elapsed = time.perf_counter() - t0
    train_accuracy = sum(correct.values()) / len(correct)
    print(f"[train pool] decoded {len(train_examples)} in {decode_elapsed:.0f}s, accuracy={train_accuracy:.3f}", flush=True)

    train_tuples, train_pairs = build_train_tuples(train_examples, correct, z_cache, train_n_recipients, da.pair_seed)
    print(f"[train pool] ladder recipients: {len(train_pairs)}, qualifying (recipient,donor,level) "
          f"tuples: {len(train_tuples)}/{2 * len(train_pairs)}", flush=True)

    # eval pairs: EXACT same 259 recipients as the unrestricted P4 runs
    questions = xc.read_questions(da.questions)
    codi_dump, coconut_dump = torch.load(da.codi_latents), torch.load(da.coconut_latents)
    n_eval = 5 if da.smoke_n else da.n_eval_recipients
    eval_pairs, _key_of = xc._ladder_pairs(questions, codi_dump, coconut_dump, n_eval, da.eval_pair_seed)
    print(f"[eval] P4-identical recipients: {len(eval_pairs)}", flush=True)

    decode_cache: dict = {}
    torch.manual_seed(da.train_seed)
    conditions = {"full": (OrthogonalRotation(HIDDEN_DIM).to(device).eval(), HIDDEN_DIM)}
    for k in k_values:
        conditions[f"untrained_k{k}"] = (OrthogonalRotation(HIDDEN_DIM).to(device).eval(), k)

    trained_rotations = {}
    t1 = time.perf_counter()
    for k in k_values:
        print(f"--- training group={group_label(group)} k={k} (n_train_tuples={len(train_tuples)}) ---", flush=True)
        rotation = train_rotation(model, tokenizer, device, n_iters, group, k, list(train_tuples), da.lr, da.epochs)
        trained_rotations[k] = rotation
        conditions[f"trained_k{k}"] = (rotation, k)
        if da.save_rotations:
            os.makedirs(da.save_rotations, exist_ok=True)
            torch.save({"W": rotation.rot.weight.detach().float().cpu(), "k": k, "group": group,
                       "mechanism": "codi", "basis": "ladder_L2L3"},
                      os.path.join(da.save_rotations, f"rot_{group_label(group)}_k{k}.pt"))
    train_elapsed = time.perf_counter() - t1

    t2 = time.perf_counter()
    rows = evaluate_conditions(model, tokenizer, device, n_iters, group, conditions, eval_pairs,
                               da.max_new_tokens, decode_cache)
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
        "compute_steps": n_iters,
        "intervention_accuracy": summary["L4"][f"trained_k{k_values[-1]}"]["rates"]["cf_joint"],
        "extra": {
            "design": "Experiment B: DAS rotation trained on CROSS-PROBLEM ladder (L2+L3) donor pairs "
                      "instead of within-problem minimal-pair twins; teacher-forcing target is the "
                      "recipient's own program re-run on the donor's step values (cf_joint), not the "
                      "donor's own answer. Evaluated on the SAME 259 eval recipients as the unrestricted "
                      "P4 runs (own-mechanism donor, subspace-only interchange -- own_sub).",
            "site_definition": "site s = z_s = latent fed into loop iteration s+1; z_0 = latent-0",
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
                "k16": {"L2": 0.120, "L3": 0.135, "L4": 0.073, "run_id": "20260927-191525_codi_xmech-subspace-coconut-to-codi-k16"},
                "k32": {"L2": 0.178, "L3": 0.208, "L4": 0.100, "run_id": "20260927-191455_codi_xmech-subspace-coconut-to-codi-k32"},
            },
            "related_runs": ["20260927-191455_codi_xmech-subspace-coconut-to-codi-k32",
                             "20260927-191525_codi_xmech-subspace-coconut-to-codi-k16",
                             "20260927-092012_codi_xmech-coconut-to-codi"],
        },
    }

    record = RunRecord(
        run_id=new_run_id("codi", da.slug),
        mechanism="codi",
        stage=da.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=da.checkpoint_label or model_args.ckpt_dir, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(eval_pairs), seed=da.eval_pair_seed),
        metrics=metrics,
        hyperparams={"inf_latent_iterations": n_iters, "site_group": group_label(group), "k_values": k_values,
                     "train_n_recipients": train_n_recipients, "epochs": da.epochs, "lr": da.lr,
                     "n_eval_recipients_cap": da.n_eval_recipients, "eval_pair_seed": da.eval_pair_seed,
                     "reps": da.reps},
        seed=da.train_seed,
        hardware=f"{da.hardware} / {torch.cuda.get_device_name(0)}",
        notes="Experiment B: DAS on cross-problem ladder donors (CODI) -- own-mechanism subspace-only "
              "transfer, trained rotation vs untrained vs raw whole-vector swap, on the same 259 P4 "
              "eval recipients as the minimal-pair-trained rotation.",
    )
    manifest = record.save(predictions=rows)
    out_dir = manifest.parent
    (out_dir / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
