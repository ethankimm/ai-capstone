#!/usr/bin/env python3
"""Latent diagnostics for a RuleChain (depth2) CODI / CODI-0 checkpoint: three checks, one forward per item and
condition (no generation).

1. Latent-count sweep (early termination): run the question, then k latent passes (k in --latent_counts, e.g.
   0,1,2,4,6,8 on a model trained with 6), then <eot> "The answer is:". Report teacher-forced accuracy of the answer
   token per level AND the distillation loss at that position. If accuracy is flat in k but the distillation loss
   rises as latents are removed, the latents carry teacher-state content that the answer does not use.
2. Per-layer distillation loss by level: the training objective, split by hidden-state layer (13 for GPT-2:
   embeddings + 12 blocks) and level, at the trained latent count (and at every k of the sweep).
3. Train vs test accuracy per level (memorisation check): teacher-forced accuracy on --train_per_level items per level
   of train.json vs the full test set, at the trained latent count.

Teacher-forced accuracy at the distillation position equals generated accuracy within +-0.004 per level on the
round-1 checkpoints (their logged test evals), so it stands in for generation here.

Distillation loss = CODI's (src/model.py with distill_loss_div_std): per layer, SmoothL1(student, teacher) at the
token before the answer (":" of "The answer is:"), divided by the teacher's std at that layer, averaged over layers.
Training takes the std over each batch of 128; here it is taken over the whole evaluated set (fixed per layer, so
conditions are comparable). Teacher = question + CoT without its last step + "The answer is:" (include_last_cot False),
tokenized per segment as in CODI's preprocess, right-padded.

Runs INSIDE the CODI venv from the CODI checkout (patched with codi_synthetic / codi_model_fixes), like
eval_synthetic_codi.py:

  cd ../codi && .venv/bin/python ../ai-capstone/scripts/diagnose_codi_latents.py \\
      --ckpt_dir ../ai-capstone/rulechain-checkpoints/gpt2_depth2_codi_bs64/checkpoint-31280 \\
      --test_json ../synthetic_data_v2/depth2/test.json --train_json ../synthetic_data_v2/depth2/train.json \\
      --latent_counts 0,1,2,3,4,5,6,8 --slug v2-depth2-codi-latent-diagnostics --hardware "MacBook (MPS)" \\
      --output_dir /tmp/unused --model_name_or_path gpt2 --seed 11 --model_max_length 512 \\
      --lora_r 128 --lora_alpha 32 --lora_init --batch_size 64 --greedy True \\
      --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \\
      --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \\
      --fix_attn_mask True --pad_invariant True --include_last_cot False
  # CODI-0: --ckpt_dir .../gpt2_depth2_codi0/checkpoint-29716 --num_latent 0 --inf_latent_iterations 0 --latent_counts 0
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers
from torch.nn import functional as F

sys.path.insert(0, os.getcwd())  # the CODI checkout: `src.model`
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.model import DataArguments, ModelArguments, TrainingArguments  # noqa: E402

from eval_codi import PadInvariant, build_model, question_position_ids, tokenize_questions  # noqa: E402
from eval_synthetic_codi import answer_token, check_flags  # noqa: E402
from synthetic_common import load_synthetic  # noqa: E402

from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402

if not torch.cuda.is_available():  # checkpoints were saved on CUDA; build_model / check_flags call torch.load bare
    _torch_load = torch.load
    torch.load = lambda *a, **kw: _torch_load(*a, **{"map_location": "cpu", **kw})


@dataclass
class DiagArguments:
    test_json: str = field(default="")
    train_json: str = field(default="")
    train_per_level: int = field(default=200)
    test_per_level: int = field(default=0, metadata={"help": "0 = every test item"})
    latent_counts: str = field(default="0,1,2,4,6")
    device: str = field(default="auto")
    slug: str = field(default="v2-depth2-codi-latent-diagnostics")
    stage: str = field(default="full_run")
    hardware: str = field(default="local")
    checkpoint_label: Optional[str] = field(default=None)
    smoke_n: int = field(default=0, metadata={"help": "items per level (smoke test, not logged)"})
    no_log: bool = field(default=False)


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def per_level(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """First n items of each level (n = 0: all), or a seeded sample of n per level for the large train file."""
    by: dict[int, list[dict]] = {}
    for r in rows:
        by.setdefault(int(r["level"]), []).append(r)
    out = []
    for lvl in sorted(by):
        items = by[lvl]
        if n and len(items) > n:
            items = random.Random(seed + lvl).sample(items, n)
        out += items
    return out


@torch.no_grad()
def teacher_states(model, tokenizer, rows, batch_size: int) -> tuple[torch.Tensor, list[int]]:
    """Teacher hidden states [N, L, d] (all layers) and argmax answer token at the ':' of "The answer is:"."""
    device = next(model.parameters()).device
    prefix = tokenizer.encode("The answer is:")
    seqs = [tokenizer.encode(r["question"].strip() + "\n") + tokenizer.encode("\n".join(r["steps"][:-1]) + "\n") + prefix
            for r in rows]
    states, preds = [], []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        width = max(len(x) for x in chunk)
        ids = torch.full((len(chunk), width), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for b, x in enumerate(chunk):
            ids[b, :len(x)] = torch.tensor(x)
            mask[b, :len(x)] = 1
        # The transformer body only (LoRA layers are injected in place, so they apply): full-sequence LM-head logits
        # would be [B, T, 50k] (~5 GB at batch 64 for L12) when only the last position is needed.
        lm = model.codi.get_base_model()
        out = lm.transformer(input_ids=ids.to(device), attention_mask=mask.to(device), output_hidden_states=True)
        last = (mask.sum(-1) - 1).to(device)
        ar = torch.arange(len(chunk), device=device)
        states.append(torch.stack([h[ar, last] for h in out.hidden_states], 1).float().cpu())
        logits = lm.lm_head(out.hidden_states[-1][ar, last])
        preds += logits[:, :model.codi.config.vocab_size - 1].argmax(-1).tolist()
        del out
        free(device.type)
    return torch.cat(states), preds


def encode_and_think_lean(model, batch, training_args):
    """eval_codi.encode_and_think, same masks / positions / projection, but through the transformer body: the shared
    helper calls the LM-head model on the question, which builds [B, T, 50k] logits (~5 GB per batch at L12) that are
    never used. LoRA layers sit inside the body, so the computation is the same."""
    body = model.codi.get_base_model().transformer
    pad_inv = getattr(training_args, "pad_invariant", False)
    q_pos = question_position_ids(batch["attention_mask"]) if pad_inv else None
    out = body(input_ids=batch["input_ids"], use_cache=True, attention_mask=batch["attention_mask"], position_ids=q_pos)
    past = out.past_key_values
    latent = out.last_hidden_state[:, -1:, :]
    if training_args.use_prj:
        latent = model.prj(latent)
    pi = PadInvariant(batch["attention_mask"]) if pad_inv else None
    for _ in range(training_args.inf_latent_iterations):
        out = body(inputs_embeds=latent, use_cache=True, past_key_values=past, **(pi.next(1) if pi else {}))
        past = out.past_key_values
        latent = out.last_hidden_state[:, -1:, :]
        if training_args.use_prj:
            latent = model.prj(latent)
    return past, pi


def free(device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()


@torch.no_grad()
def student_states(model, tokenizer, rows, k: int, data_args, training_args) -> tuple[torch.Tensor, list[int]]:
    """Student hidden states [N, L, d] and argmax answer token at the ':' after question, <bot>, k latents, <eot>,
    "The answer is:" (the distillation position), with k latent passes."""
    device = next(model.parameters()).device
    training_args.inf_latent_iterations = k
    dec = [model.eot_id] + tokenizer.encode("The answer is:")
    embed = model.get_embd(model.codi, model.model_name)
    questions = [r["question"].strip() + "\n" for r in rows]
    states, preds = [], []
    for batch in tokenize_questions(model, tokenizer, questions, data_args.batch_size, training_args, device):
        past, pi = encode_and_think_lean(model, batch, training_args)
        emb = embed(torch.tensor(dec, dtype=torch.long, device=device))[None].expand(batch["input_ids"].size(0), -1, -1)
        out = model.codi(inputs_embeds=emb, use_cache=True, past_key_values=past, output_hidden_states=True,
                         **(pi.next(len(dec)) if pi else {"attention_mask": None}))
        states.append(torch.stack([h[:, -1] for h in out.hidden_states], 1).float().cpu())
        preds += out.logits[:, -1, :model.codi.config.vocab_size - 1].argmax(-1).tolist()
        del out, past
        free(device.type)
    return torch.cat(states), preds


def distill_matrix(student: torch.Tensor, teacher: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """[N, L]: per item and layer, mean over dims of SmoothL1(student - teacher) / teacher std of that layer."""
    return F.smooth_l1_loss(student, teacher, reduction="none").mean(-1) / std[None]


def summarise(rows, correct: list[Optional[bool]], dist: Optional[torch.Tensor]) -> dict:
    lv = sorted({int(r["level"]) for r in rows})
    acc, dl, dl_layer = {}, {}, {}
    for L in lv:
        idx = [i for i, r in enumerate(rows) if int(r["level"]) == L and correct[i] is not None]
        acc[L] = round(sum(correct[i] for i in idx) / len(idx), 4) if idx else None
        if dist is not None:
            d = dist[idx]
            dl[L] = round(d.mean().item(), 5)
            dl_layer[L] = [round(x, 5) for x in d.mean(0).tolist()]
    ok = [c for c in correct if c is not None]
    out = {"accuracy": round(sum(ok) / len(ok), 4), "accuracy_per_level": acc}
    if dist is not None:
        out.update({"distill_loss": round(dist.mean().item(), 5), "distill_loss_per_level": dl,
                    "distill_loss_per_layer": [round(x, 5) for x in dist.mean(0).tolist()],
                    "distill_loss_per_level_layer": dl_layer})
    return out


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, DiagArguments))
    model_args, data_args, training_args, da = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    trained_k = training_args.num_latent
    flags = check_flags(model_args.ckpt_dir, training_args)  # needs inf_latent_iterations == num_latent here
    device = pick_device(da.device)
    model, tokenizer, load_result = build_model(model_args, training_args, device=device, dtype=torch.float32)
    if any(not k.startswith("codi.base_model.model.lm_head") for k in load_result.missing_keys):
        raise SystemExit(f"checkpoint is missing weights: {load_result.missing_keys[:5]}")
    print(f"loaded {model_args.ckpt_dir} on {device}: missing={len(load_result.missing_keys)} "
          f"unexpected={len(load_result.unexpected_keys)}", flush=True)
    ks = sorted({int(x) for x in da.latent_counts.split(",")} | {trained_k})

    test = per_level(load_synthetic(da.test_json).rows, da.smoke_n or da.test_per_level)
    train = per_level(load_synthetic(da.train_json).rows, da.smoke_n or da.train_per_level) if da.train_json else []
    gold = {id(r): answer_token(tokenizer, r["answer"]) for r in test + train}

    t0 = time.perf_counter()
    t_states, t_pred = teacher_states(model, tokenizer, test, 16)
    std = t_states.std(dim=(0, 2))  # per layer, over items and dims
    teacher_ok = [None if gold[id(r)] is None else p == gold[id(r)] for r, p in zip(test, t_pred)]
    results = {"teacher": summarise(test, teacher_ok, None), "teacher_std_per_layer": [round(x, 4) for x in std.tolist()]}
    print(f"teacher acc {results['teacher']['accuracy']} ({time.perf_counter() - t0:.0f}s)", flush=True)

    records = [{"idx": i, "id": r["id"], "level": int(r["level"]), "gold_answer": r["answer"],
                "question": r["question"]} for i, r in enumerate(test)]
    sweep = {}
    for k in ks:
        t1 = time.perf_counter()
        s_states, s_pred = student_states(model, tokenizer, test, k, data_args, training_args)
        dist = distill_matrix(s_states, t_states, std)
        ok = [None if gold[id(r)] is None else p == gold[id(r)] for r, p in zip(test, s_pred)]
        sweep[k] = summarise(test, ok, dist)
        for rec, p, c, d in zip(records, s_pred, ok, dist.mean(1).tolist()):
            rec[f"tf_pred_k{k}"] = tokenizer.decode([p]).strip()
            rec[f"tf_correct_k{k}"] = c
            rec[f"distill_k{k}"] = round(d, 5)
        print(f"k={k}: acc {sweep[k]['accuracy']} distill {sweep[k]['distill_loss']} "
              f"per-level acc {sweep[k]['accuracy_per_level']} ({time.perf_counter() - t1:.0f}s)", flush=True)
    results["latent_sweep"] = sweep

    if train:
        _, tr_pred = student_states(model, tokenizer, train, trained_k, data_args, training_args)
        tr_ok = [None if gold[id(r)] is None else p == gold[id(r)] for r, p in zip(train, tr_pred)]
        tr = summarise(train, tr_ok, None)
        te = sweep[trained_k]  # every evaluated test item at the trained latent count
        results["train_vs_test"] = {"train_per_level_n": da.smoke_n or da.train_per_level,
                                    "train_accuracy_per_level": tr["accuracy_per_level"],
                                    "test_accuracy_per_level": te["accuracy_per_level"],
                                    "gap_per_level": {L: round(tr["accuracy_per_level"][L] - te["accuracy_per_level"][L], 4)
                                                      for L in tr["accuracy_per_level"]}}
        print("train vs test:", json.dumps(results["train_vs_test"]["gap_per_level"]), flush=True)

    elapsed = time.perf_counter() - t0
    for rec in records:
        rec["predicted_answer"] = rec[f"tf_pred_k{trained_k}"]
        rec["correct"] = rec[f"tf_correct_k{trained_k}"]
    out_json = json.dumps(results, indent=1, default=str)
    if da.smoke_n or da.no_log:
        print(out_json[:3000])
        print("not logging (smoke_n / no_log)")
        return

    acc_full = sweep[trained_k]["accuracy"]
    metrics = {
        "final_answer_accuracy": acc_full,  # teacher-forced at the distillation position, trained latent count
        "unparseable_rate": 0.0,
        "compute_steps": trained_k,
        "sec_per_example": elapsed / len(test),
        "extra": {"accuracy_kind": "teacher-forced answer token at the distillation position (no generation)",
                  "trained_latent_count": trained_k, "latent_counts": ks,
                  "data_sha256": hashlib.sha256(Path(da.test_json).read_bytes()).hexdigest(),
                  "matched_flags": flags, **results},
    }
    if trained_k > 0 and 0 in sweep:
        metrics["early_termination_necessity"] = round(acc_full - sweep[0]["accuracy"], 4)
    hp = {"inference": {"latent_counts": ks, "batch_size": data_args.batch_size, "dtype": "float32", "device": device},
          "test_json": da.test_json, "train_json": da.train_json, "train_per_level": da.train_per_level}
    record = RunRecord(
        run_id=new_run_id("codi", da.slug),
        mechanism="codi",
        stage=da.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=da.checkpoint_label or model_args.ckpt_dir,
                        n_params=sum(p.numel() for p in model.parameters())),
        dataset=DatasetInfo(name="synthetic-depth2", split="test", n_examples=len(test), seed=None),
        metrics=metrics,
        hyperparams=hp,
        seed=None,
        hardware=f"{da.hardware} / {device}",
        notes="RuleChain CODI latent diagnostics: latent-count sweep, per-layer distillation loss, train vs test.",
    )
    manifest = record.save(predictions=records)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
