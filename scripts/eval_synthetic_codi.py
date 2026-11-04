#!/usr/bin/env python3
"""Per-level test eval of a CODI checkpoint trained on the synthetic depth / width tasks
(`configs/synthetic/train_gpt2_depth_codi.sh`, CODI repo + scripts/codi_synthetic.patch).

Runs INSIDE the CODI venv, from the CODI checkout, like `eval_codi.py` (whose model
construction and the reference test.py generation loop it reuses unchanged):

  cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/eval_synthetic_codi.py \\
      --ckpt_dir /workspace/codi_runs/gpt2_depth2_codi/gpt2/ep_40/lr_0.003/seed_11/checkpoint-<STEP> \\
      --test_json /workspace/synthetic_data_v2/depth2/valid.json --no_log True \\
      --output_dir /tmp/codi_eval_unused \\
      --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \\
      --lora_r 128 --lora_alpha 32 --lora_init --batch_size 64 --greedy True \\
      --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \\
      --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \\
      --fix_attn_mask True --pad_invariant True --include_last_cot False
  # logged test eval of the selected checkpoint: --test_json .../depth2/test.json --slug v2-depth2-<run> --stage full_run
  #   --hardware "RunPod RTX A5000 (secure)" --checkpoint_label <ckpt> (drop --no_log)
  # CODI-0 / student-only: --num_latent 0 --inf_latent_iterations 0

The eval flags must match the checkpoint's training_args.bin (checked; MATCHED_FLAGS). With pad_invariant the batch size
does not change results (scripts/check_codi_padding.py), so batched eval is fine. Besides generation accuracy it reports
teacher-forced student accuracy at the distillation position, teacher accuracy, and accuracy by answer value.

Prompt = the training format of the patch's `depth`/`width` branch: question.strip() + "\\n",
then <bot>, 6 latent iterations, <eot>, greedy decode. Scoring: last number in the output
(CODI test.py's metric, the target is "The answer is: N"), compared with
`latentreasoning.eval.metrics.is_correct`. Per-level accuracy / midpoint / AUC via
`scripts/depth_curve.py`.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers

sys.path.insert(0, os.getcwd())  # the CODI checkout: `src.model`
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.model import DataArguments, ModelArguments, TrainingArguments  # noqa: E402

from depth_curve import curve_metrics  # noqa: E402
from eval_codi import build_model, encode_and_think, generate_batches, hyperparams_from_ckpt, tokenize_questions  # noqa: E402
from synthetic_common import annotate_records, load_synthetic  # noqa: E402

from latentreasoning.eval.metrics import is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402


@dataclass
class SynthEvalArguments:
    test_json: str = field(default="")
    slug: str = field(default="synthetic-codi")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: Optional[str] = field(default=None)
    train_log_from: Optional[str] = field(default=None, metadata={"help": "trainer_state.json"})
    seconds_per_epoch: Optional[float] = field(default=None)
    smoke_n: int = field(default=0)
    no_log: bool = field(default=False, metadata={"help": "print accuracy only, no RunRecord"})
    allow_flag_mismatch: bool = field(default=False, metadata={"help": "skip the check that eval flags match the checkpoint's training_args.bin"})


def last_number(text: str) -> Optional[str]:
    import re
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None


# Flags that must be the same at eval as in training (from the checkpoint's training_args.bin). A mismatch silently
# evaluates a different model (e.g. pad_invariant off at eval runs the latent passes without masks).
MATCHED_FLAGS = ("pad_invariant", "fix_attn_mask", "use_prj", "prj_dim", "prj_no_ln", "remove_eos", "include_last_cot")


def check_flags(ckpt_dir: str, training_args) -> dict:
    path = Path(ckpt_dir) / "training_args.bin"
    if not path.exists():
        raise SystemExit(f"no training_args.bin in {ckpt_dir}: cannot check eval flags (pass --allow_flag_mismatch True)")
    ta = torch.load(path, weights_only=False)
    bad = {f: (getattr(ta, f, None), getattr(training_args, f, None)) for f in MATCHED_FLAGS
           if getattr(ta, f, None) != getattr(training_args, f, None)}
    if getattr(ta, "num_latent", None) != training_args.inf_latent_iterations:
        bad["num_latent vs inf_latent_iterations"] = (getattr(ta, "num_latent", None), training_args.inf_latent_iterations)
    if bad:
        raise SystemExit(f"eval flags differ from training (train, eval): {bad}  (pass --allow_flag_mismatch True to override)")
    return {f: getattr(ta, f, None) for f in MATCHED_FLAGS}


def answer_token(tokenizer, answer) -> Optional[int]:
    """Token id of the answer in the training target "The answer is: N" (the token after the prefix), or None if N is
    more than one token (then the teacher-forced readouts are undefined for that item)."""
    prefix = tokenizer.encode("The answer is:")
    full = tokenizer.encode(f"The answer is: {str(answer).strip()}")
    return full[len(prefix)] if len(full) == len(prefix) + 1 and full[:len(prefix)] == prefix else None


@torch.no_grad()
def teacher_forced_student(model, tokenizer, questions, data_args, training_args) -> list[int]:
    """Argmax answer token after question, <bot>, the latents, <eot> and the teacher-forced "The answer is:"
    (Liang & Pan's metric: one forward, no generation). This is the distillation position."""
    device = next(model.parameters()).device
    dec = [model.eot_id] if training_args.remove_eos else [model.eot_id, tokenizer.eos_token_id]
    dec = dec + tokenizer.encode("The answer is:")
    embed = model.get_embd(model.codi, model.model_name)
    preds = []
    for batch in tokenize_questions(model, tokenizer, questions, data_args.batch_size, training_args, device):
        past, pi, _ = encode_and_think(model, batch, training_args)
        emb = embed(torch.tensor(dec, dtype=torch.long, device=device))[None].expand(batch["input_ids"].size(0), -1, -1)
        out = model.codi(inputs_embeds=emb, use_cache=True, past_key_values=past,
                         **(pi.next(len(dec)) if pi else {"attention_mask": None}))
        preds += out.logits[:, -1, :model.codi.config.vocab_size - 1].argmax(-1).tolist()
    return preds


@torch.no_grad()
def teacher_answers(model, tokenizer, rows, include_last_cot: bool, batch_size: int) -> list[int]:
    """Argmax answer token of the TEACHER pass: question + CoT (last step dropped unless include_last_cot, as in
    codi_synthetic.patch) + "The answer is:", tokenized per segment like CODI's preprocess. Right-padded, so no
    position shift (as in training)."""
    device = next(model.parameters()).device
    prefix = tokenizer.encode("The answer is:")
    seqs = []
    for r in rows:
        steps = r["steps"] if include_last_cot else r["steps"][:-1]
        seqs.append(tokenizer.encode(r["question"].strip() + "\n") + tokenizer.encode("\n".join(steps) + "\n") + prefix)
    preds = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        width = max(len(x) for x in chunk)
        ids = torch.full((len(chunk), width), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for b, x in enumerate(chunk):
            ids[b, :len(x)] = torch.tensor(x)
            mask[b, :len(x)] = 1
        logits = model.codi(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        last = mask.sum(-1).to(device) - 1
        preds += logits[torch.arange(len(chunk), device=device), last, :model.codi.config.vocab_size - 1].argmax(-1).tolist()
    return preds


def by_key(records: list[dict], key: str, field: str) -> dict:
    out: dict = {}
    for r in records:
        if r.get(field) is None:
            continue
        out.setdefault(r[key], []).append(bool(r[field]))
    return {k: round(sum(v) / len(v), 4) for k, v in sorted(out.items())}


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, SynthEvalArguments))
    model_args, data_args, training_args, ea = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    flags = {} if ea.allow_flag_mismatch else check_flags(model_args.ckpt_dir, training_args)
    cuda = torch.cuda.is_available()  # CPU fallback is for smoke tests only
    model, tokenizer, load_result = build_model(model_args, training_args, device="cuda" if cuda else "cpu",
                                                dtype=torch.bfloat16 if cuda else torch.float32)
    if any(not k.startswith("codi.base_model.model.lm_head") for k in load_result.missing_keys):
        raise SystemExit(f"checkpoint is missing weights (wrong --ckpt_dir?): {load_result.missing_keys[:5]}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {model_args.ckpt_dir}: missing={len(load_result.missing_keys)} "
          f"unexpected={len(load_result.unexpected_keys)}", flush=True)

    ds = load_synthetic(ea.test_json).subset(ea.smoke_n)
    questions = [r["question"].strip() + "\n" for r in ds.rows]
    t0 = time.perf_counter()
    outputs = generate_batches(model, tokenizer, questions, data_args, training_args)
    elapsed = time.perf_counter() - t0

    records = []
    for ex, out in zip(ds.examples, outputs):
        pred = last_number(out)
        records.append({"idx": ex.idx, "question": ex.question, "gold_answer": ex.answer, "raw_output": out,
                        "predicted_answer": pred, "correct": is_correct(pred, ex.answer)})
    annotate_records(records, ds)
    # Liang & Pan-style readouts (TRAINING_PROTOCOL.md section 5.4): teacher-forced student answer at the distillation
    # position, the teacher's own answer (question + CoT minus the last step), and accuracy by gold answer value.
    tf_pred = teacher_forced_student(model, tokenizer, questions, data_args, training_args)
    t_pred = teacher_answers(model, tokenizer, ds.rows, training_args.include_last_cot, max(data_args.batch_size, 32))
    for r, tp, tt in zip(records, tf_pred, t_pred):
        gold_tok = answer_token(tokenizer, r["gold_answer"])
        r["tf_correct"] = None if gold_tok is None else tp == gold_tok
        r["teacher_correct"] = None if gold_tok is None else tt == gold_tok
    n = len(records)
    acc = sum(r["correct"] for r in records) / n
    unparseable = sum(r["predicted_answer"] is None for r in records) / n
    curve = curve_metrics(records)
    print(f"accuracy={acc:.3f} unparseable={unparseable:.3f} midpoint={curve['midpoint']} "
          f"({curve['midpoint_status']}) auc={curve['auc']:.3f} {elapsed / n:.4f}s/example", flush=True)
    print("per-level:", json.dumps(curve["per_level_accuracy"]), flush=True)
    readouts = {"teacher_forced_accuracy_per_level": by_key(records, "level", "tf_correct"),
                "teacher_accuracy_per_level": by_key(records, "level", "teacher_correct"),
                "accuracy_by_answer_value": by_key(records, "gold_answer", "correct")}
    print("teacher-forced per-level:", json.dumps(readouts["teacher_forced_accuracy_per_level"]), flush=True)
    print("teacher per-level:", json.dumps(readouts["teacher_accuracy_per_level"]), flush=True)
    for r in records[:3]:
        print(f"  [{r['id']} L{r['level']}] gold={r['gold_answer']} pred={r['predicted_answer']} "
              f"out={r['raw_output'][:120]!r}", flush=True)
    if ea.smoke_n or ea.no_log:
        print("not logging (smoke_n / no_log)")
        return

    log_history = None
    if ea.train_log_from:
        log_history = json.loads(Path(ea.train_log_from).read_text()).get("log_history", [])
    hp = hyperparams_from_ckpt(model_args.ckpt_dir) or {}
    hp["inference"] = {"inf_latent_iterations": training_args.inf_latent_iterations, "greedy": training_args.greedy,
                       "batch_size": data_args.batch_size, "max_new_tokens": 256,
                       "prompt": "question.strip() + '\\n' + <bot>"}
    hp["test_json"] = ea.test_json
    record = RunRecord(
        run_id=new_run_id("codi", ea.slug),
        mechanism="codi",
        stage=ea.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=ea.checkpoint_label or model_args.ckpt_dir, n_params=n_params),
        dataset=DatasetInfo(name=f"synthetic-{ds.task}", split="validation" if Path(ea.test_json).stem.startswith("valid") else "test", n_examples=n, seed=None),
        metrics={
            "final_answer_accuracy": acc,
            "unparseable_rate": unparseable,
            "compute_steps": training_args.inf_latent_iterations,
            "sec_per_example": elapsed / n,
            "extra": {"task": ds.task, "task_level_range": ds.level_range,
                      "seconds_per_epoch": ea.seconds_per_epoch, **curve, **readouts,
                      "data_sha256": hashlib.sha256(Path(ea.test_json).read_bytes()).hexdigest(),
                      "matched_flags": flags},
        },
        hyperparams=hp,
        seed=hp.get("seed"),
        hardware=f"{ea.hardware} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        notes=f"CODI GPT-2 on synthetic {ds.task}, per-level test eval.",
    )
    manifest = record.save(predictions=records)
    if log_history is not None:
        (manifest.parent / "train_log.json").write_text(json.dumps(log_history, indent=2))
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
