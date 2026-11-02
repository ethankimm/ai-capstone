#!/usr/bin/env python3
"""Logit lens on a RuleChain (depth2) CODI / CODI-0 checkpoint: project the hidden state at <bot>, each latent pass and
the answer position (":"), at layers 0, 2, ..., 12, through GPT-2's final LayerNorm and unembedding (layer 12 is
already normed in HF GPT-2), and check whether the top-1 / top-5 tokens are the item's intermediate values v_1..v_k
(result of rule j; RuleChain values 0..99 are single tokens, with or without a leading space) or its answer.

No training, so unlike scripts/probe_swap_codi_latents.py it cannot overfit; it only sees what is written in the
vocabulary basis. CODI's paper reads its GSM8K latents the same way (decoding continuous thoughts with the LM head).

Run from the CODI checkout in its venv, flags as for probe_swap_codi_latents.py.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_swap_codi_latents import PROBE_LAYERS, per_level, run, step_values  # noqa: E402  (also applies the CPU torch.load map)
from src.model import DataArguments, ModelArguments, TrainingArguments  # noqa: E402
from eval_codi import build_model  # noqa: E402
from eval_synthetic_codi import answer_token, check_flags  # noqa: E402
from synthetic_common import load_synthetic  # noqa: E402

from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402


@dataclass
class Args:
    test_json: str = field(default="")
    smoke_n: int = field(default=0)
    slug: str = field(default="v2-depth2-codi-logit-lens")
    stage: str = field(default="full_run")
    hardware: str = field(default="local")
    checkpoint_label: Optional[str] = field(default=None)
    no_log: bool = field(default=False)


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, Args))
    model_args, data_args, training_args, a = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    flags = check_flags(model_args.ckpt_dir, training_args)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = build_model(model_args, training_args, device=device, dtype=torch.float32)
    k = training_args.num_latent
    test = per_level(load_synthetic(a.test_json).rows, a.smoke_n)
    t0 = time.perf_counter()
    preds, _, states = run(model, tokenizer, test, k, training_args, data_args.batch_size, collect=True)
    gold = [answer_token(tokenizer, r["answer"]) for r in test]
    vals = [step_values(r) for r in test]

    lm = model.codi.get_base_model()
    ln_f, head = lm.transformer.ln_f, lm.lm_head
    vocab = model.codi.config.vocab_size - 1
    # token id -> integer value it spells (0..99, with or without leading space), else -1
    tok_val = torch.full((vocab,), -1, dtype=torch.long)
    for v in range(100):
        for s in (str(v), " " + str(v)):
            ids = tokenizer.encode(s)
            if len(ids) == 1:
                tok_val[ids[0]] = v
    tok_val = tok_val.to(device)

    results: dict = {"positions": list(states), "layers": list(PROBE_LAYERS)}
    top_tokens: dict = {}
    for pos, h in states.items():  # h: [N, len(PROBE_LAYERS), d] fp16 on CPU
        res_pos = {}
        for li, layer in enumerate(PROBE_LAYERS):
            top1_val, top5_val = [], []
            counter = collections.Counter()
            for s in range(0, len(test), 256):
                x = h[s:s + 256, li].float().to(device)
                if layer != 12:
                    x = ln_f(x)
                with torch.no_grad():
                    logits = head(x)[:, :vocab]
                top5 = logits.topk(5, -1).indices
                top1_val.append(tok_val[top5[:, 0]].cpu())
                top5_val.append(tok_val[top5].cpu())
                counter.update(tokenizer.decode([t]) for t in top5[:, 0].tolist())
            t1, t5 = torch.cat(top1_val), torch.cat(top5_val)
            row = {"top1_is_number": round((t1 >= 0).float().mean().item(), 4)}
            for j in range(1, 7):
                idx = [i for i, v in enumerate(vals) if len(v) >= j]
                tgt = torch.tensor([vals[i][j - 1] for i in idx])
                row[f"v{j}_top1"] = round((t1[idx] == tgt).float().mean().item(), 4)
                row[f"v{j}_top5"] = round((t5[idx] == tgt[:, None]).any(-1).float().mean().item(), 4)
            ans = torch.tensor([v[-1] for v in vals])
            row["answer_top1"] = round((t1 == ans).float().mean().item(), 4)
            row["answer_top5"] = round((t5 == ans[:, None]).any(-1).float().mean().item(), 4)
            # any intermediate value of the item (rules 1..k-1, excluding the answer) in top-1
            row["any_intermediate_top1"] = round(sum(
                int(t1[i]) in v[:-1] for i, v in enumerate(vals) if len(v) > 1) / sum(len(v) > 1 for v in vals), 4)
            res_pos[layer] = row
            if layer in (8, 12):
                top_tokens[f"{pos}@{layer}"] = [(t, c) for t, c in counter.most_common(8)]
        results[pos] = res_pos
        best = {m: max(res_pos[l][m] for l in PROBE_LAYERS) for m in ("v1_top1", "v2_top1", "v3_top1", "answer_top1", "any_intermediate_top1")}
        print(f"{pos}: best over layers " + json.dumps(best) + f" | top1 @12: {top_tokens.get(f'{pos}@12', [])[:5]}", flush=True)
    results["top_tokens"] = top_tokens

    if a.smoke_n or a.no_log:
        print("not logging")
        return
    acc = sum(p == g for p, g in zip(preds, gold) if g is not None) / sum(g is not None for g in gold)
    records = [{"idx": i, "id": r["id"], "level": int(r["level"]), "gold_answer": r["answer"],
                "predicted_answer": tokenizer.decode([p]).strip(), "correct": None if g is None else p == g}
               for i, (r, p, g) in enumerate(zip(test, preds, gold))]
    record = RunRecord(
        run_id=new_run_id("codi", a.slug), mechanism="codi", stage=a.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=a.checkpoint_label or model_args.ckpt_dir,
                        n_params=sum(p.numel() for p in model.parameters())),
        dataset=DatasetInfo(name="synthetic-depth2", split="test", n_examples=len(test), seed=None),
        metrics={"final_answer_accuracy": round(acc, 4), "unparseable_rate": 0.0, "compute_steps": k,
                 "sec_per_example": (time.perf_counter() - t0) / len(test),
                 "extra": {"accuracy_kind": "teacher-forced answer token at the distillation position",
                           "data_sha256": hashlib.sha256(Path(a.test_json).read_bytes()).hexdigest(),
                           "matched_flags": flags, "logit_lens": results}},
        hyperparams={"latent_count": k, "batch_size": data_args.batch_size, "layers": list(PROBE_LAYERS),
                     "lens": "final LayerNorm + tied unembedding", "device": device},
        seed=None, hardware=f"{a.hardware} / {device}",
        notes="RuleChain CODI logit lens on <bot>, latent passes and answer position.",
    )
    manifest = record.save(predictions=records)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id}")


if __name__ == "__main__":
    main()
