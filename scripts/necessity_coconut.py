#!/usr/bin/env python3
"""Necessity of Coconut's latents under every replacement type -- the Coconut
counterpart to `necessity_codi.py` (same conditions, same scoring via
`necessity_common.py`). Coconut had no necessity run at all before this.

Sites are passes 0..5: pass p = the vector spliced into the p-th <|latent|> slot (pass 0
= the hidden state at <|start-latent|>), same numbering as CODI's z_0..z_5.
  trunc_k           k <|latent|> tokens between <|start-latent|> and <|end-latent|>
                    (k=0: question + start + end), then the answer. trunc_6 = baseline.
  ablate_all_<r>    every pass's spliced vector replaced; ablate_single_<r>_<p> one pass.
  r = zero | mean (population mean of pass p's live hidden over `mean_sample_n` test
      examples) | noise (Gaussian direction scaled to this example's clean pass-p norm,
      seed idx*1000+p)
pass@k: `k_samples` answers sampled at `temperature` from the deterministic latent state
for baseline, trunc_0 and ablate_all_{zero,mean,noise}.

Data: `gsm_original_test.json` (all 1319 GSM8K test questions -- the same questions as
CODI's test split), answer after `###`.

  cd /workspace/ai-capstone && for i in 0 1; do nohup .venv_coconut/bin/python scripts/necessity_coconut.py \\
      --checkpoint_path /workspace/coconut_checkpoints/checkpoint_33 --data_dir /workspace/coconut_data \\
      --num_shards 2 --shard_index $i --shard_dir /workspace/necessity_coconut > nec_coconut_$i.log 2>&1 & done
  # then: --merge_dir /workspace/necessity_coconut --slug necessity-all-replacements --stage full_run --hardware ...
"""
from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coconut_common import (  # noqa: E402
    encode_question, extract_answer_after_delimiter, finish_and_decode, load_coconut, run_passes,
)
import necessity_common as nc  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from latentreasoning.eval.metrics import is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402


@dataclass
class NecArguments:
    checkpoint_path: str = field(default="/workspace/coconut_checkpoints/checkpoint_33")
    data_dir: str = field(default="/workspace/coconut_data")
    model_id: str = field(default="openai-community/gpt2")
    checkpoint_label: str = field(default="hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33")
    slug: str = field(default="necessity-all-replacements")
    stage: str = field(default="full_run")
    hardware: str = field(default="RunPod GPU")
    num_latents: int = field(default=6)
    smoke_n: int = field(default=0)
    mean_sample_n: int = field(default=300)
    max_new_tokens: int = field(default=48)
    k_samples: int = field(default=10)
    temperature: float = field(default=0.7)
    num_shards: int = field(default=1)
    shard_index: int = field(default=0)
    shard_dir: str = field(default="/workspace/necessity_coconut")
    merge_dir: Optional[str] = field(default=None)
    device: str = field(default="cuda")


def load_test(data_dir: str) -> list[dict]:
    rows = json.loads((Path(data_dir) / "gsm_original_test.json").read_text())
    return [{"idx": i, "question": r["question"], "answer": r["answer"].strip()} for i, r in enumerate(rows)]


def finalize(merged: dict, na: NecArguments, argv: str, gold: dict) -> None:
    meta = merged["meta"]
    n_sites = meta["n_sites"]
    records, samples = merged["records"], merged["samples"]
    baseline = f"trunc_{n_sites}"
    summary = nc.summarize(records, samples, baseline, meta["k_samples"])
    nc.print_table(summary, meta["k_samples"])
    curve = {k: summary[f"trunc_{k}"]["accuracy"] for k in range(n_sites + 1)}
    acc = summary[baseline]["accuracy"]
    n = summary[baseline]["n"]
    print(f"early_termination_necessity = {acc:.4f} - {curve[0]:.4f} = {acc - curve[0]:.4f}")
    if na.smoke_n:
        return
    metrics = {
        "final_answer_accuracy": acc,
        "unparseable_rate": summary[baseline]["unparseable_rate"],
        "compute_steps": n_sites,
        "sec_per_example": sum(merged["elapsed_sec"]) / n,
        "early_termination_necessity": acc - curve[0],
        "extra": {
            "early_termination_necessity_definition": f"acc(trunc_{n_sites}) - acc(trunc_0), greedy, all 1319 test questions",
            "site_definition": "pass p = vector spliced into the p-th <|latent|> slot; pass 0 = hidden at <|start-latent|>",
            "replacements": {"zero": "all-zeros", "mean": f"population mean of pass p over {meta['mean_sample_n']} test examples",
                             "noise": "Gaussian direction scaled to this example's clean pass-p norm, seed idx*1000+p"},
            "pwc_definition": "P(condition correct | baseline trunc_6 correct)",
            "sampling": {"k": meta["k_samples"], "temperature": meta["temperature"],
                         "what_is_sampled": "answer tokens only; latents are deterministic",
                         "conditions": nc.sampled_condition_names(n_sites)},
            "truncation_curve_accuracy_by_k": curve,
            "conditions": summary,
            "mean_norms": meta["mean_norms"],
            "num_shards": meta["num_shards"],
            "data_file": "gsm_original_test.json (Dilgren & Wiegreffe prep)",
        },
    }
    record = RunRecord(
        run_id=new_run_id("coconut", na.slug),
        mechanism="coconut",
        stage=na.stage,
        model=ModelInfo(backbone=na.model_id, checkpoint=meta["checkpoint"], n_params=meta["n_params"]),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=n, seed=None),
        metrics=metrics,
        hyperparams={"num_latents": n_sites, "greedy": True, "mean_sample_n": meta["mean_sample_n"],
                     "max_new_tokens": meta["max_new_tokens"], "k_samples": meta["k_samples"],
                     "temperature": meta["temperature"]},
        seed=0,
        hardware=f"{na.hardware} / {meta['gpu']}",
        notes="Necessity, Coconut: early termination + ablate-all/single x {zero, mean, norm-matched noise}, "
              "full test set, PWC + flip split, pass@k on ablate-all.",
    )
    path = record.save(predictions=nc.predictions_rows(records, samples, gold))
    (path.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {path.parent / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


def main() -> None:
    (na,) = transformers.HfArgumentParser((NecArguments,)).parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    test = load_test(na.data_dir)
    gold = {r["idx"]: r["answer"] for r in test}
    if na.merge_dir:
        finalize(nc.load_shards(na.merge_dir), na, argv, gold)
        return

    device = na.device
    tokenizer, base_model, embedding, special_ids = load_coconut(na.model_id, na.checkpoint_path, device)
    eos_id = tokenizer.eos_token_id
    n_sites = na.num_latents

    def state(question: str, k: int, override=None):
        input_ids, attn = encode_question(tokenizer, special_ids, question, k, device)
        if k == 0:
            return embedding(input_ids), None, (0, input_ids.shape[1]), attn, []
        recs, ie, kv, ncr = run_passes(base_model, embedding, input_ids, attn, device, k,
                                       override_at_pass=override, latent_token_id=special_ids["latent"])
        return ie, kv, ncr, attn, recs

    def greedy(st) -> Optional[str]:
        ie, kv, ncr, attn, _ = st
        raw = finish_and_decode(base_model, embedding, tokenizer, ie, kv, ncr, attn, device, na.max_new_tokens, eos_id)
        return extract_answer_after_delimiter(raw)

    @torch.no_grad()
    def sampled(st, answer: str, seed: int) -> list[bool]:
        """`finish_and_decode`, batched over k temperature samples."""
        ie, kv, ncr, attn, _ = st
        k = na.k_samples
        gen = torch.Generator(device=device).manual_seed(seed)
        seq_len = ie.shape[1]
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(k, -1)
        ie_k = ie.expand(k, -1, -1).contiguous()
        attn_k = attn.expand(k, -1)
        past = ([(a[:, :, :ncr[0], :].repeat(k, 1, 1, 1), b[:, :, :ncr[0], :].repeat(k, 1, 1, 1)) for a, b in kv]
                if kv else None)
        out = base_model(inputs_embeds=ie_k[:, ncr[0]:ncr[1], :], attention_mask=attn_k[:, :ncr[1]],
                         position_ids=pos[:, ncr[0]:ncr[1]], past_key_values=past)
        toks: list[list[int]] = [[] for _ in range(k)]
        done = [False] * k
        seq = ie_k
        for step in range(na.max_new_tokens):
            logits = out.logits[:, -1].float() / na.temperature
            ids = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=gen)  # [k, 1]
            for r in range(k):
                t = ids[r, 0].item()
                if not done[r]:
                    if t == eos_id and step > 0:  # finish_and_decode keeps the first token unconditionally
                        done[r] = True
                    else:
                        toks[r].append(t)
            if all(done):
                break
            seq = torch.cat((seq, embedding(ids)), dim=1)
            out = base_model(inputs_embeds=seq)
        return [is_correct(extract_answer_after_delimiter(tokenizer.decode(t, skip_special_tokens=True)), answer)
                for t in toks]

    # population mean of each pass's live hidden
    pool = list(test)
    random.Random(0).shuffle(pool)
    acc_sum = None
    for r in pool[:na.mean_sample_n]:
        recs = state(r["question"], n_sites)[4]
        v = torch.stack([rec["live_hidden"] for rec in recs])
        acc_sum = v if acc_sum is None else acc_sum + v
    mean_z = {p: (acc_sum[p] / na.mean_sample_n) for p in range(n_sites)}
    zero = torch.zeros_like(mean_z[0])
    mean_norms = [mean_z[p].norm().item() for p in range(n_sites)]
    print("mean pass norms: " + ", ".join(f"p{p}={v:.2f}" for p, v in enumerate(mean_norms)))

    examples = test[:na.smoke_n] if na.smoke_n else test[na.shard_index::na.num_shards]
    print(f"shard {na.shard_index}/{na.num_shards}: {len(examples)} examples")
    conds = nc.condition_names(n_sites)
    sampled_conds = nc.sampled_condition_names(n_sites)
    records = {c: {} for c in conds}
    samples = {c: {} for c in sampled_conds}
    t0 = time.perf_counter()
    for n_done, ex in enumerate(examples, 1):
        q, idx, ans = ex["question"], ex["idx"], ex["answer"]

        def put(cond, pred):
            records[cond][idx] = {"pred": pred, "correct": is_correct(pred, ans)}

        clean = None
        for k in range(n_sites + 1):
            st = state(q, k)
            if k == n_sites:
                clean = st[4]
            if f"trunc_{k}" in samples:
                samples[f"trunc_{k}"][idx] = sampled(st, ans, idx * 100 + k)
            put(f"trunc_{k}", greedy(st))

        noise_z = {}
        for p in range(n_sites):
            g = torch.Generator().manual_seed(idx * 1000 + p)
            d = torch.randn(clean[p]["live_hidden"].numel(), generator=g)
            noise_z[p] = d / d.norm() * clean[p]["live_hidden"].norm()
        repl = {"zero": lambda p: zero, "mean": lambda p: mean_z[p], "noise": lambda p: noise_z[p]}
        for ri, (r, vec) in enumerate(repl.items()):
            st = state(q, n_sites, override={p: vec(p) for p in range(n_sites)})
            samples[f"ablate_all_{r}"][idx] = sampled(st, ans, idx * 100 + 50 + ri)
            put(f"ablate_all_{r}", greedy(st))
            for p in range(n_sites):
                put(f"ablate_single_{r}_{p}", greedy(state(q, n_sites, override={p: vec(p)})))

        if n_done % 25 == 0 or n_done == len(examples):
            el = time.perf_counter() - t0
            acc = sum(v["correct"] for v in records[f"trunc_{n_sites}"].values()) / n_done
            print(f"{n_done}/{len(examples)} ({el:.0f}s, {el / n_done:.2f}s/ex, baseline acc {acc:.3f})", flush=True)
    elapsed = time.perf_counter() - t0

    meta = {"n_sites": n_sites, "num_shards": na.num_shards, "mean_sample_n": na.mean_sample_n,
            "mean_norms": mean_norms, "k_samples": na.k_samples, "temperature": na.temperature,
            "max_new_tokens": na.max_new_tokens, "checkpoint": na.checkpoint_label,
            "n_params": sum(p.numel() for p in base_model.parameters()),
            "gpu": torch.cuda.get_device_name(0), "elapsed_sec": elapsed}
    if na.smoke_n:
        finalize({"meta": meta, "records": records, "samples": samples, "elapsed_sec": [elapsed]}, na, argv, gold)
        return
    path = nc.write_shard(na.shard_dir, na.shard_index, {"meta": meta, "records": records, "samples": samples})
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
