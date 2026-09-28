#!/usr/bin/env python3
"""Experiment A2 (question-only control for P4, RESEARCH_PLAN pasted plan): dump plain HF
`gpt2` (no latent training at all) hidden states for the shared question set
(`xmech_questions.jsonl`), in the same dump shape `xmech_common.py` expects (`z: [S, 768]`
per key) so it plugs into `xmech_common.fit_site_maps` / `run_transplant` unchanged --
here S=3 "feature sites" instead of a mechanism's 6 latent sites:
  0  last-token hidden state, layer 6
  1  last-token hidden state, layer 12 (final)
  2  mean over question tokens, hidden state layer 12
No CUDA needed: ~9k short forward passes through 124M params runs fine on MPS/CPU.

  uv run --extra train python scripts/gpt2_plain_dump.py \\
      --questions ~/Documents/Penn/CIS5980/xmech_artifacts/xmech_questions.jsonl \\
      --out ~/Documents/Penn/CIS5980/xmech_artifacts/gpt2_latents.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def prep(q: str) -> str:
    return q.strip().replace("  ", " ")


def pick_device(spec: str) -> str:
    if spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def features_for(model, tokenizer, device: str, question: str) -> torch.Tensor:
    ids = tokenizer(prep(question), return_tensors="pt").to(device)
    out = model(**ids, output_hidden_states=True)
    hs = out.hidden_states  # hs[0] = embeddings, hs[i] = output of block i
    layer6_last = hs[6][0, -1, :]
    layer12_last = hs[12][0, -1, :]
    layer12_mean = hs[12][0, :, :].mean(dim=0)
    return torch.stack([layer6_last, layer12_last, layer12_mean]).float().cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_id", default="gpt2")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log_every", type=int, default=500)
    a = ap.parse_args()

    device = pick_device(a.device)
    tokenizer = GPT2TokenizerFast.from_pretrained(a.model_id)
    model = GPT2LMHeadModel.from_pretrained(a.model_id).to(device).eval()
    print(f"loaded {a.model_id} on {device}")

    questions = [json.loads(line) for line in Path(a.questions).read_text().splitlines() if line.strip()]
    rows = {}
    t0 = time.perf_counter()
    for i, q in enumerate(questions):
        rows[q["key"]] = {"z": features_for(model, tokenizer, device, q["question"]), "pred": None, "correct": None}
        if (i + 1) % a.log_every == 0:
            elapsed = time.perf_counter() - t0
            print(f"{i + 1}/{len(questions)} ({elapsed:.0f}s, {elapsed / (i + 1) * 1000:.0f}ms/ex)", flush=True)

    torch.save({"meta": {"mechanism": "gpt2_plain", "model_id": a.model_id,
                         "site_definition": "0=layer6 last-token,1=layer12 last-token,"
                                             "2=layer12 mean-over-question-tokens"}, "rows": rows}, a.out)
    print(f"dumped {len(rows)} in {time.perf_counter() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
