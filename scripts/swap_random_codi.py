#!/usr/bin/env python3
"""CODI counterpart to `swap_random_coconut.py`: Zhang et al. 2025's random-permutation
full-chain latent swap, next to the E3 twin swap on the same recipients (GSM8K only), on
`zen-E/CODI-gpt2` (GSM8K) and the Dilgren & Wiegreffe grid CODI checkpoints (ProntoQA, ProsQA).

Loads with the grid repo's device-agnostic inference wrapper (`models/codi/codi_wrapper.py`,
`CODIInference` -- it also loads zen-E, which the grid uses for GSM8K) instead of
`eval_codi.build_model`, which hard-codes CUDA, so this runs on CPU. The latent loop is
`CODIInference.generate` with one addition: the vector fed INTO iteration i (z_0 = projected
hidden state at <bot>, z_i = projected output of iteration i) can be replaced by a donor's
z_i. That is the aligned site convention of `patch_minimal_pair_codi.py` (site s = z_s feeds
iteration s+1), and the same numbering as Coconut's passes 0..5.

  .venv/bin/python scripts/swap_random_codi.py --dataset gsm8k --checkpoint_path zen-E/CODI-gpt2 \\
      --grid_repo ~/Projects/are-lrms-easily-interpretable \\
      --data_dir ~/Projects/are-lrms-easily-interpretable/data --device cpu \\
      --slug zhang-swap-gsm8k --stage pilot --hardware "local Apple Silicon CPU"
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swap_random_coconut import DATA_FILES, DELTAS, load_examples, swap_experiment  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402

DELIM = "The answer is:"


@dataclass
class CodiSwapArguments:
    checkpoint_path: str = field(metadata={"help": "zen-E/CODI-gpt2 or a grid pytorch_model.bin"})
    grid_repo: str = field(metadata={"help": "checkout of are-lrms-easily-interpretable (for CODIInference)"})
    data_dir: str = field(metadata={"help": "are-lrms-easily-interpretable/data"})
    dataset: str = field(default="gsm8k", metadata={"help": "gsm8k | prontoqa | prosqa"})
    model_id: str = field(default="gpt2")
    checkpoint_label: Optional[str] = field(default=None)
    slug: str = field(default="zhang-swap")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    num_latents: int = field(default=6)
    eval_n: int = field(default=0)
    eval_seed: int = field(default=0)
    perm_seed: int = field(default=0)
    n_twin: int = field(default=400)
    max_new_tokens: int = field(default=64)
    smoke_n: int = field(default=0)
    device: str = field(default="cuda")
    fp32: bool = field(default=False, metadata={"help": "cast to float32 after loading (much faster on CPU); "
                                                         "the reference protocol is bf16"})


def main() -> None:
    (sa,) = transformers.HfArgumentParser((CodiSwapArguments,)).parse_args_into_dataclasses()
    assert sa.dataset in DATA_FILES
    argv = " ".join(sys.argv)
    sys.path.insert(0, str(Path(sa.grid_repo).expanduser()))
    from models.codi.codi_wrapper import CODIInference  # noqa: E402

    model = CODIInference(model_name_or_path=sa.model_id, num_latent=sa.num_latents, use_prj=True,
                          prj_dim=768, use_lora=True, lora_r=128, lora_alpha=32, device=sa.device)
    model.load_checkpoint(sa.checkpoint_path)  # raises on any missing/unexpected key
    model = model.to(sa.device).eval()
    if sa.fp32:
        model = model.float()
    tok = model.tokenizer
    embed = model.get_embd(model.codi, model.model_name)
    n_params = sum(p.numel() for p in model.parameters())
    dev = torch.device(sa.device)

    @torch.no_grad()
    def run(question: str, payload=None):
        enc = tok(question.strip().replace("  ", " "), return_tensors="pt", add_special_tokens=True)
        ids = torch.cat([enc["input_ids"], torch.tensor([[model.bot_id]])], dim=1).to(dev)
        out = model.codi(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, output_hidden_states=True)
        pkv = out.past_key_values
        z = model.prj(out.hidden_states[-1][:, -1, :].unsqueeze(1))
        live = []
        for i in range(sa.num_latents):
            live.append(z.detach().clone())
            fed = payload[i].to(z.dtype) if payload is not None else z
            out = model.codi(inputs_embeds=fed, use_cache=True, output_hidden_states=True, past_key_values=pkv)
            pkv = out.past_key_values
            z = model.prj(out.hidden_states[-1][:, -1, :].unsqueeze(1))
        nxt = embed(torch.tensor([model.eot_id], device=dev)).unsqueeze(0)
        gen = []
        for _ in range(sa.max_new_tokens):
            out = model.codi(inputs_embeds=nxt, use_cache=True, past_key_values=pkv)
            pkv = out.past_key_values
            t = int(torch.argmax(out.logits[0, -1, :model.codi.config.vocab_size - 1]))
            if t == tok.eos_token_id:
                break
            gen.append(t)
            nxt = embed(torch.tensor([t], device=dev)).unsqueeze(0)
        raw = tok.decode([t for t in gen if t < len(tok)], skip_special_tokens=True)
        pred = raw.split(DELIM)[-1].strip().replace(",", "") if DELIM in raw else None
        if pred and sa.dataset == "gsm8k":
            pred = pred.split()[0].rstrip(".") if pred.split() else None
        return pred or None, live, raw

    examples = load_examples(sa.data_dir, sa.dataset, sa.eval_n, sa.eval_seed)
    if sa.smoke_n:
        examples = examples[:sa.smoke_n]
    print(f"method=codi dataset={sa.dataset} n_examples={len(examples)}")
    summary, records, acc_before, decode_elapsed, swap_elapsed = swap_experiment(
        run, examples, sa.dataset, sa.perm_seed, sa.n_twin)

    if sa.smoke_n:
        print("smoke run -- not logging")
        return
    record = RunRecord(
        run_id=new_run_id("codi", sa.slug),
        mechanism="codi",
        stage=sa.stage,
        model=ModelInfo(backbone=sa.model_id, checkpoint=sa.checkpoint_label or sa.checkpoint_path, n_params=n_params),
        dataset=DatasetInfo(name=sa.dataset if sa.dataset != "gsm8k" else "gsm8k-aug", split="test",
                            n_examples=len(examples), seed=sa.eval_seed),
        metrics={
            "final_answer_accuracy": acc_before,
            "compute_steps": sa.num_latents,
            "sec_per_example": decode_elapsed / len(examples),
            "intervention_accuracy": summary["random_swap_all"].get("accuracy_after"),
            "extra": {"design": "Zhang et al. 2025 random-permutation full-chain swap (+ E3 twin swap on gsm8k, "
                                "same recipients); IR = answer changed; steer = P(answer == donor's | donor differs). "
                                "Aligned sites: donor z_i fed into iteration i+1, i = 0..5",
                      **summary, "decode_elapsed_sec": decode_elapsed, "swap_elapsed_sec": swap_elapsed},
        },
        hyperparams={"method": "codi", "num_latents": sa.num_latents, "perm_seed": sa.perm_seed, "n_twin_cap": sa.n_twin,
                     "max_new_tokens": sa.max_new_tokens, "deltas": list(DELTAS), "dtype": "float32" if sa.fp32 else "bfloat16", "greedy": True},
        seed=sa.perm_seed,
        hardware=f"{sa.hardware} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        notes=f"Zhang-style random swap vs twin swap, codi on {sa.dataset}.",
    )
    manifest = record.save(predictions=records)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
