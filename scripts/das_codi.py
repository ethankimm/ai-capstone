#!/usr/bin/env python3
"""Distributed Alignment Search (DAS) subspace-interchange patching on CODI's
continuous thoughts -- the follow-up flagged as NOT implemented in
`scripts/probe_codi.py`'s `das_note` and in `next_experiments.md` #2's "Not started"
list.

Motivation: raw full-vector donor-interchange patching at every iteration --
including the non-decodable placeholder positions z0/z3 -- is a clean null
(`results/20260920-031925_codi_interchange-placeholder-pilot`: swapping in another
example's z0/z3 does not steer the answer toward the donor, despite z0/z3 being
load-bearing under mean-ablation). One reading of that null is that the intervention
is unconstrained: patching the FULL 768-dim vector also overwrites context/attention
state that has nothing to do with the calculation, so the recipient's forward pass is
knocked off-manifold rather than steered. DAS asks the sharper question: is there a
learned k-dimensional SUBSPACE of z (found by an orthogonal rotation R, not the raw
standard basis) that carries transferable, donor-swappable state, while the
orthogonal (d-k)-dim background subspace stays the recipient's own?

Mechanism (Geiger et al.'s DAS, adapted to CODI's inference-time thought loop):
  - Parameterize a full-rank orthogonal rotation R in SO(768) via
    `torch.nn.utils.parametrizations.orthogonal` on an `nn.Linear(768, 768, bias=False)`.
  - Split the rotated basis at k: `z_tilde = R z`, `z_tilde[:k]` = target subspace,
    `z_tilde[k:]` = orthogonal background.
  - Intervention: `z^{A<-B}(R,k) = R^T [ (R z_B)[:k] ; (R z_A)[k:] ]` -- recipient A's
    background, donor B's target subspace, un-rotated back to the model's native basis.
  - Train R (transformer + LoRA frozen, `.requires_grad_(False)`) to maximize the
    donor's own gold-answer likelihood under this intervention: cross-entropy of the
    model's OWN continuation (post-intervention forward, teacher-forced against the
    donor's gold answer string) vs `y_donor`. AdamW, cosine schedule, one R per (site,
    k) combination.
  - Evaluate the trained R on held-out pairs: `steered_to_donor` (patched answer
    matches donor's gold, base didn't) and `answer_changed`, mirroring
    `interchange_patch_placeholder_codi.py`'s summary stats exactly so the two are
    directly comparable (same metric, raw-vector vs learned-subspace intervention).

Per the user's explicit choice: sweeps ALL 6 iteration sites (1..6), not just the
z0/z3 placeholder pair -- gives the complete picture including the already-decodable
positions (z2/z4), where raw patching is ALSO null
(`20260919-184323_codi_decode-patch-full-eval`), so DAS can independently confirm or
overturn that finding too.

Scale note: the spec this follows calls for train_n=1000 pairs x 20 epochs x k in
{4,8,16,32,64} -- with this repo's batch-size-1 rollout (same constraint as every
other CODI script here: no batched KV cache), that's ~30 independent (site, k)
trainings x 20,000 forward+backward passes each, far beyond a pilot budget. Defaults
below are a PILOT-scale reduction (documented per-flag); rerun at the spec's full
scale (`--train_n_pairs 1000 --epochs 20 --k_values 4,8,16,32,64`) once a pilot slice
looks promising.

Run inside the CODI venv, from the CODI checkout (same convention as every other
CODI script):

  cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/das_codi.py \\
      --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \\
      --slug das-pilot --stage pilot --hardware "RunPod RTX A5000 (secure)" \\
      --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \\
      --lora_r 128 --lora_alpha 32 --lora_init --greedy True \\
      --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \\
      --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \\
      --sites 1,2,3,4,5,6 --k_values 8,32,64 --train_n_pairs 150 --eval_n_pairs 60 --epochs 5
"""
from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.nn.utils.parametrizations import orthogonal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())  # the CODI checkout: `src.model`

from src.model import DataArguments, ModelArguments, TrainingArguments  # noqa: E402
from eval_codi import build_model  # noqa: E402
from decode_patch_codi import (  # noqa: E402
    encode_question, decode_answer, wilson_ci, mcnemar_exact_p,
)
from interchange_patch_placeholder_codi import sample_pairs  # noqa: E402

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.eval.metrics import extract_final_number, is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402

HIDDEN_DIM = 768


@dataclass
class DASArguments:
    slug: str = field(default="das")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: Optional[str] = field(default=None)
    sites: str = field(default="1,2,3,4,5,6", metadata={"help": "comma-separated iteration indices to intervene at"})
    k_values: str = field(default="8,32,64", metadata={"help": "comma-separated subspace sizes to sweep (spec default: 4,8,16,32,64)"})
    train_n_pairs: int = field(default=150, metadata={"help": "donor/recipient training pairs per (site,k); spec default 1000"})
    eval_n_pairs: int = field(default=60, metadata={"help": "held-out donor/recipient eval pairs per (site,k); spec default 200"})
    epochs: int = field(default=5, metadata={"help": "passes over the training pairs; spec default 20"})
    lr: float = field(default=1e-3)
    train_seed: int = field(default=0)
    eval_seed: int = field(default=0)
    pair_seed: int = field(default=0)
    max_new_tokens: int = field(default=64)


class OrthogonalRotation(nn.Module):
    """R in SO(d) via PyTorch's built-in orthogonal parametrization on a square
    linear map. `forward` applies R; `inverse` applies R^T = R^-1 (R orthogonal)."""

    def __init__(self, d: int):
        super().__init__()
        self.rot = orthogonal(nn.Linear(d, d, bias=False))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.rot(z)

    def inverse(self, z_tilde: torch.Tensor) -> torch.Tensor:
        return F.linear(z_tilde, self.rot.weight.t())


def intervene(rotation: OrthogonalRotation, z_a: torch.Tensor, z_b: torch.Tensor, k: int) -> torch.Tensor:
    """z^{A<-B}(R,k): recipient A's background, donor B's target subspace, k dims."""
    tilde_a = rotation(z_a)
    tilde_b = rotation(z_b)
    patched_tilde = torch.cat([tilde_b[..., :k], tilde_a[..., k:]], dim=-1)
    return rotation.inverse(patched_tilde)


@torch.no_grad()
def prefix_pkv_and_latent(model, tokenizer, question: str, device: str, site_it: int):
    """State fed INTO iteration `site_it` (matches `override_input_at`'s semantics in
    decode_patch_codi.run_thoughts): initial encode for site_it==1, else iterations
    1..site_it-1 run normally first."""
    input_ids, attn = encode_question(model, tokenizer, question, device)
    outputs = model.codi(input_ids=input_ids, use_cache=True, output_hidden_states=True,
                          past_key_values=None, attention_mask=attn)
    pkv = outputs.past_key_values
    latent = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
    if model.use_prj:
        latent = model.prj(latent)
    for _ in range(1, site_it):
        outputs = model.codi(inputs_embeds=latent, use_cache=True, output_hidden_states=True, past_key_values=pkv)
        pkv = outputs.past_key_values
        pre_hidden = outputs.hidden_states[-1][:, -1, :]
        latent = model.prj(pre_hidden.unsqueeze(1)) if model.use_prj else pre_hidden.unsqueeze(1)
    return pkv, latent.detach()


def continue_loop(model, pkv, feed: torch.Tensor, site_it: int, n_iters: int):
    """Iterations site_it..n_iters, GRAD ENABLED, starting from `feed` (the patched
    vector). Returns the final pkv (the only thing decode_answer actually reads --
    it, like the reference CODI inference loop, ignores the trailing latent value)."""
    for it in range(site_it, n_iters + 1):
        outputs = model.codi(inputs_embeds=feed, use_cache=True, output_hidden_states=True, past_key_values=pkv)
        pkv = outputs.past_key_values
        pre_hidden = outputs.hidden_states[-1][:, -1, :]
        feed = model.prj(pre_hidden.unsqueeze(1)) if model.use_prj else pre_hidden.unsqueeze(1)
    return pkv


def teacher_forced_ce(model, tokenizer, pkv, device: str, target_text: str) -> Optional[torch.Tensor]:
    """Grad-enabled continuation from `pkv` through eot + the donor's gold-answer
    string, teacher-forced (true token embedded back in, not argmax) -- mirrors
    `decode_answer`'s generation tail but for training instead of greedy decode."""
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    if not target_ids:
        return None
    embed_fn = model.get_embd(model.codi, model.model_name)
    inp = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device=device)).unsqueeze(0)
    losses = []
    for tid in target_ids:
        out = model.codi(inputs_embeds=inp, output_hidden_states=False, attention_mask=None,
                          use_cache=True, past_key_values=pkv)
        pkv = out.past_key_values
        logits = out.logits[:, -1, :model.codi.config.vocab_size - 1]
        target = torch.tensor([tid], dtype=torch.long, device=device)
        losses.append(F.cross_entropy(logits, target))
        inp = embed_fn(target).unsqueeze(1).to(device)
    return torch.stack(losses).mean()


@torch.no_grad()
def base_answer(model, tokenizer, recipient, device, n_iters, max_new_tokens) -> Optional[str]:
    q = recipient.question.strip().replace("  ", " ")
    pkv, latent = prefix_pkv_and_latent(model, tokenizer, q, device, 1)
    pkv = continue_loop(model, pkv, latent, 1, n_iters)
    out = decode_answer(model, tokenizer, pkv, latent, device, max_new_tokens)
    return extract_final_number(out)


@torch.no_grad()
def z_at_site(model, tokenizer, ex, device, n_iters, site_it, cache: dict) -> torch.Tensor:
    key = (ex.idx, site_it)
    if key not in cache:
        q = ex.question.strip().replace("  ", " ")
        _, latent = prefix_pkv_and_latent(model, tokenizer, q, device, site_it)
        cache[key] = latent
    return cache[key]


@torch.no_grad()
def patched_answer(model, tokenizer, rotation, recipient, donor, device, n_iters, site_it, k, cache, max_new_tokens):
    q = recipient.question.strip().replace("  ", " ")
    pkv, z_a = prefix_pkv_and_latent(model, tokenizer, q, device, site_it)
    z_b = z_at_site(model, tokenizer, donor, device, n_iters, site_it, cache)
    patched = intervene(rotation, z_a.float(), z_b.float(), k).to(z_a.dtype)
    pkv = continue_loop(model, pkv, patched, site_it, n_iters)
    out = decode_answer(model, tokenizer, pkv, patched, device, max_new_tokens)
    return extract_final_number(out)


def train_rotation(model, tokenizer, pairs, device, n_iters, site_it, k, lr, epochs, hidden_dim) -> OrthogonalRotation:
    rotation = OrthogonalRotation(hidden_dim).to(device)
    total_steps = max(1, epochs * len(pairs))
    opt = torch.optim.AdamW(rotation.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    step = 0
    t0 = time.perf_counter()
    for _ in range(epochs):
        random.shuffle(pairs)
        for recipient, donor in pairs:
            q_r = recipient.question.strip().replace("  ", " ")
            with torch.no_grad():
                pkv, z_a = prefix_pkv_and_latent(model, tokenizer, q_r, device, site_it)
                q_d = donor.question.strip().replace("  ", " ")
                _, z_b = prefix_pkv_and_latent(model, tokenizer, q_d, device, site_it)
            patched = intervene(rotation, z_a.float(), z_b.float(), k).to(z_a.dtype)
            pkv_final = continue_loop(model, pkv, patched, site_it, n_iters)
            loss = teacher_forced_ce(model, tokenizer, pkv_final, device, donor.answer.strip())
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"  site={site_it} k={k} step={step}/{total_steps} loss={loss.item():.4f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    rotation.eval()
    return rotation


def evaluate_rotation(model, tokenizer, rotation, eval_pairs, device, n_iters, site_it, k, max_new_tokens):
    cache: dict = {}
    records = []
    for recipient, donor in eval_pairs:
        pred_base = base_answer(model, tokenizer, recipient, device, n_iters, max_new_tokens)
        pred_patched = patched_answer(model, tokenizer, rotation, recipient, donor, device, n_iters, site_it, k, cache, max_new_tokens)
        steered = None
        if pred_base is not None:
            steered = is_correct(pred_patched, donor.answer) and not is_correct(pred_base, donor.answer)
        records.append({
            "recipient_idx": recipient.idx, "donor_idx": donor.idx,
            "recipient_gold": recipient.answer, "donor_gold": donor.answer,
            "answer_base": pred_base, "answer_patched": pred_patched,
            "answer_changed": pred_patched != pred_base,
            "steered_to_donor": steered,
        })
    n = len(records)
    x_changed = sum(r["answer_changed"] for r in records)
    elig = [r for r in records if r["steered_to_donor"] is not None]
    x_steer = sum(r["steered_to_donor"] for r in elig)
    return records, {
        "site": site_it, "k": k, "n": n,
        "answer_changed_rate": x_changed / n if n else 0.0,
        "answer_changed_wilson_ci": list(wilson_ci(x_changed, n)),
        "steered_to_donor_rate": x_steer / len(elig) if elig else 0.0,
        "steered_to_donor_wilson_ci": list(wilson_ci(x_steer, len(elig))),
        "n_steered_eligible": len(elig),
    }


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
    sites = [int(s) for s in da.sites.split(",")]
    k_values = [int(s) for s in da.k_values.split(",")]

    train_pool = load_gsm8k_aug(split="train", n=max(da.train_n_pairs * 3, 500), seed=da.train_seed)
    eval_pool = load_gsm8k_aug(split="validation", n=min(1000, max(da.eval_n_pairs * 3, 200)), seed=da.eval_seed)
    print(f"train_pool={len(train_pool)} eval_pool={len(eval_pool)} sites={sites} k_values={k_values}")

    all_summaries = []
    all_records = []
    t0 = time.perf_counter()
    for site_it in sites:
        train_pairs = sample_pairs(train_pool, da.train_n_pairs, random.Random(da.pair_seed * 1000 + site_it))
        eval_pairs = sample_pairs(eval_pool, da.eval_n_pairs, random.Random(da.pair_seed * 2000 + site_it))
        for k in k_values:
            print(f"--- training site={site_it} k={k} (n_train_pairs={len(train_pairs)}) ---", flush=True)
            rotation = train_rotation(model, tokenizer, list(train_pairs), device, n_iters, site_it, k,
                                       da.lr, da.epochs, HIDDEN_DIM)
            records, summary = evaluate_rotation(model, tokenizer, rotation, eval_pairs, device, n_iters,
                                                  site_it, k, da.max_new_tokens)
            all_records.extend([{"site": site_it, "k": k, **r} for r in records])
            all_summaries.append(summary)
            print(f"site={site_it} k={k}: answer_changed={summary['answer_changed_rate']:.3f} "
                  f"steered_to_donor={summary['steered_to_donor_rate']:.3f} "
                  f"(n={summary['n']}, n_steered_eligible={summary['n_steered_eligible']}, "
                  f"{time.perf_counter() - t0:.0f}s elapsed)", flush=True)

    best = max(all_summaries, key=lambda s: s["steered_to_donor_rate"])
    print(f"BEST: site={best['site']} k={best['k']} steered_to_donor={best['steered_to_donor_rate']:.3f}")

    nondecodable_summaries = [s for s in all_summaries if s["site"] in (1, 4)]
    decodable_summaries = [s for s in all_summaries if s["site"] in (3, 5)]

    metrics = {
        "compute_steps": n_iters,
        "intervention_accuracy": best["steered_to_donor_rate"],
        "extra": {
            "sweep": all_summaries,
            "best_site": best["site"], "best_k": best["k"],
            "nondecodable_site_summaries": nondecodable_summaries,
            "decodable_site_summaries": decodable_summaries,
            "sites": sites, "k_values": k_values,
            "train_n_pairs": da.train_n_pairs, "eval_n_pairs": da.eval_n_pairs, "epochs": da.epochs,
            "n_trainable_params_backbone": n_params,
            "related_runs": [
                "20260920-031925_codi_interchange-placeholder-pilot",
                "20260920-032053_codi_probe-pilot",
            ],
            "spec_scale_note": "pilot-scale reduction of the DAS spec (train_n=1000, "
                                "epochs=20, k in {4,8,16,32,64}) -- see script docstring.",
        },
    }

    record = RunRecord(
        run_id=new_run_id("codi", da.slug),
        mechanism="codi",
        stage=da.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=da.checkpoint_label or model_args.ckpt_dir, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="validation", n_examples=len(eval_pool), seed=da.eval_seed),
        metrics=metrics,
        hyperparams={
            "inf_latent_iterations": n_iters, "num_latent": training_args.num_latent,
            "sites": sites, "k_values": k_values,
            "train_n_pairs": da.train_n_pairs, "eval_n_pairs": da.eval_n_pairs,
            "epochs": da.epochs, "lr": da.lr,
        },
        seed=da.train_seed,
        hardware=f"{da.hardware} / {torch.cuda.get_device_name(0)}",
        notes="Distributed Alignment Search: learned orthogonal rotation R isolates a "
              "k-dim subspace of z for donor-interchange patching, trained to maximize "
              "the donor's own gold-answer likelihood, evaluated for steered_to_donor / "
              "answer_changed at every iteration site 1..6 across a k sweep -- tests "
              "whether a learned subspace succeeds where raw full-vector interchange "
              "patching (20260920-031925) was a clean null.",
    )
    manifest = record.save(predictions=all_records)
    out_dir = manifest.parent
    (out_dir / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
