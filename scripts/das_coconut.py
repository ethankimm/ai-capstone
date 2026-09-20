#!/usr/bin/env python3
"""Distributed Alignment Search (DAS) subspace-interchange patching on Coconut's
continuous-thought hidden states -- the Coconut counterpart to `das_codi.py`, run in
the same session for the same reason: raw full-vector donor-interchange patching is
one of the two null results this project's decode+patch sweep has produced so far
(CODI: `20260920-031925_codi_interchange-placeholder-pilot`; Coconut itself has only
had single-slot RAW patching tested in `20260920-031246_coconut_decode-patch-pilot`,
not yet a learned-subspace version). DAS asks whether a learned k-dim subspace of the
patched hidden vector succeeds where the raw full vector does not.

Where CODI's continuous thoughts are a separate loop feeding `inputs_embeds` for a
fresh forward call each iteration, Coconut splices each pass's hidden state directly
into a `<|latent|>` token's position in ONE growing sequence (see
`coconut_common.run_passes`'s docstring). So the DAS hook here is structurally
different, per the execution-difference note this follows: CODI hooks
`run_thoughts`'s per-iteration INPUT; Coconut hooks the continuous input-embedding
stack at the `<|latent|>` token position for pass i, i.e. the value
`coconut_common.run_passes` would otherwise splice in as `fill_value` for that pass.

Mechanism is identical to `das_codi.py` otherwise: rotate z into a basis with
`torch.nn.utils.parametrizations.orthogonal`, split at k, intervene as
`z^{A<-B}(R,k) = R^T[(Rz_B)[:k]; (Rz_A)[k:]]` (recipient's own hidden at that pass as
the background z_A, donor's hidden at the same pass as z_B), train R by
backpropagating the donor's-own-gold-answer cross-entropy through the SPLICE and the
remaining passes (all downstream computation the way `override_at_pass` already
affects it in the no-grad eval path -- transformer weights frozen throughout).
`coconut_common.run_passes`/`finish_and_decode` are hardwired `@torch.no_grad()`, so
this script reimplements a resumable, optionally-grad-enabled version of the same
loop (`run_pass_range`) rather than editing that shared module for one script's use.

Teacher-forcing target: since Coconut's post-latent continuation is free-form
reasoning text ending "... ### <answer>" (not a single explicit answer boundary token
the way CODI's `eot_id` is), this script simplifies the training signal to
teacher-force directly on `" ### {donor.answer}"` right after the last latent token
-- an approximation of `y_donor`, not the model's natural generation path, same
spirit as the DAS spec's CrossEntropy(Logits(z^{A<-B}), y_donor) objective.

Sweeps ALL 6 latent passes (0..5), matching `das_codi.py`'s all-sites choice, plus a
k sweep. Same pilot-scale reduction versus the spec (n=1000 pairs, 20 epochs,
k in {4,8,16,32,64}) for the same batch-size-1 rollout cost reason -- see
`das_codi.py`'s docstring for the compute-budget accounting, identical here.

Run inside a venv pinned to Coconut's own requirements.txt (same convention as
`decode_patch_coconut.py`):

  cd /workspace/ai-capstone && .venv_coconut/bin/python scripts/das_coconut.py \\
      --checkpoint_path /workspace/coconut_checkpoints/gsm-coconut/checkpoint_33 \\
      --data_dir /workspace/coconut_data \\
      --slug das-pilot --stage pilot --hardware "RunPod RTX A5000 (secure)" \\
      --num_latents 6 --sites 0,1,2,3,4,5 --k_values 8,32,64 \\
      --train_n_pairs 150 --eval_n_pairs 60 --epochs 5
"""
from __future__ import annotations

import contextlib
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.nn.utils.parametrizations import orthogonal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coconut_common import (  # noqa: E402
    load_coconut, encode_question, finish_and_decode, extract_answer_after_delimiter,
    parse_step_value, wilson_ci, mcnemar_exact_p, _to_legacy_cache,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from latentreasoning.eval.metrics import is_correct  # noqa: E402
from latentreasoning.mechanisms.coconut import name as mechanism_name  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402

HIDDEN_DIM = 768


@dataclass
class DASArguments:
    checkpoint_path: str = field(metadata={"help": "path to the downloaded checkpoint_33 file"})
    data_dir: str = field(metadata={"help": "dir with gsm_valid-gold-reasoning-trace_test.json"})
    model_id: str = field(default="openai-community/gpt2")
    slug: str = field(default="das")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: str = field(default="hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33")
    num_latents: int = field(default=6)
    sites: str = field(default="0,1,2,3,4,5", metadata={"help": "comma-separated 0-indexed latent passes to intervene at"})
    k_values: str = field(default="8,32,64", metadata={"help": "comma-separated subspace sizes to sweep (spec default: 4,8,16,32,64)"})
    train_n_pairs: int = field(default=150, metadata={"help": "donor/recipient training pairs per (site,k); spec default 1000"})
    eval_n_pairs: int = field(default=60, metadata={"help": "held-out donor/recipient eval pairs per (site,k); spec default 200"})
    epochs: int = field(default=5, metadata={"help": "passes over the training pairs; spec default 20"})
    lr: float = field(default=1e-3)
    train_seed: int = field(default=0)
    eval_seed: int = field(default=1)  # disjoint shuffle seed from decode_patch_coconut's eval slice
    pair_seed: int = field(default=0)
    max_new_tokens: int = field(default=48)
    device: str = field(default="cuda")


class Example:
    __slots__ = ("idx", "question", "answer", "steps")

    def __init__(self, idx, question, answer, steps):
        self.idx, self.question, self.answer, self.steps = idx, question, answer, steps


def load_examples(data_dir: str, n: int, seed: int, offset_pool: int = 0) -> list[Example]:
    rows = json.loads((Path(data_dir) / "gsm_valid-gold-reasoning-trace_test.json").read_text())
    examples = [Example(i, r["question"], r["answer"].strip(), r["steps"]) for i, r in enumerate(rows)]
    random.Random(seed).shuffle(examples)
    return examples[offset_pool:offset_pool + n]


def sample_pairs(examples: list[Example], n: int, rng: random.Random) -> list[tuple]:
    pool = list(examples)
    pairs = []
    attempts, max_attempts = 0, n * 20
    while len(pairs) < n and attempts < max_attempts:
        attempts += 1
        r, d = rng.sample(pool, 2)
        if r.answer.strip() != d.answer.strip():
            pairs.append((r, d))
    return pairs


class OrthogonalRotation(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.rot = orthogonal(nn.Linear(d, d, bias=False))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.rot(z)

    def inverse(self, z_tilde: torch.Tensor) -> torch.Tensor:
        return F.linear(z_tilde, self.rot.weight.t())


def intervene(rotation: OrthogonalRotation, z_a: torch.Tensor, z_b: torch.Tensor, k: int) -> torch.Tensor:
    tilde_a, tilde_b = rotation(z_a), rotation(z_b)
    patched_tilde = torch.cat([tilde_b[..., :k], tilde_a[..., k:]], dim=-1)
    return rotation.inverse(patched_tilde)


def run_pass_range(base_model, embedding, input_ids, attn, device, num_latents, latent_token_id,
                    pass_start: int, pass_end: int, state=None, override_at_pass: dict | None = None,
                    grad: bool = False):
    """Resumable version of `coconut_common.run_passes`'s loop body (same MIT-licensed
    Coconut.forward adaptation), split so a DAS-patched continuation can run with
    autograd enabled from a no-grad-computed prefix. `state` (from a previous call)
    resumes mid-sequence; `grad=True` leaves the whole range's autograd on instead of
    wrapping it in `torch.no_grad()`."""
    ctx = contextlib.nullcontext() if grad else torch.no_grad()
    with ctx:
        if state is None:
            latent_indices = (input_ids == latent_token_id).nonzero()
            latent_positions = [idx[1].item() for idx in latent_indices if idx[0] == 0]
            assert len(latent_positions) == num_latents
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
            inputs_embeds = embedding(input_ids)
            next_compute_range = (0, latent_positions[0])
            kv_cache = None
        else:
            inputs_embeds, kv_cache, next_compute_range, latent_positions, seq_len, position_ids = state

        pass_records = []
        for pass_idx in range(pass_start, pass_end):
            if kv_cache is None:
                outputs = base_model(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attn[:, next_compute_range[0]:next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0
            else:
                past_key_values = [(k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :]) for k, v in kv_cache]
                outputs = base_model(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attn[:, :next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                hidden_states_offset = next_compute_range[0]

            next_compute_range = (next_compute_range[1], seq_len if pass_idx + 1 >= num_latents else next_compute_range[1] + 1)
            hidden_states = outputs.hidden_states[-1]
            kv_cache = _to_legacy_cache(outputs.past_key_values)

            token_idx = latent_positions[pass_idx]
            live_hidden = hidden_states[0, token_idx - 1 - hidden_states_offset, :]
            if not grad:
                live_hidden = live_hidden.detach().clone()
            pass_records.append({"pass": pass_idx, "live_hidden": live_hidden})

            fill_value = override_at_pass[pass_idx] if (override_at_pass and pass_idx in override_at_pass) else live_hidden
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[0, token_idx, :] = fill_value.to(inputs_embeds.dtype).to(device)

        new_state = (inputs_embeds, kv_cache, next_compute_range, latent_positions, seq_len, position_ids)
        return pass_records, new_state


def teacher_forced_ce(base_model, embedding, tokenizer, state, attn, device, target_text: str) -> Optional[torch.Tensor]:
    """Grad-enabled continuation from `state` teacher-forced against `target_text`
    (mirrors `finish_and_decode`'s no-cache-reuse continuation exactly, but with the
    true next token embedded back in at each step instead of argmax)."""
    inputs_embeds, kv_cache, next_compute_range, _, _, position_ids = state
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    if not target_ids:
        return None
    past_key_values = (
        [(k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :]) for k, v in kv_cache]
        if kv_cache else None
    )
    outputs = base_model(
        inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
        attention_mask=attn[:, :next_compute_range[1]],
        position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
        past_key_values=past_key_values,
        output_hidden_states=False,
    )
    losses = []
    target0 = torch.tensor([target_ids[0]], device=device)
    losses.append(F.cross_entropy(outputs.logits[:, -1, :], target0))
    cur_embeds = torch.cat([inputs_embeds, embedding(target0).view(1, 1, -1)], dim=1)
    for tid in target_ids[1:]:
        out = base_model(inputs_embeds=cur_embeds)
        target_t = torch.tensor([tid], device=device)
        losses.append(F.cross_entropy(out.logits[:, -1, :], target_t))
        cur_embeds = torch.cat([cur_embeds, embedding(target_t).view(1, 1, -1)], dim=1)
    return torch.stack(losses).mean()


@torch.no_grad()
def full_trace(base_model, embedding, tokenizer, special_ids, question, device, num_latents, cache: dict, idx):
    if idx not in cache:
        input_ids, attn = encode_question(tokenizer, special_ids, question, num_latents, device)
        records, state = run_pass_range(base_model, embedding, input_ids, attn, device, num_latents,
                                         special_ids["latent"], 0, num_latents, grad=False)
        cache[idx] = (records, state, input_ids, attn)
    return cache[idx]


@torch.no_grad()
def base_answer(base_model, embedding, tokenizer, eos_id, ex, device, num_latents, cache, max_new_tokens):
    _, state, input_ids, attn = full_trace(base_model, embedding, tokenizer, cache["special_ids"], ex.question,
                                            device, num_latents, cache["traces"], ex.idx)
    inputs_embeds, kv_cache, next_compute_range, *_ = state
    raw = finish_and_decode(base_model, embedding, tokenizer, inputs_embeds, kv_cache, next_compute_range,
                             attn, device, max_new_tokens, eos_id)
    return extract_answer_after_delimiter(raw)


@torch.no_grad()
def patched_answer_eval(base_model, embedding, tokenizer, eos_id, special_ids, rotation, recipient, donor,
                         device, num_latents, site_pass, k, cache, max_new_tokens):
    input_ids, attn = encode_question(tokenizer, special_ids, recipient.question, num_latents, device)
    prefix_records, prefix_state = run_pass_range(base_model, embedding, input_ids, attn, device, num_latents,
                                                   special_ids["latent"], 0, site_pass + 1, grad=False)
    z_a = prefix_records[site_pass]["live_hidden"]
    donor_records, _, _, _ = full_trace(base_model, embedding, tokenizer, special_ids, donor.question,
                                         device, num_latents, cache, donor.idx)
    z_b = donor_records[site_pass]["live_hidden"]
    patched = intervene(rotation, z_a.float(), z_b.float(), k).to(z_a.dtype)

    inputs_embeds, kv_cache, next_compute_range, latent_positions, seq_len, position_ids = prefix_state
    inputs_embeds = inputs_embeds.clone()
    inputs_embeds[0, latent_positions[site_pass], :] = patched.to(inputs_embeds.dtype)
    new_state = (inputs_embeds, kv_cache, next_compute_range, latent_positions, seq_len, position_ids)
    _, final_state = run_pass_range(base_model, embedding, input_ids, attn, device, num_latents,
                                     special_ids["latent"], site_pass + 1, num_latents, state=new_state, grad=False)
    fe, fk, fr, *_ = final_state
    raw = finish_and_decode(base_model, embedding, tokenizer, fe, fk, fr, attn, device, max_new_tokens, eos_id)
    return extract_answer_after_delimiter(raw)


def train_rotation(base_model, embedding, tokenizer, special_ids, pairs, device, num_latents, site_pass, k,
                    lr, epochs, hidden_dim, trace_cache) -> OrthogonalRotation:
    rotation = OrthogonalRotation(hidden_dim).to(device)
    total_steps = max(1, epochs * len(pairs))
    opt = torch.optim.AdamW(rotation.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    step = 0
    t0 = time.perf_counter()
    for _ in range(epochs):
        random.shuffle(pairs)
        for recipient, donor in pairs:
            input_ids, attn = encode_question(tokenizer, special_ids, recipient.question, num_latents, device)
            prefix_records, prefix_state = run_pass_range(base_model, embedding, input_ids, attn, device,
                                                            num_latents, special_ids["latent"], 0, site_pass + 1, grad=False)
            z_a = prefix_records[site_pass]["live_hidden"]
            donor_records, _, _, _ = full_trace(base_model, embedding, tokenizer, special_ids, donor.question,
                                                 device, num_latents, trace_cache, donor.idx)
            z_b = donor_records[site_pass]["live_hidden"]
            patched = intervene(rotation, z_a.float(), z_b.float(), k).to(z_a.dtype)

            inputs_embeds, kv_cache, next_compute_range, latent_positions, seq_len, position_ids = prefix_state
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[0, latent_positions[site_pass], :] = patched.to(inputs_embeds.dtype)
            new_state = (inputs_embeds, kv_cache, next_compute_range, latent_positions, seq_len, position_ids)

            _, final_state = run_pass_range(base_model, embedding, input_ids, attn, device, num_latents,
                                             special_ids["latent"], site_pass + 1, num_latents, state=new_state, grad=True)
            loss = teacher_forced_ce(base_model, embedding, tokenizer, final_state, attn, device,
                                      f" ### {donor.answer.strip()}")
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"  site={site_pass} k={k} step={step}/{total_steps} loss={loss.item():.4f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    rotation.eval()
    return rotation


def evaluate_rotation(base_model, embedding, tokenizer, eos_id, special_ids, rotation, eval_pairs, device,
                       num_latents, site_pass, k, cache, max_new_tokens):
    records = []
    for recipient, donor in eval_pairs:
        pred_base = base_answer(base_model, embedding, tokenizer, eos_id, recipient, device, num_latents, cache, max_new_tokens)
        pred_patched = patched_answer_eval(base_model, embedding, tokenizer, eos_id, special_ids, rotation,
                                            recipient, donor, device, num_latents, site_pass, k,
                                            cache["traces"], max_new_tokens)
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
        "site": site_pass, "k": k, "n": n,
        "answer_changed_rate": x_changed / n if n else 0.0,
        "answer_changed_wilson_ci": list(wilson_ci(x_changed, n)),
        "steered_to_donor_rate": x_steer / len(elig) if elig else 0.0,
        "steered_to_donor_wilson_ci": list(wilson_ci(x_steer, len(elig))),
        "n_steered_eligible": len(elig),
    }


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

    sites = [int(s) for s in da.sites.split(",")]
    k_values = [int(s) for s in da.k_values.split(",")]

    train_pool = load_examples(da.data_dir, max(da.train_n_pairs * 3, 300), da.train_seed, offset_pool=0)
    eval_pool = load_examples(da.data_dir, max(da.eval_n_pairs * 3, 150), da.eval_seed, offset_pool=0)
    print(f"train_pool={len(train_pool)} eval_pool={len(eval_pool)} sites={sites} k_values={k_values}")

    all_summaries = []
    all_records = []
    t0 = time.perf_counter()
    for site_pass in sites:
        train_pairs = sample_pairs(train_pool, da.train_n_pairs, random.Random(da.pair_seed * 1000 + site_pass))
        eval_pairs = sample_pairs(eval_pool, da.eval_n_pairs, random.Random(da.pair_seed * 2000 + site_pass))
        trace_cache: dict = {}
        for k in k_values:
            print(f"--- training site={site_pass} k={k} (n_train_pairs={len(train_pairs)}) ---", flush=True)
            rotation = train_rotation(base_model, embedding, tokenizer, special_ids, list(train_pairs), device,
                                       da.num_latents, site_pass, k, da.lr, da.epochs, HIDDEN_DIM, trace_cache)
            eval_cache = {"special_ids": special_ids, "traces": trace_cache}
            records, summary = evaluate_rotation(base_model, embedding, tokenizer, eos_id, special_ids, rotation,
                                                  eval_pairs, device, da.num_latents, site_pass, k, eval_cache,
                                                  da.max_new_tokens)
            all_records.extend([{"site": site_pass, "k": k, **r} for r in records])
            all_summaries.append(summary)
            print(f"site={site_pass} k={k}: answer_changed={summary['answer_changed_rate']:.3f} "
                  f"steered_to_donor={summary['steered_to_donor_rate']:.3f} "
                  f"(n={summary['n']}, n_steered_eligible={summary['n_steered_eligible']}, "
                  f"{time.perf_counter() - t0:.0f}s elapsed)", flush=True)

    best = max(all_summaries, key=lambda s: s["steered_to_donor_rate"])
    print(f"BEST: site={best['site']} k={best['k']} steered_to_donor={best['steered_to_donor_rate']:.3f}")

    metrics = {
        "compute_steps": da.num_latents,
        "intervention_accuracy": best["steered_to_donor_rate"],
        "extra": {
            "sweep": all_summaries,
            "best_site": best["site"], "best_k": best["k"],
            "sites": sites, "k_values": k_values,
            "train_n_pairs": da.train_n_pairs, "eval_n_pairs": da.eval_n_pairs, "epochs": da.epochs,
            "n_trainable_params_backbone": n_params,
            "teacher_forcing_target": "' ### {donor.answer}' spliced directly after the last "
                                       "latent token -- a simplification of y_donor, not the "
                                       "model's natural free-form reasoning continuation.",
            "related_runs": ["20260920-031246_coconut_decode-patch-pilot"],
            "spec_scale_note": "pilot-scale reduction of the DAS spec (train_n=1000, "
                                "epochs=20, k in {4,8,16,32,64}) -- see script docstring.",
        },
    }

    record = RunRecord(
        run_id=new_run_id("coconut", da.slug),
        mechanism=mechanism_name,
        stage=da.stage,
        model=ModelInfo(backbone=da.model_id, checkpoint=da.checkpoint_label, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(eval_pool), seed=da.eval_seed),
        metrics=metrics,
        hyperparams={
            "num_latents": da.num_latents, "sites": sites, "k_values": k_values,
            "train_n_pairs": da.train_n_pairs, "eval_n_pairs": da.eval_n_pairs,
            "epochs": da.epochs, "lr": da.lr,
        },
        seed=da.train_seed,
        hardware=f"{da.hardware} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        notes="Distributed Alignment Search on Coconut: learned orthogonal rotation R "
              "isolates a k-dim subspace of the spliced latent hidden state for "
              "donor-interchange patching at each pass 0..5, trained to maximize the "
              "donor's own gold-answer likelihood, evaluated for steered_to_donor / "
              "answer_changed across a k sweep -- Coconut counterpart to das_codi.py, "
              "same question: does a learned subspace succeed where raw full-vector "
              "interchange patching does not.",
    )
    manifest = record.save(predictions=all_records)
    out_dir = manifest.parent
    (out_dir / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
