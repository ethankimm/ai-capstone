#!/usr/bin/env python3
"""E4 (`steered_to_donor_audit.md` §5, `next_experiments.md` pasted plan): Distributed
Alignment Search on SAME-PROBLEM MINIMAL-PAIR donors for Coconut -- counterpart to
`das_minimal_pair_codi.py`, conditional on E3(a)
(`results/20260920-195725_coconut_minimal-pair-patch`), which found ALL-SLOT
minimal-pair patching steers to the twin's answer 77.1% of the time, with pass 1 ALONE
already carrying 48.6% and pass 4 alone an independent 28.6% (passes 0/2/3/5 near-inert).
E4 asks: does a learned k-dim subspace (rotation R, not the raw full vector) explain
pass 1's signal alone, and does a JOINT subspace spanning passes {1,4} explain more of
the ALL-SLOT upper bound than either pass alone?

New relative to `das_coconut.py` (which only patches ONE pass per rotation): multi-site
JOINT interchange -- one shared rotation R applied, in increasing pass order, at every
site in a group. `--site_groups` is a semicolon-separated list of comma-separated
0-indexed pass lists, e.g. `"1;1,4"` = pass-1-alone and the joint {1,4} group (E4's two
Coconut target sites per the plan).

Mechanism: identical DAS recipe to `das_coconut.py` (rotate into a learned basis, swap
the first k rotated dims from the donor, k..767 stays the recipient's own, un-rotate),
generalized to a SEQUENCE of sites within one growing spliced sequence. For each site in
the group (ascending), the segment from the previous site (exclusive) up to this site
(inclusive) is run via `run_pass_range` -- grad-enabled once the first patch has
happened, since downstream passes then depend on it; the very first segment (before any
patch) needs no grad, matching `das_coconut.py`'s original single-site precedent of
treating the recipient's OWN live value at the first site as an ordinary (non-graph)
input to the rotation. After each such segment, the resulting live hidden at that site
is intervened (rotation-mixed with the twin's donor value at that same pass, precomputed
from the twin's own full 6-pass trace) and manually spliced back into `inputs_embeds`
before continuing. The final segment (after the last site) is grad-enabled through to
the teacher-forced CE against the twin's own gold answer.

Donor values: the twin's own live hidden at each pass, from one `run_pass_range` full
trace per twin question, cached once per pair (same as `patch_minimal_pair_coconut.py`
/ `das_coconut.py`'s `full_trace`), NOT another random example's.

Scale: n=500 minimal-pair training tuples requested by the plan; k in {8,16,32,64}, not
pilot-reduced (see `das_minimal_pair_codi.py`'s docstring for the compute-budget
accounting, identical reasoning here -- Coconut's forward pass is cheaper per step than
CODI's so this should be faster in wall-clock, not slower).

Run inside a venv pinned to Coconut's own requirements.txt (same convention as
`das_coconut.py`):

  cd /workspace/ai-capstone && .venv_coconut/bin/python scripts/das_minimal_pair_coconut.py \\
      --checkpoint_path /workspace/coconut_checkpoints/gsm-coconut/checkpoint_33 \\
      --data_dir /workspace/coconut_data \\
      --slug das-minimal-pair --stage pilot --hardware "RunPod RTX A5000 (secure)" \\
      --num_latents 6 --site_groups "1;1,4" --k_values 8,16,32,64 \\
      --train_n_pairs 500 --eval_n_pairs 150 --epochs 5
"""
from __future__ import annotations

import json
import os
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
    wilson_ci, _to_legacy_cache,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from latentreasoning.data.gsm8k_aug import Example  # noqa: E402
from latentreasoning.data.minimal_pairs import generate_minimal_pair  # noqa: E402
from latentreasoning.eval.counterfactual import parse_steps, qualifying_steps  # noqa: E402
from latentreasoning.eval.metrics import is_correct  # noqa: E402
from latentreasoning.mechanisms.coconut import name as mechanism_name  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402

HIDDEN_DIM = 768
DELTAS = (1, -1, 2, -2, 3, -3)


@dataclass
class DASArguments:
    checkpoint_path: str = field(metadata={"help": "path to the downloaded checkpoint_33 file"})
    data_dir: str = field(metadata={"help": "dir with gsm_valid-gold-reasoning-trace_test.json"})
    model_id: str = field(default="openai-community/gpt2")
    slug: str = field(default="das-minimal-pair-fixed")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: str = field(default="hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33")
    num_latents: int = field(default=6)
    site_groups: str = field(
        default="1;1,4",
        metadata={"help": "semicolon-separated groups, each a comma-separated list of "
                           "0-indexed latent passes sharing one rotation, e.g. '1;1,4' "
                           "= pass-1-alone and joint {1,4} (E4's two Coconut target sites)"},
    )
    k_values: str = field(default="8,16,32,64", metadata={"help": "comma-separated subspace sizes to sweep"})
    train_n_pairs: int = field(default=500, metadata={"help": "qualified minimal-pair training tuples per (group,k)"})
    eval_n_pairs: int = field(default=150, metadata={"help": "held-out qualified minimal-pair tuples per (group,k)"})
    train_pool_n: int = field(default=2500, metadata={"help": "gsm_original_train.json examples to decode looking for train pairs"})
    eval_pool_n: int = field(default=500, metadata={"help": "gsm_original_valid.json examples to decode looking for eval pairs"})
    train_max_candidate_checks: int = field(default=2500)
    eval_max_candidate_checks: int = field(default=900)
    epochs: int = field(default=5, metadata={"help": "passes over the training pairs"})
    lr: float = field(default=1e-3)
    train_seed: int = field(default=0)
    eval_seed: int = field(default=0)
    pair_seed: int = field(default=0)
    max_new_tokens: int = field(default=48)
    save_rotations: str = field(default="", metadata={"help": "dir: save each trained R as rot_<group>_k<k>.pt"})
    device: str = field(default="cuda")


def load_examples(data_dir: str, n: int, seed: int, fname: str) -> list[Example]:
    rows = json.loads((Path(data_dir) / fname).read_text())
    examples = [Example(question=r["question"], rationale=" ".join(r["steps"]),
                         answer=r["answer"].strip(), idx=i) for i, r in enumerate(rows)]
    random.Random(seed).shuffle(examples)
    return examples[:n]


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


def parse_site_groups(spec: str) -> list[list[int]]:
    return [[int(s) for s in part.split(",")] for part in spec.split(";") if part.strip()]


def group_label(group: list[int]) -> str:
    return "+".join(str(s) for s in group) if len(group) > 1 else str(group[0])


def run_pass_range(base_model, embedding, input_ids, attn, device, num_latents, latent_token_id,
                    pass_start: int, pass_end: int, state=None, grad: bool = False):
    """Resumable version of `coconut_common.run_passes`'s loop body, lifted from
    `das_coconut.py` unchanged. `grad=True` leaves autograd on for the whole range
    instead of wrapping it in `torch.no_grad()`; no `override_at_pass` here -- E4's
    multi-site splice is done by the caller between segments (see `run_intervened`),
    since the intervened value for site i needs i's own just-computed live_hidden,
    which isn't available until this function returns."""
    import contextlib
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

            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[0, token_idx, :] = live_hidden.to(inputs_embeds.dtype).to(device)

        new_state = (inputs_embeds, kv_cache, next_compute_range, latent_positions, seq_len, position_ids)
        return pass_records, new_state


def run_intervened(base_model, embedding, tokenizer, special_ids, question: str, device, num_latents,
                    rotation: Optional[OrthogonalRotation], sites: list[int],
                    donor_vecs: Optional[dict[int, torch.Tensor]], k: int, grad: bool):
    """Run the full num_latents-pass sequence, splicing z^{A<-B}(R,k) in at every pass
    in `sites` (ascending), one shared rotation. `grad=False` wraps the whole thing in
    `torch.no_grad()` for eval; `grad=True` leaves the post-first-patch segments
    differentiable for training. Returns the final `state` tuple (as `run_pass_range`)."""
    sites_sorted = sorted(sites)
    input_ids, attn = encode_question(tokenizer, special_ids, question, num_latents, device)
    state = None
    prev_end = 0
    for i, site in enumerate(sites_sorted):
        seg_grad = grad and i > 0  # first segment: recipient's own untouched trace, no grad needed
        records, state = run_pass_range(base_model, embedding, input_ids, attn, device, num_latents,
                                         special_ids["latent"], prev_end, site + 1, state=state, grad=seg_grad)
        z_a = records[-1]["live_hidden"]
        z_b = donor_vecs[site].to(device=device, dtype=torch.float32)
        patched = intervene(rotation, z_a.float(), z_b, k).to(z_a.dtype)
        inputs_embeds, kv_cache, next_compute_range, latent_positions, seq_len, position_ids = state
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[0, latent_positions[site], :] = patched.to(inputs_embeds.dtype)
        state = (inputs_embeds, kv_cache, next_compute_range, latent_positions, seq_len, position_ids)
        prev_end = site + 1
    _, final_state = run_pass_range(base_model, embedding, input_ids, attn, device, num_latents,
                                     special_ids["latent"], prev_end, num_latents, state=state, grad=grad)
    return final_state, input_ids, attn


def teacher_forced_ce(base_model, embedding, tokenizer, state, attn, device, target_text: str) -> Optional[torch.Tensor]:
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
def full_trace(base_model, embedding, tokenizer, special_ids, question, device, num_latents):
    input_ids, attn = encode_question(tokenizer, special_ids, question, num_latents, device)
    records, _ = run_pass_range(base_model, embedding, input_ids, attn, device, num_latents,
                                 special_ids["latent"], 0, num_latents, grad=False)
    return {r["pass"]: r["live_hidden"] for r in records}


@torch.no_grad()
def decode_one(base_model, embedding, tokenizer, eos_id, special_ids, question, device, num_latents, max_new_tokens):
    input_ids, attn = encode_question(tokenizer, special_ids, question, num_latents, device)
    _, state = run_pass_range(base_model, embedding, input_ids, attn, device, num_latents,
                               special_ids["latent"], 0, num_latents, grad=False)
    inputs_embeds, kv_cache, next_compute_range, *_ = state
    raw = finish_and_decode(base_model, embedding, tokenizer, inputs_embeds, kv_cache, next_compute_range,
                             attn, device, max_new_tokens, eos_id)
    return extract_answer_after_delimiter(raw)


def build_minimal_pairs(base_model, embedding, tokenizer, eos_id, special_ids, device, num_latents,
                         max_new_tokens, examples, n_pairs: int, max_candidate_checks: int,
                         pair_seed: int, tag: str):
    t0 = time.perf_counter()
    base_info: dict[int, dict] = {}
    for ex in examples:
        pred = decode_one(base_model, embedding, tokenizer, eos_id, special_ids, ex.question, device, num_latents, max_new_tokens)
        steps = parse_steps(ex.rationale)
        base_info[ex.idx] = {"pred": pred, "correct": is_correct(pred, ex.answer), "qualifying": qualifying_steps(steps)}
    decode_elapsed = time.perf_counter() - t0
    accuracy = sum(v["correct"] for v in base_info.values()) / len(base_info)
    correct_examples = [ex for ex in examples if base_info[ex.idx]["correct"]]
    print(f"[{tag}] decoded {len(examples)} in {decode_elapsed:.1f}s, accuracy={accuracy:.3f}, "
          f"base-correct pool={len(correct_examples)}", flush=True)

    rng = random.Random(pair_seed)
    candidates = []
    for ex in correct_examples:
        for step in base_info[ex.idx]["qualifying"]:
            for delta in DELTAS:
                mp = generate_minimal_pair(ex, step, delta, rng=rng)
                if mp is not None:
                    candidates.append(mp)
                    break
    rng.shuffle(candidates)
    print(f"[{tag}] minimal-pair candidates: {len(candidates)}", flush=True)

    pairs = []
    n_checked = 0
    for mp in candidates:
        if len(pairs) >= n_pairs or n_checked >= max_candidate_checks:
            break
        n_checked += 1
        twin_pred = decode_one(base_model, embedding, tokenizer, eos_id, special_ids, mp.twin.question, device, num_latents, max_new_tokens)
        if is_correct(twin_pred, mp.twin.answer):
            twin_vecs = full_trace(base_model, embedding, tokenizer, special_ids, mp.twin.question, device, num_latents)
            pairs.append((mp, base_info[mp.original.idx]["pred"], twin_vecs))
        if n_checked % 100 == 0:
            print(f"[{tag}] checked {n_checked} candidates, {len(pairs)} qualified so far", flush=True)
    print(f"[{tag}] qualified pairs: {len(pairs)}/{n_checked} candidates checked", flush=True)
    return pairs, accuracy, len(correct_examples)


def train_rotation(base_model, embedding, tokenizer, special_ids, device, num_latents, group: list[int],
                    k: int, train_pairs, lr: float, epochs: int) -> OrthogonalRotation:
    rotation = OrthogonalRotation(HIDDEN_DIM).to(device)
    total_steps = max(1, epochs * len(train_pairs))
    opt = torch.optim.AdamW(rotation.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    step = 0
    t0 = time.perf_counter()
    for _ in range(epochs):
        random.shuffle(train_pairs)
        for mp, _pred_base, twin_vecs in train_pairs:
            final_state, _input_ids, attn = run_intervened(
                base_model, embedding, tokenizer, special_ids, mp.original.question, device, num_latents,
                rotation, group, twin_vecs, k, grad=True)
            loss = teacher_forced_ce(base_model, embedding, tokenizer, final_state, attn, device,
                                      f"### {mp.twin.answer.strip()}")  # "###" (21017) as the model emits it, not " ###" (44386)
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
def evaluate_rotation(base_model, embedding, tokenizer, eos_id, special_ids, device, num_latents,
                       group: list[int], k: int, rotation: OrthogonalRotation, eval_pairs, max_new_tokens: int):
    records = []
    for mp, pred_base, twin_vecs in eval_pairs:
        final_state, _input_ids, attn = run_intervened(
            base_model, embedding, tokenizer, special_ids, mp.original.question, device, num_latents,
            rotation, group, twin_vecs, k, grad=False)
        inputs_embeds, kv_cache, next_compute_range, *_ = final_state
        raw = finish_and_decode(base_model, embedding, tokenizer, inputs_embeds, kv_cache, next_compute_range,
                                 attn, device, max_new_tokens, eos_id)
        pred_patched = extract_answer_after_delimiter(raw)
        matches_twin = is_correct(pred_patched, mp.twin.answer) and not is_correct(pred_base, mp.twin.answer)
        records.append({
            "recipient_idx": mp.original.idx, "step": mp.step, "perturbed_number": mp.perturbed_number,
            "delta": mp.delta, "answer_base": pred_base, "recipient_gold": mp.original.answer,
            "twin_answer": mp.twin.answer, "answer_patched": pred_patched,
            "answer_changed": pred_patched != pred_base, "matches_twin": matches_twin,
        })
    n = len(records)
    x_changed = sum(r["answer_changed"] for r in records)
    x_matched = sum(r["matches_twin"] for r in records)
    return records, {
        "group": group, "group_label": group_label(group), "k": k, "n": n,
        "answer_changed_rate": x_changed / n if n else 0.0,
        "answer_changed_wilson_ci": list(wilson_ci(x_changed, n)),
        "matches_twin_rate": x_matched / n if n else 0.0,
        "matches_twin_wilson_ci": list(wilson_ci(x_matched, n)),
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

    groups = parse_site_groups(da.site_groups)
    k_values = [int(s) for s in da.k_values.split(",")]
    print(f"site_groups={groups} k_values={k_values}")

    # train and eval from Coconut's own train / valid files (before 2026-09-27 both came from the
    # 1194-example test file with train_pool_n >= 1194, so every eval recipient was also a train recipient)
    train_examples = load_examples(da.data_dir, da.train_pool_n, da.train_seed, "gsm_original_train.json")
    eval_examples = load_examples(da.data_dir, da.eval_pool_n, da.eval_seed, "gsm_original_valid.json")

    train_pairs, train_acc, train_n_correct = build_minimal_pairs(
        base_model, embedding, tokenizer, eos_id, special_ids, device, da.num_latents, da.max_new_tokens,
        train_examples, da.train_n_pairs, da.train_max_candidate_checks, da.pair_seed, tag="train")
    eval_pairs, eval_acc, eval_n_correct = build_minimal_pairs(
        base_model, embedding, tokenizer, eos_id, special_ids, device, da.num_latents, da.max_new_tokens,
        eval_examples, da.eval_n_pairs, da.eval_max_candidate_checks, da.pair_seed + 1, tag="eval")

    all_summaries = []
    all_records = []
    t0 = time.perf_counter()
    reference_summaries = []
    for group in groups:
        # untrained references on the same eval pairs: full-vector swap (k=768), random rotation per k
        torch.manual_seed(da.train_seed)
        for ref_k, tag in [(HIDDEN_DIM, "full")] + [(k, "untrained") for k in k_values]:
            ref_rot = OrthogonalRotation(HIDDEN_DIM).to(device).eval()
            records, summary = evaluate_rotation(base_model, embedding, tokenizer, eos_id, special_ids, device,
                                                  da.num_latents, group, ref_k, ref_rot, eval_pairs, da.max_new_tokens)
            summary["reference"] = tag
            reference_summaries.append(summary)
            all_records.extend([{"group": group_label(group), "k": ref_k, "reference": tag, **r} for r in records])
            print(f"[ref {tag}] group={group_label(group)} k={ref_k}: matches_twin={summary['matches_twin_rate']:.3f} "
                  f"answer_changed={summary['answer_changed_rate']:.3f}", flush=True)
        for k in k_values:
            print(f"--- training group={group_label(group)} k={k} (n_train_pairs={len(train_pairs)}) ---", flush=True)
            rotation = train_rotation(base_model, embedding, tokenizer, special_ids, device, da.num_latents,
                                       group, k, list(train_pairs), da.lr, da.epochs)
            records, summary = evaluate_rotation(base_model, embedding, tokenizer, eos_id, special_ids, device,
                                                  da.num_latents, group, k, rotation, eval_pairs, da.max_new_tokens)
            all_records.extend([{"group": group_label(group), "k": k, **r} for r in records])
            if da.save_rotations:
                os.makedirs(da.save_rotations, exist_ok=True)
                torch.save({"W": rotation.rot.weight.detach().float().cpu(), "k": k, "group": group,
                            "mechanism": "coconut", "matches_twin_rate": summary["matches_twin_rate"]},
                           os.path.join(da.save_rotations, f"rot_{group_label(group)}_k{k}.pt"))
            all_summaries.append(summary)
            print(f"group={group_label(group)} k={k}: matches_twin={summary['matches_twin_rate']:.3f} "
                  f"answer_changed={summary['answer_changed_rate']:.3f} (n={summary['n']}, "
                  f"{time.perf_counter() - t0:.0f}s elapsed)", flush=True)

    best = max(all_summaries, key=lambda s: s["matches_twin_rate"])
    print(f"BEST: group={best['group_label']} k={best['k']} matches_twin={best['matches_twin_rate']:.3f}")

    metrics = {
        "compute_steps": da.num_latents,
        "intervention_accuracy": best["matches_twin_rate"],
        "extra": {
            "design": "E4: DAS on same-problem minimal-pair donors, joint multi-site rotation "
                      "(steered_to_donor_audit.md #5, next_experiments.md pasted plan)",
            "sweep": all_summaries,
            "references": reference_summaries,
            "best_group": best["group_label"], "best_k": best["k"],
            "site_groups": [group_label(g) for g in groups], "k_values": k_values,
            "train_n_pairs_requested": da.train_n_pairs, "train_n_pairs_actual": len(train_pairs),
            "eval_n_pairs_requested": da.eval_n_pairs, "eval_n_pairs_actual": len(eval_pairs),
            "train_base_accuracy": train_acc, "eval_base_accuracy": eval_acc,
            "epochs": da.epochs, "lr": da.lr,
            "n_trainable_params_backbone": n_params,
            "teacher_forcing_target": "'### {twin.answer}' (token 21017 as the model emits it) after the final "
                                       "pass; pools: train = gsm_original_train.json, eval = gsm_original_valid.json",
            "site_definition": "pass p = vector spliced into the p-th <|latent|> slot",
            "related_runs": [
                "20260920-195725_coconut_minimal-pair-patch",
                "20260920-053935_coconut_das-pilot",
            ],
        },
    }

    record = RunRecord(
        run_id=new_run_id("coconut", da.slug),
        mechanism=mechanism_name,
        stage=da.stage,
        model=ModelInfo(backbone=da.model_id, checkpoint=da.checkpoint_label, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="validation", n_examples=len(eval_examples), seed=da.eval_seed),
        metrics=metrics,
        hyperparams={
            "num_latents": da.num_latents, "site_groups": [group_label(g) for g in groups], "k_values": k_values,
            "train_n_pairs": da.train_n_pairs, "eval_n_pairs": da.eval_n_pairs,
            "epochs": da.epochs, "lr": da.lr,
        },
        seed=da.train_seed,
        hardware=f"{da.hardware} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        notes="E4: Distributed Alignment Search on same-problem minimal-pair donors for Coconut -- "
              "a shared learned orthogonal rotation R isolates a k-dim subspace, jointly patched "
              "across every pass in a group (pass-1-alone vs. joint {1,4}), trained to maximize the "
              "twin's own gold-answer likelihood. Tests whether E3(a)'s pass-1 48.6% and pass-4 "
              "28.6% minimal-pair signals are explained by a low-dimensional linear subspace.",
    )
    manifest = record.save(predictions=all_records)
    out_dir = manifest.parent
    (out_dir / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
