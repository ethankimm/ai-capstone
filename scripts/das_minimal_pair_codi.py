#!/usr/bin/env python3
"""E4 (`steered_to_donor_audit.md` §5, `next_experiments.md` pasted plan): Distributed
Alignment Search on SAME-PROBLEM MINIMAL-PAIR donors for CODI -- conditional on E3(a)
(`results/20260920-190420_codi_minimal-pair-patch`), which found ALL-SLOT minimal-pair
patching steers to the twin's answer 70.1% of the time but no SINGLE iteration carries
more than 10.9% alone, with a sharp jump between cumulative prefix 1..3 (8.0%) and
prefix 1..4 (46.7%). E4 asks: is that prefix-1..4 jump explained by a k-dimensional
LINEAR SUBSPACE (a learned orthogonal rotation R, not the raw standard basis) rather
than the full 768-dim vector?

Two of `das_codi.py`'s three problems for this question are already fixed by reusing
minimal pairs instead of cross-problem donors:
  - Training target is well-defined: the twin's OWN gold answer via `matches_cf`==
    `matches_twin` by construction (no cross-problem "recipient never saw the donor's
    remaining program" confound -- see `latentreasoning/data/minimal_pairs.py`).
  - The undertrained-loss floor `das_codi.py` hit (30-40 nats, "plausibly unachievable
    by construction") should no longer apply if the value really is in the thoughts.

What's new here (`das_codi.py` only does ONE site per rotation):
  - JOINT multi-site interchange: one shared rotation R patches k-dims at EVERY site in
    a group (e.g. iterations {1,2,3,4} together), matching the "cumulative prefix"
    condition's site set but with a learned subspace instead of the raw full vector.
    `--site_groups` is a semicolon-separated list of comma-separated site lists, e.g.
    `"4;1,2,3,4"` = two groups: iteration 4 alone, and the joint {1,2,3,4} prefix span.
  - Donor vectors come from a per-pair minimal-pair twin's own live thought trajectory
    (`run_thoughts`, all 6 iterations decoded and cached once per pair at construction
    time), not another random example's.

Mechanism (same core DAS recipe as `das_codi.py`, generalized to a site SET): for each
site in the group (in increasing order), z^{A<-B}(R,k) = R^T[(Rz_B)[:k]; (Rz_A)[k:]] is
spliced in as the input feeding that iteration -- z_A is the recipient's own live value
(reflecting any earlier patch in the same group), z_B is the twin's own precomputed
value at that same iteration. One rotation R is shared across every site in the group
(all iterations are instances of the same 768-dim "thought slot" type). R is trained by
backpropagating the twin's-own-gold-answer teacher-forced cross-entropy through the
splice(s) and the rest of the loop (transformer + LoRA weights frozen throughout, via
`model.requires_grad_(False)`). No manual grad-context toggling: each rollout is short
(<=6 iterations) so running the WHOLE per-example forward under default autograd is
cheap even though only the early, pre-first-site segment doesn't strictly need it.

Scale: n=500 minimal-pair TRAINING tuples requested by the plan. Default k sweep
{8,16,32,64} matches the plan exactly (not pilot-reduced, unlike `das_codi.py`'s
cross-problem sweep, since n=500 train x default 5 epochs x 2 site_groups x 4 k = 8
(site_group,k) combos x 2500 steps each = 20,000 total forward+backward passes,
similar order of magnitude to the earlier `20260920-050602_codi_das-pilot`'s 13,500
steps (150 x 5 x 18 combos) despite fewer combos, because train_n_pairs is ~3x larger
per the plan's explicit n=500 spec.

Sites (aligned 2026-09-27): site s = z_s, the latent fed INTO loop iteration s+1,
s = 0..n_iters-1 (z_0 = latent-0). At iteration s+1 the recipient's own z_s is the
background and the twin's z_s the donor. The first E4 run
(`20260920-235312_codi_das-minimal-pair`, groups "4" and "1,2,3,4" in the old 1-indexed
labels) mixed the recipient's z_{i-1} background with the twin's z_i -- see the correction
note on that run. Default groups are now "4" and "0,2,4", the sites the aligned E3 rerun
(`20260927-004340_codi_minimal-pair-patch-aligned`) found carry the value. Each group also
gets two untrained references on the same eval pairs: `full` (k=768, i.e. the raw
full-vector swap -- any orthogonal R gives exactly z_B) and `untrained` (a random
orthogonal R at each k: the floor for what training adds).

Run inside the CODI venv, from the CODI checkout (same convention as `das_codi.py`):

  cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/das_minimal_pair_codi.py \\
      --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \\
      --slug das-minimal-pair --stage pilot --hardware "RunPod RTX A5000 (secure)" \\
      --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \\
      --lora_r 128 --lora_alpha 32 --lora_init --greedy True \\
      --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \\
      --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \\
      --site_groups "4;0,2,4" --k_values 8,16,32,64 \\
      --train_n_pairs 500 --eval_n_pairs 150 --epochs 5
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
from decode_patch_codi import encode_question, decode_answer, run_thoughts, wilson_ci  # noqa: E402

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.data.minimal_pairs import generate_minimal_pair  # noqa: E402
from latentreasoning.eval.counterfactual import parse_steps, qualifying_steps  # noqa: E402
from latentreasoning.eval.metrics import extract_final_number, is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402

HIDDEN_DIM = 768
DELTAS = (1, -1, 2, -2, 3, -3)


@dataclass
class DASArguments:
    slug: str = field(default="das-minimal-pair-aligned")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: Optional[str] = field(default=None)
    site_groups: str = field(
        default="4;0,2,4",
        metadata={"help": "semicolon-separated groups, each a comma-separated list of "
                           "sites z_s (0-indexed; z_s feeds iteration s+1) sharing one rotation"},
    )
    k_values: str = field(default="8,16,32,64", metadata={"help": "comma-separated subspace sizes to sweep"})
    train_n_pairs: int = field(default=500, metadata={"help": "qualified minimal-pair training tuples per (group,k)"})
    eval_n_pairs: int = field(default=150, metadata={"help": "held-out qualified minimal-pair tuples per (group,k)"})
    train_pool_n: int = field(default=2500, metadata={"help": "gsm8k-aug train examples to decode looking for train pairs"})
    eval_pool_n: int = field(default=800, metadata={"help": "gsm8k-aug validation examples to decode looking for eval pairs"})
    train_max_candidate_checks: int = field(default=2500)
    eval_max_candidate_checks: int = field(default=900)
    epochs: int = field(default=5, metadata={"help": "passes over the training pairs"})
    lr: float = field(default=1e-3)
    train_seed: int = field(default=0)
    eval_seed: int = field(default=0)
    pair_seed: int = field(default=0)
    max_new_tokens: int = field(default=64)


class OrthogonalRotation(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.rot = orthogonal(nn.Linear(d, d, bias=False))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.rot(z)

    def inverse(self, z_tilde: torch.Tensor) -> torch.Tensor:
        return F.linear(z_tilde, self.rot.weight.t())


def intervene(rotation: OrthogonalRotation, z_a: torch.Tensor, z_b: torch.Tensor, k: int) -> torch.Tensor:
    """z^{A<-B}(R,k): recipient A's background, donor B's target subspace, k dims."""
    tilde_a, tilde_b = rotation(z_a), rotation(z_b)
    patched_tilde = torch.cat([tilde_b[..., :k], tilde_a[..., k:]], dim=-1)
    return rotation.inverse(patched_tilde)


def parse_site_groups(spec: str) -> list[list[int]]:
    return [[int(s) for s in part.split(",")] for part in spec.split(";") if part.strip()]


def group_label(group: list[int]) -> str:
    return "+".join(str(s) for s in group) if len(group) > 1 else str(group[0])


def run_intervened(model, tokenizer, question: str, device: str, n_iters: int,
                    rotation: Optional[OrthogonalRotation], sites: set[int],
                    donor_vecs: Optional[dict[int, torch.Tensor]], k: int):
    """Full n_iters loop, splicing z^{A<-B}(R,k) in as the input feeding every
    iteration in `sites` (recipient's own current latent as background, donor's
    precomputed value at that iteration as the target-subspace donor). Returns
    (final_pkv, final_latent). No grad-context management -- rollouts are short
    (<=6 iterations) so tracking the whole thing under default autograd during
    training is cheap; callers doing eval wrap this in `torch.no_grad()`."""
    input_ids, attn = encode_question(model, tokenizer, question, device)
    outputs = model.codi(input_ids=input_ids, use_cache=True, output_hidden_states=True,
                          past_key_values=None, attention_mask=attn)
    pkv = outputs.past_key_values
    latent = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
    if model.use_prj:
        latent = model.prj(latent)
    for it in range(1, n_iters + 1):
        if it - 1 in sites:  # site s = z_s, fed into iteration s+1
            z_b = donor_vecs[it - 1].to(device=device, dtype=torch.float32)
            latent = intervene(rotation, latent.float(), z_b, k).to(latent.dtype)
        outputs = model.codi(inputs_embeds=latent, use_cache=True, output_hidden_states=True, past_key_values=pkv)
        pkv = outputs.past_key_values
        pre_hidden = outputs.hidden_states[-1][:, -1, :]
        latent = model.prj(pre_hidden.unsqueeze(1)) if model.use_prj else pre_hidden.unsqueeze(1)
    return pkv, latent


def teacher_forced_ce(model, tokenizer, pkv, device: str, target_text: str) -> Optional[torch.Tensor]:
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


def decode_one(model, tokenizer, device, n_iters, max_new_tokens, question: str):
    q = question.strip().replace("  ", " ")
    pkv, thoughts, latent = run_thoughts(model, tokenizer, q, device, n_iters, include_latent0=True)
    raw = decode_answer(model, tokenizer, pkv, latent, device, max_new_tokens)
    return extract_final_number(raw), thoughts


def build_minimal_pairs(model, tokenizer, device, n_iters, max_new_tokens, examples,
                         n_pairs: int, max_candidate_checks: int, pair_seed: int, tag: str):
    """Decode-then-filter minimal-pair pool, mirroring `patch_minimal_pair_codi.py`:
    base-correct originals, propagation-qualified perturbation, twin also base-correct.
    Returns a list of (mp, pred_base, twin_vecs) tuples, twin_vecs = {s: the twin's z_s}
    for s = 0..n_iters-1, for use at any site/group."""
    t0 = time.perf_counter()
    base_info: dict[int, dict] = {}
    for ex in examples:
        pred, _ = decode_one(model, tokenizer, device, n_iters, max_new_tokens, ex.question)
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
        twin_pred, twin_thoughts = decode_one(model, tokenizer, device, n_iters, max_new_tokens, mp.twin.question)
        if is_correct(twin_pred, mp.twin.answer):
            twin_vecs = {rec["iter"]: rec["post"] for rec in twin_thoughts if rec["iter"] < n_iters}
            pairs.append((mp, base_info[mp.original.idx]["pred"], twin_vecs))
        if n_checked % 100 == 0:
            print(f"[{tag}] checked {n_checked} candidates, {len(pairs)} qualified so far", flush=True)
    print(f"[{tag}] qualified pairs: {len(pairs)}/{n_checked} candidates checked", flush=True)
    return pairs, accuracy, len(correct_examples)


def train_rotation(model, tokenizer, device, n_iters, group: list[int], k: int,
                    train_pairs, lr: float, epochs: int) -> OrthogonalRotation:
    rotation = OrthogonalRotation(HIDDEN_DIM).to(device)
    sites = set(group)
    total_steps = max(1, epochs * len(train_pairs))
    opt = torch.optim.AdamW(rotation.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    step = 0
    t0 = time.perf_counter()
    for _ in range(epochs):
        random.shuffle(train_pairs)
        for mp, _pred_base, twin_vecs in train_pairs:
            q = mp.original.question.strip().replace("  ", " ")
            pkv, _latent = run_intervened(model, tokenizer, q, device, n_iters, rotation, sites, twin_vecs, k)
            loss = teacher_forced_ce(model, tokenizer, pkv, device, mp.twin.answer.strip())
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
def evaluate_rotation(model, tokenizer, device, n_iters, group: list[int], k: int,
                       rotation: OrthogonalRotation, eval_pairs, max_new_tokens: int):
    sites = set(group)
    records = []
    for mp, pred_base, twin_vecs in eval_pairs:
        q = mp.original.question.strip().replace("  ", " ")
        pkv, latent = run_intervened(model, tokenizer, q, device, n_iters, rotation, sites, twin_vecs, k)
        pred_patched = extract_final_number(decode_answer(model, tokenizer, pkv, latent, device, max_new_tokens))
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
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, DASArguments))
    model_args, data_args, training_args, da = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = "cuda"

    model, tokenizer, load_result = build_model(model_args, training_args)
    n_params = sum(p.numel() for p in model.parameters())
    model.requires_grad_(False)
    print(f"loaded {model_args.ckpt_dir}: missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}")

    n_iters = training_args.inf_latent_iterations
    groups = parse_site_groups(da.site_groups)
    k_values = [int(s) for s in da.k_values.split(",")]
    print(f"site_groups={groups} k_values={k_values}")

    train_examples = load_gsm8k_aug(split="train", n=da.train_pool_n, seed=da.train_seed)
    eval_examples = load_gsm8k_aug(split="validation", n=min(1000, da.eval_pool_n), seed=da.eval_seed)

    train_pairs, train_acc, train_n_correct = build_minimal_pairs(
        model, tokenizer, device, n_iters, da.max_new_tokens, train_examples,
        da.train_n_pairs, da.train_max_candidate_checks, da.pair_seed, tag="train")
    eval_pairs, eval_acc, eval_n_correct = build_minimal_pairs(
        model, tokenizer, device, n_iters, da.max_new_tokens, eval_examples,
        da.eval_n_pairs, da.eval_max_candidate_checks, da.pair_seed + 1, tag="eval")

    all_summaries = []
    reference_summaries = []
    all_records = []
    t0 = time.perf_counter()
    for group in groups:
        # untrained references on the same eval pairs: full-vector swap, random rotation per k
        torch.manual_seed(da.train_seed)
        for ref_k, tag in [(HIDDEN_DIM, "full")] + [(k, "untrained") for k in k_values]:
            ref_rot = OrthogonalRotation(HIDDEN_DIM).to(device).eval()
            records, summary = evaluate_rotation(model, tokenizer, device, n_iters, group, ref_k, ref_rot,
                                                  eval_pairs, da.max_new_tokens)
            summary["reference"] = tag
            reference_summaries.append(summary)
            all_records.extend([{"group": group_label(group), "k": ref_k, "reference": tag, **r} for r in records])
            print(f"[ref {tag}] group={group_label(group)} k={ref_k}: matches_twin={summary['matches_twin_rate']:.3f} "
                  f"answer_changed={summary['answer_changed_rate']:.3f}", flush=True)
        for k in k_values:
            print(f"--- training group={group_label(group)} k={k} (n_train_pairs={len(train_pairs)}) ---", flush=True)
            rotation = train_rotation(model, tokenizer, device, n_iters, group, k, list(train_pairs), da.lr, da.epochs)
            records, summary = evaluate_rotation(model, tokenizer, device, n_iters, group, k, rotation,
                                                  eval_pairs, da.max_new_tokens)
            all_records.extend([{"group": group_label(group), "k": k, **r} for r in records])
            all_summaries.append(summary)
            print(f"group={group_label(group)} k={k}: matches_twin={summary['matches_twin_rate']:.3f} "
                  f"answer_changed={summary['answer_changed_rate']:.3f} (n={summary['n']}, "
                  f"{time.perf_counter() - t0:.0f}s elapsed)", flush=True)

    best = max(all_summaries, key=lambda s: s["matches_twin_rate"])
    print(f"BEST: group={best['group_label']} k={best['k']} matches_twin={best['matches_twin_rate']:.3f}")

    metrics = {
        "compute_steps": n_iters,
        "intervention_accuracy": best["matches_twin_rate"],
        "extra": {
            "design": "E4: DAS on same-problem minimal-pair donors, joint multi-site rotation "
                      "(steered_to_donor_audit.md #5, next_experiments.md pasted plan)",
            "site_definition": "site s = z_s = latent fed into loop iteration s+1; z_0 = latent-0",
            "sweep": all_summaries,
            "references": reference_summaries,
            "best_group": best["group_label"], "best_k": best["k"],
            "site_groups": [group_label(g) for g in groups], "k_values": k_values,
            "train_n_pairs_requested": da.train_n_pairs, "train_n_pairs_actual": len(train_pairs),
            "eval_n_pairs_requested": da.eval_n_pairs, "eval_n_pairs_actual": len(eval_pairs),
            "train_base_accuracy": train_acc, "eval_base_accuracy": eval_acc,
            "epochs": da.epochs, "lr": da.lr,
            "n_trainable_params_backbone": n_params,
            "related_runs": [
                "20260920-235312_codi_das-minimal-pair",
                "20260927-004340_codi_minimal-pair-patch-aligned",
            ],
        },
    }

    record = RunRecord(
        run_id=new_run_id("codi", da.slug),
        mechanism="codi",
        stage=da.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=da.checkpoint_label or model_args.ckpt_dir, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="validation", n_examples=len(eval_examples), seed=da.eval_seed),
        metrics=metrics,
        hyperparams={
            "inf_latent_iterations": n_iters, "num_latent": training_args.num_latent,
            "site_groups": [group_label(g) for g in groups], "k_values": k_values,
            "train_n_pairs": da.train_n_pairs, "eval_n_pairs": da.eval_n_pairs,
            "epochs": da.epochs, "lr": da.lr,
        },
        seed=da.train_seed,
        hardware=f"{da.hardware} / {torch.cuda.get_device_name(0)}",
        notes="E4 (aligned sites z_0..z_5): Distributed Alignment Search on same-problem minimal-pair donors -- a shared "
              "learned orthogonal rotation R isolates a k-dim subspace, jointly patched across "
              "every site in a group (single-slot vs. joint prefix span), trained to maximize "
              "the twin's own gold-answer likelihood. Tests whether E3(a)'s ALL-SLOT upper bound "
              "(70.1% matches_twin) and the prefix-1..4 jump are explained by a low-dimensional "
              "linear subspace rather than the full 768-dim vector.",
    )
    manifest = record.save(predictions=all_records)
    out_dir = manifest.parent
    (out_dir / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
