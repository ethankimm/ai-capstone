#!/usr/bin/env python3
"""Diagnostics extending scripts/patch_recurrent_depth.py's causal-patching test
(results/20260919-075647_recurrent_depth_patch-pilot/): (1) zero/mean ablation at the
same injection position as that run's real/control patch conditions, replaying the exact
same (recipient, step, donor, control_donor) tuples from its patch_pairs.jsonl for direct
comparability; (2) an attention diagnostic on the coda's self-attention at the
answer-readout position, checking whether the model even attends to the position
patching targets vs. other prompt positions.

Architectural note on the attention diagnostic: LoopedGPT2's recurrent state s_i lives on
the SAME sequence positions at every iteration -- there's no separate "iteration i's
number-position token" for the coda to attend back over; the state at the last position
evolves in place across iterations, it isn't a growing set of positions. So "attention to
the number position at earlier iterations" isn't well-formed here. What's measured
instead: at the point the coda produces the answer-trained read-out, how much attention
mass does the last-position query assign to ITSELF (the exact vector patching overwrites)
vs. to other prompt positions. Near-100% self-attention would mean the read-out is (at
least structurally) positioned to depend on exactly the vector we patch; substantial mass
elsewhere would mean the model has an alternate route to the answer that doesn't go
through that vector at all -- either would help explain the patching null.

Usage (inside the recurrent_depth training env, GPU):
  uv run python scripts/patch_recurrent_depth_diagnostics.py \\
      --ckpt-dir results/20260918-214656_recurrent_depth_stepsup-split4-4-4-full/ckpt \\
      --pairs-file results/20260919-075647_recurrent_depth_patch-pilot/patch_pairs.jsonl \\
      --attn-n 80 --slug ablate-attn --stage pilot --hardware "RunPod RTX A5000 (secure)"
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer

from latentreasoning.data.gsm8k_aug import Example, load_gsm8k_aug
from latentreasoning.mechanisms.recurrent_depth import ANSWER_PREFIX, build_eval_prompt_ids
from latentreasoning.mechanisms.recurrent_depth import name as mechanism_name
from latentreasoning.mechanisms.recurrent_depth_model import LoopedGPT2
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id

MODEL_ID = "gpt2"


def prompt_ids(tokenizer, ex: Example) -> list[int]:
    prefix = tokenizer(ANSWER_PREFIX, add_special_tokens=False)["input_ids"]
    return build_eval_prompt_ids(tokenizer, ex.question) + prefix


@torch.no_grad()
def states_for_example(model, tokenizer, ex, n_iter, seed, device):
    ids = prompt_ids(tokenizer, ex)
    input_ids = torch.tensor([ids], device=device)
    attn = torch.ones_like(input_ids)
    x, cm, pos = model.embed(input_ids, attn)
    e = model._run(model.prelude, x, cm, pos)
    gen = torch.Generator(device=device).manual_seed(seed)
    s = model.initial_state(e, gen)
    states = []
    for _ in range(n_iter):
        s = model.iterate(e, s, cm, pos)
        states.append(s.float())
    return states, e, cm, pos


@torch.no_grad()
def patched_trajectory(model, e, cm, pos, s_start, start_iter, n_iter, donor_vec):
    s = s_start.clone().to(next(model.parameters()).dtype)
    s[:, -1, :] = donor_vec.to(s.dtype)
    out = []
    for _ in range(start_iter, n_iter):
        s = model.iterate(e, s, cm, pos)
        out.append(s.float())
    return out


def readout_top1(model, s, cm, pos) -> int:
    return int(model.readout(s.to(next(model.parameters()).dtype), cm, pos)[0, -1].argmax(-1).item())


@torch.no_grad()
def readout_with_attn(model, s, cm, pos):
    """Same computation as model.readout but also returns, per coda layer, the
    last-position query's attention distribution over all key positions.

    In this transformers version, GPT2Block.forward always returns a bare hidden_states
    tensor -- it calls `attn_output, _ = self.attn(...)` internally and discards the
    attention weights, even when output_attentions=True is threaded through **kwargs.
    So we hook block.attn directly and capture its own (attn_output, attn_weights)
    return, and require attn_implementation="eager" (SDPA/flash backends don't
    materialize weights to return, hook or no hook)."""
    x = s.to(next(model.parameters()).dtype)
    attn_per_layer: list[torch.Tensor] = []
    captured: dict = {}

    def make_hook(idx):
        def hook(module, args, kwargs, output):
            captured[idx] = output[1]  # (attn_output, attn_weights)
        return hook

    handles = []
    for i, block in enumerate(model.coda):
        handles.append(block.attn.register_forward_hook(make_hook(i), with_kwargs=True))
    try:
        for i, block in enumerate(model.coda):
            x = block(x, None, cm, position_ids=pos, output_attentions=True)
            a = captured.get(i)  # [batch, n_head, q_len, k_len]
            if a is None or a.dim() != 4:
                raise RuntimeError(f"expected attention weights [B,H,Q,K] from layer {i}, got "
                                    f"{type(a)} {getattr(a, 'shape', None)} -- check attn_implementation='eager'")
            attn_per_layer.append(a[0, :, -1, :].float().cpu())  # [n_head, k_len] at the last query position
    finally:
        for h in handles:
            h.remove()
    logits = model.lm_head(model.ln_f(x))
    return logits, attn_per_layer


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p_ = k / n
    denom = 1 + z * z / n
    center = (p_ + z * z / (2 * n)) / denom
    half = z * math.sqrt(p_ * (1 - p_) / n + z * z / (4 * n * n)) / denom
    return (p_, max(0.0, center - half), min(1.0, center + half))


def binom_two_sided_p(k, n, p=0.5):
    if n == 0:
        return 1.0
    from math import comb
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1)))


def mcnemar(recs, a_key, b_key):
    both = sum(r[a_key] and r[b_key] for r in recs)
    a_only = sum(r[a_key] and not r[b_key] for r in recs)
    b_only = sum(not r[a_key] and r[b_key] for r in recs)
    disc = a_only + b_only
    p = binom_two_sided_p(a_only, disc) if disc > 0 else 1.0
    return {"both": both, "a_only": a_only, "b_only": b_only, "p_value": p}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--pairs-file", required=True)
    p.add_argument("--eval-n", type=int, default=200)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attn-n", type=int, default=80)
    p.add_argument("--slug", default="ablate-attn")
    p.add_argument("--stage", default="pilot", choices=["pilot", "full_run"])
    p.add_argument("--hardware", default="RunPod GPU")
    p.add_argument("--source-run-id", default="20260918-214656_recurrent_depth_stepsup-split4-4-4-full")
    p.add_argument("--patch-run-id", default="20260919-075647_recurrent_depth_patch-pilot")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = LoopedGPT2.load(args.ckpt_dir, backbone=MODEL_ID, attn_implementation="eager").to(device).eval()
    n_params = sum(pp.numel() for pp in model.parameters())

    examples = load_gsm8k_aug(split="test", n=args.eval_n, seed=args.eval_seed)

    # ---- Task 1: ablation, replaying the exact pairs from the original patch run ----
    pairs = [json.loads(l) for l in Path(args.pairs_file).read_text().splitlines() if l.strip()]
    print(f"replaying {len(pairs)} pairs from {args.pairs_file}", flush=True)

    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    for i, ex in enumerate(examples):
        n = len(ex.intermediate_values)
        if n < 2:
            continue
        r_seed = args.seed * 100003 + i
        states, _, _, _ = states_for_example(model, tokenizer, ex, n, r_seed, device)
        for s in range(1, n):
            v = states[s - 1][:, -1, :].squeeze(0).cpu()
            sums[s] = sums.get(s, torch.zeros_like(v)) + v
            counts[s] = counts.get(s, 0) + 1
    mean_vec = {s: (sums[s] / counts[s]) for s in sums}
    print(f"per-s mean vectors computed from: {sorted(counts.items())}", flush=True)

    recs = []
    for k, rec in enumerate(pairs):
        ri, s, n_steps = rec["recipient_idx"], rec["step"], rec["n_steps"]
        rex = examples[ri]
        r_seed = args.seed * 100003 + ri
        r_states, e_r, cm_r, pos_r = states_for_example(model, tokenizer, rex, n_steps, r_seed, device)

        zero_vec = torch.zeros_like(r_states[s - 1][:, -1, :])
        mvec = mean_vec[s].to(device).unsqueeze(0)

        zero_traj = patched_trajectory(model, e_r, cm_r, pos_r, r_states[s - 1], s, n_steps, zero_vec)
        mean_traj = patched_trajectory(model, e_r, cm_r, pos_r, r_states[s - 1], s, n_steps, mvec)

        baseline_final = readout_top1(model, r_states[n_steps - 1], cm_r, pos_r)
        zero_final = readout_top1(model, zero_traj[-1], cm_r, pos_r)
        mean_final = readout_top1(model, mean_traj[-1], cm_r, pos_r)

        recs.append({
            "recipient_idx": ri, "step": s, "n_steps": n_steps,
            "donor_idx": rec.get("donor_idx"), "control_donor_idx": rec.get("control_donor_idx"),
            "baseline_final": baseline_final,
            "zero_final": zero_final, "mean_final": mean_final,
            "zero_final_changed": bool(zero_final != baseline_final),
            "mean_final_changed": bool(mean_final != baseline_final),
            "real_final_changed": rec["real_final_changed"],
            "ctrl_final_changed": rec["ctrl_final_changed"],
        })
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(pairs)} ablation pairs done", flush=True)

    n = len(recs)
    real_changed = sum(r["real_final_changed"] for r in recs) / n
    ctrl_changed = sum(r["ctrl_final_changed"] for r in recs) / n
    zero_changed = sum(r["zero_final_changed"] for r in recs) / n
    mean_changed = sum(r["mean_final_changed"] for r in recs) / n

    ablation_summary = {
        "n_pairs": n,
        "answer_proxy_changed": {
            "real": real_changed, "control": ctrl_changed,
            "zero_ablation": zero_changed, "mean_ablation": mean_changed,
            "wilson_zero": wilson(sum(r["zero_final_changed"] for r in recs), n),
            "wilson_mean": wilson(sum(r["mean_final_changed"] for r in recs), n),
        },
        "mcnemar_zero_vs_real": mcnemar(recs, "zero_final_changed", "real_final_changed"),
        "mcnemar_mean_vs_real": mcnemar(recs, "mean_final_changed", "real_final_changed"),
        "mcnemar_zero_vs_control": mcnemar(recs, "zero_final_changed", "ctrl_final_changed"),
        "mcnemar_mean_vs_control": mcnemar(recs, "mean_final_changed", "ctrl_final_changed"),
    }
    print("ABLATION SUMMARY:", json.dumps(ablation_summary, indent=2, default=str), flush=True)

    # ---- Task 2: attention diagnostic ----
    attn_examples = [ex for ex in examples if len(ex.intermediate_values) >= 2][: args.attn_n]
    n_coda = len(model.coda)
    self_mass_by_layer = [[] for _ in range(n_coda)]
    other_mass_by_layer = [[] for _ in range(n_coda)]
    for i, ex in enumerate(attn_examples):
        n_ex_steps = len(ex.intermediate_values)
        seed_i = args.seed * 100003 + i
        states, e, cm, pos = states_for_example(model, tokenizer, ex, n_ex_steps, seed_i, device)
        _, attn_per_layer = readout_with_attn(model, states[n_ex_steps - 1], cm, pos)
        for li, a in enumerate(attn_per_layer):
            head_avg = a.mean(dim=0)  # [k_len]
            self_mass_by_layer[li].append(float(head_avg[-1].item()))
            other_mass_by_layer[li].append(float(head_avg[:-1].sum().item()))
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(attn_examples)} attention examples done", flush=True)

    attn_summary = {
        "n_examples": len(attn_examples),
        "self_attention_mass_by_coda_layer": [sum(v) / len(v) for v in self_mass_by_layer],
        "other_positions_mass_by_coda_layer": [sum(v) / len(v) for v in other_mass_by_layer],
    }
    print("ATTENTION SUMMARY:", json.dumps(attn_summary, indent=2, default=str), flush=True)

    record = RunRecord(
        run_id=new_run_id(mechanism_name, args.slug),
        mechanism=mechanism_name,
        stage=args.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=f"local:{args.source_run_id}/ckpt", n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(examples), seed=args.eval_seed),
        metrics={
            "final_answer_accuracy": None, "unparseable_rate": 0.0, "compute_steps": None,
            "extra": {
                "ablation": ablation_summary, "attention": attn_summary,
                "source_run_id": args.source_run_id, "patch_run_id": args.patch_run_id,
                "design": "zero/mean ablation replaying patch-pilot pairs; coda self-attention diagnostic",
            },
        },
        hyperparams={"n_pairs": n, "attn_n": len(attn_examples), "eval_n": args.eval_n,
                      "eval_seed": args.eval_seed, "seed": args.seed},
        seed=args.seed,
        hardware=f"{args.hardware} / {torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}",
        notes=f"recurrent_depth ablation+attention diagnostics {args.slug}: "
              f"{n} ablation pairs, {len(attn_examples)} attention examples",
    )
    manifest = record.save(predictions=recs)
    out_dir = manifest.parent
    (out_dir / "ablation_pairs.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
