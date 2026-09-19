#!/usr/bin/env python3
"""Causal-patching test for the step-supervised recurrent_depth checkpoint -- the Oct 16
causality item flagged in `results/20260918-214656_recurrent_depth_stepsup-split4-4-4-full/
notes.md`'s "Next" section, designed to mirror `results/20260919-073312_codi_decode-patch-
pilot/notes.md`'s CODI test so the two mechanisms are directly comparable.

Model recap (`LoopedGPT2`, see `recurrent_depth_model.py`): for a probe sequence
`prompt + " ####"`, `s_0 ~ N(0, std^2)`, and `s_i = iterate(e, s_{i-1})` for i=1..r, where
`e = prelude(embed(x))` is fixed per example and `iterate` runs the looped core blocks with
full causal self-attention over every position. `readout(s_i)[:, -1]` reads the last
position (right after " ####"). Training target (see the full-run notes): iteration i<n is
supervised on step i's value, iteration n (n = the example's own step count) on the answer
-- so "iteration n's read-out" is a literal, trained proxy for the generated answer's first
token, not a post-hoc correlation (the full-run's post-hoc check found it matches the
*actual* generated answer 84%/49.7% of the time at r=1; here we use the trained target
directly instead of full multi-step generation, which this architecture makes expensive to
patch into -- `LoopedGPT2.generate` has no KV cache and recomputes s_0..s_r from fresh noise
at every new token, so a patch at one iteration doesn't persist token-to-token the way it
would with a cache. Restricting to the trained probe context sidesteps that entirely).

Patch mechanic: run the recipient's own iterate() through iteration `s-1` (its own e, s_0,
causal mask -- nothing borrowed). Compute iteration s normally, then overwrite ONLY the last
position's state vector with the donor's independently-computed `s_s[:, -1, :]` (donor's own
e, s_0, probe). Continue iterating s+1..n on this hybrid state. Because attention is causal
and position -1 is the sequence's last position, nothing attends TO it during iterations
1..s-1 (already computed) and only the last position's own future trajectory is touched --
a clean, single-vector patch, same spirit as CODI's single-latent-token swap.

Two conditions per recipient, both patched at the same iteration `s` and compared to the
recipient's own unpatched baseline (same s_0, same seed):
  - real:    donor = a same-step-count example with a DIFFERENT gold value at step s.
  - control: donor = an unrelated (example, iteration) pair, not step/value-matched.

Metrics:
  - `readout_next`: does iteration s+1's read-out (if s+1 <= n_steps-1, i.e. it has a step
    target) move to the donor's step-s value's first token? (real only -- "moved toward
    donor" isn't defined for an unrelated control.)
  - `answer_proxy_changed`: does iteration n_steps's read-out (the answer-trained iteration)
    change from the unpatched baseline's? Reported for both real and control, paired
    (McNemar) since both run against the same recipient/baseline.

Run per-example (batch=1) rather than batched: donor and recipient prompts have different
lengths, and correctness matters far more than throughput here -- even at n=200 pairs x 3
conditions x <=8 iterations of a 12-effective-layer, 125M model, this is seconds of GPU time.

Usage (inside the recurrent_depth training env, GPU):
  uv run python scripts/patch_recurrent_depth.py \\
      --ckpt-dir results/20260918-214656_recurrent_depth_stepsup-split4-4-4-full/ckpt \\
      --n-pairs 200 --seed 42 --slug patch-pilot --stage pilot \\
      --hardware "RunPod RTX A5000 (secure)"
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from latentreasoning.data.gsm8k_aug import Example, load_gsm8k_aug
from latentreasoning.mechanisms.recurrent_depth import ANSWER_PREFIX, build_eval_prompt_ids, first_value_token
from latentreasoning.mechanisms.recurrent_depth import name as mechanism_name
from latentreasoning.mechanisms.recurrent_depth_model import LoopedGPT2
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id

MODEL_ID = "gpt2"


def prompt_ids(tokenizer, ex: Example) -> list[int]:
    prefix = tokenizer(ANSWER_PREFIX, add_special_tokens=False)["input_ids"]
    return build_eval_prompt_ids(tokenizer, ex.question) + prefix


@torch.no_grad()
def states_for_example(model: LoopedGPT2, tokenizer, ex: Example, n_iter: int, seed: int, device: str):
    """Run `n_iter` honest iterations on `ex`'s own probe; return the list of per-iteration
    states s_1..s_n (each [1, seq_len, h], fp32) and (causal_mask, position_ids) for re-use
    by the caller (patched continuations reuse the recipient's own mask/positions, not the
    donor's -- only the donor's last-position vector is borrowed)."""
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
def patched_trajectory(model, e, cm, pos, s_start, start_iter: int, n_iter: int, donor_vec: torch.Tensor):
    """`s_start` = recipient's own state AT `start_iter` (already computed honestly); patch
    its last position with `donor_vec`, then continue iterating to `n_iter`. Returns the
    list of post-patch states for iterations start_iter+1..n_iter (index 0 = iteration
    start_iter+1)."""
    s = s_start.clone().to(next(model.parameters()).dtype)
    s[:, -1, :] = donor_vec.to(s.dtype)
    out = []
    for _ in range(start_iter, n_iter):
        s = model.iterate(e, s, cm, pos)
        out.append(s.float())
    return out


def readout_top1(model, s, cm, pos) -> int:
    return int(model.readout(s.to(next(model.parameters()).dtype), cm, pos)[0, -1].argmax(-1).item())


def build_pairs(examples: list[Example], n_pairs: int, rng: random.Random):
    """(recipient_idx, step s [1-indexed, s <= n_steps-1], donor_idx) with donor sharing
    recipient's step count and differing at step s. Sampled from all valid (example, step)
    tuples without replacement on the tuple (donors may repeat)."""
    by_nsteps: dict[int, list[int]] = {}
    for i, ex in enumerate(examples):
        n = len(ex.intermediate_values)
        if n >= 2:
            by_nsteps.setdefault(n, []).append(i)
    tuples = []
    for i, ex in enumerate(examples):
        n = len(ex.intermediate_values)
        if n < 2:
            continue
        for s in range(1, n):  # s = 1..n-1 (iterations with a step target)
            tuples.append((i, s))
    rng.shuffle(tuples)
    pairs = []
    for i, s in tuples:
        if len(pairs) >= n_pairs:
            break
        n = len(examples[i].intermediate_values)
        v_s = examples[i].intermediate_values[s - 1]
        candidates = [j for j in by_nsteps[n] if j != i and examples[j].intermediate_values[s - 1] != v_s]
        if not candidates:
            continue
        pairs.append((i, s, rng.choice(candidates)))
    return pairs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--n-pairs", type=int, default=200)
    p.add_argument("--eval-n", type=int, default=200)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--slug", default="patch-pilot")
    p.add_argument("--stage", default="pilot", choices=["pilot", "full_run"])
    p.add_argument("--hardware", default="RunPod GPU")
    p.add_argument("--source-run-id", default="20260918-214656_recurrent_depth_stepsup-split4-4-4-full")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = LoopedGPT2.load(args.ckpt_dir, backbone=MODEL_ID).to(device).eval()
    n_params = sum(p_.numel() for p_ in model.parameters())

    examples = load_gsm8k_aug(split="test", n=args.eval_n, seed=args.eval_seed)
    pairs = build_pairs(examples, args.n_pairs, rng)
    print(f"built {len(pairs)} (recipient, step, donor) tuples from {len(examples)} examples", flush=True)

    recs = []
    t0 = time.perf_counter()
    for k, (ri, s, di) in enumerate(pairs):
        rex, dex = examples[ri], examples[di]
        n_steps = len(rex.intermediate_values)
        r_seed = args.seed * 100003 + ri  # fixed per recipient, shared by baseline/real/control
        d_seed = args.seed * 100003 + di

        r_states, e_r, cm_r, pos_r = states_for_example(model, tokenizer, rex, n_steps, r_seed, device)
        d_states, _, _, _ = states_for_example(model, tokenizer, dex, s, d_seed, device)  # only need s_s
        donor_vec = d_states[s - 1][:, -1, :]  # [1, h]

        # unrelated control donor: any other example, any of ITS valid iterations
        cj = rng.choice([j for j in range(len(examples)) if j != ri and j != di])
        c_iter = rng.randint(1, max(1, len(examples[cj].intermediate_values)))
        c_states, _, _, _ = states_for_example(model, tokenizer, examples[cj], c_iter, args.seed * 100003 + cj, device)
        control_vec = c_states[c_iter - 1][:, -1, :]

        baseline_next = int((s < n_steps - 1)) and readout_top1(model, r_states[s], cm_r, pos_r)  # iter s+1 = index s
        baseline_final = readout_top1(model, r_states[n_steps - 1], cm_r, pos_r)  # iter n_steps = index n_steps-1

        real_traj = patched_trajectory(model, e_r, cm_r, pos_r, r_states[s - 1], s, n_steps, donor_vec)
        ctrl_traj = patched_trajectory(model, e_r, cm_r, pos_r, r_states[s - 1], s, n_steps, control_vec)

        real_next = readout_top1(model, real_traj[0], cm_r, pos_r) if s < n_steps - 1 else None
        real_final = readout_top1(model, real_traj[-1], cm_r, pos_r)
        ctrl_final = readout_top1(model, ctrl_traj[-1], cm_r, pos_r)

        donor_val_tok = first_value_token(tokenizer, dex.intermediate_values[s - 1])
        recs.append({
            "recipient_idx": ri, "donor_idx": di, "control_donor_idx": cj, "control_iter": c_iter,
            "step": s, "n_steps": n_steps, "donor_value": dex.intermediate_values[s - 1],
            "recipient_value": rex.intermediate_values[s - 1],
            "baseline_next_matches_donor": bool(real_next is not None and baseline_next == donor_val_tok),
            "real_next_matches_donor": bool(real_next is not None and real_next == donor_val_tok),
            "has_next_target": s < n_steps - 1,
            "baseline_final": baseline_final, "real_final": real_final, "ctrl_final": ctrl_final,
            "real_final_changed": bool(real_final != baseline_final),
            "ctrl_final_changed": bool(ctrl_final != baseline_final),
        })
        if (k + 1) % 25 == 0:
            print(f"{k + 1}/{len(pairs)} pairs done ({time.perf_counter() - t0:.0f}s)", flush=True)

    elapsed = time.perf_counter() - t0
    n = len(recs)
    n_with_next = sum(r["has_next_target"] for r in recs)
    baseline_coincidence = sum(r["baseline_next_matches_donor"] for r in recs) / max(1, n_with_next)
    real_moved = sum(r["real_next_matches_donor"] for r in recs) / max(1, n_with_next)
    real_changed = sum(r["real_final_changed"] for r in recs) / n
    ctrl_changed = sum(r["ctrl_final_changed"] for r in recs) / n
    both = sum(r["real_final_changed"] and r["ctrl_final_changed"] for r in recs)
    real_only = sum(r["real_final_changed"] and not r["ctrl_final_changed"] for r in recs)
    ctrl_only = sum(not r["real_final_changed"] and r["ctrl_final_changed"] for r in recs)
    neither = n - both - real_only - ctrl_only

    # McNemar exact (binomial on the discordant pairs), no scipy dependency assumed present
    import math
    def binom_two_sided_p(k, n, p=0.5):
        if n == 0:
            return 1.0
        from math import comb
        k = min(k, n - k)
        return min(1.0, 2 * sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1)))
    disc = real_only + ctrl_only
    mcnemar_p = binom_two_sided_p(real_only, disc) if disc > 0 else 1.0

    def wilson(k, n, z=1.96):
        if n == 0:
            return (0.0, 0.0, 0.0)
        p_ = k / n
        denom = 1 + z * z / n
        center = (p_ + z * z / (2 * n)) / denom
        half = z * math.sqrt(p_ * (1 - p_) / n + z * z / (4 * n * n)) / denom
        return (p_, max(0.0, center - half), min(1.0, center + half))

    summary = {
        "n_pairs": n, "n_with_next_target": n_with_next,
        "readout_next_moved_to_donor": {"real": real_moved, "baseline_coincidence": baseline_coincidence, "wilson_real": wilson(sum(r["real_next_matches_donor"] for r in recs), n_with_next)},
        "answer_proxy_changed": {"real": real_changed, "control": ctrl_changed, "wilson_real": wilson(sum(r["real_final_changed"] for r in recs), n), "wilson_control": wilson(sum(r["ctrl_final_changed"] for r in recs), n)},
        "mcnemar_2x2": {"both_changed": both, "real_only": real_only, "control_only": ctrl_only, "neither": neither, "p_value": mcnemar_p},
        "elapsed_s": elapsed,
    }
    print(json.dumps(summary, indent=2, default=str))

    record = RunRecord(
        run_id=new_run_id(mechanism_name, args.slug),
        mechanism=mechanism_name,
        stage=args.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=f"local:{args.source_run_id}/ckpt", n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(examples), seed=args.eval_seed),
        metrics={
            "final_answer_accuracy": None, "unparseable_rate": 0.0, "compute_steps": None,
            "intervention_accuracy": real_moved,  # RQ2: does patching move the read-out toward the predicted value
            "extra": {**summary, "source_run_id": args.source_run_id, "design": "causal patching, step-supervised checkpoint, per-pair batch=1"},
        },
        hyperparams={"n_pairs": args.n_pairs, "eval_n": args.eval_n, "eval_seed": args.eval_seed, "patch_seed": args.seed},
        seed=args.seed,
        hardware=f"{args.hardware} / {torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}",
        notes=f"recurrent_depth causal-patching {args.slug}: {n} pairs against {args.source_run_id}",
    )
    manifest = record.save(predictions=recs)
    out_dir = manifest.parent
    (out_dir / "patch_pairs.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
