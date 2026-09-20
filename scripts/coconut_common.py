"""Shared model-loading + intervention primitives for the Coconut interpretability
sweep (decode_patch_coconut.py, probe_coconut.py) -- the Coconut-side counterpart to
decode_patch_codi.py's `run_thoughts`/`decode_answer`.

Uses the VANILLA `facebookresearch/coconut` `Coconut` module (from the checkout at
`~/Projects/coconut`, copied onto the pod -- same convention as CODI: run inside the
reference repo's own checkout so `coconut.py`/`dataset.py` import cleanly) with the
released checkpoint `connordilgren/gpt2-gsm8k-coconut` (from the "Are Latent Reasoning
Models Easily Interpretable?" (Dilgren & Wiegreffe, COLM 2026) checkpoint collection --
their fork of Coconut only *adds* attention-ablation hooks on top of the same
`base_causallm`/`embedding`, so the state dict loads unchanged into the vanilla class).

Coconut's continuous thoughts are NOT a separate loop like CODI's: `<|latent|>` tokens
sit at fixed positions in one sequence, and each pass's hidden state (from the
position immediately before the next `<|latent|>`) is spliced into that position's
`inputs_embeds` before the next forward pass. `run_passes` below is `Coconut.forward`
(github.com/facebookresearch/coconut, MIT) with two additions: (1) it records the
LIVE hidden vector that would normally fill each pass's latent position -- for the
logit-lens / probe extraction, via `lm_head` applied directly to that vector (the
paper's own "vocabulary projection" method, no extra forward pass needed); (2)
`override_at_pass={pass_idx: donor_vector}` lets a donor's vector be spliced in
instead, for causal patching -- the Coconut analogue of CODI's `override_input_at`.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2 import GPT2LMHeadModel

LATENT_TOKEN = "<|latent|>"
START_LATENT_TOKEN = "<|start-latent|>"
END_LATENT_TOKEN = "<|end-latent|>"


def load_coconut(model_id: str, checkpoint_path: str, device: str):
    """Mirrors `models/coconut/coconut_model.py`'s `load()` (Dilgren & Wiegreffe repo)
    exactly, but returns the plain (tokenizer, base_causallm, embedding, lm_head,
    special_token_ids) tuple this module's functions operate on directly -- no
    ablation-hook wrapper."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(START_LATENT_TOKEN)
    tokenizer.add_tokens(END_LATENT_TOKEN)
    tokenizer.add_tokens(LATENT_TOKEN)
    special_ids = {
        "latent": tokenizer.convert_tokens_to_ids(LATENT_TOKEN),
        "start_latent": tokenizer.convert_tokens_to_ids(START_LATENT_TOKEN),
        "end_latent": tokenizer.convert_tokens_to_ids(END_LATENT_TOKEN),
    }
    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    base_model.resize_token_embeddings(len(tokenizer))

    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        # checkpoints save the whole Coconut module (`base_causallm.*`, no separate
        # embedding params -- `self.embedding` is a reference, not new weights)
        missing, unexpected = base_model.load_state_dict(
            {k.removeprefix("base_causallm."): v for k, v in state.items() if k.startswith("base_causallm.")},
            strict=False,
        )
        print(f"loaded {checkpoint_path}: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        raise FileNotFoundError(checkpoint_path)

    base_model = base_model.to(device)
    base_model.eval()
    embedding = base_model.transformer.get_input_embeddings() if isinstance(base_model, GPT2LMHeadModel) else base_model.get_input_embeddings()
    return tokenizer, base_model, embedding, special_ids


def encode_question(tokenizer, special_ids: dict, question: str, num_latents: int, device: str):
    q = question if question.endswith("\n") else question + "\n"
    q_ids = tokenizer.encode(q, add_special_tokens=True)
    tokens = q_ids + [special_ids["start_latent"]] + [special_ids["latent"]] * num_latents + [special_ids["end_latent"]]
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)
    return input_ids, attn


@torch.no_grad()
def run_passes(base_model, embedding, input_ids, attn, device, num_latents: int,
                override_at_pass: dict[int, torch.Tensor] | None = None, latent_token_id: int = None):
    """Runs the Coconut latent-filling loop (adapted from `Coconut.forward`, MIT
    licensed, github.com/facebookresearch/coconut). Returns
    (pass_records, final_inputs_embeds, final_kv_cache, next_compute_range) --
    `pass_records[p]` = {"pass": p, "live_hidden": tensor, "logits": tensor} where
    `live_hidden` is the UN-overridden hidden vector (for logit-lens/probes) and
    `logits = lm_head(live_hidden)`; the value actually spliced into `inputs_embeds`
    for pass p+1 is the override if given, else `live_hidden`."""
    latent_indices = (input_ids == latent_token_id).nonzero()
    latent_positions = [idx[1].item() for idx in latent_indices if idx[0] == 0]
    max_n_latents = len(latent_positions)
    assert max_n_latents == num_latents, f"expected {num_latents} latent tokens, found {max_n_latents}"

    seq_len = input_ids.shape[1]
    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
    inputs_embeds = embedding(input_ids)
    next_compute_range = (0, latent_positions[0])
    kv_cache = None
    pass_records = []

    for pass_idx in range(max_n_latents):
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

        next_compute_range = (next_compute_range[1], seq_len if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1)
        hidden_states = outputs.hidden_states[-1]
        kv_cache = _to_legacy_cache(outputs.past_key_values)

        token_idx = latent_positions[pass_idx]
        live_hidden = hidden_states[0, token_idx - 1 - hidden_states_offset, :].detach().clone()
        logits = base_model.lm_head(live_hidden)
        pass_records.append({"pass": pass_idx, "live_hidden": live_hidden.float().cpu(), "logits": logits.float().cpu()})

        fill_value = override_at_pass[pass_idx] if (override_at_pass and pass_idx in override_at_pass) else live_hidden
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[0, token_idx, :] = fill_value.to(inputs_embeds.dtype).to(device)

    return pass_records, inputs_embeds, kv_cache, next_compute_range


@torch.no_grad()
def finish_and_decode(base_model, embedding, tokenizer, inputs_embeds, kv_cache, next_compute_range,
                       attn, device, max_new_tokens: int, eos_token_id: int) -> str:
    """Final pass (question+filled-latents -> first output token) + autoregressive
    continuation, mirroring `Coconut.generate`'s tail exactly."""
    seq_len = inputs_embeds.shape[1]
    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
    past_key_values = (
        [(k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :]) for k, v in kv_cache]
        if kv_cache else None
    )
    outputs = base_model(
        inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
        attention_mask=attn[:, :next_compute_range[1]],
        position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
        past_key_values=past_key_values,
        output_hidden_states=True,
    )
    next_token = torch.argmax(outputs.logits[0, -1]).item()
    tokens = [next_token]
    new_embed = embedding(torch.tensor(next_token, device=device)).view(1, 1, -1)
    new_inputs_embeds = torch.cat((inputs_embeds, new_embed), dim=1)

    for _ in range(max_new_tokens - 1):
        outputs = base_model(inputs_embeds=new_inputs_embeds)
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        if next_token == eos_token_id:
            break
        tokens.append(next_token)
        new_embed = embedding(torch.tensor(next_token, device=device)).view(1, 1, -1)
        new_inputs_embeds = torch.cat((new_inputs_embeds, new_embed), dim=1)

    return tokenizer.decode(tokens, skip_special_tokens=True)


def _to_legacy_cache(kv_cache):
    """Newer `transformers` returns a `DynamicCache` object from `past_key_values`
    (per-layer `.keys`/`.values` via `cache.layers[i]`, no `__getitem__`/
    `to_legacy_cache` in current releases); older ones (incl. the version pinned in
    coconut's requirements.txt) return plain `((k, v), ...)` tuples directly. This
    module's slicing (`k[:, :, :n, :]`) needs the tuple form -- normalize to it."""
    if kv_cache is not None and hasattr(kv_cache, "layers"):
        return tuple((layer.keys, layer.values) for layer in kv_cache.layers)
    return kv_cache


def topk_token_strings(tokenizer, logits_1d: torch.Tensor, k: int) -> list[str]:
    top = torch.topk(logits_1d, k).indices.tolist()
    return [tokenizer.decode([t]).strip() for t in top]


def extract_answer_after_delimiter(text: str, delimiter: str = "###") -> Optional[str]:
    """Coconut/CoT GSM8K format is `... ### <answer>` (per this checkpoint's training
    data / the Dilgren & Wiegreffe repo's own `extract_answer`)."""
    if delimiter in text:
        parts = text.split(delimiter)
        if len(parts) >= 2:
            return parts[-1].strip().replace(",", "")
    return None


def parse_step_value(step: str) -> Optional[str]:
    """`"<<16-3-4=9>>"` -> `"9"` -- the gold-trace `steps` field's per-step format
    (identical `<<...=x>>` convention to our own `gsm8k_aug.parse_intermediate_values`,
    just pre-split into a list here instead of embedded in one rationale string)."""
    if "=" not in step:
        return None
    return step.split("=")[-1].strip().rstrip(">").strip()


def num_match(candidates: list[str], gold: str) -> bool:
    gold = gold.strip().lstrip("+")
    for c in candidates:
        c = c.strip()
        if c == gold:
            return True
        try:
            if float(c.replace(",", "")) == float(gold.replace(",", "")):
                return True
        except ValueError:
            continue
    return False


def wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = x / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_le_k = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p_le_k)
