"""`LoopedGPT2`: pretrained GPT-2 re-wired into Geiping et al.'s prelude / looped-core /
coda form. Design notes and the mapping onto the paper live in `recurrent_depth.py`
(this file needs torch + transformers; that one doesn't).

Forward, for r iterations:

    e   = prelude(embed(x))                    # blocks [0, n_prelude)
    s_0 ~ N(0, init_state_std^2)
    s_i = core(adapter([e ; LN(s_{i-1})]))     # blocks [n_prelude, n_prelude + n_core), i = 1..r
    y   = lm_head(ln_f(coda(s_r)))             # remaining blocks

`forward(..., return_states=True)` hands back every `s_i` -- those are the per-iteration
intermediate states the Oct 9 decoding work will probe.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import GPT2LMHeadModel
from transformers.masking_utils import create_causal_mask

from latentreasoning.mechanisms.recurrent_depth import DEFAULT_INIT_STATE_STD


class LoopedGPT2(nn.Module):
    CONFIG_FILE = "looped_config.json"
    WEIGHTS_FILE = "model.pt"

    def __init__(
        self,
        gpt2: GPT2LMHeadModel,
        n_prelude: int = 4,
        n_core: int = 4,
        n_coda: int = 4,
        init_state_std: float = DEFAULT_INIT_STATE_STD,
    ):
        super().__init__()
        t = gpt2.transformer
        if n_prelude + n_core + n_coda > len(t.h):
            raise ValueError(f"{n_prelude}+{n_core}+{n_coda} blocks requested, backbone has {len(t.h)}")
        self.config = gpt2.config
        self.n_prelude, self.n_core, self.n_coda = n_prelude, n_core, n_coda
        self.init_state_std = init_state_std

        self.wte, self.wpe, self.drop = t.wte, t.wpe, t.drop
        self.prelude = nn.ModuleList(t.h[:n_prelude])
        self.core = nn.ModuleList(t.h[n_prelude : n_prelude + n_core])
        self.coda = nn.ModuleList(t.h[n_prelude + n_core : n_prelude + n_core + n_coda])
        self.ln_f = t.ln_f
        self.lm_head = gpt2.lm_head  # weight-tied to wte, as in GPT2LMHeadModel

        h = self.config.n_embd
        self.state_norm = nn.LayerNorm(h, eps=self.config.layer_norm_epsilon)
        self.adapter = nn.Linear(2 * h, h)
        with torch.no_grad():  # A[e; s] = e at init -> r=1 (and any r) is plain GPT-2
            self.adapter.weight.zero_()
            self.adapter.bias.zero_()
            self.adapter.weight[:, :h].copy_(torch.eye(h))

    # ---- pieces -------------------------------------------------------------------
    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)  # left-pad safe
        x = self.drop(self.wte(input_ids) + self.wpe(position_ids))
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=x,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        return x, causal_mask, position_ids

    @staticmethod
    def _run(blocks: nn.ModuleList, x: torch.Tensor, causal_mask, position_ids) -> torch.Tensor:
        for block in blocks:
            x = block(x, None, causal_mask, position_ids=position_ids)
        return x

    def initial_state(self, like: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(like.shape, generator=generator, device=like.device, dtype=like.dtype)
        return noise * self.init_state_std

    def iterate(self, e: torch.Tensor, s: torch.Tensor, causal_mask, position_ids) -> torch.Tensor:
        x = self.adapter(torch.cat([e, self.state_norm(s)], dim=-1))
        return self._run(self.core, x, causal_mask, position_ids)

    def readout(self, s: torch.Tensor, causal_mask, position_ids) -> torch.Tensor:
        return self.lm_head(self.ln_f(self._run(self.coda, s, causal_mask, position_ids)))

    # ---- forward ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        num_recurrences: int = 1,
        backprop_last_k: int | None = None,
        initial_state: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        return_states: bool = False,
        checkpoint_iterations: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Returns `(logits, states)`; `states` is `[s_1, ..., s_r]` if `return_states`
        else `[]`. `backprop_last_k` (training only) runs all but the last k iterations
        under no_grad -- the paper's truncated backprop. `checkpoint_iterations`
        (training only) recomputes each with-grad iteration's internals in backward so
        memory is O(k) states instead of O(k) full iterations -- needed for full BPTT
        (k >= r) on a 24 GB card."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        x, causal_mask, position_ids = self.embed(input_ids, attention_mask)
        e = self._run(self.prelude, x, causal_mask, position_ids)
        s = initial_state if initial_state is not None else self.initial_state(e, generator)

        n_nograd = 0
        if self.training and backprop_last_k is not None:
            n_nograd = max(0, num_recurrences - backprop_last_k)
        states = []
        for i in range(num_recurrences):
            if i < n_nograd:
                with torch.no_grad():
                    s = self.iterate(e, s, causal_mask, position_ids)
            elif checkpoint_iterations and self.training and torch.is_grad_enabled():
                s = checkpoint(self.iterate, e, s, causal_mask, position_ids, use_reentrant=False)
            else:
                s = self.iterate(e, s, causal_mask, position_ids)
            if return_states:
                states.append(s)
        logits = self.readout(s, causal_mask, position_ids)
        return logits, states

    # ---- generation ---------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_recurrences: int,
        max_new_tokens: int,
        eos_token_id: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Greedy, batched (left-padded inputs), no KV cache -- each step re-runs the
        looped forward on the whole prefix. Returns `[B, <=max_new_tokens]` of new
        tokens, EOS-padded once a row has finished."""
        bsz, prompt_len = input_ids.shape
        h = self.config.n_embd
        # One noise draw per example for the whole sequence, sliced per step, so the
        # prefix's random states don't change under the model as it decodes.
        noise = torch.randn(
            (bsz, prompt_len + max_new_tokens, h), generator=generator, device=input_ids.device,
            dtype=self.wte.weight.dtype,
        ) * self.init_state_std
        done = torch.zeros(bsz, dtype=torch.bool, device=input_ids.device)
        out = []
        for _ in range(max_new_tokens):
            logits, _ = self.forward(
                input_ids, attention_mask, num_recurrences, initial_state=noise[:, : input_ids.shape[1]]
            )
            next_tok = logits[:, -1].argmax(-1)
            next_tok = torch.where(done, torch.full_like(next_tok, eos_token_id), next_tok)
            out.append(next_tok)
            done |= next_tok == eos_token_id
            if bool(done.all()):
                break
            input_ids = torch.cat([input_ids, next_tok[:, None]], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_tok)[:, None]], dim=1)
        return torch.stack(out, dim=1)

    # ---- diagnostics --------------------------------------------------------------
    @torch.no_grad()
    def iteration_diagnostics(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, num_recurrences: int,
        generator: torch.Generator | None = None,
    ) -> dict[str, list[float]]:
        """Per-iteration convergence at the last (non-pad) position, batch-averaged:
        relative state change `||s_i - s_{i-1}|| / ||s_i||` and the KL between the
        next-token distributions read out from `s_{i-1}` and `s_i` (the paper's
        adaptive-exit criterion). Index 0 compares against s_0 (noise)."""
        x, causal_mask, position_ids = self.embed(input_ids, attention_mask)
        e = self._run(self.prelude, x, causal_mask, position_ids)
        # Left-padded (or unpadded) batches, as in `generate`: the last position is the
        # last real token of every row.
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        last = torch.full_like(rows, input_ids.shape[1] - 1)
        s = self.initial_state(e, generator)
        prev_logp = F.log_softmax(self.readout(s, causal_mask, position_ids)[rows, last].float(), dim=-1)
        rel_delta, kl = [], []
        for _ in range(num_recurrences):
            s_new = self.iterate(e, s, causal_mask, position_ids)
            d = (s_new - s)[rows, last].float().norm(dim=-1) / s_new[rows, last].float().norm(dim=-1)
            logp = F.log_softmax(self.readout(s_new, causal_mask, position_ids)[rows, last].float(), dim=-1)
            kl.append(F.kl_div(logp, prev_logp, log_target=True, reduction="none").sum(-1).mean().item())
            rel_delta.append(d.mean().item())
            s, prev_logp = s_new, logp
        return {"state_rel_delta": rel_delta, "next_token_kl": kl}

    # ---- persistence --------------------------------------------------------------
    def looped_config(self) -> dict:
        return {
            "n_prelude": self.n_prelude, "n_core": self.n_core, "n_coda": self.n_coda,
            "init_state_std": self.init_state_std,
        }

    def save(self, ckpt_dir: str | Path) -> None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / self.CONFIG_FILE).write_text(json.dumps(self.looped_config(), indent=2))
        torch.save(self.state_dict(), ckpt_dir / self.WEIGHTS_FILE)

    @classmethod
    def load(cls, ckpt_dir: str | Path, backbone: str = "gpt2", **from_pretrained_kwargs) -> "LoopedGPT2":
        ckpt_dir = Path(ckpt_dir)
        cfg = json.loads((ckpt_dir / cls.CONFIG_FILE).read_text())
        model = cls(GPT2LMHeadModel.from_pretrained(backbone, **from_pretrained_kwargs), **cfg)
        model.load_state_dict(torch.load(ckpt_dir / cls.WEIGHTS_FILE, map_location="cpu"))
        return model


def unrolled_depth(n_prelude: int, n_core: int, n_coda: int, r: int) -> int:
    """Effective layer count at r iterations, e.g. (4,4,4) at r=32 -> 136."""
    return n_prelude + n_core * r + n_coda
