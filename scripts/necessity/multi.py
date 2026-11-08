"""Batched conditions for batch-1 adapters: run one question B times in one batch, each row with its own edits (no padding needed,
since every row is the same question). ~20x faster than core.run per condition. Equivalence with core.run is checked on the fly by
grid_split.py (SMOKE / VERIFY): batched matmuls can differ from batch-1 in the last bits, so answers are compared, not assumed.

Edit dict per row: feed {s: [1, d] or [d]}, kv {s: kv list of [1, H, D]}, hide (sites masked from the answer).
"""
from __future__ import annotations

import torch

import adapters
from core import _layers


def _greedy_multi(body, head, embed, first, past, start_len, hides, vocab, eos, max_new, dev):
    B = first.size(0); toks = [[] for _ in range(B)]; done = [False] * B; nxt = first
    any_hide = any(hides)
    for t in range(max_new):
        kw = {}
        if any_hide:
            m = torch.ones((B, start_len + t + 1), dtype=torch.long, device=dev)
            for b, hp in enumerate(hides):
                for p in hp: m[b, p] = 0
            kw['attention_mask'] = m
        o = body(inputs_embeds=nxt, use_cache=True, past_key_values=past, **kw)
        past = o.past_key_values
        nid = head(o.last_hidden_state[:, -1])[:, :vocab].argmax(-1).tolist()
        for b, x in enumerate(nid):
            if done[b]: continue
            if x == eos: done[b] = True
            else: toks[b].append(x)
        if all(done): break
        nxt = embed(torch.tensor(nid, device=dev)).unsqueeze(1)
    return toks


class _Multi:
    def begin_multi(self, row, B):
        ids = self._ids(row).to(self.dev).expand(B, -1).contiguous()
        o = self.body(input_ids=ids, use_cache=True)
        return dict(Q=ids.size(1), B=B), o.past_key_values, self._z(o.last_hidden_state[:, -1, :])

    def latent_multi(self, ctx, past, z, s):
        o = self.body(inputs_embeds=z[:, None], use_cache=True, past_key_values=past)
        return o.past_key_values, self._z(o.last_hidden_state[:, -1, :])

    def finish_multi(self, ctx, past, hides):
        first = self.embed(torch.full((ctx['B'],), self._first_id, device=self.dev)).unsqueeze(1)
        T = _greedy_multi(self.body, self.head, self.embed, first, past, ctx['Q'] + self.n_latents,
                          [[ctx['Q'] + h for h in hp] for hp in hides], self.vocab, self.tok.eos_token_id, self.max_new, self.dev)
        return [self._extract(self.tok.decode([t for t in tt if t < len(self.tok)], skip_special_tokens=True)) for tt in T]


class GridCODIMulti(adapters.GridCODI, _Multi):
    def _ids(self, row): return self.qids(row['question'])
    def _z(self, h): return self.m.prj(h[:, None])[:, 0]
    @property
    def _first_id(self): return self.m.eot_id
    def _extract(self, raw): return adapters.norm_text(raw.split(self.DELIM)[-1]) if self.DELIM in raw else ''


class GridCoconutMulti(adapters.GridCoconut, _Multi):
    def _ids(self, row): return torch.tensor([self.qids(row['question']) + [self.sl]])
    def _z(self, h): return h
    @property
    def _first_id(self): return self.el
    def _extract(self, raw): return self.extract(raw)


@torch.no_grad()
def run_multi(ad, row, edits):
    """One batch: row repeated len(edits) times, row b gets edits[b]. Returns answers list."""
    B = len(edits); ctx, past, z = ad.begin_multi(row, B)
    for s in range(ad.n_latents):
        rows_f = [(b, e['feed'][s]) for b, e in enumerate(edits) if s in e.get('feed', {})]
        if rows_f:
            z = z.clone()
            for b, v in rows_f: z[b] = v.reshape(-1).to(device=z.device, dtype=z.dtype)
        past, z_next = ad.latent_multi(ctx, past, z, s)
        pos = ad.pos(ctx, s)
        for b, e in enumerate(edits):
            if s in e.get('kv', {}):
                for (k, v), (sk, sv) in zip(_layers(past), e['kv'][s]):
                    k[b, :, pos] = sk.reshape(k[b, :, pos].shape).to(k.device); v[b, :, pos] = sv.reshape(v[b, :, pos].shape).to(v.device)
        z = z_next
    return ad.finish_multi(ctx, past, [tuple(e.get('hide', ())) for e in edits])
