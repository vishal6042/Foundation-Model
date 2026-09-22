"""Event read-out: event-level transformer conditioned on moment context (DESIGN.md §7.2).

Causal mode conditions each event on the context of the *previous* moment and attends only to earlier
events, so event i never sees information from after event i.
"""

from __future__ import annotations

import torch
from torch import nn

from .stream import causal_mask


class EventReadout(nn.Module):
    def __init__(self, d: int, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.start_ctx = nn.Parameter(torch.zeros(1, 1, d))
        self.ctx_proj = nn.Linear(d, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, ev: torch.Tensor, ctx: torch.Tensor, moment_idx: torch.Tensor, valid: torch.Tensor,
                causal: bool) -> torch.Tensor:
        B, N, d = ev.shape
        if causal:
            ctx = torch.cat([self.start_ctx.expand(B, 1, d), ctx[:, :-1]], dim=1)
        c = torch.gather(ctx, 1, moment_idx.unsqueeze(-1).expand(B, N, d))
        h = ev + self.ctx_proj(c)
        mask = causal_mask(N, ev.device) if causal else None
        # padding is at the end and never attended to by valid events under the causal mask; for the
        # bidirectional case use key padding. Keep at least one key per row to avoid NaNs.
        pad = ~valid
        pad = pad & valid.any(dim=1, keepdim=True)
        out = self.encoder(h, mask=mask, src_key_padding_mask=None if causal else pad)
        return self.norm(out)
