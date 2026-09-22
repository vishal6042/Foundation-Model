"""Stream model over moments (DESIGN.md §7.2, L2). Causal for streaming/forecasting, bidirectional for summaries."""

from __future__ import annotations

import torch
from torch import nn


def causal_mask(n: int, device) -> torch.Tensor:
    return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)


class StreamModel(nn.Module):
    def __init__(self, d: int, n_layers: int = 4, n_heads: int = 4, max_moments: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.pos = nn.Embedding(max_moments, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, causal: bool) -> torch.Tensor:
        B, M, _ = x.shape
        x = x + self.pos(torch.arange(M, device=x.device))
        mask = causal_mask(M, x.device) if causal else None
        return self.norm(self.encoder(x, mask=mask))
