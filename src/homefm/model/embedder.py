"""Event embedder with attribute fusion (DESIGN.md §7.2)."""

from __future__ import annotations

import math

import torch
from torch import nn

N_ATTRS = 4  # entity, value, time, meta
ATTR_ENTITY, ATTR_VALUE, ATTR_TIME, ATTR_META = range(N_ATTRS)


def mlp(d_in: int, d: int, d_out: int | None = None) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d_out or d))


class CyclicTime(nn.Module):
    """sin/cos harmonics of hour-of-day and day-of-week, then a learned projection."""

    def __init__(self, d: int, n_harmonics: int = 4):
        super().__init__()
        self.register_buffer("k", torch.arange(1, n_harmonics + 1, dtype=torch.float32), persistent=False)
        self.proj = nn.Linear(4 * n_harmonics, d)

    def forward(self, hour: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        h = 2 * math.pi * hour.unsqueeze(-1) / 24.0 * self.k
        w = 2 * math.pi * dow.unsqueeze(-1) / 7.0 * self.k
        return self.proj(torch.cat([h.sin(), h.cos(), w.sin(), w.cos()], dim=-1))


class EventEmbedder(nn.Module):
    def __init__(self, text_dim: int, d: int, n_heads: int = 4, n_modalities: int = 5, n_states: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.entity_proj = nn.Linear(text_dim, d)
        self.state_emb = nn.Embedding(n_states, d)
        self.value_mlp = mlp(1, d)
        self.time = CyclicTime(d)
        self.dt_mlp = mlp(2, d)
        self.meta_emb = nn.Embedding(n_modalities, d)
        self.conf_proj = nn.Linear(1, d)
        self.attr_type = nn.Parameter(torch.randn(N_ATTRS, d) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(N_ATTRS, d) * 0.02)
        self.fuse = nn.TransformerEncoderLayer(d, n_heads, 2 * d, dropout, batch_first=True, norm_first=True)
        self.norm = nn.LayerNorm(d)

    def entity_keys(self, entity_table: torch.Tensor) -> torch.Tensor:
        """Semantic embeddings of every entity in the vocabulary, in model space ([V, d])."""
        return self.entity_proj(entity_table)

    def forward(self, batch, entity_table: torch.Tensor, attr_mask: torch.Tensor | None = None) -> torch.Tensor:
        """attr_mask: [B, N, 4] bool; True replaces that attribute with its MASK token."""
        ent = self.entity_proj(entity_table[batch.entity_idx])
        val = self.state_emb(batch.state) + self.value_mlp(batch.value.unsqueeze(-1)) * batch.has_value.unsqueeze(-1)
        dts = torch.stack([torch.log1p(batch.dt_prev.clamp(min=0)), torch.log1p(batch.dt_entity.clamp(min=0))], -1)
        tim = self.time(batch.hour, batch.dow) + self.dt_mlp(dts)
        meta = self.meta_emb(batch.modality) + self.conf_proj(batch.confidence.unsqueeze(-1))
        attrs = torch.stack([ent, val, tim, meta], dim=2)  # [B, N, 4, d]
        if attr_mask is not None:
            attrs = torch.where(attr_mask.unsqueeze(-1), self.mask_token.expand_as(attrs), attrs)
        B, N, A, d = attrs.shape
        x = self.fuse((attrs + self.attr_type).reshape(B * N, A, d))
        return self.norm(x.mean(dim=1).reshape(B, N, d))
