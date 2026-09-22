"""HomeFM backbone assembly (DESIGN.md §7.1)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .embedder import EventEmbedder
from .heads import NextEventHead
from .moment_encoder import MomentEncoder
from .readout import EventReadout
from .stream import StreamModel


@dataclass
class ModelConfig:
    text_dim: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_latents: int = 4
    n_stream_layers: int = 4
    n_readout_layers: int = 2
    max_moments: int = 1024
    dropout: float = 0.1
    n_mixture: int = 8


@dataclass
class Encoded:
    events: torch.Tensor          # [B, N, d] event embeddings
    moments: torch.Tensor         # [B, M, d] moment tokens (before the stream model)
    stream: torch.Tensor          # [B, M, d] contextual moment embeddings
    readout: torch.Tensor | None  # [B, N, d] contextual event embeddings
    valid: torch.Tensor           # [B, N] events that were visible to the encoder


class HomeFM(nn.Module):
    def __init__(self, cfg: ModelConfig, entity_table: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.register_buffer("entity_table", entity_table.float())
        self.embedder = EventEmbedder(cfg.text_dim, d, cfg.n_heads, dropout=cfg.dropout)
        self.moment_encoder = MomentEncoder(d, cfg.n_latents, cfg.n_heads)
        self.stream = StreamModel(d, cfg.n_stream_layers, cfg.n_heads, cfg.max_moments, cfg.dropout)
        self.readout = EventReadout(d, cfg.n_readout_layers, cfg.n_heads, cfg.dropout)
        self.next_event = NextEventHead(d, n_mixture=cfg.n_mixture)
        self.moment_mask_token = nn.Parameter(torch.randn(d) * 0.02)

    def entity_keys(self) -> torch.Tensor:
        return self.embedder.entity_keys(self.entity_table)

    def encode(self, batch, causal: bool, attr_mask: torch.Tensor | None = None,
               drop_mask: torch.Tensor | None = None, moment_mask: torch.Tensor | None = None,
               with_readout: bool = False) -> Encoded:
        """
        attr_mask   [B, N, 4] bool: replace attributes with MASK tokens (event keeps its slot).
        drop_mask   [B, N] bool: remove events entirely (they are invisible to the encoder).
        moment_mask [B, M] bool: replace whole moment tokens with a MASK token before the stream model.
        """
        valid = batch.valid if drop_mask is None else batch.valid & ~drop_mask
        ev = self.embedder(batch, self.entity_table, attr_mask)
        mom = self.moment_encoder(ev, batch.moment_idx, valid, batch.n_moments, batch.moment_hour, batch.moment_dow)
        if moment_mask is not None:
            mom = torch.where(moment_mask.unsqueeze(-1), self.moment_mask_token.expand_as(mom), mom)
        st = self.stream(mom, causal=causal)
        ro = self.readout(ev, st, batch.moment_idx, valid, causal=causal) if with_readout else None
        return Encoded(ev, mom, st, ro, valid)

    @staticmethod
    def pooled(enc: Encoded) -> torch.Tensor:
        return enc.stream.mean(dim=1)

    @torch.no_grad()
    def moment_embeddings(self, batch) -> torch.Tensor:
        """Frozen bidirectional moment embeddings for probing, tagging and retrieval."""
        return self.encode(batch, causal=False).stream

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
