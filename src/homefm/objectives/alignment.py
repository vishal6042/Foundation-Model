"""Variant F additions: language alignment (SigLIP) and home-adversarial invariance (DESIGN.md §8.2 Stage 3)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from homefm.model.heads import ProjectionHead, grad_reverse

from .base import Objective


class LanguageAlignment(Objective):
    """Sigmoid (SigLIP) loss between window embeddings and caption embeddings.

    Windows whose captions describe the same concept are all positives, so repetitive routines are not
    pushed apart as false negatives (DESIGN.md §8.1 #3).
    """

    name = "language"

    def __init__(self, d: int, text_dim: int, proj_dim: int = 128):
        super().__init__()
        self.window_proj = ProjectionHead(d, proj_dim)
        self.text_proj = ProjectionHead(text_dim, proj_dim)
        self.log_t = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = nn.Parameter(torch.tensor(-10.0))

    def embed_windows(self, model, batch) -> torch.Tensor:
        return self.window_proj(model.pooled(model.encode(batch, causal=False)))

    def embed_text(self, text_emb: torch.Tensor) -> torch.Tensor:
        return self.text_proj(text_emb)

    def forward(self, model, batch):
        zx = self.embed_windows(model, batch)
        zt = self.embed_text(batch.caption_emb)
        logits = zx @ zt.t() * self.log_t.exp() + self.bias
        g = batch.caption_group
        labels = (g[:, None] == g[None, :]).float() * 2 - 1
        loss = -F.logsigmoid(labels * logits).sum() / len(zx)
        return loss, {"loss": loss.item()}


class HomeAdversarial(Objective):
    """Predict the home from the pooled embedding through a gradient-reversal layer."""

    name = "home_adv"

    def __init__(self, d: int, n_homes: int, lam: float = 0.1):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_homes))
        self.lam = lam

    def forward(self, model, batch):
        z = model.pooled(model.encode(batch, causal=False))
        logits = self.clf(grad_reverse(z, self.lam))
        loss = F.cross_entropy(logits, batch.home_idx)
        acc = (logits.argmax(-1) == batch.home_idx).float().mean()
        return loss, {"loss": loss.item(), "home_acc": acc.item()}
