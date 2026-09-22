"""Task heads (DESIGN.md §8.2 Stage 1, §9)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .embedder import mlp


class LogNormalMixture(nn.Module):
    """Mixture density over y = log(1 + Δt seconds)."""

    def __init__(self, d: int, n_components: int = 8):
        super().__init__()
        self.proj = nn.Linear(d, 3 * n_components)

    def nll(self, h: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        logits, mu, log_sigma = self.proj(h).chunk(3, dim=-1)
        log_sigma = log_sigma.clamp(-5, 3)
        y = torch.log1p(dt.clamp(min=0)).unsqueeze(-1)
        log_n = -0.5 * ((y - mu) / log_sigma.exp()) ** 2 - log_sigma - 0.5 * math.log(2 * math.pi)
        return -torch.logsumexp(F.log_softmax(logits, -1) + log_n, dim=-1)


class NextEventHead(nn.Module):
    """Marked temporal point process head: what (entity), state, value and when (Δt) of the next event.

    The entity is scored against the semantic entity embeddings of the sample's home, so it transfers
    across homes (no fixed sensor-id output layer).
    """

    def __init__(self, d: int, n_states: int = 3, n_mixture: int = 8):
        super().__init__()
        self.what = mlp(d, d)
        self.cond = nn.Linear(d, d)
        self.state = mlp(d, d, n_states)
        self.value = mlp(d, d, 1)
        self.when = LogNormalMixture(d, n_mixture)

    def entity_logits(self, h: torch.Tensor, entity_keys: torch.Tensor, home_mask: torch.Tensor) -> torch.Tensor:
        """h [B, N, d], entity_keys [V, d], home_mask [B, V] → logits [B, N, V]."""
        logits = torch.einsum("bnd,vd->bnv", self.what(h), entity_keys) / math.sqrt(h.shape[-1])
        return logits.masked_fill(~home_mask[:, None, :], float("-inf"))

    def conditioned(self, h: torch.Tensor, target_entity_key: torch.Tensor) -> torch.Tensor:
        return h + self.cond(target_entity_key)


class ProjectionHead(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_in), nn.GELU(), nn.Linear(d_in, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lam)
