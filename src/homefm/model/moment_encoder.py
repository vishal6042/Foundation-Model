"""Perceiver-style moment encoder (DESIGN.md §7.2, L1).

K learned latents attend to the events inside each fixed-duration moment. Because every event belongs
to exactly one moment, attention is computed with a segment softmax (scatter), so memory is
O(B·N·K·d) instead of O(B·M·N·d).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .embedder import CyclicTime, mlp


class MomentEncoder(nn.Module):
    def __init__(self, d: int, n_latents: int = 4, n_heads: int = 4):
        super().__init__()
        assert d % n_heads == 0
        self.d, self.K, self.H, self.dh = d, n_latents, n_heads, d // n_heads
        self.latents = nn.Parameter(torch.randn(n_latents, d) * 0.02)
        self.quiet = nn.Parameter(torch.randn(1, d) * 0.02)  # always-present key so empty moments are defined
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.norm_kv = nn.LayerNorm(d)
        self.norm_ff = nn.LayerNorm(d)
        self.ff = mlp(d, 2 * d, d)
        self.count_proj = nn.Linear(1, d)
        self.time = CyclicTime(d)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, ev: torch.Tensor, moment_idx: torch.Tensor, valid: torch.Tensor, n_moments: int,
                moment_hour: torch.Tensor, moment_dow: torch.Tensor) -> torch.Tensor:
        B, N, d = ev.shape
        M, K, H, dh = n_moments, self.K, self.H, self.dh
        ev = self.norm_kv(ev)
        q = self.q(self.latents).view(K, H, dh)
        k = self.k(ev).view(B, N, H, dh)
        v = self.v(ev).view(B, N, H, dh)
        quiet = self.norm_kv(self.quiet)
        kq, vq = self.k(quiet).view(H, dh), self.v(quiet).view(H, dh)

        scale = 1.0 / math.sqrt(dh)
        s = torch.einsum("bnhd,khd->bnhk", k, q) * scale                     # [B, N, H, K]
        s = s.masked_fill(~valid[:, :, None, None], float("-inf"))
        sq = torch.einsum("hd,khd->hk", kq, q) * scale                       # [H, K]

        # invalid events go to a dump bucket M
        seg = torch.where(valid, moment_idx, torch.full_like(moment_idx, M))
        seg4 = seg[:, :, None, None].expand(B, N, H, K)
        mx = torch.full((B, M + 1, H, K), float("-inf"), device=ev.device, dtype=s.dtype)
        mx = mx.scatter_reduce(1, seg4, s, reduce="amax", include_self=True)
        mx = torch.maximum(mx, sq.expand(B, M + 1, H, K)).detach()           # quiet key is in every moment
        ex = torch.exp(s - mx.gather(1, seg4))                               # 0 for invalid events
        exq = torch.exp(sq - mx)                                             # [B, M+1, H, K]

        denom = torch.zeros(B, M + 1, H, K, device=ev.device, dtype=s.dtype).scatter_add(1, seg4, ex) + exq
        num = torch.zeros(B, M + 1, H, K, dh, device=ev.device, dtype=s.dtype)
        num = num.scatter_add(1, seg4[..., None].expand(B, N, H, K, dh), ex[..., None] * v[:, :, :, None, :])
        num = num + exq[..., None] * vq[None, None, :, None, :]
        att = (num / denom[..., None])[:, :M]                                # [B, M, H, K, dh]
        lat = self.o(att.permute(0, 1, 3, 2, 4).reshape(B, M, K, d)) + self.latents
        lat = lat + self.ff(self.norm_ff(lat))

        counts = torch.zeros(B, M + 1, device=ev.device).scatter_add(1, seg, valid.float())[:, :M]
        out = lat.mean(dim=2) + self.count_proj(torch.log1p(counts).unsqueeze(-1)) + self.time(moment_hour, moment_dow)
        return self.out_norm(out)
