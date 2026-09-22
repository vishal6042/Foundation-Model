"""Variant C: structured latent prediction (JEPA-style, DESIGN.md §8.2 Stage 2).

A context encoder sees the window with a structured mask; a predictor regresses the moment latents
that an EMA target encoder produces from the full window. Loss only on masked moments.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from .base import Objective
from .masking import structured_mask

DEFAULT_WEIGHTS = {"time_block": 0.4, "entity": 0.2, "room": 0.2, "modality": 0.1, "info_weighted": 0.1}


class LatentMask(Objective):
    name = "latent_mask"

    def __init__(self, d: int, ema: float = 0.996, mask_weights: dict | None = None,
                 block_moments: tuple[int, int] = (5, 15), entity_freq: torch.Tensor | None = None,
                 predictor_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.ema = ema
        self.mask_weights = mask_weights or DEFAULT_WEIGHTS
        self.block_moments = tuple(block_moments)
        self.register_buffer("entity_freq", entity_freq if entity_freq is not None else torch.ones(1), persistent=False)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, 0.0, batch_first=True, norm_first=True)
        self.predictor = nn.Sequential(nn.TransformerEncoder(layer, predictor_layers, enable_nested_tensor=False),
                                       nn.LayerNorm(d), nn.Linear(d, d))
        self._target: list[nn.Module] = []  # list so the EMA copy is not registered as a submodule / optimised

    def target(self, model) -> nn.Module:
        if not self._target:
            t = copy.deepcopy(model).eval()
            t.requires_grad_(False)
            self._target.append(t)
        return self._target[0]

    def forward(self, model, batch):
        target_model = self.target(model)
        freq = self.entity_freq if self.entity_freq.numel() > 1 else None
        drop, tgt_m, tok_m = structured_mask(batch, self.mask_weights, freq, self.block_moments)
        ctx = model.encode(batch, causal=False, drop_mask=drop, moment_mask=tok_m)
        pred = self.predictor(ctx.stream)
        with torch.no_grad():
            tgt = target_model.encode(batch, causal=False).stream
            tgt = F.layer_norm(tgt, tgt.shape[-1:])
        loss = F.smooth_l1_loss(pred[tgt_m], tgt[tgt_m])
        return loss, {"loss": loss.item(), "masked_frac": float(tgt_m.float().mean())}

    @torch.no_grad()
    def after_step(self, model):
        if not self._target:
            return
        for pt, p in zip(self._target[0].parameters(), model.parameters()):
            pt.mul_(self.ema).add_(p.detach(), alpha=1 - self.ema)

    def _apply(self, fn, *args, **kwargs):  # keep the EMA copy on the same device as the objective
        super()._apply(fn, *args, **kwargs)
        if self._target:
            self._target[0]._apply(fn, *args, **kwargs)
        return self
