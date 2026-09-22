"""Variant A: DomusFM-style dual contrastive pretraining (baseline, DESIGN.md §8.3).

Stage 1 (progress < stage_switch): mask one attribute of some events; InfoNCE between original and
augmented windows. Stage 2: freeze the event embedder, mask whole events; InfoNCE again.
"""

from __future__ import annotations

from homefm.model.heads import ProjectionHead

from .base import Objective, info_nce
from .masking import attribute_mask, random_event_mask, whole_event_mask


class DualContrastive(Objective):
    name = "contrastive"

    def __init__(self, d: int, attr_mask_p: float = 0.3, event_mask_p: float = 0.15, temperature: float = 0.1,
                 stage_switch: float = 0.5, proj_dim: int = 128):
        super().__init__()
        self.proj = ProjectionHead(d, proj_dim)
        self.attr_mask_p, self.event_mask_p = attr_mask_p, event_mask_p
        self.temperature, self.stage_switch = temperature, stage_switch
        self.stage = "attribute"

    def set_progress(self, frac, model):
        stage = "attribute" if frac < self.stage_switch else "event"
        if stage != self.stage:
            self.stage = stage
            model.embedder.requires_grad_(stage == "attribute")

    def forward(self, model, batch):
        if self.stage == "attribute":
            aug = attribute_mask(batch.valid, self.attr_mask_p)
        else:
            aug = whole_event_mask(random_event_mask(batch.valid, self.event_mask_p))
        z1 = self.proj(model.pooled(model.encode(batch, causal=False)))
        z2 = self.proj(model.pooled(model.encode(batch, causal=False, attr_mask=aug)))
        loss = info_nce(z1, z2, self.temperature)
        return loss, {"loss": loss.item(), "stage2": float(self.stage == "event")}
