"""Variant B: BERT-style random event masking with reconstruction of entity, state and value."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Objective
from .masking import random_event_mask, whole_event_mask


class MaskedRecon(Objective):
    name = "masked_recon"

    def __init__(self, mask_p: float = 0.15, value_weight: float = 1.0):
        super().__init__()
        self.mask_p, self.value_weight = mask_p, value_weight

    def forward(self, model, batch):
        mask = random_event_mask(batch.valid, self.mask_p)
        if mask.sum() == 0:
            mask = batch.valid & (torch.cumsum(batch.valid.long(), 1) == 1)  # at least one event
        enc = model.encode(batch, causal=False, attr_mask=whole_event_mask(mask), with_readout=True)
        head, keys = model.next_event, model.entity_keys()
        h = enc.readout
        logits = head.entity_logits(h, keys, batch.home_entity_mask)[mask]           # [K, V]
        tgt = batch.entity_idx[mask]
        ce_ent = F.cross_entropy(logits, tgt)
        hc = head.conditioned(h[mask], keys[tgt])
        ce_state = F.cross_entropy(head.state(hc), batch.state[mask])
        hv = batch.has_value[mask]
        val = F.mse_loss(head.value(hc).squeeze(-1)[hv], batch.value[mask][hv]) if hv.any() else h.sum() * 0
        loss = ce_ent + ce_state + self.value_weight * val
        acc = (logits.argmax(-1) == tgt).float().mean()
        return loss, {"loss": loss.item(), "entity_ce": ce_ent.item(), "entity_acc": acc.item()}
