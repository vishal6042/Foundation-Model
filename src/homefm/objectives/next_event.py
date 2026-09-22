"""Variant D: causal next-event prediction as a marked temporal point process (DESIGN.md §8.2 Stage 1).

For each event i, predict event i+1: what (entity), state/value, and when (Δt). The per-event negative
log-likelihood doubles as the anomaly "surprise" score (DESIGN.md §9.2).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Objective


class NextEvent(Objective):
    name = "next_event"

    def __init__(self, value_weight: float = 1.0):
        super().__init__()
        self.value_weight = value_weight

    def terms(self, model, batch) -> dict[str, torch.Tensor]:
        enc = model.encode(batch, causal=True, with_readout=True)
        head, keys = model.next_event, model.entity_keys()
        h = enc.readout[:, :-1]
        pair = batch.valid[:, :-1] & batch.valid[:, 1:]
        tgt_ent = batch.entity_idx[:, 1:]
        # padded targets may not exist in the home; point them at a real entity, they are masked out below
        fallback = batch.home_entity_mask.float().argmax(-1, keepdim=True).expand_as(tgt_ent)
        tgt_ent = torch.where(pair, tgt_ent, fallback)

        logits = head.entity_logits(h, keys, batch.home_entity_mask)
        ce_ent = F.cross_entropy(logits.transpose(1, 2), tgt_ent, reduction="none")
        hc = head.conditioned(h, keys[tgt_ent])
        ce_state = F.cross_entropy(head.state(hc).transpose(1, 2), batch.state[:, 1:], reduction="none")
        has_v = batch.has_value[:, 1:]
        val = (head.value(hc).squeeze(-1) - batch.value[:, 1:]) ** 2 * has_v
        dt = (batch.t[:, 1:] - batch.t[:, :-1]).clamp(min=0)
        when = head.when.nll(h, dt)
        zero = torch.zeros_like(ce_ent)
        nll = torch.where(pair, ce_ent + ce_state + self.value_weight * val + when, zero)
        return dict(pair=pair, nll=nll, ce_ent=torch.where(pair, ce_ent, zero), when=torch.where(pair, when, zero),
                    acc=(logits.argmax(-1) == tgt_ent) & pair)

    def forward(self, model, batch):
        t = self.terms(model, batch)
        n = t["pair"].sum().clamp(min=1)
        loss = t["nll"].sum() / n
        return loss, {"loss": loss.item(), "entity_ce": (t["ce_ent"].sum() / n).item(),
                      "when_nll": (t["when"].sum() / n).item(), "entity_acc": float(t["acc"].sum() / n)}

    @torch.no_grad()
    def surprise(self, model, batch) -> torch.Tensor:
        """[B, N] negative log-likelihood of each event given its past (0 for the first event)."""
        t = self.terms(model, batch)
        out = torch.zeros_like(batch.t)
        out[:, 1:] = t["nll"]
        return out
