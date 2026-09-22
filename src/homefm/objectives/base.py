from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Objective(nn.Module):
    """A pretraining objective. Owns any objective-specific parameters (projection heads, predictors)."""

    name = "objective"

    def forward(self, model, batch) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    def set_progress(self, frac: float, model) -> None:
        """Called before each step with training progress in [0, 1]."""

    def after_step(self, model) -> None:
        """Called after each optimiser step (e.g. EMA updates)."""


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(len(z1), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


class CompositeObjective(Objective):
    name = "composite"

    def __init__(self, parts: list[tuple[Objective, float]]):
        super().__init__()
        self.parts = nn.ModuleList([p for p, _ in parts])
        self.weights = [w for _, w in parts]

    def forward(self, model, batch):
        total, logs = 0.0, {}
        for obj, w in zip(self.parts, self.weights):
            loss, l = obj(model, batch)
            total = total + w * loss
            logs.update({f"{obj.name}/{k}": v for k, v in l.items()})
        logs["loss"] = float(total.detach())
        return total, logs

    def set_progress(self, frac, model):
        for p in self.parts:
            p.set_progress(frac, model)

    def after_step(self, model):
        for p in self.parts:
            p.after_step(model)
