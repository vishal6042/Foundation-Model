"""Pretraining objectives and the A–F ablation variants (DESIGN.md §8.3)."""

from __future__ import annotations

from .alignment import HomeAdversarial, LanguageAlignment
from .base import CompositeObjective, Objective
from .contrastive import DualContrastive
from .latent_mask import LatentMask
from .masked_recon import MaskedRecon
from .next_event import NextEvent


def build_objective(cfg: dict, d_model: int, text_dim: int, n_homes: int, entity_freq=None) -> Objective:
    name = cfg["name"]
    kw = {k: v for k, v in cfg.items() if k not in ("name", "weight", "parts", "run_name")}
    if name == "contrastive":
        return DualContrastive(d_model, **kw)
    if name == "masked_recon":
        return MaskedRecon(**kw)
    if name == "latent_mask":
        return LatentMask(d_model, entity_freq=entity_freq, **kw)
    if name == "next_event":
        return NextEvent(**kw)
    if name == "language":
        return LanguageAlignment(d_model, text_dim, **kw)
    if name == "home_adv":
        return HomeAdversarial(d_model, n_homes, **kw)
    if name == "composite":
        return CompositeObjective([
            (build_objective(p, d_model, text_dim, n_homes, entity_freq), float(p.get("weight", 1.0)))
            for p in cfg["parts"]
        ])
    raise ValueError(f"unknown objective: {name}")


def find_part(obj: Objective, cls):
    """Return the first objective of type `cls` inside obj (or obj itself), else None."""
    if isinstance(obj, cls):
        return obj
    if isinstance(obj, CompositeObjective):
        for p in obj.parts:
            found = find_part(p, cls)
            if found is not None:
                return found
    return None


__all__ = [
    "CompositeObjective", "DualContrastive", "HomeAdversarial", "LanguageAlignment", "LatentMask",
    "MaskedRecon", "NextEvent", "Objective", "build_objective", "find_part",
]
