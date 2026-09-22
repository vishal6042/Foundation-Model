"""Masking strategies (DESIGN.md §8.1–8.2).

Masks are sampled per sample in a Python loop over the batch; B is small and this keeps the logic
readable. Returns boolean tensors on the batch's device.
"""

from __future__ import annotations

import torch

from homefm.model.embedder import N_ATTRS

MASK_FAMILIES = ("time_block", "entity", "room", "modality", "info_weighted")


def random_event_mask(valid: torch.Tensor, p: float) -> torch.Tensor:
    return (torch.rand(valid.shape, device=valid.device) < p) & valid


def attribute_mask(valid: torch.Tensor, p: float) -> torch.Tensor:
    """DomusFM stage 1: for a fraction p of events, mask exactly one random attribute. → [B, N, 4]"""
    sel = random_event_mask(valid, p)
    which = torch.randint(0, N_ATTRS, valid.shape, device=valid.device)
    return torch.nn.functional.one_hot(which, N_ATTRS).bool() & sel.unsqueeze(-1)


def whole_event_mask(event_mask: torch.Tensor) -> torch.Tensor:
    """DomusFM stage 2 / BERT-style: every attribute of the selected events. → [B, N, 4]"""
    return event_mask.unsqueeze(-1).expand(*event_mask.shape, N_ATTRS).clone()


def structured_mask(batch, weights: dict[str, float], entity_freq: torch.Tensor | None = None,
                    block_moments: tuple[int, int] = (5, 15), info_p: float = 0.3,
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one structured mask family per sample.

    Returns
      drop        [B, N] events removed from the context encoder
      target      [B, M] moments whose latent the predictor must recover
      moment_tok  [B, M] moments replaced by a MASK token (time blocks only)
    """
    B, N = batch.valid.shape
    M = batch.n_moments
    dev = batch.valid.device
    drop = torch.zeros(B, N, dtype=torch.bool, device=dev)
    target = torch.zeros(B, M, dtype=torch.bool, device=dev)
    moment_tok = torch.zeros(B, M, dtype=torch.bool, device=dev)

    fams = [f for f in MASK_FAMILIES if weights.get(f, 0) > 0]
    probs = torch.tensor([weights[f] for f in fams], dtype=torch.float)
    choice = torch.multinomial(probs / probs.sum(), B, replacement=True)

    def time_block(b):
        L = int(torch.randint(block_moments[0], block_moments[1] + 1, (1,)))
        L = min(L, M)
        s = int(torch.randint(0, M - L + 1, (1,)))
        target[b, s:s + L] = moment_tok[b, s:s + L] = True
        drop[b] = batch.valid[b] & (batch.moment_idx[b] >= s) & (batch.moment_idx[b] < s + L)

    for b in range(B):
        fam = fams[int(choice[b])]
        v = batch.valid[b]
        if fam == "time_block" or v.sum() < 2:
            time_block(b)
            continue
        if fam == "entity":
            present = batch.entity_idx[b][v].unique()
            k = min(len(present), int(torch.randint(1, 4, (1,))))
            hidden = present[torch.randperm(len(present), device=dev)[:k]]
            d = v & torch.isin(batch.entity_idx[b], hidden)
        elif fam == "room":
            rooms = batch.room_idx[b][v].unique()
            r = rooms[torch.randint(0, len(rooms), (1,))]
            d = v & (batch.room_idx[b] == r)
        elif fam == "modality":
            mods = batch.modality[b][v].unique()
            m = mods[torch.randint(0, len(mods), (1,))]
            d = v & (batch.modality[b] == m)
        else:  # info_weighted: rare entities are hidden more often
            freq = entity_freq.to(dev)[batch.entity_idx[b]] if entity_freq is not None else torch.ones(N, device=dev)
            w = 1.0 / (freq + 1e-6)
            w = w * v
            p = (w / w[v].mean().clamp(min=1e-9) * info_p).clamp(max=0.9)
            d = v & (torch.rand(N, device=dev) < p)
        if d.sum() == 0 or d.sum() == v.sum():
            time_block(b)
            continue
        drop[b] = d
        target[b, batch.moment_idx[b][d].unique()] = True
    return drop, target, moment_tok
