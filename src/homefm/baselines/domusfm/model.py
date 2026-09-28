"""DomusFM architecture (paper §4, Fig. 1–2, §6.2.1).

Event-level feature extraction:
  house item, sensor type, room  → frozen Sentence-BERT embeddings (all-MiniLM-L6-v2, 384-d)
  status                         → learned embedding {OFF, ON, MASK}
  timestamp                      → day-of-week and hour (cyclic sin/cos harmonics + projection)
                                   + seconds-within-hour (learned embedding, 3600 values)
  5 attribute vectors            → attribute self-attention → one event embedding
Contextualised event-level feature extraction:
  12-layer, 12-head transformer encoder (post-norm, as in Fig. 2b), d = 384, no positional encoding
  (the paper relies on the temporal encoding for order, §7.3).

Details the paper does not specify are marked ASSUMPTION and listed in docs/DOMUSFM_REPRODUCTION.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

N_ATTR = 5
A_ITEM, A_TYPE, A_ROOM, A_STATUS, A_TIME = range(N_ATTR)
STATUS_MASK = 2


@dataclass
class DomusConfig:
    d: int = 384
    n_layers: int = 12
    n_heads: int = 12
    ff: int = 2048              # ASSUMPTION: PyTorch default feed-forward width
    attr_heads: int = 4         # ASSUMPTION
    n_harmonics: int = 4        # ASSUMPTION: harmonics for the cyclic day/hour encodings
    dropout: float = 0.1        # ASSUMPTION


class Cyclic(nn.Module):
    def __init__(self, period: float, n_harmonics: int, d: int):
        super().__init__()
        self.period = period
        self.register_buffer("k", torch.arange(1, n_harmonics + 1, dtype=torch.float32), persistent=False)
        self.proj = nn.Linear(2 * n_harmonics, d)  # learns which frequencies matter (§4.1.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = 2 * math.pi * x.float().unsqueeze(-1) / self.period * self.k
        return self.proj(torch.cat([a.sin(), a.cos()], -1))


class EventFeatureExtractor(nn.Module):
    def __init__(self, cfg: DomusConfig, text_table: torch.Tensor):
        super().__init__()
        d = cfg.d
        self.register_buffer("text_table", text_table.float())
        # MiniLM is 384-d = d; otherwise project. The LLM embeddings themselves stay frozen.
        self.text_proj = nn.Identity() if text_table.shape[1] == d else nn.Linear(text_table.shape[1], d)
        self.status_emb = nn.Embedding(3, d)
        self.dow = Cyclic(7, cfg.n_harmonics, d)
        self.hour = Cyclic(24, cfg.n_harmonics, d)
        self.sec = nn.Embedding(3600, d)
        self.mask_vec = nn.Parameter(torch.randn(N_ATTR, d) * 0.02)   # MASK token per attribute
        self.attr_type = nn.Parameter(torch.randn(N_ATTR, d) * 0.02)  # ASSUMPTION: attribute-type embeddings
        self.attn = nn.TransformerEncoderLayer(d, cfg.attr_heads, 2 * d, cfg.dropout, batch_first=True)

    def forward(self, x: dict, attr_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: dict of [B, L] tensors. attr_mask: [B, L, 5] bool → MASK those attributes. Returns [B, L, d]."""
        tt = self.text_proj(self.text_table)
        status = x["status"]
        if attr_mask is not None:
            status = torch.where(attr_mask[..., A_STATUS], torch.full_like(status, STATUS_MASK), status)
        attrs = torch.stack([tt[x["item"]], tt[x["stype"]], tt[x["room"]], self.status_emb(status),
                             self.dow(x["dow"]) + self.hour(x["hour"]) + self.sec(x["sec"])], dim=2)
        if attr_mask is not None:
            m = attr_mask.clone()
            m[..., A_STATUS] = False  # status is masked through its own MASK index
            attrs = torch.where(m.unsqueeze(-1), self.mask_vec.expand_as(attrs), attrs)
        B, L, A, d = attrs.shape
        h = self.attn((attrs + self.attr_type).reshape(B * L, A, d))
        return h.mean(1).reshape(B, L, d)  # ASSUMPTION: mean-pool the attended attributes


class DomusFM(nn.Module):
    """The Window Encoder φ = contextualised(event-level(·)) — the pretrained backbone."""

    def __init__(self, cfg: DomusConfig, text_table: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.event = EventFeatureExtractor(cfg, text_table)
        layer = nn.TransformerEncoderLayer(cfg.d, cfg.n_heads, cfg.ff, cfg.dropout, batch_first=True)
        self.context = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)

    def forward(self, x: dict, attr_mask: torch.Tensor | None = None, contextualise: bool = True) -> torch.Tensor:
        e = self.event(x, attr_mask)
        return self.context(e) if contextualise else e

    @staticmethod
    def window_embedding(h: torch.Tensor) -> torch.Tensor:
        return h.mean(1)  # ASSUMPTION: sequence-level embedding = mean over events

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PretrainHeads(nn.Module):
    """Pretraining-only heads (not part of the backbone; discarded before fine-tuning). Not in the paper.

    projector: SimCLR-style MLP so InfoNCE shapes the projection, not the backbone features.
    mlm: predicts masked attributes from context. Text attributes are scored against the frozen text table
    (works across homes with different sensor vocabularies); status is a 2-way classifier. Unlike the
    contrastive game, this cannot be solved by recognising the window, because the answer is hidden.
    """

    def __init__(self, d: int, text_table: torch.Tensor, proj_dim: int = 128):
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, proj_dim))
        self.register_buffer("text_table", F.normalize(text_table.float(), dim=-1))
        self.text_q = nn.ModuleList(nn.Linear(d, text_table.shape[1]) for _ in range(3))  # item, type, room
        self.status = nn.Linear(d, 2)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def mlm_loss(self, h: torch.Tensor, x: dict, mask: torch.Tensor) -> torch.Tensor:
        """h: [B, L, d] contextualised masked view; mask: [B, L, 5]. Mean CE over masked attributes."""
        losses = []
        for a, (key, q) in enumerate(zip(("item", "stype", "room"), self.text_q)):
            sel = mask[..., a]
            if sel.any():
                logits = F.normalize(q(h[sel]), dim=-1) @ self.text_table.t() * self.log_scale.exp()
                losses.append(F.cross_entropy(logits, x[key][sel]))
        sel = mask[..., A_STATUS]
        if sel.any():
            losses.append(F.cross_entropy(self.status(h[sel]), x["status"][sel]))
        return torch.stack(losses).mean() if losses else h.sum() * 0


class ADLHead(nn.Module):
    """Linear layer on the contextualised embedding of the window's last event (§6.4.3; last-event is an ASSUMPTION)."""

    def __init__(self, d: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d, n_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h[:, -1])

    @staticmethod
    def loss(logits, batch):
        return F.cross_entropy(logits, batch["label"])


class NextKHead(nn.Module):
    """Dual head (§6.5.3): distribution over future event types + expected count per type."""

    def __init__(self, d: int, n_types: int):
        super().__init__()
        self.presence = nn.Linear(d, n_types)
        self.count = nn.Linear(d, n_types)

    def forward(self, h: torch.Tensor):
        z = DomusFM.window_embedding(h)  # ASSUMPTION: pooled window embedding
        return self.presence(z), F.softplus(self.count(z))

    @staticmethod
    def loss(out, batch):
        logits, counts = out
        y = batch["next_counts"]
        present = (y > 0).float()
        count_loss = ((counts - y) ** 2 * present).sum() / present.sum().clamp(min=1)
        return F.binary_cross_entropy_with_logits(logits, present) + count_loss

    @staticmethod
    def predict(out) -> torch.Tensor:
        logits, counts = out
        return torch.where(torch.sigmoid(logits) > 0.5, counts.round().clamp(min=1), torch.zeros_like(counts))
