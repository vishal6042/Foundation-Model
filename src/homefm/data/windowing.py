"""Time-based windows of Home Tokens → padded tensors (DESIGN.md §7.2).

A window covers `span_s` seconds and is cut into `span_s / moment_s` fixed-duration moments.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import torch
from torch.utils.data import Dataset

from homefm.schema import Entity, Episode, HomeStream, Modality, State

from .text_encoder import TextEncoder

QUIET_CAPTION = "a quiet period at home"


@dataclass
class WindowConfig:
    span_s: float = 3600.0
    moment_s: float = 60.0
    stride_s: float = 900.0
    max_events: int = 768

    @property
    def n_moments(self) -> int:
        return int(round(self.span_s / self.moment_s))


def _local_hour_dow(ts: np.ndarray, tz_hours: float) -> tuple[np.ndarray, np.ndarray]:
    local = ts + tz_hours * 3600.0
    hour = (local % 86400.0) / 3600.0
    dow = (np.floor(local / 86400.0) + 3) % 7  # 1970-01-01 was a Thursday; Monday = 0
    return hour.astype(np.float32), dow.astype(np.float32)


class _StreamArrays:
    """Per-home event arrays, precomputed once (vectorised so corpus-scale homes load quickly)."""

    def __init__(self, home_id, tz, ts, ent, room, state, modality, conf, raw, entities, episodes, anomalies):
        self.home_id, self.tz = home_id, tz
        self.ts = np.asarray(ts, dtype=np.float64)
        self.ent = np.asarray(ent, dtype=np.int64)
        self.room = np.asarray(room, dtype=np.int64)
        self.state = np.asarray(state, dtype=np.int64)
        self.modality = np.asarray(modality, dtype=np.int64)
        self.conf = np.asarray(conf, dtype=np.float32)
        raw = np.asarray(raw, dtype=np.float64)
        self.has_value = ~np.isnan(raw)
        # per-entity z-score for scalar values (no binarisation, DESIGN.md �4.1 L2)
        value = np.zeros_like(raw)
        for e in np.unique(self.ent[self.has_value]):
            sel = (self.ent == e) & self.has_value
            mu, sd = raw[sel].mean(), max(raw[sel].std(), 1e-3)
            value[sel] = np.clip((raw[sel] - mu) / sd, -5, 5)
        self.value = value.astype(np.float32)
        self.dt_prev = np.diff(self.ts, prepend=self.ts[0] if len(self.ts) else 0).astype(np.float32)
        self.dt_entity = np.full(len(self.ts), 86400.0, dtype=np.float32)
        if len(self.ts) > 1:  # time since the previous event of the same entity
            order = np.lexsort((self.ts, self.ent))
            same = self.ent[order][1:] == self.ent[order][:-1]
            gaps = np.minimum(np.diff(self.ts[order]), 86400.0)
            self.dt_entity[order[1:][same]] = gaps[same]
        self.hour, self.dow = _local_hour_dow(self.ts, self.tz)
        self.episodes = episodes
        self.anomalies = anomalies
        self.entities = entities

    @classmethod
    def from_stream(cls, stream: HomeStream, ent_index: dict[str, int], room_index: dict[str, int]):
        toks = stream.tokens
        return cls(stream.home_id, stream.tz_offset_hours, [t.ts for t in toks],
                   [ent_index[t.entity.text()] for t in toks], [room_index[t.entity.room] for t in toks],
                   [int(t.state) for t in toks], [int(t.modality) for t in toks], [t.confidence for t in toks],
                   [np.nan if t.value is None else t.value for t in toks],
                   {ent_index[e.text()] for e in stream.entities}, stream.episodes, stream.anomalies)

    @classmethod
    def from_home_arrays(cls, h, ent_index: dict[str, int], room_index: dict[str, int]):
        ents = home_entities(h)
        ent_map = np.array([ent_index[e.text()] for e in ents], dtype=np.int64)
        room_map = np.array([room_index[e.room] for e in ents], dtype=np.int64)
        mod_map = np.array([int(e.modality) for e in ents], dtype=np.int64)
        episodes = [Episode(h.concepts[c], float(s), float(e)) for s, e, c in zip(h.ep_start, h.ep_end, h.ep_concept)]
        return cls(h.home_id, 0.0, h.ts, ent_map[h.entity], room_map[h.entity], h.state, mod_map[h.entity],
                   np.ones(len(h.ts), np.float32), h.value, set(ent_map.tolist()), episodes, [])


def home_entities(h) -> list[Entity]:
    """Entities of a HomeStream or a casas_fast HomeArrays."""
    if isinstance(h, HomeStream):
        return list(h.entities) + [t.entity for t in h.tokens]
    return [Entity(eid, item, room, stype, Modality(mod)) for eid, item, room, stype, mod in h.entities]


class HomeFMData:
    """Holds vocabularies (entities, rooms, homes, concepts) and per-home arrays for a set of streams."""

    def __init__(self, streams: list, text_encoder: TextEncoder, concepts: list[str],
                 cfg: WindowConfig | None = None):
        """streams: HomeStream objects and/or casas_fast HomeArrays."""
        self.cfg = cfg or WindowConfig()
        self.text_encoder = text_encoder
        self.concepts = list(concepts)
        self.concept_index = {c: i for i, c in enumerate(self.concepts)}
        ents = {s.home_id: home_entities(s) for s in streams}
        texts = sorted({e.text() for es in ents.values() for e in es})
        self.entity_texts = texts
        self.entity_index = {t: i for i, t in enumerate(texts)}
        rooms = sorted({e.room for es in ents.values() for e in es})
        self.room_index = {r: i for i, r in enumerate(rooms)}
        self.home_ids = [s.home_id for s in streams]
        self.home_index = {h: i for i, h in enumerate(self.home_ids)}
        self.entity_table = torch.from_numpy(text_encoder.encode(texts))
        self.arrays = {s.home_id: (_StreamArrays.from_stream if isinstance(s, HomeStream) else
                                   _StreamArrays.from_home_arrays)(s, self.entity_index, self.room_index)
                       for s in streams}
        counts = np.zeros(len(texts), dtype=np.float64)
        for a in self.arrays.values():
            np.add.at(counts, a.ent, 1)
        self.entity_freq = torch.from_numpy(counts / max(counts.sum(), 1)).float()
        self._caption_cache: dict[str, torch.Tensor] = {}

    @property
    def n_entities(self) -> int:
        return len(self.entity_texts)

    def caption_embedding(self, caption: str) -> torch.Tensor:
        if caption not in self._caption_cache:
            self._caption_cache[caption] = torch.from_numpy(self.text_encoder.encode([caption])[0])
        return self._caption_cache[caption]

    def concept_text_embeddings(self) -> torch.Tensor:
        return torch.from_numpy(self.text_encoder.encode(self.concepts))

    def _episode_arrays(self, home_id: str):
        cache = self.__dict__.setdefault("_ep_cache", {})
        if home_id not in cache:
            eps = self.arrays[home_id].episodes
            cache[home_id] = (np.array([e.start_ts for e in eps], np.float64),
                              np.array([e.end_ts for e in eps], np.float64),
                              np.array([self.concept_index.get(e.concept, -1) for e in eps], np.int64))
        return cache[home_id]

    def window_starts(self, home_id: str, stride_s: float | None = None) -> list[float]:
        a = self.arrays[home_id]
        if len(a.ts) == 0:
            return []
        stride = stride_s or self.cfg.stride_s
        t0 = np.floor(a.ts[0] / self.cfg.moment_s) * self.cfg.moment_s
        return list(np.arange(t0, a.ts[-1] - self.cfg.span_s, stride))

    def window(self, home_id: str, start: float, keep: str = "first") -> dict:
        """keep: which events to keep when a window exceeds max_events ("last" for windows anchored at their end)."""
        cfg, a = self.cfg, self.arrays[home_id]
        M, end = cfg.n_moments, start + cfg.span_s
        lo, hi = int(np.searchsorted(a.ts, start)), int(np.searchsorted(a.ts, end))
        truncated = hi - lo > cfg.max_events
        if keep == "last":
            lo = max(lo, hi - cfg.max_events)
        else:
            hi = min(hi, lo + cfg.max_events)
        sl = slice(lo, hi)
        t = (a.ts[sl] - start).astype(np.float32)
        moment_idx = np.clip((t // cfg.moment_s).astype(np.int64), 0, M - 1)

        m_start = start + np.arange(M) * cfg.moment_s
        m_hour, m_dow = _local_hour_dow(m_start, a.tz)
        labels = np.zeros((M, len(self.concepts)), dtype=np.float32)
        coverage: dict[int, float] = {}
        caption_of: dict[int, str] = {}
        es, ee, ec = self._episode_arrays(home_id)
        for j in np.flatnonzero((ee > start) & (es < end) & (ec >= 0)):  # vectorised overlap test
            ep, c = a.episodes[j], int(ec[j])
            s_m = int(max(0, (es[j] - start) // cfg.moment_s))
            e_m = int(min(M - 1, (ee[j] - start) // cfg.moment_s))
            labels[s_m:e_m + 1, c] = 1.0
            cov = min(ee[j], end) - max(es[j], start)
            if cov > coverage.get(c, 0):
                coverage[c], caption_of[c] = cov, ep.caption or ep.concept
        anomaly = np.zeros(M, dtype=bool)
        for an in a.anomalies:
            if an.end_ts <= start or an.start_ts >= end:
                continue
            s_m = int(max(0, (an.start_ts - start) // cfg.moment_s))
            e_m = int(min(M - 1, (an.end_ts - start) // cfg.moment_s))
            anomaly[s_m:e_m + 1] = True
        # caption = the concept covering most of the window (used for language alignment)
        group, caption = -1, QUIET_CAPTION
        if coverage:
            c = max(coverage, key=lambda k: coverage[k])
            group, caption = c, caption_of[c]

        return dict(
            entity_idx=a.ent[sl], room_idx=a.room[sl], state=a.state[sl], value=a.value[sl],
            has_value=a.has_value[sl], modality=a.modality[sl], confidence=a.conf[sl], t=t,
            hour=a.hour[sl], dow=a.dow[sl], dt_prev=a.dt_prev[sl], dt_entity=a.dt_entity[sl],
            moment_idx=moment_idx, moment_hour=m_hour, moment_dow=m_dow, labels=labels, anomaly=anomaly,
            home_idx=self.home_index[home_id], caption=caption, caption_group=group,
            home_entities=a.entities, truncated=truncated, start=start,
        )


@dataclass
class Batch:
    entity_idx: torch.Tensor   # [B, N] long
    room_idx: torch.Tensor     # [B, N] long
    state: torch.Tensor        # [B, N] long (OFF, ON, NA)
    value: torch.Tensor        # [B, N] float (z-scored)
    has_value: torch.Tensor    # [B, N] bool
    modality: torch.Tensor     # [B, N] long
    confidence: torch.Tensor   # [B, N] float
    t: torch.Tensor            # [B, N] float, seconds from window start
    hour: torch.Tensor         # [B, N] float
    dow: torch.Tensor          # [B, N] float
    dt_prev: torch.Tensor      # [B, N] float
    dt_entity: torch.Tensor    # [B, N] float
    moment_idx: torch.Tensor   # [B, N] long
    valid: torch.Tensor        # [B, N] bool (False = padding)
    moment_hour: torch.Tensor  # [B, M] float
    moment_dow: torch.Tensor   # [B, M] float
    labels: torch.Tensor       # [B, M, C] float multi-hot concepts
    anomaly: torch.Tensor      # [B, M] bool
    home_idx: torch.Tensor     # [B] long
    caption_emb: torch.Tensor  # [B, Dt] float
    caption_group: torch.Tensor  # [B] long, -1 = quiet
    home_entity_mask: torch.Tensor  # [B, V] bool, entities that exist in each sample's home

    @property
    def n_moments(self) -> int:
        return self.moment_hour.shape[1]

    def to(self, device) -> "Batch":
        return Batch(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})


class WindowDataset(Dataset):
    def __init__(self, data: HomeFMData, home_ids: list[str] | None = None, stride_s: float | None = None):
        self.data = data
        self.index = [(h, s) for h in (home_ids or data.home_ids) for s in data.window_starts(h, stride_s)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        return self.data.window(*self.index[i])

    def collate(self, items: list[dict]) -> Batch:
        B = len(items)
        N = max(1, max(len(it["t"]) for it in items))
        V = self.data.n_entities

        def pad(key, dtype, fill=0):
            out = torch.full((B, N), fill, dtype=dtype)
            for i, it in enumerate(items):
                n = len(it[key])
                if n:
                    out[i, :n] = torch.as_tensor(np.asarray(it[key]), dtype=dtype)
            return out

        valid = torch.zeros(B, N, dtype=torch.bool)
        home_mask = torch.zeros(B, V, dtype=torch.bool)
        for i, it in enumerate(items):
            valid[i, :len(it["t"])] = True
            home_mask[i, list(it["home_entities"])] = True
        return Batch(
            entity_idx=pad("entity_idx", torch.long), room_idx=pad("room_idx", torch.long),
            state=pad("state", torch.long, int(State.NA)), value=pad("value", torch.float),
            has_value=pad("has_value", torch.bool, False), modality=pad("modality", torch.long, int(Modality.BINARY)),
            confidence=pad("confidence", torch.float, 1.0), t=pad("t", torch.float), hour=pad("hour", torch.float),
            dow=pad("dow", torch.float), dt_prev=pad("dt_prev", torch.float), dt_entity=pad("dt_entity", torch.float),
            moment_idx=pad("moment_idx", torch.long), valid=valid,
            moment_hour=torch.as_tensor(np.stack([it["moment_hour"] for it in items])),
            moment_dow=torch.as_tensor(np.stack([it["moment_dow"] for it in items])),
            labels=torch.as_tensor(np.stack([it["labels"] for it in items])),
            anomaly=torch.as_tensor(np.stack([it["anomaly"] for it in items])),
            home_idx=torch.tensor([it["home_idx"] for it in items], dtype=torch.long),
            caption_emb=torch.stack([self.data.caption_embedding(it["caption"]) for it in items]),
            caption_group=torch.tensor([it["caption_group"] for it in items], dtype=torch.long),
            home_entity_mask=home_mask,
        )
