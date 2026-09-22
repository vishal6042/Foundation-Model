"""DomusFM data pipeline (paper §3.1, §5, §6.2.2).

- Only binary ON/OFF events. Continuous sensors are binarised into "virtual" ON/OFF events (§3.1).
- Each sensor is described by three text attributes: house item, sensor type, room (§4.1.1).
- Fixed event-based windows of L = 30 events, sliding by one event (§6.2.2).
- ADL label of a window = activity at its last event, with an explicit "Other" class (§6.1.1, §6.4.1).
- Next-k target = bag of (sensor, status) counts over the k events after the window (§6.5.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from homefm.schema import HomeStream, Modality, State

OTHER = "Other"


def binarize_stream(stream: HomeStream) -> list[tuple[float, str, int]]:
    """Home Tokens → [(ts, entity_id, status)] with scalar sensors turned into ON/OFF virtual events.

    Threshold per scalar sensor = midpoint of its 10th and 90th percentile; only state changes are emitted.
    Vector modalities (audio/vision tags) are kept as ON events.
    """
    scalars: dict[str, list[float]] = {}
    for t in stream.tokens:
        if t.modality == Modality.SCALAR and t.value is not None:
            scalars.setdefault(t.entity.entity_id, []).append(t.value)
    thr = {k: 0.5 * (np.percentile(v, 10) + np.percentile(v, 90)) for k, v in scalars.items()}

    out, last = [], {}
    for t in stream.tokens:
        eid = t.entity.entity_id
        if t.modality == Modality.SCALAR:
            if t.value is None or eid not in thr:
                continue
            s = int(t.value > thr[eid])
            if last.get(eid) == s:
                continue
            last[eid] = s
            out.append((t.ts, eid, s))
        elif t.state in (State.ON, State.OFF):
            out.append((t.ts, eid, int(t.state == State.ON)))
    return out


@dataclass
class DomusDataset:
    """One smart-home dataset in DomusFM form. Arrays are aligned per event."""

    name: str
    sensor_ids: list[str]
    sensor_text: list[tuple[str, str, str]]  # (house item, sensor type, room)
    activities: list[str]                    # index 0 is "Other"
    sensor: np.ndarray
    status: np.ndarray
    ts: np.ndarray
    label: np.ndarray

    @property
    def n_events(self) -> int:
        return len(self.sensor)

    @property
    def n_event_types(self) -> int:
        return 2 * len(self.sensor_ids)


def build_domus_dataset(name: str, streams: list[HomeStream], activities: list[str] | None = None) -> DomusDataset:
    """Concatenate one or more homes' streams (same sensor layout) into a DomusFM dataset."""
    sensor_index: dict[str, int] = {}
    sensor_text: list[tuple[str, str, str]] = []
    for s in streams:
        for e in s.entities:
            if e.entity_id not in sensor_index:
                sensor_index[e.entity_id] = len(sensor_index)
                sensor_text.append((e.item, e.sensor_type, e.room))

    acts = [OTHER] + sorted(activities or {ep.concept for s in streams for ep in s.episodes})
    act_index = {a: i for i, a in enumerate(acts)}
    sensor, status, ts, label = [], [], [], []
    for s in streams:
        events = binarize_stream(s)
        e_ts = np.array([e[0] for e in events])
        lab = np.zeros(len(events), dtype=np.int64)
        best = np.full(len(events), np.inf)
        # overlapping annotations: the most specific (shortest) episode covering the event wins
        for ep in s.episodes:
            if ep.concept not in act_index:
                continue
            lo, hi = np.searchsorted(e_ts, ep.start_ts), np.searchsorted(e_ts, ep.end_ts, side="right")
            dur = ep.end_ts - ep.start_ts
            sel = slice(lo, hi)
            better = best[sel] > dur
            lab[sel] = np.where(better, act_index[ep.concept], lab[sel])
            best[sel] = np.where(better, dur, best[sel])
        sensor += [sensor_index[e[1]] for e in events]
        status += [e[2] for e in events]
        ts += list(e_ts)
        label += list(lab)
    return DomusDataset(name, list(sensor_index), sensor_text, acts, np.array(sensor, np.int64),
                        np.array(status, np.int64), np.array(ts, np.float64), np.array(label, np.int64))


def build_domus_from_arrays(h, activities: list[str] | None = None) -> DomusDataset:
    """Vectorised equivalent of build_domus_dataset for a `HomeArrays` (casas_fast) home."""
    keep = h.state != int(State.NA)
    scalar = ~keep & ~np.isnan(h.value)
    status = np.where(keep, h.state == int(State.ON), False).astype(np.int64)
    for e in np.unique(h.entity[scalar]):  # virtual ON/OFF events on threshold crossings
        idx = np.flatnonzero(scalar & (h.entity == e))
        v = h.value[idx]
        s = v > 0.5 * (np.percentile(v, 10) + np.percentile(v, 90))
        change = np.concatenate([[True], s[1:] != s[:-1]])
        keep[idx[change]] = True
        status[idx] = s
    ts, sensor, status = h.ts[keep], h.entity[keep].astype(np.int64), status[keep]

    acts = [OTHER] + sorted(activities or set(h.concepts))
    act_index = {a: i for i, a in enumerate(acts)}
    label = np.zeros(len(ts), dtype=np.int64)
    best = np.full(len(ts), np.inf)
    for s, e, c in zip(h.ep_start, h.ep_end, h.ep_concept):
        a = act_index.get(h.concepts[c])
        if a is None:
            continue
        lo, hi = np.searchsorted(ts, s), np.searchsorted(ts, e, side="right")
        better = best[lo:hi] > (e - s)
        label[lo:hi] = np.where(better, a, label[lo:hi])
        best[lo:hi] = np.where(better, e - s, best[lo:hi])
    return DomusDataset(h.home_id, [e[0] for e in h.entities], [(e[1], e[3], e[2]) for e in h.entities], acts,
                        sensor, status, ts, label)


class TextTable:
    """Frozen text embeddings for every attribute string (items, sensor types, rooms) across datasets."""

    def __init__(self, datasets: list[DomusDataset], encoder):
        texts = sorted({t for d in datasets for triple in d.sensor_text for t in triple})
        self.index = {t: i for i, t in enumerate(texts)}
        self.table = torch.from_numpy(encoder.encode(texts)).float()

    def sensor_attr_ids(self, d: DomusDataset) -> np.ndarray:
        """[n_sensors, 3] text ids of (item, type, room)."""
        return np.array([[self.index[t] for t in triple] for triple in d.sensor_text], dtype=np.int64)


class DomusWindows(Dataset):
    """Sliding windows of `window` events. `ends` are indices of each window's last event."""

    def __init__(self, d: DomusDataset, text: TextTable, window: int = 30, stride: int = 1, k: int = 0,
                 ends: np.ndarray | None = None):
        self.d, self.window, self.k = d, window, k
        self.attr = torch.from_numpy(text.sensor_attr_ids(d))
        local = d.ts  # CASAS stores local time; synthetic homes use UTC = local
        self.dow = torch.from_numpy(((np.floor(local / 86400) + 3) % 7).astype(np.int64))
        self.hour = torch.from_numpy(((local % 86400) // 3600).astype(np.int64))
        self.sec = torch.from_numpy((local % 3600).astype(np.int64))
        self.sensor = torch.from_numpy(d.sensor)
        self.status = torch.from_numpy(d.status)
        self.label = torch.from_numpy(d.label)
        self.ends = ends if ends is not None else np.arange(window - 1, d.n_events - k, stride)

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, i: int) -> dict:
        e = int(self.ends[i])
        sl = slice(e - self.window + 1, e + 1)
        sens = self.sensor[sl]
        item = dict(item=self.attr[sens, 0], stype=self.attr[sens, 1], room=self.attr[sens, 2],
                    status=self.status[sl], dow=self.dow[sl], hour=self.hour[sl], sec=self.sec[sl],
                    label=self.label[e])
        if self.k:
            nxt = slice(e + 1, e + 1 + self.k)
            types = self.sensor[nxt] * 2 + self.status[nxt]
            item["next_counts"] = torch.bincount(types, minlength=self.d.n_event_types).float()
        return item
