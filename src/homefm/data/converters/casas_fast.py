"""Fast, cached CASAS (2025 Zenodo) loader for corpus-scale pretraining.

`load_casas_zenodo` builds one Python object per event, which is too slow and memory-hungry for the
~20M events of the 82 labelled homes. This loader parses a home straight into numpy arrays with the
same semantics (sensor naming, sensor types, episodes) and caches the result as .npz next to a cache dir.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from homefm.schema import Modality, State

from .casas_zenodo import parse_sensor_name

_LABEL = re.compile(r'^(.*?)(?:="(begin|end)")?$')


@dataclass
class HomeArrays:
    """One home's events as arrays. Entities are (entity_id, item, room, sensor_type, modality)."""

    home_id: str
    entities: list[tuple[str, str, str, str, int]]
    ts: np.ndarray          # float64 seconds (local time stored as UTC)
    entity: np.ndarray      # int32 index into entities
    state: np.ndarray       # int8 (OFF, ON, NA)
    value: np.ndarray       # float32, NaN when not scalar
    concepts: list[str]
    ep_start: np.ndarray    # float64
    ep_end: np.ndarray      # float64
    ep_concept: np.ndarray  # int32 index into concepts

    @property
    def n_events(self) -> int:
        return len(self.ts)


def _parse(path: Path, home_id: str) -> HomeArrays:
    dates, times, ent_idx, states, values = [], [], [], [], []
    entities: dict[str, int] = {}
    ent_list: list[tuple[str, str, str, str, int]] = []
    concepts: dict[str, int] = {}
    ep_s, ep_e, ep_c = [], [], []
    open_acts: dict[int, int] = {}  # concept -> event index of begin
    labels_at: list[tuple[int, str, str | None]] = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 4 or not parts[2]:
                continue
            msg = parts[3].strip().upper()
            if msg in ("ON", "OFF"):
                stype, mod, st, val = "motion", int(Modality.BINARY), int(State.ON if msg == "ON" else State.OFF), np.nan
            elif msg in ("OPEN", "CLOSE"):
                stype, mod, st, val = "door contact", int(Modality.BINARY), int(State.ON if msg == "OPEN" else State.OFF), np.nan
            else:
                try:
                    val = float(msg)
                except ValueError:
                    continue
                stype = "temperature" if "Temperature" in parts[2] else "scalar"
                mod, st = int(Modality.SCALAR), int(State.NA)
            sensor = parts[2]
            eid = f"{sensor}|{stype}"
            if eid not in entities:
                item, room = parse_sensor_name(sensor.replace("Temperature", ""))
                if stype == "temperature":
                    item = "air"
                entities[eid] = len(ent_list)
                ent_list.append((eid, item, room, stype, mod))
            i = len(dates)
            dates.append(parts[0])
            times.append(parts[1])
            ent_idx.append(entities[eid])
            states.append(st)
            values.append(val)
            if len(parts) > 4 and parts[4].strip():
                m = _LABEL.match(parts[4].strip())
                labels_at.append((i, m.group(1).strip().lower().replace("_", " "), m.group(2)))

    ts = (np.array([f"{d}T{t}" for d, t in zip(dates, times)], dtype="datetime64[us]")
          .astype("int64") / 1e6).astype(np.float64)
    for i, concept, edge in labels_at:
        c = concepts.setdefault(concept, len(concepts))
        if edge == "begin":
            open_acts[c] = i
        elif edge == "end":
            if c in open_acts:
                b = open_acts.pop(c)
                ep_s.append(ts[b]); ep_e.append(ts[i]); ep_c.append(c)
        else:
            ep_s.append(ts[i]); ep_e.append(ts[i]); ep_c.append(c)

    order = np.argsort(ts, kind="stable")
    return HomeArrays(home_id, ent_list, ts[order], np.array(ent_idx, np.int32)[order],
                      np.array(states, np.int8)[order], np.array(values, np.float32)[order],
                      list(concepts), np.array(ep_s, np.float64), np.array(ep_e, np.float64),
                      np.array(ep_c, np.int32))


def load_casas_fast(path: str | Path, home_id: str | None = None, cache_dir: str | Path | None = None) -> HomeArrays:
    path = Path(path)
    home_id = home_id or path.stem
    cache = Path(cache_dir) / f"{home_id}.npz" if cache_dir else None
    if cache and cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        z = np.load(cache, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return HomeArrays(home_id, [tuple(e) for e in meta["entities"]], z["ts"], z["entity"], z["state"],
                          z["value"], meta["concepts"], z["ep_start"], z["ep_end"], z["ep_concept"])
    h = _parse(path, home_id)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps({"entities": h.entities, "concepts": h.concepts})
        np.savez(cache, ts=h.ts, entity=h.entity, state=h.state, value=h.value, ep_start=h.ep_start,
                 ep_end=h.ep_end, ep_concept=h.ep_concept, meta=np.array(meta))
    return h
