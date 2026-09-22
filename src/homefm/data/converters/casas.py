"""CASAS dataset converter (Milan, Aruba and other CASAS homes) → HomeStream.

CASAS lines look like:
    2009-10-16 00:01:04.000059 M017 ON
    2009-10-16 08:07:10.084419 M021 ON Sleep end
    2010-11-04 00:03:57.399391 T004 21.5

Sensor ids carry no semantics, so a sensor map is required (JSON):
    {"M017": {"item": "bed", "room": "bedroom", "sensor_type": "motion"}, ...}
Write one map per home under data/raw/casas/<home>/sensor_map.json (see docs/DESIGN.md §6.1).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from homefm.schema import Entity, Episode, HomeStream, HomeToken, Modality, State

_LINE = re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)(?:\s+(.+?))?\s*$")
_ON = {"ON", "OPEN", "PRESENT"}
_OFF = {"OFF", "CLOSE", "ABSENT"}


def _parse_ts(date: str, time: str) -> float:
    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in time else "%Y-%m-%d %H:%M:%S"
    # CASAS timestamps are local time; treat them as UTC and keep tz_offset_hours=0 so hour/day features stay local.
    return datetime.strptime(f"{date} {time}", fmt).replace(tzinfo=timezone.utc).timestamp()


def load_casas(data_path: str | Path, sensor_map_path: str | Path, home_id: str,
               label_map: dict[str, str] | None = None) -> HomeStream:
    sensor_map = json.loads(Path(sensor_map_path).read_text())
    label_map = {k.lower(): v for k, v in (label_map or {}).items()}
    entities: dict[str, Entity] = {}
    tokens: list[HomeToken] = []
    episodes: list[Episode] = []
    open_acts: dict[str, float] = {}
    skipped = 0

    for line in Path(data_path).read_text(errors="ignore").splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        date, time, sid, raw, annot = m.groups()
        try:
            ts = _parse_ts(date, time)
        except ValueError:
            skipped += 1
            continue
        meta = sensor_map.get(sid)
        if meta is None:
            skipped += 1
            continue
        up = raw.upper()
        if up in _ON or up in _OFF:
            modality, state, value = Modality.BINARY, State.ON if up in _ON else State.OFF, None
        else:
            try:
                modality, state, value = Modality.SCALAR, State.NA, float(raw)
            except ValueError:
                skipped += 1
                continue
        if sid not in entities:
            entities[sid] = Entity(sid, meta["item"], meta["room"], meta["sensor_type"], modality)
        tokens.append(HomeToken(ts, entities[sid], state, value, home_id=home_id))

        if annot:
            parts = annot.rsplit(" ", 1)
            if len(parts) == 2 and parts[1].lower() in ("begin", "end"):
                act, edge = parts[0].strip(), parts[1].lower()
                concept = label_map.get(act.lower(), act.lower().replace("_", " "))
                if edge == "begin":
                    open_acts[concept] = ts
                elif concept in open_acts:
                    episodes.append(Episode(concept, open_acts.pop(concept), ts, meta["room"],
                                            caption=f"the resident is {concept}"))

    if skipped:
        print(f"[casas] {home_id}: skipped {skipped} unparseable or unmapped lines")
    return HomeStream(home_id, list(entities.values()), tokens, episodes).sort()
