"""UCI ADL Binary dataset (Ordóñez et al., UCI repository id 271) → HomeStream.

OrdonezX_Sensors.txt rows: start, end, Location (item), Type, Place (room) → ON at start, OFF at end.
OrdonezX_ADLs.txt rows:    start, end, Activity.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from homefm.schema import Entity, Episode, HomeStream, HomeToken, Modality, State

_TS = r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)"
_SENSOR = re.compile(rf"^{_TS}\s+{_TS}\s+(\S+)\s+(\S+)\s+(\S+)")
_ADL = re.compile(rf"^{_TS}\s+{_TS}\s+(\S+)")
_TYPES = {"PIR": "motion", "Magnetic": "door contact", "Pressure": "pressure", "Electric": "power plug", "Flush": "flush"}


def _ts(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def load_uci_adl(folder: str | Path, home: str = "B") -> HomeStream:
    folder = Path(folder)
    home_id = f"uci_{home.lower()}"
    entities: dict[str, Entity] = {}
    tokens: list[HomeToken] = []
    for line in (folder / f"Ordonez{home}_Sensors.txt").read_text(errors="ignore").splitlines():
        m = _SENSOR.match(line.strip())
        if not m:
            continue
        start, end, item, stype, room = m.groups()
        eid = f"{room}|{item}|{stype}"
        if eid not in entities:
            entities[eid] = Entity(eid, item.lower().replace("_", " "), room.lower(), _TYPES.get(stype, stype.lower()),
                                   Modality.BINARY)
        tokens.append(HomeToken(_ts(start), entities[eid], State.ON, home_id=home_id))
        tokens.append(HomeToken(_ts(end), entities[eid], State.OFF, home_id=home_id))
    episodes = []
    for line in (folder / f"Ordonez{home}_ADLs.txt").read_text(errors="ignore").splitlines():
        m = _ADL.match(line.strip())
        if m:
            s, e, act = m.groups()
            concept = act.lower().replace("_", " ")
            episodes.append(Episode(concept, _ts(s), _ts(e), caption=f"the resident is doing: {concept}"))
    return HomeStream(home_id, list(entities.values()), tokens, episodes).sort()
