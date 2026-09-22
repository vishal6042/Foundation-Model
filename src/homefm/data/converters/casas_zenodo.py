"""CASAS 2025 Zenodo release (record 15708568) → HomeStream.

Lines look like:
    2012-07-20,10:38:54.512364,OutsideDoor,ON,Step_Out="begin"
    2011-06-15,00:00:00.956024,Bathroom,ON,Toilet
    2011-06-15,02:11:07.1,KitchenATemperature,21

Sensor ids are compositional location names (Room + optional instance letter + item, e.g. KitchenAStove,
BathroomBToilet), so the house item / room / sensor type attributes DomusFM needs are derived
automatically — no hand-written sensor map. Sensor type comes from the message: ON/OFF → motion,
OPEN/CLOSE → door contact, numeric → temperature/scalar.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from homefm.schema import Entity, Episode, HomeStream, HomeToken, Modality, State

# (prefix, room text, default item) — longest prefixes first
_ROOMS = [
    ("MainEntryway", "entryway", "main entryway"), ("OutsideDoor", "entrance", "outside door"),
    ("MainDoor", "entrance", "main door"), ("LoungeChair", "living room", "lounge chair"),
    ("LaundryRoom", "laundry room", None), ("SewingRoom", "sewing room", None),
    ("LivingRoom", "living room", None), ("Livingroom", "living room", None),
    ("DiningRoom", "dining room", None), ("Diningroom", "dining room", None),
    ("GuestRoom", "guest room", None), ("OtherRoom", "other room", None), ("WorkArea", "office", "work area"),
    ("Bathroom", "bathroom", None), ("Bedroom", "bedroom", None), ("Kitchen", "kitchen", None),
    ("Hallway", "hallway", None), ("Entryway", "entryway", None), ("MOffice", "office", None),
    ("Office", "office", None), ("Hall", "hallway", None),
]
_ORDINAL = {"B": "second ", "C": "third ", "D": "fourth ", "E": "fifth "}
_LABEL = re.compile(r'^(.*?)(?:="(begin|end)")?$')


def _words(camel: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", camel).lower().strip()


def parse_sensor_name(name: str) -> tuple[str, str]:
    """'KitchenAStove' → ('stove', 'kitchen'); 'BedroomBArea' → ('second bedroom area', 'second bedroom')."""
    for prefix, room, default_item in _ROOMS:
        if name.startswith(prefix):
            rest = name[len(prefix):]
            letter = ""
            if len(rest) >= 1 and rest[0] in "ABCDE" and (len(rest) == 1 or rest[1].isupper()):
                letter, rest = rest[0], rest[1:]
            room_text = _ORDINAL.get(letter, "") + room
            if rest.endswith("B") and len(rest) > 1 and rest[-2].islower():
                rest = rest[:-1]  # BedB → Bed
            item = _words(rest) if rest else (default_item or f"{room_text} area")
            if item == "area":
                item = f"{room_text} area"
            return item, room_text
    return _words(name), "unknown room"


def _ts(date: str, time: str) -> float:
    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in time else "%Y-%m-%d %H:%M:%S"
    # CASAS times are local; stored as UTC so hour/day features stay local (tz_offset_hours = 0)
    return datetime.strptime(f"{date} {time}", fmt).replace(tzinfo=timezone.utc).timestamp()


def load_casas_zenodo(path: str | Path, home_id: str | None = None) -> HomeStream:
    path = Path(path)
    home_id = home_id or path.stem
    entities: dict[str, Entity] = {}
    tokens: list[HomeToken] = []
    episodes: list[Episode] = []
    open_acts: dict[str, float] = {}
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.strip().split(",")
        if len(parts) < 4 or not parts[2]:
            continue
        date, time, sensor, msg = parts[:4]
        label = parts[4].strip() if len(parts) > 4 else ""
        try:
            ts = _ts(date, time)
        except ValueError:
            continue
        up = msg.upper()
        if up in ("ON", "OFF"):
            stype, modality, state, value = "motion", Modality.BINARY, State.ON if up == "ON" else State.OFF, None
        elif up in ("OPEN", "CLOSE"):
            stype, modality, state, value = "door contact", Modality.BINARY, State.ON if up == "OPEN" else State.OFF, None
        else:
            try:
                value = float(msg)
            except ValueError:
                continue
            stype = "temperature" if "Temperature" in sensor else "scalar"
            modality, state = Modality.SCALAR, State.NA
        eid = f"{sensor}|{stype}"
        if eid not in entities:
            item, room = parse_sensor_name(sensor.replace("Temperature", ""))
            if stype == "temperature":
                item = "air"
            entities[eid] = Entity(eid, item, room, stype, modality)
        tokens.append(HomeToken(ts, entities[eid], state, value, home_id=home_id))

        if label:
            m = _LABEL.match(label)
            act, edge = m.group(1).strip(), m.group(2)
            concept = act.lower().replace("_", " ")
            if edge == "begin":
                open_acts[concept] = ts
            elif edge == "end":
                if concept in open_acts:
                    episodes.append(Episode(concept, open_acts.pop(concept), ts, entities[eid].room,
                                            caption=f"the resident is doing: {concept}"))
            else:  # event annotated individually with its activity
                episodes.append(Episode(concept, ts, ts, entities[eid].room, caption=f"the resident is doing: {concept}"))
    return HomeStream(home_id, list(entities.values()), tokens, episodes).sort()
