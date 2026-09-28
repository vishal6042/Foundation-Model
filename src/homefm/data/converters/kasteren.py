"""Kasteren houses A and C (van Kasteren et al., "Accurate Activity Recognition in a Home Setting") → HomeStream.

The original download (sites.google.com/site/tim0306/tlDatasets.zip) is gone. Copies used here:
  House A: du-phan/Human-Activity-Recognition on GitHub, sensorData.txt + activitiesData.txt (original format,
           tab separated, numeric ids). Ids are named below; the names were checked by matching every event's start
           time against aitoralmeida/c4a_activity_recognition experiments/kasteren_dataset/base_kasteren.csv.
  House C: aitoralmeida/c4a_activity_recognition experiments/kasterenC_dataset, sensors.csv + activities.csv
           (original format, comma separated) with sensor_labels.txt / activity_labels.txt for the names.

Sensor rows: start, end, id, value → ON at start, OFF at end. Activity rows: start, end, id → episodes.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from homefm.schema import Entity, Episode, HomeStream, HomeToken, Modality, State

_TS = r"(\d\d-[A-Za-z]{3}-\d{4} \d\d:\d\d:\d\d)"
_ROW = re.compile(rf"^{_TS}[\t,]+{_TS}[\t,]+(\d+)")
_MONTHS = {m: i for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}

# House A: id → (item, room). Rooms are not in the release; assigned from the item (3-room apartment).
_A_SENSORS = {1: ("Microwave", "Kitchen"), 5: ("HallToiletDoor", "Hall"), 6: ("HallBathroomDoor", "Hall"),
              7: ("CupsCupboard", "Kitchen"), 8: ("Fridge", "Kitchen"), 9: ("PlatesCupboard", "Kitchen"),
              12: ("Frontdoor", "Hall"), 13: ("Dishwasher", "Kitchen"), 14: ("ToiletFlush", "Toilet"),
              17: ("Freezer", "Kitchen"), 18: ("PansCupboard", "Kitchen"), 20: ("Washingmachine", "Kitchen"),
              23: ("GroceriesCupboard", "Kitchen"), 24: ("HallBedroomDoor", "Hall")}
_A_ACTIVITIES = {1: "LeaveHouse", 4: "UseToilet", 5: "TakeShower", 10: "GoToBed", 13: "PrepareBreakfast",
                 15: "PrepareDinner", 17: "GetDrink"}
_FILES = {"A": ("sensorData.txt", "activitiesData.txt"), "C": ("sensors.csv", "activities.csv")}


def _ts(s: str) -> float:
    day, mon, rest = s.split("-", 2)
    s = f"{rest[:4]}-{_MONTHS[mon.title()]:02d}-{day} {rest[5:]}"
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def _words(name: str) -> str:
    """'HallToiletDoor' → 'hall toilet door'."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).lower()


def _sensor_type(item: str) -> str:
    i = item.lower()
    if "flush" in i:
        return "float"
    if i.startswith("bed") or i == "couch":
        return "pressure"
    return "door contact" if "door" in i else "contact switch"


def _labels(path: Path) -> dict[int, list[str]]:
    out = {}
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if parts and parts[0].isdigit():
            out[int(parts[0])] = [p for p in parts[1:] if not p.startswith("(")]
    return out


def _rows(path: Path):
    for line in path.read_text(errors="ignore").splitlines():
        m = _ROW.match(line.strip())
        if m:
            yield _ts(m[1]), _ts(m[2]), int(m[3])


def load_kasteren(folder: str | Path, house: str = "A") -> HomeStream:
    folder, house = Path(folder), house.upper()
    home_id = f"kasteren_{house.lower()}"
    if house == "A":
        sensors, activities = _A_SENSORS, _A_ACTIVITIES
    else:
        sensors = {k: (v[0], " ".join(v[1:])) for k, v in _labels(folder / "sensor_labels.txt").items()}
        activities = {k: v[0] for k, v in _labels(folder / "activity_labels.txt").items()}
    sensor_file, activity_file = _FILES.get(house, _FILES["C"])

    entities: dict[int, Entity] = {}
    tokens: list[HomeToken] = []
    for start, end, sid in _rows(folder / sensor_file):
        if sid not in entities:
            item, room = sensors[sid]
            entities[sid] = Entity(f"{sid}|{item}", _words(item), _words(room), _sensor_type(item), Modality.BINARY)
        tokens.append(HomeToken(start, entities[sid], State.ON, home_id=home_id))
        tokens.append(HomeToken(end, entities[sid], State.OFF, home_id=home_id))
    episodes = []
    for start, end, aid in _rows(folder / activity_file):
        concept = _words(activities[aid])
        episodes.append(Episode(concept, start, end, caption=f"the resident is doing: {concept}"))
    return HomeStream(home_id, list(entities.values()), tokens, episodes).sort()
