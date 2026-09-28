"""MuRAL (Chen et al., 2026; https://mural.imag.fr/) → HomeStream. One stream for all 21 sessions.

MuRAL/NN/data.csv rows: uid, time, sensor, action, Subject, Description, activity (id into activities.json).
MuRAL/NN/context.json: session start time, "weekday"/"weekend", number of residents.
MuRAL/sensors.json: sensor name, type and location.

Only times of day are given, so session NN is placed in week NN of a dummy calendar (weekday sessions on a
Wednesday, weekend sessions on a Saturday); sessions never overlap and keep their time of day and day type.
Every event carries its own activity label and resident, so each event becomes a zero-length episode. Events that
share a second are spread by 1 ms so each episode covers exactly its event. Rows without a sensor (annotation-only)
and activity 0 ("others") are skipped.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from homefm.schema import Entity, Episode, HomeStream, HomeToken, Modality, State

_STATE = {"turned ON": State.ON, "OPENED": State.ON, "turned OFF": State.OFF, "CLOSED": State.OFF}
_TYPES = {"wattmeter": "power meter", "PIR motion sensor": "motion", "motion sensor": "motion",
          "magnetic contact sensor": "door contact"}
_EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)  # a Monday


def _sensors(root: Path) -> dict[str, Entity]:
    """data.csv names ('kitchen counter mov') are the sensors.json names ('kitchen counter motion sensor') shortened."""
    out = {}
    for s in json.loads((root / "sensors.json").read_text(encoding="utf-8")):
        name, room = s["sensor name"], s["sensor location"].removesuffix(" door")
        short = name.removesuffix(" sensor").replace("motion", "mov").replace("microwave oven", "microwave")
        item = name.removeprefix(room).removesuffix(" sensor").strip().replace("_", " ")
        out[short] = Entity(short, item, room.replace("_", " "), _TYPES.get(s["sensor type"], s["sensor type"]),
                            Modality.BINARY)
    return out


def load_mural(folder: str | Path) -> HomeStream:
    root = Path(folder)
    root = root / "MuRAL" if (root / "MuRAL").is_dir() else root
    entities = _sensors(root)
    acts = {a["id"]: a["name"] for a in json.loads((root / "activities.json").read_text(encoding="utf-8"))}
    tokens, episodes, used = [], [], {}
    for week, session in enumerate(sorted(p for p in root.iterdir() if (p / "data.csv").exists())):
        ctx = json.loads((session / "context.json").read_text(encoding="utf-8"))
        day = _EPOCH + timedelta(weeks=week, days=5 if ctx.get("day") == "weekend" else 2)
        prev, last_sec, dup = None, None, 0
        with open(session / "data.csv", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                state = _STATE.get(r["action"])
                if not r["sensor"] or state is None:
                    continue
                t = datetime.strptime(r["time"], "%H:%M:%S").time()
                if prev is not None and t < prev:  # session runs past midnight
                    day += timedelta(days=1)
                prev = t
                sec = datetime.combine(day.date(), t, timezone.utc).timestamp()
                dup = dup + 1 if sec == last_sec else 0
                last_sec, ts = sec, sec + dup * 1e-3
                e = entities[r["sensor"]]
                used[e.entity_id] = e
                tokens.append(HomeToken(ts, e, state, person_id=r["Subject"], home_id="mural"))
                concept = acts.get(int(r["activity"] or 0), "others")
                if concept != "others":
                    episodes.append(Episode(concept, ts, ts, room=e.room, persons=[r["Subject"]],
                                            caption=r["Description"]))
    return HomeStream("mural", list(used.values()), tokens, episodes).sort()
