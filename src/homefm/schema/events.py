"""Home Token schema: the common event format every signal is converted into (DESIGN.md §6.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Modality(IntEnum):
    BINARY = 0
    SCALAR = 1
    AUDIO_TAG = 2
    VISION_DET = 3
    EMBEDDING = 4


class State(IntEnum):
    OFF = 0
    ON = 1
    NA = 2  # not applicable (scalar / vector modalities)


class Source(IntEnum):
    SENSOR = 0
    EXPERT = 1
    HOMEFM = 2


@dataclass(frozen=True)
class Entity:
    """A sensor or virtual sensor. The model only ever sees its text description, never the id."""

    entity_id: str
    item: str
    room: str
    sensor_type: str
    modality: Modality = Modality.BINARY
    capability: str = ""

    def text(self) -> str:
        return f"{self.item} in {self.room}, {self.sensor_type} sensor"


@dataclass
class HomeToken:
    ts: float  # unix seconds
    entity: Entity
    state: State = State.NA
    value: float | None = None
    vector: list[float] | None = None
    confidence: float = 1.0
    source: Source = Source.SENSOR
    person_id: str | None = None
    home_id: str = ""

    @property
    def modality(self) -> Modality:
        return self.entity.modality


@dataclass
class Episode:
    """A labelled or predicted occurrence of a concept (activity, event) with boundaries."""

    concept: str
    start_ts: float
    end_ts: float
    room: str = ""
    confidence: float = 1.0
    persons: list[str] = field(default_factory=list)
    caption: str = ""


@dataclass
class AnomalyLabel:
    kind: str
    start_ts: float
    end_ts: float
    description: str = ""


@dataclass
class HomeStream:
    """A home's event stream plus whatever ground truth is available."""

    home_id: str
    entities: list[Entity]
    tokens: list[HomeToken]
    episodes: list[Episode] = field(default_factory=list)
    anomalies: list[AnomalyLabel] = field(default_factory=list)
    tz_offset_hours: float = 0.0

    def sort(self) -> "HomeStream":
        self.tokens.sort(key=lambda t: t.ts)
        self.episodes.sort(key=lambda e: e.start_ts)
        return self
