"""Home ontology and capability registry (DESIGN.md §6.2)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .events import Entity


@dataclass
class ConceptRule:
    """How raw per-moment detections become countable episodes (DESIGN.md §9.1)."""

    concept: str
    merge_gap_s: float = 120.0
    min_duration_s: float = 0.0
    requires: list[str] = field(default_factory=list)  # sensor types needed to observe it


DEFAULT_CONCEPTS: dict[str, ConceptRule] = {
    r.concept: r
    for r in [
        ConceptRule("cooking", merge_gap_s=600, min_duration_s=120, requires=["power", "motion"]),
        ConceptRule("sleeping", merge_gap_s=1800, min_duration_s=1200, requires=["pressure", "motion"]),
        ConceptRule("watching tv", merge_gap_s=300, min_duration_s=300, requires=["power"]),
        ConceptRule("bathroom visit", merge_gap_s=60, requires=["motion"]),
        ConceptRule("baby crying", merge_gap_s=60, requires=["audio"]),
        ConceptRule("dog barking", merge_gap_s=60, requires=["audio"]),
        ConceptRule("parcel delivered", merge_gap_s=300, requires=["camera"]),
        ConceptRule("guest visit", merge_gap_s=900, requires=["contact", "camera"]),
        ConceptRule("away from home", merge_gap_s=600, min_duration_s=900, requires=["contact", "motion"]),
    ]
}


@dataclass
class HomeOntology:
    home_id: str
    entities: list[Entity]
    residents: list[str] = field(default_factory=list)
    pets: list[str] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)  # "kid" -> "Aarav", "grandma's room" -> "bedroom 2"
    concepts: dict[str, ConceptRule] = field(default_factory=lambda: dict(DEFAULT_CONCEPTS))

    def rooms(self) -> list[str]:
        return sorted({e.room for e in self.entities})

    def sensor_types(self) -> set[str]:
        return {e.sensor_type for e in self.entities}

    def can_observe(self, concept: str) -> bool:
        """Capability registry: is there any sensor that could observe this concept?"""
        rule = self.concepts.get(concept)
        if rule is None or not rule.requires:
            return True  # unknown concepts go to the open path, which reports its own confidence
        return bool(self.sensor_types() & set(rule.requires))

    def resolve(self, phrase: str) -> str:
        return self.aliases.get(phrase.lower().strip(), phrase)
