"""Synthetic smart-home simulator (DESIGN.md §11).

Generates a multi-modal Home Token stream with ground-truth episodes, captions and injected faults.
It is deliberately simple: its job is to make every objective, head and metric in the scaffold
runnable end to end before real datasets are wired in. It is not a substitute for real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from homefm.schema import AnomalyLabel, Entity, Episode, HomeStream, HomeToken, Modality, Source, State

BASE_TS = 1767571200.0  # 2026-01-05 00:00 UTC, a Monday
DAY = 86400.0
MIN = 60.0
HOUR = 3600.0

CAPTIONS: dict[str, list[str]] = {
    "cooking": ["someone is cooking in the kitchen", "a meal is being prepared on the stove"],
    "sleeping": ["the resident is sleeping in bed", "someone is asleep in the bedroom"],
    "watching tv": ["someone is watching tv in the living room", "the television is on in the lounge"],
    "bathroom visit": ["someone is using the bathroom", "a person is in the bathroom"],
    "baby crying": ["the baby is crying", "an infant is crying in the nursery"],
    "dog barking": ["the dog is barking", "the pet dog barks loudly"],
    "parcel delivered": ["a parcel was delivered at the front door", "a courier dropped off a package"],
    "guest visit": ["a guest is visiting the home", "visitors are in the living room"],
    "away from home": ["nobody is home", "the residents are out of the house"],
}


def make_entities() -> dict[str, Entity]:
    B, S, A, V = Modality.BINARY, Modality.SCALAR, Modality.AUDIO_TAG, Modality.VISION_DET
    specs = {
        "kitchen_motion": ("kitchen", "kitchen", "motion", B),
        "stove": ("stove", "kitchen", "power", S),
        "fridge_door": ("fridge door", "kitchen", "contact", B),
        "fridge_power": ("fridge", "kitchen", "power", S),
        "living_motion": ("living room", "living room", "motion", B),
        "tv": ("television", "living room", "power", S),
        "dog_audio": ("dog barking", "living room", "audio", A),
        "bedroom_motion": ("bedroom", "bedroom", "motion", B),
        "bed": ("bed", "bedroom", "pressure", B),
        "bathroom_motion": ("bathroom", "bathroom", "motion", B),
        "humidity": ("air", "bathroom", "humidity", S),
        "nursery_audio": ("baby crying", "nursery", "audio", A),
        "front_door": ("front door", "entrance", "contact", B),
        "entrance_motion": ("hallway", "entrance", "motion", B),
        "cam_person": ("person", "entrance", "camera", V),
        "cam_parcel": ("parcel", "entrance", "camera", V),
        "meter": ("whole home electricity", "utility", "energy meter", S),
    }
    return {k: Entity(k, item, room, stype, mod) for k, (item, room, stype, mod) in specs.items()}


@dataclass
class SimConfig:
    days: int = 14
    has_baby: bool = True
    has_dog: bool = True
    away_p: float = 0.6
    inject_faults: bool = True


class _Sim:
    def __init__(self, home_id: str, cfg: SimConfig, rng: np.random.Generator):
        self.home_id, self.cfg, self.rng = home_id, cfg, rng
        self.ent = make_entities()
        self.tokens: list[HomeToken] = []
        self.episodes: list[Episode] = []
        self.anomalies: list[AnomalyLabel] = []
        self.loads: list[tuple[float, float, float]] = []  # (start, end, watts) for the energy meter
        self.fridge_on_s, self.fridge_watts = 12 * MIN, 110.0

    # ---- primitives -------------------------------------------------------------------------
    def emit(self, ts, key, state=State.NA, value=None, conf=1.0, source=Source.SENSOR):
        self.tokens.append(HomeToken(float(ts), self.ent[key], state, value, None, conf, source, None, self.home_id))

    def binary(self, ts, key, on_for):
        self.emit(ts, key, State.ON)
        self.emit(ts + on_for, key, State.OFF)

    def motion(self, key, start, end, mean_gap):
        t = start + self.rng.exponential(mean_gap / 2)
        while t < end:
            self.binary(t, key, self.rng.uniform(2, 6))
            t += self.rng.exponential(mean_gap)

    def power(self, key, start, end, watts, every=5 * MIN):
        self.loads.append((start, end, watts))
        t = start
        while t < end:
            self.emit(t, key, value=max(0.0, watts * self.rng.normal(1, 0.05)))
            t += every
        self.emit(end, key, value=0.0)

    def audio(self, key, start, end, gap=(4, 10)):
        t = start
        while t < end:
            self.emit(t, key, State.ON, conf=self.rng.uniform(0.6, 0.95), source=Source.EXPERT)
            t += self.rng.uniform(*gap)

    def episode(self, concept, start, end, room):
        cap = CAPTIONS[concept][self.rng.integers(len(CAPTIONS[concept]))]
        self.episodes.append(Episode(concept, float(start), float(end), room, 1.0, [], cap))

    # ---- activities -------------------------------------------------------------------------
    def cooking(self, start, dur):
        end = start + dur
        self.motion("kitchen_motion", start, end, 45)
        self.power("stove", start + 2 * MIN, end - MIN, self.rng.uniform(1200, 2000), every=2 * MIN)
        for _ in range(self.rng.integers(2, 5)):
            self.binary(self.rng.uniform(start, end), "fridge_door", self.rng.uniform(5, 40))
        self.episode("cooking", start, end, "kitchen")

    def bathroom(self, start, dur, shower=False):
        end = start + dur
        self.motion("bathroom_motion", start, end, 40)
        if shower:
            h, t = 50.0, start
            while t < end + 20 * MIN:
                h = h + (85 - h) * 0.3 if t < end else h + (50 - h) * 0.15
                self.emit(t, "humidity", value=h + self.rng.normal(0, 1))
                t += MIN
        self.episode("bathroom visit", start, end, "bathroom")

    def tv(self, start, end):
        self.power("tv", start, end, self.rng.uniform(80, 140))
        self.motion("living_motion", start, end, 10 * MIN)
        self.episode("watching tv", start, end, "living room")

    def leave_return(self, leave, back):
        for t in (leave, back):
            self.motion("entrance_motion", t - MIN, t + MIN, 20)
            self.binary(t, "front_door", self.rng.uniform(5, 15))
        self.episode("away from home", leave, back, "entrance")

    def cries(self, lo, hi, n):
        for _ in range(n):
            s = self.rng.uniform(lo, hi)
            e = s + self.rng.uniform(1, 6) * MIN
            self.audio("nursery_audio", s, e)
            self.episode("baby crying", s, e, "nursery")

    def barks(self, lo, hi, n):
        for _ in range(n):
            s = self.rng.uniform(lo, hi)
            e = s + self.rng.uniform(0.5, 3) * MIN
            self.audio("dog_audio", s, e, gap=(3, 8))
            if self.rng.random() < 0.5:  # pet motion confuses PIR-only occupancy
                self.motion("living_motion", s, e, 30)
            self.episode("dog barking", s, e, "living room")

    def parcel(self, t, home):
        self.emit(t, "cam_person", State.ON, conf=self.rng.uniform(0.7, 0.99), source=Source.EXPERT)
        self.emit(t + self.rng.uniform(20, 60), "cam_parcel", State.ON, conf=self.rng.uniform(0.6, 0.95),
                  source=Source.EXPERT)
        self.episode("parcel delivered", t, t + 2 * MIN, "entrance")
        if home and self.rng.random() < 0.7:
            pick = t + self.rng.uniform(3, 30) * MIN
            self.binary(pick, "front_door", self.rng.uniform(5, 15))
            self.motion("entrance_motion", pick - 30, pick + 30, 15)

    def guest(self, start, dur):
        end = start + dur
        self.emit(start - 20, "cam_person", State.ON, conf=0.9, source=Source.EXPERT)
        for t in (start, end):
            self.binary(t, "front_door", self.rng.uniform(5, 20))
            self.motion("entrance_motion", t - MIN, t + MIN, 15)
        self.motion("living_motion", start, end, 90)
        self.episode("guest visit", start, end, "living room")

    def fridge_cycles(self, start, end, degrade_from):
        t = start + self.rng.uniform(0, 40 * MIN)
        while t < end:
            on_s, watts = self.fridge_on_s, self.fridge_watts
            if t >= degrade_from:
                on_s, watts = 25 * MIN, 140.0
            self.power("fridge_power", t, t + on_s * self.rng.normal(1, 0.1), watts * self.rng.normal(1, 0.05))
            t += on_s + self.rng.normal(28, 4) * MIN

    def meter(self, start, end):
        loads = np.array(self.loads) if self.loads else np.zeros((0, 3))
        t = start
        while t < end:
            active = loads[(loads[:, 0] <= t) & (loads[:, 1] > t), 2].sum() if len(loads) else 0.0
            self.emit(t, "meter", value=150 + active + self.rng.normal(0, 10))
            t += 5 * MIN

    # ---- faults (DESIGN.md §9.2, §9.3) ------------------------------------------------------
    def stuck_sensor(self, start, dur):
        t = start
        while t < start + dur:
            self.emit(t, "bathroom_motion", State.ON)
            t += self.rng.uniform(20, 40)
        self.anomalies.append(AnomalyLabel("stuck_sensor", start, start + dur, "bathroom motion stuck ON"))

    def night_door(self, t):
        self.motion("entrance_motion", t - MIN, t + 2 * MIN, 20)
        self.binary(t, "front_door", self.rng.uniform(10, 60))
        self.anomalies.append(AnomalyLabel("night_door", t - MIN, t + 2 * MIN, "front door opened at night"))

    # ---- day loop ---------------------------------------------------------------------------
    def run(self) -> HomeStream:
        cfg, rng = self.cfg, self.rng
        D = cfg.days
        end_ts = BASE_TS + D * DAY
        wake = [BASE_TS + d * DAY + rng.normal(6.5, 0.4) * HOUR for d in range(D + 1)]
        bed = [BASE_TS + d * DAY + rng.normal(23.0, 0.4) * HOUR for d in range(D)]
        degrade_from = BASE_TS + int(0.7 * D) * DAY if cfg.inject_faults else float("inf")
        fault_days = set(rng.choice(D, size=min(3, D), replace=False).tolist()) if cfg.inject_faults else set()
        fault_kinds = ["stuck", "night_door", "missed_meal"]
        faults = dict(zip(sorted(fault_days), fault_kinds))

        # sleep episodes span midnight
        sleeps = [(BASE_TS, wake[0])] + [(bed[d], min(wake[d + 1], end_ts)) for d in range(D)]
        for s, e in sleeps:
            if e - s < HOUR:
                continue
            self.binary(s, "bed", e - s)
            self.motion("bedroom_motion", s - 5 * MIN, s, 60)
            self.episode("sleeping", s, e, "bedroom")
            if e - s > 3 * HOUR and rng.random() < 0.5:  # night bathroom trip
                t = rng.uniform(s + HOUR, e - HOUR)
                self.motion("bedroom_motion", t - MIN, t, 20)
                self.bathroom(t, rng.uniform(3, 6) * MIN)
            if cfg.has_baby and e - s > 3 * HOUR and rng.random() < 0.5:
                self.cries(s + HOUR, e - HOUR, 1)

        for d in range(D):
            day0, w, b = BASE_TS + d * DAY, wake[d], bed[d]
            fault = faults.get(d)
            self.motion("bedroom_motion", w, w + 5 * MIN, 60)
            self.bathroom(w + 5 * MIN, rng.uniform(10, 20) * MIN, shower=True)
            self.cooking(w + rng.uniform(30, 45) * MIN, rng.uniform(15, 25) * MIN)

            away = d % 7 < 5 and rng.random() < cfg.away_p
            if away:
                leave, back = day0 + rng.normal(8.8, 0.3) * HOUR, day0 + rng.normal(17.5, 0.6) * HOUR
                self.leave_return(leave, back)
                home_spans = [(w, leave), (back, b)]
                if cfg.has_dog:
                    self.barks(leave, back, rng.poisson(2))
            else:
                self.cooking(day0 + rng.normal(12.5, 0.4) * HOUR, rng.uniform(25, 40) * MIN)
                home_spans = [(w, b)]
                if rng.random() < 0.3:
                    s = day0 + rng.uniform(14, 16) * HOUR
                    self.tv(s, s + rng.uniform(30, 90) * MIN)
                if rng.random() < 0.3:
                    self.guest(day0 + rng.uniform(15, 18) * HOUR, rng.uniform(1, 2.5) * HOUR)

            if fault != "missed_meal":
                self.cooking(day0 + rng.normal(18.8, 0.4) * HOUR, rng.uniform(30, 50) * MIN)
            else:
                t = day0 + 18.8 * HOUR
                self.anomalies.append(AnomalyLabel("missed_routine", t - HOUR, t + HOUR, "no dinner preparation"))
            if rng.random() < 0.8:
                s = day0 + rng.normal(20.0, 0.3) * HOUR
                if b - s > 30 * MIN:
                    self.tv(s, b - 15 * MIN)
            for _ in range(rng.poisson(1.5)):
                self.bathroom(rng.uniform(w + 2 * HOUR, b - HOUR), rng.uniform(2, 6) * MIN)
            for lo, hi in home_spans:  # background presence
                self.motion("living_motion", lo, hi, 8 * MIN)
            if cfg.has_baby:
                self.cries(w, b, rng.poisson(2.5))
            if cfg.has_dog:
                self.barks(w, b, rng.poisson(1.5))
            for _ in range(rng.poisson(0.7)):
                t = day0 + rng.uniform(10, 17) * HOUR
                self.parcel(t, home=not away or not (leave <= t <= back))
            if fault == "stuck":
                self.stuck_sensor(day0 + rng.uniform(9, 15) * HOUR, 3 * HOUR)
            elif fault == "night_door":
                self.night_door(day0 + DAY + rng.uniform(2, 4) * HOUR)

        self.fridge_cycles(BASE_TS, end_ts, degrade_from)
        if cfg.inject_faults and degrade_from < end_ts:
            self.anomalies.append(AnomalyLabel("device_degradation", degrade_from, end_ts, "fridge compressor cycles longer"))
        self.meter(BASE_TS, end_ts)

        tokens = [t for t in self.tokens if BASE_TS <= t.ts < end_ts]
        return HomeStream(self.home_id, list(self.ent.values()), tokens, self.episodes, self.anomalies).sort()


def simulate_home(home_id: str, cfg: SimConfig | None = None, seed: int = 0) -> HomeStream:
    return _Sim(home_id, cfg or SimConfig(), np.random.default_rng(seed)).run()


def simulate_homes(n_homes: int, days: int = 14, seed: int = 0, inject_faults: bool = True) -> list[HomeStream]:
    rng = np.random.default_rng(seed)
    homes = []
    for i in range(n_homes):
        cfg = SimConfig(days=days, has_baby=bool(rng.random() < 0.6), has_dog=bool(rng.random() < 0.6),
                        away_p=float(rng.uniform(0.3, 0.9)), inject_faults=inject_faults)
        homes.append(simulate_home(f"sim_{i:03d}", cfg, seed=int(rng.integers(1 << 31))))
    return homes
