"""Data cleaning before segmentation (paper Appendix A), plus fixes for faults found in the EDA
(notebooks/domusfm_eda.ipynb). Switched on by `datasets.clean: true`; off by default so earlier runs reproduce.

Rules, applied per dataset in this order:
  1. drop_types        Drop sensors of the given types. Default: "scalar", the undocumented numeric codes that the
                       2025 CASAS release puts on motion-sensor names (e.g. `BedroomABed,67`); the loader turns them
                       into invented ON/OFF events. Temperature sensors are kept.
  2. merge_case        Merge sensor ids that differ only in letter case (tm004: DiningroomAArea / DiningRoomAArea).
  3. alternate         [paper, Appendix A] A binary sensor's events must alternate ON/OFF; a repeated state is a
                       duplicate event and is removed. This also removes exact duplicate rows.
Malformed readings (`OF`, `ONf`, `0ta082`, …) are already skipped by the loaders.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .data import DomusDataset


def clean_dataset(d: DomusDataset, drop_types: tuple[str, ...] = ("scalar",), merge_case: bool = True,
                  alternate: bool = True) -> tuple[DomusDataset, dict]:
    """Returns the cleaned dataset and a report of how many events each rule removed."""
    report = {"events_in": d.n_events, "sensors_in": len(d.sensor_ids)}
    keep = np.ones(d.n_events, bool)

    # 1 + 2: build the new sensor table (dropped sensors map to -1, case duplicates to one id)
    new_ids, new_text, remap, seen = [], [], np.full(len(d.sensor_ids), -1, np.int64), {}
    for i, (sid, text) in enumerate(zip(d.sensor_ids, d.sensor_text)):
        if text[1] in drop_types:
            continue
        key = sid.lower() if merge_case else sid
        if key not in seen:
            seen[key] = len(new_ids)
            new_ids.append(sid)
            new_text.append(text)
        remap[i] = seen[key]
    sensor = remap[d.sensor]
    report["dropped_type_events"] = int((sensor < 0).sum())
    report["merged_case_sensors"] = len(d.sensor_ids) - len(new_ids) - sum(t[1] in drop_types for t in d.sensor_text)
    keep &= sensor >= 0

    # 3: per sensor, drop an event whose status repeats the previous kept event of that sensor
    if alternate:
        idx = np.flatnonzero(keep)
        order = idx[np.lexsort((d.ts[idx], sensor[idx]))]  # by sensor, then time (stable within equal times)
        s, st = sensor[order], d.status[order]
        repeat = np.zeros(len(order), bool)
        repeat[1:] = (s[1:] == s[:-1]) & (st[1:] == st[:-1])
        keep[order[repeat]] = False
        report["dropped_repeat_events"] = int(repeat.sum())

    out = replace(d, sensor_ids=new_ids, sensor_text=new_text, sensor=sensor[keep], status=d.status[keep],
                  ts=d.ts[keep], label=d.label[keep])
    report.update(events_out=out.n_events, sensors_out=len(new_ids))
    return out, report
