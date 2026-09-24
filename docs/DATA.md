# Training data

What data HomeFM and the DomusFM baseline are trained and tested on, how it is structured on disk, and how it is turned into model input. For results see [DOMUSFM_REPRODUCTION.md](DOMUSFM_REPRODUCTION.md); for the long-term data plan see [DESIGN.md](DESIGN.md) §11.

## 1. The four data groups at a glance

All real data comes from two public sources: the **CASAS 2025 release** on Zenodo (record 15708568, CC BY 4.0) and the **UCI ADL Binary dataset** (UCI ML Repository, dataset 271). The split is set in `configs/domusfm_corpus.yaml` (`held_out`, `pretrain_only`). Counts below come from the run log (`results/domusfm_corpus/train.log`, line 5) and are events **after** DomusFM's ON/OFF conversion.

| Group | Homes | Used for | Labels? | Raw format | Events |
|---|---|---|---|---|---|
| **A. CASAS pretraining homes** | 75 | Pretraining only | Yes (ignored in pretraining); mv001 has none | CASAS CSV, 5 columns (§2.1) | 25,231,288 |
| **B. CASAS Milan and Aruba** | 2 | Pretraining only | No | CASAS CSV, 4 columns (§2.2) | 2,024,213 |
| **C. CASAS test homes** | 6 | Fine-tuning and testing | Yes | CASAS CSV, 5 columns (§2.1) | 803,768 |
| **D. UCI ADL Home B** | 1 | Fine-tuning and testing | Yes | Two tab-separated text files (§2.3) | 4,668 |
| **Pretraining total (A + B)** | **77** | | | | **27,255,501** |

Groups A and C have the **same file format**; they differ only in which homes are held out. No test-home data is ever seen in pretraining.

Where the files go (git-ignored):

| Group | Path | Converter |
|---|---|---|
| A, C | `data/raw/casas/labeled/<home>.csv` (from `labeled_data.zip`, 236 MB) | `casas_fast.py` (fast, cached as `.npz` in `data/cache/casas/`); reference: `casas_zenodo.py` |
| B | `data/raw/casas/data/milan.csv`, `aruba.csv` (from `data.zip`) | same |
| D | `data/raw/uci/UCI ADL Binary Dataset/OrdonezB_Sensors.txt`, `OrdonezB_ADLs.txt` | `uci_adl.py` |

## 2. Raw data formats, group by group

### 2.1 Groups A and C: CASAS labelled homes

**File:** one CSV per home, one row per sensor event, no header row, comma-separated.

```
date,       time,             sensor,              message, label (optional)
2012-07-20, 10:00:00.0,       Kitchen,             ON,      Cook="begin"
2012-07-20, 10:00:05.0,       Kitchen,             OFF
2012-07-20, 10:01:00.0,       OutsideDoor,         OPEN
2012-07-20, 10:02:00.0,       KitchenATemperature, 21
2012-07-20, 10:10:00.0,       Kitchen,             ON,      Cook="end"
2012-07-20, 10:20:00.0,       Bathroom,            ON,      Toilet
```

(Format example from `tests/test_converters.py`; real files follow the same layout.)

| Column | Type | Meaning |
|---|---|---|
| 1 date | `YYYY-MM-DD` | Local date |
| 2 time | `HH:MM:SS[.ffffff]` | Local time, microseconds optional |
| 3 sensor | text | A **location name**, not an ID: room + optional instance letter + item, e.g. `Kitchen`, `KitchenAStove`, `BathroomBToilet`, `OutsideDoor` |
| 4 message | `ON`/`OFF`, `OPEN`/`CLOSE`, or a number | `ON`/`OFF` → motion sensor; `OPEN`/`CLOSE` → door contact; a number → temperature (if the name contains `Temperature`) or another scalar reading |
| 5 label | text, optional | Activity annotation: `Name="begin"` … `Name="end"` around an episode, or a plain `Name` on a single event |

**What each home looks like:**

- 5–13 distinct sensor names (room-level; the original per-sensor ids such as M001 were merged in the 2025 release).
- Motion, door and temperature sensors only. No power, audio or camera.
- One resident is assumed; no column says who caused an event.

**Group A: the 75 pretraining homes**, by CASAS home family (activity classes include "Other"):

| Family | Homes | Events | Activity classes per home |
|---|---|---|---|
| hh | 24: hh102, hh104, hh106–hh109, hh111–hh118, hh120, hh121, hh123–hh130 | 9,291,578 | 24–38 |
| tm | 26: tm001–tm011, tm013–tm022, tm026, tm029, tm035, tm037, tm038 | 6,822,062 | 16–36 |
| mv | 1: mv001 | 3,368,850 | 1 (no labels) |
| ihs | 5: ihs06–ihs09, ihs11 | 2,387,214 | 28–34 |
| mn | 8: mn50, mn51, mn57, mn61, mn71, mn77, mn82, mn83 | 1,513,126 | 26–35 |
| mva | 4: mva001–mva004 | 1,046,291 | 26–27 |
| rw | 7: rw101, rw103–rw107, rw110 | 802,167 | 30–38 |
| **Total** | **75** | **25,231,288** | |

Homes vary a lot in size: from 13,374 events (hh124) to 3,368,850 (mv001). Labels in these homes are loaded but not used, because pretraining is self-supervised.

**Group C: the 6 CASAS test homes:**

| Home | Events | Activity classes (incl. "Other") |
|---|---|---|
| hh101 | 222,858 | 35 |
| hh103 | 112,474 | 33 |
| hh105 | 147,074 | 34 |
| hh110 | 98,686 | 32 |
| hh119 | 92,740 | 33 |
| hh122 | 129,936 | 34 |
| **Total** | **803,768** | |

These stand in for the paper's Kasteren A/C, Orange4Home and MuRAL targets, which were not available.

### 2.2 Group B: CASAS Milan and Aruba

**File:** same CSV layout as §2.1, but **only 4 columns**: there is no label column.

```
date,       time,             sensor,       message
2010-11-04, 00:03:50.209589,  Bedroom,      ON
2010-11-04, 00:03:57.399391,  Bedroom,      OFF
2010-11-04, 00:15:08.984841,  Kitchen,      ON
```

(Illustrative lines showing the layout only.)

| Home | Events | Sensor names | Labels |
|---|---|---|---|
| Milan | 421,392 | 9 | None |
| Aruba | 1,602,821 | 10 | None |

- The original annotated versions of these datasets (with per-sensor ids and activity labels, as used in the DomusFM paper) are no longer on the CASAS site.
- The 2025 release merges sensors into location names such as `Kitchen`, `Bedroom`, `LoungeChair`.
- Because there are no labels, they can only be used for pretraining.

### 2.3 Group D: UCI ADL Binary, Home B (test home)

**Files:** two tab-separated text files, each with a header line and a dashed line, then one row per interval.

`OrdonezB_Sensors.txt`: one row per sensor activation (start and end of the ON period):

```
Start time           End time             Location  Type      Place
----------           --------             --------  ----      -----
2012-11-11 21:14:21  2012-11-12 00:21:49  Seat      Pressure  Living
```

| Column | Meaning | Becomes |
|---|---|---|
| Start time, End time | `YYYY-MM-DD HH:MM:SS` | An ON event at start and an OFF event at end |
| Location | The item (e.g. Seat, Door, Toilet) | Entity item text |
| Type | PIR, Magnetic, Pressure, Electric, Flush | motion, door contact, pressure, power plug, flush |
| Place | The room (e.g. Living, Kitchen, Bathroom) | Entity room text |

`OrdonezB_ADLs.txt`: one row per activity:

```
Start time           End time             Activity
----------           --------             --------
2012-11-11 21:14:00  2012-11-12 00:22:59  Spare_Time/TV
```

| Column | Meaning | Becomes |
|---|---|---|
| Start time, End time | `YYYY-MM-DD HH:MM:SS` | Episode start and end |
| Activity | Activity name, e.g. Spare_Time/TV, Sleeping, Toileting | Episode concept, lower-cased: "spare time/tv" |

| Home | Events (ON + OFF) | Activity classes (incl. "Other") |
|---|---|---|
| UCI Home B | 4,668 | 11 |

- Very small compared with the CASAS homes (4,668 events vs about 100,000–220,000 for the CASAS test homes).
- It has a different sensor mix (pressure, power plug, flush) and different room and activity names from CASAS, so it is the hardest transfer test.
- It is the only target shared with the DomusFM paper, so it is the only one whose numbers can be compared with the paper's.

### 2.4 Same input, four sources: side by side

| | A. CASAS pretraining | B. Milan / Aruba | C. CASAS test | D. UCI Home B |
|---|---|---|---|---|
| File | 1 CSV per home | 1 CSV per home | 1 CSV per home | 2 TXT files |
| Separator | comma | comma | comma | tab |
| One row = | one event | one event | one event | one ON period, or one activity |
| Sensor naming | location names (`KitchenAStove`) | location names (`Kitchen`) | location names | item + type + room columns |
| Sensor types | motion, door, temperature | motion, door, temperature | motion, door, temperature | motion, door, pressure, power plug, flush |
| Scalar values | temperature | temperature | temperature | none |
| Activity labels | inline begin/end or per event | none | inline begin/end or per event | separate file with intervals |
| Who did it | not recorded | not recorded | not recorded | not recorded |
| Used for | pretraining | pretraining | fine-tune + test | fine-tune + test |

All four are converted into the same Home Token format (§3), so the models never see these differences in file layout, only in sensors and activities.

## 3. The common format all groups are converted into (Home Tokens)

Every source is converted into the same structure (`src/homefm/schema/events.py`, DESIGN.md §6.1):

```
HomeStream            one home
├── home_id           "hh101"
├── entities[]        Entity(entity_id, item, room, sensor_type, modality)
│                     the model only sees the text: "stove in kitchen, motion sensor"
├── tokens[]          HomeToken(ts, entity, state, value, confidence, source, person_id, home_id)
├── episodes[]        Episode(concept, start_ts, end_ts, room, caption)
│                     caption = "the resident is doing: <concept>"
├── anomalies[]       AnomalyLabel(kind, start_ts, end_ts, description)
└── tz_offset_hours
```

| Field | Values |
|---|---|
| `ts` | Unix seconds. CASAS local time is stored as if UTC, so hour and weekday features stay local |
| `state` | `ON`, `OFF`, or `NA` for scalar and vector readings |
| `value` | The number for scalar sensors (e.g. temperature), otherwise empty |
| `modality` | `BINARY`, `SCALAR`, `AUDIO_TAG`, `VISION_DET`, `EMBEDDING` |
| `person_id` | Always empty for public data: the datasets do not say who caused an event |
| `anomalies` | Empty for public data; filled by synthetic homes with injected faults |

**How CASAS names become text** (`parse_sensor_name`): the room prefix is matched (`Kitchen`, `Bathroom`, `LoungeChair`, …), an instance letter becomes an ordinal (`B` → "second"), and the rest is split into words: `KitchenAStove` → item "stove", room "kitchen"; `BedroomBArea` → "second bedroom area". No hand-written sensor maps are needed.

**How labels become episodes:** `begin`/`end` pairs become one episode with a start and end; a single labelled event becomes an episode of zero length. Episodes are used only for fine-tuning and testing, never in pretraining.

## 4. What DomusFM trains on

`src/homefm/baselines/domusfm/data.py` reshapes Home Tokens into the paper's format:

1. **Binarise.** Scalar sensors become virtual ON/OFF events: threshold = midpoint of the sensor's 10th and 90th percentile, and only changes are emitted. The actual reading is discarded (DESIGN.md L2).
2. **Encode each event as integers:** sensor index, status (0/1), timestamp, and the ids of its three text attributes (item, sensor type, room), which are embedded by a frozen text encoder.
3. **Label each event** with the activity whose episode covers it. If episodes overlap, the shortest wins. Events outside any episode get `Other` (class 0).
4. **Cut windows of the last 30 events, sliding by one event.** Because the stride is one event, there is about one window per event.

One window:

| Field | Shape | Meaning |
|---|---|---|
| `item`, `stype`, `room` | 30 | Text-attribute ids of each event's sensor |
| `status` | 30 | ON/OFF |
| `dow`, `hour`, `sec` | 30 | Day of week, hour, second within the hour |
| `label` | 1 | Activity at the window's last event (ADL task) |
| `next_counts` | 2 × number of sensors | Counts of each (sensor, ON/OFF) in the next 30 events (next-30 task) |

Tasks: **ADL** (weighted F1) and **next-30** (multiset F1), fine-tuned on 5 % and 30 % of the target home's labels, with 3 time-contiguous folds.

## 5. What HomeFM trains on

`src/homefm/data/windowing.py` keeps the Home Tokens as they are (no binarisation) and cuts **time-based** windows instead of event-count windows:

| Setting | Default (`WindowConfig`) | 77-home config (`configs/homefm_corpus.yaml`) |
|---|---|---|
| Window length | 3,600 s | 1,800 s (30 minutes) |
| Moment length | 60 s | 60 s |
| Stride | 900 s | default |
| Max events per window | — | 256 (keep the most recent when busier) |

Each window is split into fixed 1-minute moments, so a busy and a quiet half hour both become 30 moments. Scalar values stay numbers, and minutes with no events are kept as empty moments. Each window gets one caption for language alignment: the activity that covers most of it, or "a quiet period at home" if none does.

## 6. What this data cannot provide

| Missing | Why it matters | Planned source (DESIGN.md §11) |
|---|---|---|
| Audio, camera, power readings | CASAS is motion, doors and temperature only (L2, L9) | Energy datasets (REDD, UK-DALE, REFIT), on-device experts, pilot homes |
| Who did what | `person_id` is empty; homes are mostly single-resident (L4) | MuRAL, ARAS, synthetic multi-occupant homes |
| Fine-grained sensors | 2025 release merged per-sensor ids into room-level names | Original annotated releases, Kasteren, Orange4Home |
| Device faults and anomalies | No anomaly labels in public data (L6) | Synthetic homes with injected faults |
| Modern device types | No locks, cameras or robot vacuums (L10) | Pilot homes |

Converters for Kasteren, Orange4Home, MuRAL, ARAS and the energy datasets are not written yet (`src/homefm/data/converters/README.md`).
