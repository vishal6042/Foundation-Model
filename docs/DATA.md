# Training data

What data the DomusFM reproduction (and later HomeFM) is trained and tested on, how it is structured on disk, and how it is turned into model input. §1–3 describe the DomusFM reproduction exactly as it runs; §4 is HomeFM only. For results see [DOMUSFM_REPRODUCTION.md](DOMUSFM_REPRODUCTION.md); for the long-term data plan see [DESIGN.md](DESIGN.md) §11.

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

All four end up in the same DomusFM arrays and 30-event windows (§3), so the model never sees these differences in file layout, only in sensors and activities.

## 3. How the DomusFM reproduction reads the data

This is the path the 77-home DomusFM run actually takes (`src/homefm/baselines/domusfm/run.py`, `load_datasets`). It follows the paper: sensors become text attributes, everything becomes ON/OFF, and windows are the last 30 events. **Home Tokens (§4) are not part of it.**

### 3.1 The pipeline for each group

```
Groups A, B, C (CASAS CSV)
  raw CSV ──load_casas_fast──► per-home arrays (.npz cache) ──build_domus_from_arrays──► DomusFM arrays ──► 30-event windows

Group D (UCI Home B)
  2 TXT files ──load_uci_adl──► small in-memory list ──build_domus_dataset──► DomusFM arrays ──► 30-event windows
```

- **CASAS goes straight to numpy arrays.** `casas_fast.py` parses each CSV into arrays and caches them in `data/cache/casas/<home>.npz`, so later runs load in about 0.1 s. It was written because building one Python object per event was too slow for 27M events.
- **UCI Home B** is tiny (4,668 events), so it is read into a small list of events first and then flattened the same way. The result is identical in form.

### 3.2 Step 1: raw file → per-home arrays

Each CASAS home becomes one set of arrays (`HomeArrays` in `casas_fast.py`):

| Array | Type | One entry per | Meaning |
|---|---|---|---|
| `entities` | list of (id, item, room, sensor type, modality) | sensor | e.g. (`KitchenAStove\|motion`, "stove", "kitchen", "motion", binary) |
| `ts` | float64 | event | Seconds; local time stored as if UTC |
| `entity` | int32 | event | Index into `entities` |
| `state` | int8 | event | OFF, ON, or NA for numeric readings |
| `value` | float32 | event | The number for temperature and other readings, NaN otherwise |
| `concepts` | list of text | activity | Activity names, lower-case, `_` → space |
| `ep_start`, `ep_end`, `ep_concept` | arrays | labelled episode | Start, end and activity of each labelled episode |

**Sensor name → text attributes** (`parse_sensor_name`): the room prefix is matched (`Kitchen`, `Bathroom`, `LoungeChair`, …), an instance letter becomes an ordinal (`B` → "second"), and the rest is split into words.

| Raw sensor name | Item | Room | Type (from message) |
|---|---|---|---|
| `KitchenAStove` | stove | kitchen | motion (ON/OFF) |
| `BathroomBToilet` | toilet | second bathroom | motion (ON/OFF) |
| `OutsideDoor` | outside door | entrance | door contact (OPEN/CLOSE) |
| `Kitchen` | kitchen area | kitchen | motion (ON/OFF) |
| `KitchenATemperature` | air | kitchen | temperature (number) |

**Labels → episodes:** a `Name="begin"` … `Name="end"` pair becomes one episode from begin to end; a plain `Name` on one event becomes an episode of zero length at that event. Milan and Aruba produce no episodes.

### 3.3 Step 2: per-home arrays → DomusFM arrays

`build_domus_from_arrays` (CASAS) and `build_domus_dataset` (UCI) in `src/homefm/baselines/domusfm/data.py` do three things:

1. **Make everything ON/OFF** (paper §3.1). Door OPEN/CLOSE become ON/OFF. For each temperature or other numeric sensor, threshold = midpoint of its 10th and 90th percentile; a reading above it is ON, below is OFF, and only changes are kept. The actual number is dropped.
2. **Encode each event as integers:** sensor index and status (0/1). Each sensor also keeps its three text attributes (item, sensor type, room), which a frozen MiniLM text encoder turns into vectors.
3. **Give each event an activity label:** the activity whose episode covers the event's time. If episodes overlap, the shortest one wins. Events outside any episode get `Other` (class 0).

Result per home (`DomusDataset`):

| Field | One entry per | Meaning |
|---|---|---|
| `sensor_ids`, `sensor_text` | sensor | Sensor id and its (item, sensor type, room) text |
| `activities` | activity | `["Other", …sorted activity names]` |
| `sensor` | event | Sensor index |
| `status` | event | 0 = OFF, 1 = ON |
| `ts` | event | Seconds |
| `label` | event | Activity index (0 = Other) |

The event counts in §1 and §2 are counted at this stage.

### 3.4 Step 3: DomusFM arrays → 30-event windows

`DomusWindows` cuts a window ending at every event (stride 1), so there is about one window per event: 27,255,501 events gave 27,253,268 windows. One window:

| Field | Shape | Meaning |
|---|---|---|
| `item`, `stype`, `room` | 30 | Text-attribute ids of each event's sensor |
| `status` | 30 | ON/OFF of each event |
| `dow`, `hour`, `sec` | 30 | Day of week, hour, second within the hour of each event |
| `label` | 1 | Activity at the window's **last** event (ADL task) |
| `next_counts` | 2 × number of sensors | How often each (sensor, ON/OFF) occurs in the **next** 30 events (next-30 task) |

- **Pretraining** (groups A + B) uses only the inputs: `item`, `stype`, `room`, `status`, `dow`, `hour`, `sec`. `label` and `next_counts` are ignored.
- **Fine-tuning and testing** (groups C + D) use `label` for ADL (weighted F1) and `next_counts` for next-30 (multiset F1), with 5 % or 30 % of the home's labels and 3 time-contiguous folds.

### 3.5 Worked example: five CASAS lines

Raw lines (from `tests/test_converters.py`):

```
2012-07-20,10:00:00.0,Kitchen,ON,Cook="begin"
2012-07-20,10:00:05.0,Kitchen,OFF
2012-07-20,10:01:00.0,OutsideDoor,OPEN
2012-07-20,10:10:00.0,Kitchen,ON,Cook="end"
2012-07-20,10:20:00.0,Bathroom,ON,Toilet
```

After steps 1 and 2:

| # | Time | Sensor text (item, type, room) | Status | Label |
|---|---|---|---|---|
| 0 | 10:00:00 | kitchen area, motion, kitchen | 1 | cook |
| 1 | 10:00:05 | kitchen area, motion, kitchen | 0 | cook |
| 2 | 10:01:00 | outside door, door contact, entrance | 1 | cook |
| 3 | 10:10:00 | kitchen area, motion, kitchen | 1 | cook |
| 4 | 10:20:00 | bathroom area, motion, bathroom | 1 | toilet |

`activities = ["Other", "cook", "toilet"]`. Rows 0–3 fall inside the cook episode (10:00–10:10), so all are labelled cook, including the door opening. That is how the paper's labelling works: the label is about time, not about which sensor fired. In a real file, step 3 would then take each run of 30 consecutive rows like these as one window.

## 4. HomeFM only: Home Tokens and time-based windows

**Not used by the DomusFM reproduction.** This is the format HomeFM (variants A–G) reads. It is described here because the HomeFM comparison run on the same homes uses the same raw files.

### 4.1 Why HomeFM needs a richer format

DomusFM's format drops things HomeFM needs: numeric readings (cut to ON/OFF), confidence of a detection, sound and camera tags, and who caused an event. The Home Token format keeps them (`src/homefm/schema/events.py`, DESIGN.md §6.1):

```
HomeStream            one home
├── home_id           "hh101"
├── entities[]        Entity(entity_id, item, room, sensor_type, modality)
│                     text seen by the model: "stove in kitchen, motion sensor"
├── tokens[]          HomeToken(ts, entity, state, value, confidence, source, person_id, home_id)
├── episodes[]        Episode(concept, start_ts, end_ts, room, caption)
│                     caption = "the resident is doing: <concept>"
├── anomalies[]       AnomalyLabel(kind, start_ts, end_ts, description)
└── tz_offset_hours
```

| Field | Values |
|---|---|
| `state` | `ON`, `OFF`, or `NA` for numeric and vector readings |
| `value` | The number for numeric sensors (e.g. 21 °C), kept as a number |
| `modality` | `BINARY`, `SCALAR`, `AUDIO_TAG`, `VISION_DET`, `EMBEDDING` |
| `confidence` | 1.0 for sensors; below 1 for audio/vision detector guesses |
| `person_id` | Always empty for public data |
| `anomalies` | Empty for public data; filled by synthetic homes with injected faults |

At corpus scale HomeFM reads the same per-home CASAS arrays as DomusFM (§3.2, via `from_home_arrays` in `windowing.py`), which carry the same fields. Individual Home Token objects are used for UCI Home B, synthetic homes and tests.

### 4.2 Time-based windows

`src/homefm/data/windowing.py` does **not** convert to ON/OFF, and cuts windows by time instead of by event count:

| Setting | Default (`WindowConfig`) | 77-home config (`configs/homefm_corpus.yaml`) |
|---|---|---|
| Window length | 3,600 s | 1,800 s (30 minutes) |
| Moment length | 60 s | 60 s |
| Stride | 900 s | default |
| Max events per window | — | 256 (keep the most recent when busier) |

Each window is split into fixed 1-minute moments, so a busy and a quiet half hour both become 30 moments. Numeric values stay numbers, and minutes with no events are kept as empty moments. Each window gets one caption for language alignment: the activity that covers most of it, or "a quiet period at home" if none does.

## 5. What this data cannot provide

| Missing | Why it matters | Planned source (DESIGN.md §11) |
|---|---|---|
| Audio, camera, power readings | CASAS is motion, doors and temperature only (L2, L9) | Energy datasets (REDD, UK-DALE, REFIT), on-device experts, pilot homes |
| Who did what | `person_id` is empty; homes are mostly single-resident (L4) | MuRAL, ARAS, synthetic multi-occupant homes |
| Fine-grained sensors | 2025 release merged per-sensor ids into room-level names | Original annotated releases, Kasteren, Orange4Home |
| Device faults and anomalies | No anomaly labels in public data (L6) | Synthetic homes with injected faults |
| Modern device types | No locks, cameras or robot vacuums (L10) | Pilot homes |

Converters for Kasteren, Orange4Home, MuRAL, ARAS and the energy datasets are not written yet (`src/homefm/data/converters/README.md`).
