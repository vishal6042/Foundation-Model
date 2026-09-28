# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DomusFM data: exploratory analysis and cleaning
#
# Every dataset the DomusFM reproduction trains or tests on, before and after cleaning.
# DomusFM only; HomeFM is not covered here.
#
# **Run:** `.venv/Scripts/python notebooks/domusfm_eda.py` (script) or open `notebooks/domusfm_eda.ipynb`.
# Figures are written to `notebooks/figures/`, tables to `notebooks/tables/`.
#
# | Group | Datasets | Role in the run |
# |---|---|---|
# | CASAS pretraining homes | 75 homes (hh, tm, mn, rw, ihs, mva, mv families), 2025 Zenodo release | pretraining only |
# | CASAS Milan, Aruba | 2 homes, no labels | pretraining only |
# | CASAS test homes | hh101, hh103, hh105, hh110, hh119, hh122 | fine-tuning and testing |
# | Paper test datasets | UCI Home B, Kasteren A, Kasteren C, MuRAL | fine-tuning and testing |
#
# The paper (arXiv 2602.01910, §5) used Milan, Aruba, Kasteren A and C, UCI B, Orange4Home and MuRAL.
# Orange4Home is available only by email request and is not included.

# %%
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))

from homefm.baselines.domusfm.clean import clean_dataset  # noqa: E402
from homefm.baselines.domusfm.run import load_datasets  # noqa: E402
from homefm.train.pretrain import load_config  # noqa: E402

FIG, TAB = ROOT / "notebooks" / "figures", ROOT / "notebooks" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# Reference categorical palette (fixed order, light mode) and chart styling: thin marks, recessive axes.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "legend.frameon": False, "axes.axisbelow": True,
})
GROUPS = ["CASAS pretraining", "CASAS Milan/Aruba", "CASAS test", "Paper test"]
GROUP_COLOR = dict(zip(GROUPS, SLOTS))


def save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    plt.show()


# %% [markdown]
# ## 1. Load every dataset
#
# The loaders are the ones the training run uses (`homefm.baselines.domusfm.run.load_datasets`), so what is
# analysed here is exactly what the model sees: events after DomusFM's ON/OFF conversion, with one activity label per
# event (label 0 = "Other"). Cleaning is off at this point.

# %%
cfg = load_config(str(ROOT / "configs/domusfm_corpus_clean.yaml"), [],
                  base=str(ROOT / "configs/domusfm.yaml"))
cfg["datasets"]["clean"] = False
datasets, pretrain_only = load_datasets(cfg)
held_out = set(cfg["held_out"])
PAPER = {"uci_b", "kasteren_a", "kasteren_c", "mural"}


def group(name):
    if name in PAPER:
        return "Paper test"
    if name in held_out:
        return "CASAS test"
    return "CASAS Milan/Aruba" if name in ("milan", "aruba") else "CASAS pretraining"


def family(name):
    return name if name in PAPER | {"milan", "aruba"} else re.sub(r"\d+$", "", name)


print(len(datasets), "datasets,", sum(d.n_events for d in datasets), "events")

# %% [markdown]
# ## 2. Overview

# %%
rows = []
for d in datasets:
    days = (d.ts[-1] - d.ts[0]) / 86400
    start = pd.Timestamp(d.ts[0], unit="s")
    rows.append(dict(dataset=d.name, group=group(d.name), family=family(d.name), events=d.n_events,
                     days=round(days, 1), start=start.date(), end=pd.Timestamp(d.ts[-1], unit="s").date(),
                     year=start.year, sensors=len(d.sensor_ids), classes=len(d.activities),
                     events_per_day=round(d.n_events / max(days, 1e-9)),
                     labelled_share=round(float((d.label > 0).mean()), 3)))
overview = pd.DataFrame(rows)
overview.to_csv(TAB / "overview.csv", index=False)
summary = overview.groupby("group").agg(datasets=("dataset", "count"), events=("events", "sum"),
                                        sensors_min=("sensors", "min"), sensors_max=("sensors", "max"),
                                        first_year=("year", "min"), last_year=("year", "max"),
                                        median_days=("days", "median")).reindex(GROUPS)
summary

# %%
overview[overview.group.isin(["CASAS test", "Paper test"])].sort_values("group")

# %% [markdown]
# ## 3. How old is the data?
#
# Recording period of every dataset. The CASAS homes span 2009–2023; the paper test datasets span 2008
# (Kasteren) to 2024–25 (MuRAL, whose sessions have only times of day and are placed on a dummy 2024 calendar by
# the converter, so its position here is not a real date).

# %%
tl = overview.sort_values(["group", "start"])
fig, ax = plt.subplots(figsize=(9, 11))
for i, r in enumerate(tl.itertuples()):
    s, e = pd.Timestamp(r.start), pd.Timestamp(r.end)
    ax.barh(i, (e - s).days + 1, left=s, height=0.7, color=GROUP_COLOR[r.group])
ax.set_yticks(range(len(tl)), tl.dataset, fontsize=6)
ax.invert_yaxis()
ax.set_title("Recording period per dataset")
ax.set_xlabel("date")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in GROUPS], labels=GROUPS,
          loc="lower right")
ax.grid(axis="y", visible=False)
save(fig, "01_recording_timeline")

# %% [markdown]
# ## 4. Size: events per dataset
#
# Sizes differ by more than three orders of magnitude (log scale). Pretraining samples each dataset equally often
# (the paper's dataset-level oversampling, §6.1.2), so a big home does not dominate.

# %%
o = overview.sort_values("events", ascending=False)
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.bar(range(len(o)), o.events, color=[GROUP_COLOR[g] for g in o.group], width=0.8)
ax.set_yscale("log")
ax.set_xticks(range(len(o)), o.dataset, rotation=90, fontsize=6)
ax.set_ylabel("events (log)")
ax.set_title("Events per dataset")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in GROUPS], labels=GROUPS)
ax.grid(axis="x", visible=False)
save(fig, "02_events_per_dataset")

# %% [markdown]
# ## 5. What kinds of sensors are there?
#
# Share of events by sensor type. CASAS homes are almost entirely motion sensors, with a few doors and
# thermometers. "Numeric codes" are undocumented numbers on motion-sensor names in the 2025 CASAS release, which
# the loader turns into invented ON/OFF events. The paper test datasets add item-level contacts (cupboards, fridge,
# toilet flush), pressure mats and power meters.

# %%
TYPE_BUCKET = {"motion": "motion", "door contact": "door", "temperature": "temperature",
               "scalar": "numeric codes (undocumented)", "power meter": "power / plug", "power plug": "power / plug"}
BUCKETS = ["motion", "door", "temperature", "numeric codes (undocumented)", "item contact / pressure / float",
           "power / plug"]
comp = []
for d in datasets:
    types = np.array([TYPE_BUCKET.get(t[1], "item contact / pressure / float") for t in d.sensor_text])
    c = Counter(types[d.sensor])
    comp.append({"dataset": d.name, "group": group(d.name), **{b: c.get(b, 0) / d.n_events for b in BUCKETS}})
comp = pd.DataFrame(comp).set_index("dataset")
comp.to_csv(TAB / "sensor_type_share.csv")
order = comp.sort_values(["group", "motion"]).index

fig, ax = plt.subplots(figsize=(11, 3.8))
bottom = np.zeros(len(order))
for b, col in zip(BUCKETS, SLOTS):
    v = comp.loc[order, b].values
    ax.bar(range(len(order)), v, bottom=bottom, color=col, width=0.85, label=b, edgecolor="white", linewidth=0.5)
    bottom += v
ax.set_xticks(range(len(order)), order, rotation=90, fontsize=6)
ax.set_ylabel("share of events")
ax.set_title("Sensor-type mix per dataset (grouped: pretraining, Milan/Aruba, CASAS test, paper test)")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
ax.grid(axis="x", visible=False)
save(fig, "03_sensor_type_mix")
comp.groupby("group")[BUCKETS].mean().reindex(GROUPS).round(3)

# %% [markdown]
# ### Sensor names per home, by family
#
# The 2025 CASAS release merges the physical sensors of a room into one name. The hh homes keep 5–8 names (the
# hh101 floorplan shows about 40 physical sensors); newer families keep more detail.

# %%
fams = overview.groupby("family").sensors.median().sort_values().index
fig, ax = plt.subplots(figsize=(9, 3.2))
for i, f in enumerate(fams):
    v = overview[overview.family == f]
    ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(v)), v.sensors, s=18,
               color=[GROUP_COLOR[g] for g in v.group], edgecolor="white", linewidth=0.8, zorder=3)
ax.set_xticks(range(len(fams)), fams, rotation=45, ha="right")
ax.set_ylabel("sensor names")
ax.set_title("Distinct sensors per dataset, by family")
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=GROUP_COLOR[g]) for g in GROUPS], labels=GROUPS,
          fontsize=7)
save(fig, "04_sensors_per_family")

# %% [markdown]
# ## 6. Daily rhythm
#
# Share of events per hour of day, averaged over the datasets in each group (each dataset weighted equally).
# The shape is similar across groups (quiet at night, peaks in the morning and evening), which is why routines
# learned on old CASAS homes can carry over. MuRAL sessions were recorded only at set times, so its profile is
# not a full day.

# %%
def hour_profile(d):
    h = ((d.ts % 86400) // 3600).astype(int)
    return np.bincount(h, minlength=24) / d.n_events


prof = {g: np.mean([hour_profile(d) for d in datasets if group(d.name) == g], axis=0) for g in GROUPS}
fig, ax = plt.subplots(figsize=(8, 3.2))
for g in GROUPS:
    ax.plot(range(24), prof[g], color=GROUP_COLOR[g], lw=2, label=g)
ax.set_xticks(range(0, 24, 3))
ax.set_xlabel("hour of day")
ax.set_ylabel("share of events")
ax.set_title("Events by hour of day")
ax.legend()
save(fig, "05_hour_profile")

# %%
tests = [d for d in datasets if d.name in held_out]
mat = np.array([hour_profile(d) for d in tests])
fig, ax = plt.subplots(figsize=(8, 3.4))
im = ax.imshow(mat, aspect="auto", cmap="Blues")
ax.set_yticks(range(len(tests)), [d.name for d in tests])
ax.set_xticks(range(0, 24, 3))
ax.set_xlabel("hour of day")
ax.set_title("Hourly share of events, test datasets")
ax.grid(False)
fig.colorbar(im, ax=ax, label="share of events")
save(fig, "06_hour_heatmap_tests")

# %% [markdown]
# ## 7. How much time does one 30-event window cover?
#
# DomusFM reads windows of 30 events (§6.2.2). The paper notes (§7.2) that the time a window covers varies across
# datasets. Here: the time from the first to the last event of every window, per test dataset (log scale).

# %%
def window_spans(d, w=30):
    return (d.ts[w - 1:] - d.ts[: len(d.ts) - w + 1]) / 60.0


spans = {d.name: window_spans(d) for d in tests}
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.boxplot([np.clip(v, 1e-2, None) for v in spans.values()], showfliers=False, widths=0.5,
           medianprops=dict(color=SLOTS[1], lw=2), boxprops=dict(color=INK2), whiskerprops=dict(color=INK2),
           capprops=dict(color=INK2))
ax.set_xticks(range(1, len(spans) + 1), spans.keys(), rotation=30)
ax.set_yscale("log")
ax.set_ylabel("minutes (log)")
ax.set_title("Time covered by a 30-event window (before cleaning)")
save(fig, "07_window_span_before")
pd.DataFrame({k: np.percentile(v, [10, 50, 90]) for k, v in spans.items()}, index=["p10 min", "median min",
                                                                                     "p90 min"]).round(1)

# %% [markdown]
# ## 8. Labels on the test datasets
#
# What the ADL head is trained to predict: the activity at the last event of each window. "Other" (label 0) is kept
# as a class, as in the paper (§6.1.1). Kasteren C is dominated by "go to bed" because the bed pressure mat fires
# every few seconds during sleep.

# %%
fig, axes = plt.subplots(2, 5, figsize=(15, 6.5))
for ax, d in zip(axes.flat, tests):
    c = pd.Series(np.array(d.activities)[d.label]).value_counts(normalize=True).head(8)[::-1]
    ax.barh(range(len(c)), c.values, color=SLOTS[0], height=0.7)
    ax.set_yticks(range(len(c)), [s[:22] for s in c.index], fontsize=7)
    ax.set_title(f"{d.name} ({len(d.activities)} classes)", fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1)
fig.suptitle("Top activity classes per test dataset (share of events)", fontweight="bold")
fig.tight_layout()
save(fig, "08_label_distribution_tests")

# %% [markdown]
# ## 9. Data quality
#
# ### 9.1 Raw-file checks (CASAS)
#
# Things the loaders silently skip or that cannot be seen after loading: malformed readings, rows out of time
# order, activity labels with a begin and no end (or the reverse), and recording outages.

# %%
LAB = re.compile(r'^(.*?)(?:="(begin|end)")?$')
NUM = re.compile(r"^-?\d+(\.\d+)?$")


def raw_scan(path):
    r = Counter()
    prev, open_ = None, Counter()
    with open(path, errors="ignore") as f:
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 4 or not p[2]:
                continue
            r["rows"] += 1
            m = p[3].strip()
            if m.upper() not in ("ON", "OFF", "OPEN", "CLOSE"):
                if NUM.match(m):
                    r["numeric" if "Temperature" in p[2] else "numeric_on_non_temperature"] += 1
                else:
                    r["malformed"] += 1
            t = p[0] + "T" + p[1]
            r["out_of_order"] += bool(prev and t < prev)
            prev = t
            if len(p) > 4 and p[4].strip():
                c, edge = LAB.match(p[4].strip()).groups()
                if edge == "begin":
                    r["dangling_begin"] += open_[c]
                    open_[c] = 1
                elif edge == "end":
                    r["unmatched_end"] += 1 - open_[c]
                    open_[c] = 0
    r["dangling_begin"] += sum(open_.values())
    return r


cache = ROOT / "data/cache/eda_raw_scan.json"
casas_files = {p.stem: p for p in sorted((ROOT / "data/raw/casas/labeled").glob("*.csv"))}
casas_files.update(milan=ROOT / "data/raw/casas/data/milan.csv", aruba=ROOT / "data/raw/casas/data/aruba.csv")
scan = json.loads(cache.read_text()) if cache.exists() else {}
for name, p in casas_files.items():
    if name not in scan:
        scan[name] = raw_scan(p)
cache.write_text(json.dumps(scan))
raw = pd.DataFrame(scan).T.fillna(0).astype(int)
raw = raw[[c for c in ["rows", "numeric", "numeric_on_non_temperature", "malformed", "out_of_order",
                       "dangling_begin", "unmatched_end"] if c in raw]]
raw.to_csv(TAB / "casas_raw_checks.csv")
raw.sum().to_frame("total over all CASAS files")

# %%
gaps = pd.Series({d.name: int((np.diff(d.ts) > 86400).sum()) for d in datasets})
issues = raw.loc[raw.index.isin(overview.dataset)].assign(outages_over_1_day=gaps)
issues[(issues.drop(columns="rows") > 0).any(axis=1)].sort_values("numeric_on_non_temperature", ascending=False) \
    .head(20)

# %% [markdown]
# ### 9.2 Repeated states (paper, Appendix A)
#
# The paper: a binary sensor's events must alternate ON/OFF, and a repeated state is a duplicate event that is
# "safely removed during the data cleaning process prior to segmentation". Our reproduction did not do this until
# now. Share of events that repeat their sensor's previous state:

# %%
def repeat_share(d):
    order = np.lexsort((d.ts, d.sensor))
    s, st = d.sensor[order], d.status[order]
    return float(((s[1:] == s[:-1]) & (st[1:] == st[:-1])).sum() / max(d.n_events, 1))


rep = pd.Series({d.name: repeat_share(d) for d in datasets})
o = rep.sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.bar(range(len(o)), o.values, color=[GROUP_COLOR[group(n)] for n in o.index], width=0.8)
ax.set_xticks(range(len(o)), o.index, rotation=90, fontsize=6)
ax.set_ylabel("share of events")
ax.set_title("Events that repeat the sensor's previous state (removed by the paper's cleaning)")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in GROUPS], labels=GROUPS)
ax.grid(axis="x", visible=False)
save(fig, "09_repeated_states")
rep.groupby(rep.index.map(group)).describe()[["mean", "min", "max"]].reindex(GROUPS).round(3)

# %% [markdown]
# ## 10. Cleaning
#
# `homefm.baselines.domusfm.clean.clean_dataset`, switched on with `datasets.clean: true`:
#
# 1. drop the undocumented numeric codes (sensor type "scalar"); temperature is kept;
# 2. merge sensor names that differ only in letter case;
# 3. drop repeated states (paper Appendix A), which also removes exact duplicate rows.
#
# Malformed readings are already skipped by the loaders. Out-of-order rows are re-sorted by the loaders.
# Unmatched activity labels and outages are left as they are: they are rare and there is no safe automatic fix.

# %%
cleaned, reports = [], {}
for d in datasets:
    c, r = clean_dataset(d)
    cleaned.append(c)
    reports[d.name] = r
rep_df = pd.DataFrame(reports).T
rep_df["removed_share"] = 1 - rep_df.events_out / rep_df.events_in
rep_df["group"] = rep_df.index.map(group)
rep_df.to_csv(TAB / "cleaning_report.csv")
by_group = rep_df.groupby("group")[["events_in", "dropped_type_events", "dropped_repeat_events",
                                    "events_out"]].sum().reindex(GROUPS)
by_group["removed_share"] = (1 - by_group.events_out / by_group.events_in).round(3)
by_group

# %%
o = rep_df.sort_values("removed_share", ascending=False)
fig, ax = plt.subplots(figsize=(11, 3.4))
a = (o.dropped_type_events / o.events_in).values
b = (o.dropped_repeat_events / o.events_in).values
ax.bar(range(len(o)), a, color=SLOTS[0], width=0.8, label="numeric codes dropped")
ax.bar(range(len(o)), b, bottom=a, color=SLOTS[1], width=0.8, label="repeated states dropped",
       edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(o)), o.index, rotation=90, fontsize=6)
ax.set_ylabel("share of events removed")
ax.set_title("What cleaning removes, per dataset")
ax.legend()
ax.grid(axis="x", visible=False)
save(fig, "10_cleaning_removed")
rep_df.loc[sorted(held_out), ["events_in", "events_out", "removed_share", "sensors_in", "sensors_out"]]

# %% [markdown]
# ### After cleaning: time covered by a window, and label mix
#
# With repeats removed, a 30-event window covers more real time and more distinct actions.

# %%
tests_c = [c for c in cleaned if c.name in held_out]
spans_c = {d.name: window_spans(d) for d in tests_c}
fig, ax = plt.subplots(figsize=(9, 3.4))
pos = np.arange(len(spans))
for k, (sp, col, lab) in enumerate([(spans, SLOTS[0], "before"), (spans_c, SLOTS[1], "after")]):
    bp = ax.boxplot([np.clip(v, 1e-2, None) for v in sp.values()], positions=pos + (k - 0.5) * 0.35,
                    widths=0.3, showfliers=False, patch_artist=True, medianprops=dict(color=INK, lw=1.5),
                    boxprops=dict(facecolor=col, color=col), whiskerprops=dict(color=col), capprops=dict(color=col))
ax.set_xticks(pos, spans.keys(), rotation=30)
ax.set_yscale("log")
ax.set_ylabel("minutes (log)")
ax.set_title("Time covered by a 30-event window, before and after cleaning")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SLOTS[0]), plt.Rectangle((0, 0), 1, 1, color=SLOTS[1])],
          labels=["before", "after"])
save(fig, "11_window_span_after")
pd.DataFrame({k: [np.median(spans[k]), np.median(spans_c[k])] for k in spans}, index=["median min before",
                                                                                      "median min after"]).round(1)

# %%
lab = pd.DataFrame({d.name: {"labelled share before": float((d.label > 0).mean()),
                             "labelled share after": float((c.label > 0).mean())}
                    for d, c in zip(datasets, cleaned) if d.name in held_out}).T.round(3)
lab

# %% [markdown]
# ## 11. Findings
#
# Printed from the numbers above, so they stay correct if the data changes.

# %%
pre = overview[overview.group.isin(["CASAS pretraining", "CASAS Milan/Aruba"])]
casas_types = comp[comp.group != "Paper test"][BUCKETS].mean()
print(f"Pretraining corpus: {len(pre)} homes, {pre.events.sum():,} events, recorded {pre.year.min()}–"
      f"{overview[overview.group != 'Paper test'].end.max().year}.")
print(f"CASAS event mix (mean over homes): motion {casas_types['motion']:.1%}, door {casas_types['door']:.1%}, "
      f"temperature {casas_types['temperature']:.1%}, undocumented numeric codes "
      f"{casas_types['numeric codes (undocumented)']:.1%}. No power, water, light, audio or camera sensors.")
casas_sensors = rep_df[rep_df.group != "Paper test"].sensors_out
print(f"Sensors per CASAS home after cleaning: {casas_sensors.min()}–{casas_sensors.max()} "
      f"(before: up to {rep_df[rep_df.group != 'Paper test'].sensors_in.max()}, because numeric codes on a motion "
      f"name were counted as a second sensor).")
print(f"Raw CASAS problems: {raw.numeric_on_non_temperature.sum():,} numeric codes on motion names, "
      f"{raw.malformed.sum():,} malformed readings, {raw.out_of_order.sum()} rows out of order, "
      f"{raw.dangling_begin.sum() + raw.unmatched_end.sum():,} unmatched activity labels, "
      f"{int(gaps.sum())} outages over 1 day.")
print(f"Cleaning removes {1 - by_group.events_out.sum() / by_group.events_in.sum():.1%} of all events: "
      + ", ".join(f"{g} {by_group.loc[g, 'removed_share']:.1%}" for g in GROUPS) + ".")
