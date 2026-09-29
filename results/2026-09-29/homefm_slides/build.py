"""Builds the HomeFM deck's project/ files (slides + deck.json) from docs/DESIGN.md content."""
import html
import json
import datetime
from pathlib import Path

ROOT = Path(__file__).parent
SL = ROOT / "project" / "slides"
SL.mkdir(parents=True, exist_ok=True)

FH = "'Space Grotesk', Arial, sans-serif"
FB = "'IBM Plex Sans', Arial, sans-serif"
FM = "'JetBrains Mono', 'Courier New', monospace"
BG, CARD, LINE = "#F6F5F1", "#FDFCFA", "#E3E0D7"
INK, BODY, MUTED = "#14213D", "#3D4757", "#5B6472"
HOME, HOMET = "#1E5AA8", "#E6EEF8"
DOM, DOMT = "#A8470B", "#FBEBDD"
OK, OKT = "#2F6B45", "#E3F0E7"
DARK, ON, SOFT, HOMED, DOMD = "#14213D", "#F6F5F1", "#C5CFDD", "#8DB8F2", "#F2A05E"
ARROW = "#9AA3B2"

slides, order = {}, []


def esc(s):
    return html.escape(s, quote=False)


def add(sid, body, notes="", dark=False, gap=28, extra=""):
    bg, col = (DARK, ON) if dark else (BG, BODY)
    s = (f'<section id="{sid}" data-transition="fade" style="background:{bg}; color:{col}; font-family:{FB}; '
         f'padding:104px 128px 96px; display:flex; flex-direction:column; gap:{gap}px{extra}">\n{body}\n')
    if notes:
        s += f"<aside>{esc(notes)}</aside>\n"
    slides[sid] = s + "</section>\n"
    order.append(sid)


def head(eyebrow, title, color=HOME):
    return (f'<div style="display:flex; flex-direction:column; gap:8px">'
            f'<p style="font-size:24px; font-weight:600; letter-spacing:2px; text-transform:uppercase; color:{color}">{eyebrow}</p>'
            f'<h2 style="font-family:{FH}; font-size:56px; font-weight:600; line-height:1.1; color:{INK}">{title}</h2></div>')


def p(text, size=24, color=BODY, extra=""):
    return f'<p style="font-size:{size}px; line-height:1.4; color:{color}{extra}">{text}</p>'


def h3(text, color=INK, size=30):
    return f'<h3 style="font-family:{FH}; font-size:{size}px; font-weight:600; line-height:1.2; color:{color}">{text}</h3>'


def card(inner, flex="1", bg=CARD, border=LINE, pad=28, gap=10, extra=""):
    return (f'<div style="flex:{flex}; background:{bg}; border:2px solid {border}; border-radius:16px; padding:{pad}px; '
            f'display:flex; flex-direction:column; gap:{gap}px{extra}">{inner}</div>')


def row(*items, gap=24, extra=""):
    return f'<div style="display:flex; gap:{gap}px{extra}">{"".join(items)}</div>'


def col(*items, gap=16, extra=""):
    return f'<div style="display:flex; flex-direction:column; gap:{gap}px{extra}">{"".join(items)}</div>'


def grid(items, n, gap=24):
    return f'<div style="display:grid; grid-template-columns:repeat({n}, 1fr); gap:{gap}px">{"".join(items)}</div>'


def pill(text, bg=HOMET, color=HOME, size=24):
    return (f'<p style="font-size:{size}px; font-weight:600; background:{bg}; color:{color}; padding:6px 16px; '
            f'border-radius:999px; align-self:flex-start">{text}</p>')


def box(text, bg=CARD, border=LINE, color=INK, size=24, flex="1", weight=500, extra=""):
    return (f'<p style="flex:{flex}; font-size:{size}px; line-height:1.3; font-weight:{weight}; color:{color}; background:{bg}; '
            f'border:2px solid {border}; border-radius:12px; padding:14px 16px; text-align:center{extra}">{text}</p>')


def arrow_r(color=ARROW):
    return f'<x-shape kind="arrow-right" style="flex:none; width:44px; height:22px; background:{color}; align-self:center"></x-shape>'


def arrow_d(color=ARROW):
    return f'<x-shape kind="arrow-down" style="flex:none; width:22px; height:32px; background:{color}; align-self:center"></x-shape>'


def mono(text, size=24, bg="#EDEBE4", color=INK):
    return (f'<p style="font-family:{FM}; font-size:{size}px; line-height:1.45; color:{color}; background:{bg}; '
            f'padding:16px 20px; border-radius:12px">{text}</p>')


def table(headers, rows, widths, size=24, hl=None, head_bg=INK):
    pad = "padding:10px 14px"
    out = [f'<table style="width:1664px; font-size:{size}px; color:{BODY}">',
           f'<tr style="background:{head_bg}">' + "".join(
               f'<th style="width:{w}%; color:{ON}; text-align:left; {pad}">{h}</th>' for h, w in zip(headers, widths)) + "</tr>"]
    for i, r in enumerate(rows):
        bg = HOMET if hl is not None and i in hl else (CARD if i % 2 == 0 else BG)
        out.append(f'<tr style="background:{bg}">' + "".join(f'<td style="{pad}">{c}</td>' for c in r) + "</tr>")
    out.append("</table>")
    return "".join(out)


def table_w(width, headers, rows, widths, size=24, hl=None):
    return table(headers, rows, widths, size, hl).replace("width:1664px", f"width:{width}px", 1)


# ---------------------------------------------------------------- 1. Intro
add("cover", f"""
<div style="display:flex; flex-direction:column; gap:20px">
<p style="font-size:24px; font-weight:600; letter-spacing:3px; text-transform:uppercase; color:{HOMED}">Design overview · 29 September 2026</p>
<h1 style="font-family:{FH}; font-size:160px; font-weight:700; line-height:1; color:{ON}">HomeFM</h1>
<p style="font-family:{FH}; font-size:44px; line-height:1.25; color:{ON}; width:1400px">A foundation model for the whole smart home: architecture, use cases, pretraining, and what we change from DomusFM</p>
</div>
<div style="flex:1"></div>
<div style="display:flex; gap:48px">
{p("Status: design and scaffold built; DomusFM baseline reproduced on 77 homes; HomeFM corpus runs not started", 28, SOFT, "; width:900px")}
{p("Sources: docs/DESIGN.md · LIMITATIONS_AND_REMEDIES.md · DOMUSFM_REPRODUCTION.md", 24, SOFT, "; width:700px")}
</div>""", notes="HomeFM design deck. Every number and diagram comes from the design doc and our DomusFM reproduction runs.", dark=True)

agenda = [("1", "Why HomeFM", "The questions a home should answer, goals and requirements"),
          ("2", "DomusFM, the baseline", "What it is, what it keeps, what our reproduction showed"),
          ("3", "11 limitations", "Each DomusFM limitation, an example that breaks it, and HomeFM's fix"),
          ("4", "Architecture", "System layers, answer paths, Home Tokens, model blocks, heads"),
          ("5", "Pretraining", "Why DomusFM's masking games go, and the three games that replace them"),
          ("6", "From model to answers", "Episodes, anomaly and device health, the query agent"),
          ("7", "Plan", "Data, evaluation, deployment, roadmap, build status, risks, next steps")]
items = [row(f'<p style="font-family:{FH}; font-size:56px; font-weight:700; line-height:1; color:{HOME}; width:64px">{n}</p>',
             col(h3(t, size=32), p(d, 24, MUTED), gap=4), gap=16) for n, t, d in agenda]
add("agenda", head("Agenda", "Seven parts") + grid(items, 2, 36))

qs = [("Energy", "How much energy was used in the last hour?"), ("Childcare", "How many times did my kid cry today?"),
      ("Activities", "How many times did cooking happen in the last 4 hours?"), ("Deliveries", "How many parcels did we receive yesterday?"),
      ("Occupancy", "How many people are in the living room?"), ("Device health", "Is anything wrong with the fridge?"),
      ("Elderly care", "Did grandma eat lunch?"), ("Security", "Anything unusual last night?"), ("Pets", "Did the dog bark while we were out?")]
add("goal", head("Why HomeFM", "Ask the home anything, in plain words")
    + grid([card(p(d, 24, HOME, "; font-weight:600") + p(q, 28, INK), gap=6, pad=24) for d, q in qs], 3, 20)
    + p("Domains are open-ended, so the system is built around a <b>small set of question types</b> and an <b>open-vocabulary model</b>, not one detector per question.", 26),
    notes="Questions span activities, security, pets, childcare, elderly care, device defects, energy and anomalies.")

add("principle", f"""
<p style="font-size:24px; font-weight:600; letter-spacing:2px; text-transform:uppercase; color:{HOMED}">Core principle</p>
<h2 style="font-family:{FH}; font-size:72px; font-weight:600; line-height:1.1; color:{ON}">The model understands. The database counts. The agent plans.</h2>
<div style="flex:1"></div>
<div style="display:flex; gap:32px">
{card(h3("Understand", HOMED, 36) + p("HomeFM and the perception experts turn raw signals into structured, searchable facts: events, episodes, states, scores.", 28, ON), bg="#1F2E4F", border="#2E4270")}
{card(h3("Count", HOMED, 36) + p("A database does the counting. Every number in an answer comes from a tool, never from the LLM.", 28, ON), bg="#1F2E4F", border="#2E4270")}
{card(h3("Plan", HOMED, 36) + p("An LLM agent reads the question, picks the tools and writes the answer with evidence and confidence.", 28, ON), bg="#1F2E4F", border="#2E4270")}
</div>""", dark=True)

qt = [("Count / aggregate", "How many times did the dog bark today?", "Event / episode store (SQL)"),
      ("Current state", "How many people are in the living room?", "State table"),
      ("When / last time", "When did grandma last take her medicine?", "Episodes"),
      ("Duration", "How long did the kids watch TV?", "Episodes"),
      ("Comparison / trend", "Am I cooking less than last month?", "Aggregates + baselines"),
      ("Anomaly", "Anything unusual last night?", "Anomaly engine"),
      ("Diagnosis", "Is my fridge OK? Why is the bill high?", "Device health + energy breakdown"),
      ("Well-being / routine", "Did Dad eat lunch? Is Mum sleeping well?", "Routine model + episodes"),
      ("Summary", "What happened while I was away?", "Semantic retrieval + LLM summary"),
      ("Prediction", "Will the guest room be warm by 6?", "Forecast head"),
      ("Unanswerable", "Were the plants watered? (no sensor)", "Capability registry: honest refusal")]
add("qtypes", head("Why HomeFM", "Domains are open; question types are a closed set of 11")
    + table(["Question type", "Example", "Answered by"], qt, [22, 44, 34]),
    notes="Every question type maps to a tool path. The 'unanswerable' type matters: the capability registry lets the agent refuse honestly.")

goals = [("G1", "One pretrained model that transfers to unseen homes with any set of devices"),
         ("G2", "Open vocabulary: recognise and search concepts never labelled (“vacuuming”, “guest arrived”)"),
         ("G3", "Binary events, continuous telemetry and audio/vision detector outputs in one representation"),
         ("G4", "Reasons over seconds, hours and weeks (cries, cooking, routines, device degradation)"),
         ("G5", "Exact counts and durations, calibrated anomaly and device-health scores, forecasts"),
         ("G6", "Streams on an edge hub; raw audio and video never leave the home")]
gl = col(*[row(pill(g, HOMET, HOME), p(t, 26, INK), gap=16, extra="; align-items:center") for g, t in goals], gap=14)
side = col(card(h3("Not in v1") + p("Raw video or audio inside HomeFM: on-device experts handle it", 24)
                + p("Home automation control", 24) + p("Medical diagnosis: elderly-care outputs are behavioural signals", 24)),
           card(h3("Non-functional targets") + p("Answers in under 2 s for stored facts", 24)
                + p("Hub student under 500 MB RAM, under 20 ms per minute update", 24)
                + p("Only events and embeddings stored; every answer carries evidence and confidence", 24)), gap=20)
add("goals", head("Why HomeFM", "Goals, non-goals and targets") + row(f'<div style="flex:3">{gl}</div>', f'<div style="flex:2">{side}</div>', gap=40))

# ---------------------------------------------------------------- 2. DomusFM
steps = ["Event<br>time · sensor · ON/OFF", "Sensor as text<br>item, type, room → MiniLM (frozen) + status + cyclic time",
         "Attribute self-attention<br>→ one event vector", "12-layer transformer<br>over the last 30 events, d = 384",
         "Heads<br>activity (linear) · next-k bag of events"]
flow = []
for i, s in enumerate(steps):
    flow.append(box(s, DOMT, "#E9C3A2"))
    if i < len(steps) - 1:
        flow.append(arrow_r())
add("domus-overview", head("DomusFM, the baseline", "DomusFM in one slide (Fiori et al., arXiv 2602.01910)", DOM)
    + row(*flow, gap=12, extra="; align-items:stretch")
    + row(card(h3("Pretraining, no labels") + p("Two contrastive phases (InfoNCE): hide one attribute of some events, then hide whole events with the event layers frozen.", 24)),
          card(h3("Fine-tuning") + p("A small head trained on 5–30 % of the target dataset's human-made activity labels. Leave-one-dataset-out over 7 public datasets.", 24)),
          card(h3("Edge-sized") + p("36M parameters, under 500 MB, ~10 ms per window on a Celeron mini-PC. Beats DeepCASAS, Chronos and a GPT-2 baseline.", 24)))
    + p("Our reimplementation: <b>28.6M parameters</b> (the paper leaves some sizes unspecified), in <b>src/homefm/baselines/domusfm/</b>.", 24, MUTED),
    notes="DomusFM is the primary baseline. It is a strong idea: describing sensors in words is what makes cross-home transfer work.")

gaps = [("R1 Open vocabulary", "Only activities with labels", "Missing", "Minutes aligned with sentences: tagger, search"),
        ("R2 Multi-modal", "Binary only; numbers binarised", "Missing", "Numbers stay numbers; audio/vision tags"),
        ("R3 Multi-scale time", "Fixed 30-event window", "Missing", "1-minute steps, hours of stream, day tokens"),
        ("R4 Multi-occupant", "One resident assumed", "Missing", "Person cards, per-room counts"),
        ("R5 Episodes", "One label per window", "Partial", "Start/end head + per-activity episode rules"),
        ("R6 Anomaly / health", "Not addressed", "Missing", "Surprise score, anomaly and health engines"),
        ("R7 Forecasting", "Next-k bag, no timing", "Partial", "Next event + time until (Δt)"),
        ("R8 Transfer", "Sensor attributes as text", "Has it", "Kept: devices described in words"),
        ("R9 Edge streaming", "Small, recomputes each window", "Partial", "Causal mode + cache, int8 student"),
        ("R10 Searchable", "Not aligned with text", "Missing", "Shared text–minute space, vector store")]
add("domus-gaps", head("DomusFM, the baseline", "DomusFM against our 10 model requirements", DOM)
    + table(["Requirement", "DomusFM", "Gap", "HomeFM answer"], gaps, [18, 29, 11, 42], hl=[7]),
    notes="Seven requirements missing, three partial, one met. R8, transfer through text-described sensors, is the one we keep.")

keep = [("Devices described in words", "Item, room and type go through a frozen text encoder. This is what makes transfer to unseen homes work."),
        ("Attribute fusion", "What, where, state and time attend to each other before being pooled into one event vector."),
        ("Cyclic time", "Hour and weekday as positions on circles, so 23:59 and 00:01 are neighbours."),
        ("Self-supervised pretraining", "Learn from unlabelled streams first; labels only name things later."),
        ("Edge-sized", "36M parameters, under 500 MB, about 10 ms per window on a Celeron CPU."),
        ("Leave-one-dataset-out", "Test on homes never seen in pretraining. We adopt it as our protocol.")]
add("domus-keep", head("DomusFM, the baseline", "What DomusFM gets right, and HomeFM keeps", DOM)
    + grid([card(f'<x-icon name="Check" style="color:{OK}; width:36px; height:36px"></x-icon>' + h3(t) + p(d, 24)) for t, d in keep], 3, 24))

stats = [("0.0005", "contrastive loss by step 5,000 of 40,000 with the paper's game on 77 homes (chance 4.85): the game was solved, not learned", DOM),
         ("6 of 7", "held-out homes where the paper's pretraining made activity recognition <b>worse</b> (mean 0.385 vs 0.435 at 5 % labels)", DOM),
         ("26 of 28", "settings where pretraining <b>helps</b> after our six fixes to the game (activity, 5 % labels: 0.517 vs 0.441)", OK)]
add("domus-results", head("DomusFM, the baseline", "Our reproduction: the paper's game collapsed; a harder game fixed it", DOM)
    + row(*[card(f'<p style="font-family:{FH}; font-size:96px; font-weight:700; line-height:1; color:{c}">{n}</p>' + p(t, 26), pad=32, gap=16) for n, t, c in stats], gap=28)
    + card(p("<b>The six fixes:</b> wrong answers from the same home, ±15-min clock jitter, both copies masked at 40 %, a separate projector head, a fill-in-the-blanks loss, gentler fine-tuning. "
             "<b>Run 3 (29 Sep):</b> cleaned data (paper Appendix A), the paper's own Kasteren and MuRAL test sets, leave-one-dataset-out pretraining: pretraining now wins <b>34 of 40</b> settings (next two slides).", 24), bg=CARD)
    + p("<b>Lesson for HomeFM:</b> the pretraining game decides what the model learns.", 28, INK),
    notes="Numbers from results/domusfm_corpus and results/domusfm_corpus_fixed. The fixed DomusFM is the baseline HomeFM must beat.")

RUN3 = json.loads((ROOT.parent / "domusfm_corpus_clean" / "results.json").read_text(encoding="utf-8"))
RF = json.loads((ROOT.parent / "domusfm_corpus_clean" / "random_folds" / "results.json").read_text(encoding="utf-8"))


def r3(res, h, t, pct, name="DomusFM"):
    return res[h]["results"][f"{t}|{pct}|{name}"]["mean"]


# ---- why run 1 failed, the six fixes, the two training losses, and how we differ from the paper (plain words)
C_A, C_B = "#1E5AA8", "#C46A1B"   # chart series; validated with the dataviz palette checker on CARD
GRID, ZERO = "#E3E0D7", "#9AA3B2"
RUN1 = json.loads((ROOT.parent.parent / "domusfm_corpus" / "results.json").read_text(encoding="utf-8"))
GROUP_FIRST = ["uci_b", "kasteren_a", "kasteren_c", "mural", "hh101"]   # first target of each run 3 pretraining group


def hist_xy(h):
    return [(r["step"] + (20000 if r["phase"] == "event" else 0)) for r in h]


def line_chart(series, xmax, ymax, yticks, xticks, W, H, spec, hlines=(), vlines=(), label=""):
    """Line chart: an <svg> for the lines (no text inside, per the slide format) with <p> axis labels around it.
    spec (JSON) lets build_pptx.js swap the whole block for a native PowerPoint line chart."""
    ins = 17  # half a 24px label's line box, so y labels (space-between) line up with the gridlines
    X = lambda x: x / xmax * W
    Y = lambda y: ins + (1 - y / ymax) * (H - 2 * ins)
    parts = [f'<line x1="0" y1="{Y(t):.1f}" x2="{W}" y2="{Y(t):.1f}" stroke="{GRID}" stroke-width="1.5"/>' for t, _ in yticks]
    parts += [f'<line x1="{X(x):.1f}" y1="{ins}" x2="{X(x):.1f}" y2="{H - ins}" stroke="{ZERO}" stroke-width="2" stroke-dasharray="8 8"/>' for x in vlines]
    parts += [f'<line x1="0" y1="{Y(y):.1f}" x2="{W}" y2="{Y(y):.1f}" stroke="{c}" stroke-width="2.5" stroke-dasharray="4 7"/>' for y, c in hlines]
    for _, c, pts in series:
        d = " ".join(f"{X(x):.1f},{Y(min(y, ymax)):.1f}" for x, y in pts)
        parts.append(f'<polyline points="{d}" fill="none" stroke="{c}" stroke-width="3.5" stroke-linejoin="round" stroke-linecap="round"/>')
    svg = (f'<svg aria-label="{html.escape(label, quote=True)}" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'style="flex:none">{"".join(parts)}</svg>')
    ylab = "".join(f'<p style="font-size:24px; line-height:34px; color:{MUTED}; text-align:right">{t}</p>' for _, t in reversed(yticks))
    lw = 90
    xlab = "".join(f'<p style="flex:none; width:{lw}px; font-size:24px; color:{MUTED}; text-align:center">{t}</p>' for _, t in xticks)
    spec_attr = html.escape(json.dumps(spec), quote=True)
    return (f'<div data-chart="{spec_attr}" style="display:flex; flex-direction:column; gap:4px">'
            f'<div style="display:flex; gap:12px"><div style="flex:none; width:44px; height:{H}px; display:flex; flex-direction:column; '
            f'justify-content:space-between">{ylab}</div>{svg}</div>'
            f'<div style="display:flex"><div style="flex:none; width:{56 - lw // 2}px"></div>'
            f'<div style="flex:none; width:{W + lw}px; display:flex; justify-content:space-between">{xlab}</div></div></div>')


def key(items):
    out = []
    for name, c, dashed in items:
        sw = (f'<div style="flex:none; width:32px; height:0px; border-top:3px dashed {c}"></div>' if dashed
              else f'<div style="flex:none; width:32px; height:6px; background:{c}; border-radius:3px"></div>')
        out.append(f'<div style="display:flex; gap:10px; align-items:center">{sw}<p style="font-size:24px; color:{BODY}">{name}</p></div>')
    return f'<div style="display:flex; gap:28px; flex-wrap:wrap">{"".join(out)}</div>'


fix_rows = [
    ("The wrong answers came from other homes, so sensor names alone gave the right one away", "Wrong answers now come from the same home"),
    ("Every event carries its exact second: a fingerprint", "Shift each copy's clock by up to ±15 min"),
    ("One copy was the untouched window, so 85 % of it matched exactly", "Hide 40 % of both copies"),
    ("The game bent the model itself toward telling windows apart", "A small add-on plays the game, then is thrown away"),
    ("Nothing hidden ever had to be guessed", "New task: fill in the blanks (hidden sensor, room, ON/OFF)"),
    ("Fine-tuning moved the pretrained weights as fast as the new head", "Move pretrained weights gently, with a warm-up"),
]
add("run1-fixes", head("DomusFM, the baseline", "Why run 1 failed: the model learned to cheat at its training game", DOM)
    + card(p("<b>The game:</b> take a 30-event window, hide 15 % of it, then pick that hidden copy out of 128 windows. "
             "<b>Run 1:</b> the model won the game after 1,000 of 40,000 steps, by using shortcuts, not by learning how people live. "
             "Pretraining then made activity recognition <b>worse</b> in 6 of 7 homes.", 26, INK), flex="none", bg=DOMT, border="#E9C3A2")
    + table(["How it cheated", "Our fix (run 2 onwards)"], fix_rows, [55, 45])
    + p("<b>None of the six is in the paper.</b> It describes the game but not how much to hide, how batches are built or how fast to learn, "
        "and it has no add-on or fill-in-the-blanks task. So from run 2 on we test a <b>fixed</b> DomusFM, not the paper's exact recipe.", 24, INK),
    notes="Technical names: 1 same-home negatives (homes_per_batch 4), 2 time jitter ±900 s, 3 both views masked at 0.4 with temperature 0.2, "
          "4 projection head, 5 masked-attribute prediction (mlm_weight 1), 6 backbone LR 5e-5, head LR 1e-3, 10 % warm-up. "
          "Details: docs/DOMUSFM_REPRODUCTION.md, 'Why pretraining did not help, and the fix'. Config: configs/domusfm_corpus_fixed.yaml.")

r1h = RUN1["uci_b"]["pretrain_history"]
r1 = list(zip(hist_xy(r1h), [r["loss"] for r in r1h]))
r3h = [RUN3[k]["pretrain_history"] for k in GROUP_FIRST]
r3x = hist_xy(r3h[0])
r3c = [sum(h[i]["contrastive"] for h in r3h) / len(r3h) for i in range(len(r3x))]
r3m = [sum(h[i]["mlm"] for h in r3h) / len(r3h) for i in range(len(r3x))]
CHANCE = 4.85   # ln 128: guessing among the 128 windows of a batch
xt = [(0, "0"), (10000, "10k"), (20000, "20k"), (30000, "30k"), (40000, "40k")]
steps_lbl = [f"{x // 1000}k" if x % 10000 == 0 else "" for x in r3x]
match_spec = {"type": "line", "categories": steps_lbl, "min": 0, "max": 5, "step": 1, "legend": False, "fmt": "0.00",
              "series": [{"name": "Run 1, paper's game", "values": [round(y, 4) for _, y in r1], "color": C_B},
                         {"name": "Run 3, fixed game", "values": [round(y, 4) for y in r3c], "color": C_A},
                         {"name": "Chance", "values": [CHANCE] * len(r3x), "color": ZERO}]}
fill_spec = {"type": "line", "categories": steps_lbl, "min": 0, "max": 5, "step": 1, "legend": False, "fmt": "0.00",
             "series": [{"name": "Run 3, fill-in-the-blanks", "values": [round(y, 4) for y in r3m], "color": C_A}]}
yt = [(v, str(v)) for v in range(0, 6)]
phase_row = lambda: (f'<div style="display:flex; gap:0"><div style="flex:none; width:56px"></div>'
                     f'<p style="flex:none; width:350px; font-size:24px; color:{MUTED}; text-align:center">Phase 1: hide facts</p>'
                     f'<p style="flex:none; width:350px; font-size:24px; color:{MUTED}; text-align:center">Phase 2: hide events</p></div>')
chart_col = lambda title, chart, legend_html: col(h3(title, INK, 28), legend_html, chart, phase_row(), gap=10, extra="; flex:none; width:790px")
add("run1-losses", head("DomusFM, the baseline", "The two training losses: run 1 stopped learning, the fixed game kept learning", DOM)
    + row(chart_col("Matching loss: find your own window among 128",
                    line_chart([("run1", C_B, r1), ("run3", C_A, list(zip(r3x, r3c)))], 40000, 5, yt, xt, 700, 300, match_spec,
                               hlines=[(CHANCE, ZERO)], vlines=[20000], label="Matching loss: run 1 falls to zero by step 1,000; run 3 stays at 0.7 to 0.8"),
                    key([("Run 1, paper's game", C_B, False), ("Run 3, fixed game", C_A, False), ("Chance 4.85", ZERO, True)])),
          chart_col("Fill-in-the-blanks loss: guess what was hidden",
                    line_chart([("run3", C_A, list(zip(r3x, r3m)))], 40000, 5, yt, xt, 700, 300, fill_spec,
                               vlines=[20000], label="Fill-in-the-blanks loss: falls from 4.1 to about 0.2 in phase 1, 0.5 to 0.7 in phase 2"),
                    key([("Run 3: hidden sensor, room and ON/OFF (not in run 1)", C_A, False)])), gap=84)
    + row(card(p(f"<b>Matching:</b> 4.85 means pure guessing, 0 means never wrong. Run 1 hit {r1[1][1]:.3f} by step 1,000: the game was too easy "
                 f"to teach anything. With the fixes it stays at {min(r3c[1:20] + r3c[21:]):.1f}–{max(r3c[1:20] + r3c[21:]):.1f}: hard enough to keep learning. "
                 f"The jump at 20k is the switch to the harder phase-2 game.", 24)),
          card(p(f"<b>Fill in the blanks:</b> lower means better guesses. It falls from {r3m[0]:.1f} to about {min(r3m[:20]):.1f} when single facts are hidden, "
                 f"and sits at {min(r3m[21:]):.1f}–{max(r3m[21:]):.1f} when whole events are hidden (harder).", 24)), gap=84),
    notes="Matching loss = the contrastive (InfoNCE) loss; fill-in-the-blanks = the masked-attribute loss added by fix 5. Run 1: results/domusfm_corpus "
          "(one pretraining, logged every 1,000 steps). Run 3: mean of its five pretrained models, which are nearly identical; run 2 looks the same "
          "(0.70–0.82). Chance = ln 128 = 4.85 for both. Phase 1 hides single facts (attribute masking); phase 2 hides whole events.")

diff_cols = [
    (OK, OKT, "From the paper (added in run 3)", [
        "<b>Clean the data:</b> drop repeated ON/ON or OFF/OFF events (10 % of events)",
        "<b>Test on the paper's own datasets:</b> Kasteren A, Kasteren C, MuRAL",
        "<b>Leave each test dataset out</b> of its own pretraining (5 models)"]),
    (DOM, DOMT, "Our own changes", [
        "<b>The six game fixes</b> (two slides back)",
        "<b>Far more pretraining data:</b> 77 CASAS homes (the paper used 2)",
        "<b>6 extra test homes</b> from CASAS",
        "<b>Guessed settings</b> the paper does not give (28.6M vs 36.1M parameters)"]),
    (HOME, HOMET, "Stricter testing", [
        "<b>Test on a later time</b> than training. The paper probably splits at random, which lets near-copies of test windows into training",
        "<b>The 6 CASAS test homes</b> are never seen in any pretraining",
        "<b>Random split only as a check</b>, to compare with the paper"]),
]
add("vs-paper", head("DomusFM, the baseline", "How our DomusFM differs from the paper, in plain words", DOM)
    + row(*[card(pill(t, bgt, c) + col(*[p(x, 26, INK) for x in items], gap=16), gap=18) for c, bgt, t, items in diff_cols], gap=24)
    + card(p("<b>What it means:</b> our absolute scores are lower than the paper's because our test is harder, not because the model is weaker. "
             "The gain from pretraining for activity recognition, which the stricter test measures fairly, is as large as the paper's or larger on 3 of the 4 shared datasets.", 26, INK),
           flex="none", bg=HOMET, border="#B9CDE8")
    + p("<b>Smaller, to save time:</b> 3 test splits (paper 5), 5 % and 30 % labels (paper 5–30 %), next-30 only. "
        "<b>Not done yet:</b> Orange4Home (email-only dataset), the clustering task, the paper's other baseline models.", 24),
    notes="Paper: Fiori et al., arXiv 2602.01910. Full list with paper sections: docs/DOMUSFM_REPRODUCTION.md ('What follows the paper', 'Assumptions', run 3). "
          "Why a later-time test is stricter: consecutive windows share 29 of 30 events, so a random split puts near-copies of each test window in training.")


gain_rows = [("Activity, 5 % labels", "+0.097", "+0.068", "+0.140", "10 / 10", "+0.076"),
             ("Activity, 30 % labels", "+0.099", "+0.027", "+0.208", "9 / 10", "+0.059"),
             ("Next-30, 5 % labels", "+0.038", "+0.055", "+0.014", "9 / 10", "+0.053"),
             ("Next-30, 30 % labels", "+0.009", "+0.027", "−0.017", "6 / 10", "+0.030")]
r3stats = [("34 of 40", "settings where pretraining helps (10 test homes × 2 tasks × 2 label amounts)", OK),
           ("10 of 10", "homes where pretraining helps activity recognition with 5 % of labels", OK),
           ("+0.087", "activity gain at 5 % labels on the 7 homes shared with run 2 (run 2: +0.076)", HOME)]
add("run3", head("DomusFM, the baseline", "Run 3: cleaned data, the paper's test sets, one pretraining per test set", DOM)
    + row(*[card(f'<p style="font-family:{FH}; font-size:80px; font-weight:700; line-height:1; color:{c}">{n}</p>' + p(t, 24), pad=28, gap=12) for n, t, c in r3stats], gap=24)
    + p("Mean gain from pretraining (pretrained − no pretraining), 3 time-ordered folds", 24, INK, "; font-weight:600")
    + table(["Setting", "All 10 homes", "6 CASAS homes", "4 paper datasets", "Pretraining wins", "Run 2 (7 homes)"], gain_rows, [24, 15, 15, 16, 15, 15])
    + p("Changes from run 2: repeated ON/ON states removed (paper Appendix A, 10.1 % of events), Kasteren A/C and MuRAL added, each paper test set pretrained on the others plus 77 CASAS homes. Pretrained activity scores on the 7 shared homes barely move (0.530 vs 0.517 at 5 %); next-30 is about 0.02 lower because cleaning changes what the next 30 events are.", 24, MUTED),
    notes="Full tables: results/2026-09-29/domusfm_corpus_clean/comparison.md. Run took 10.1 h on one RTX 4090, 5 pretraining runs of 40,000 steps.")

# ---- run 3: the five pretrained models, which one helped most, per-home gains, overall findings
PAPER_SETS = ["uci_b", "kasteren_a", "kasteren_c", "mural"]
CASAS_TEST = ["hh101", "hh103", "hh105", "hh110", "hh119", "hh122"]
NICE = {"uci_b": "UCI B", "kasteren_a": "Kasteren A", "kasteren_c": "Kasteren C", "mural": "MuRAL"}
SETTINGS = [("adl", "5%"), ("adl", "30%"), ("next30", "5%"), ("next30", "30%")]


def gain(h, t, pct):
    return r3(RUN3, h, t, pct) - r3(RUN3, h, t, pct, "w/o Pretrain")


def sg(v, nd=3):
    return ("+" if v >= 0 else "−") + f"{abs(v):.{nd}f}"


def hbars(rows, series, scale, neg_px, pos_px, row_h, bar_h, label_w, value_w, ticks, spec):
    """Horizontal bar chart from painted boxes. rows: ("header", text) or (label, [value per series], value text).
    spec (JSON) lets build_pptx.js swap the whole block for a native PowerPoint chart."""
    rcol = f"height:{row_h}px; flex:none; display:flex; flex-direction:column; justify-content:center; gap:2px"
    labels, negs, poss, vals = [], [], [], []
    for r in rows:
        if r[0] == "header":
            labels.append(f'<div style="{rcol}; align-items:flex-end"><p style="font-size:24px; font-weight:600; color:{MUTED}; text-align:right">{r[1]}</p></div>')
            negs.append(f'<div style="{rcol}"></div>'); poss.append(f'<div style="{rcol}"></div>'); vals.append(f'<div style="{rcol}"></div>')
            continue
        name, values, vtext = r
        labels.append(f'<div style="{rcol}; align-items:flex-end"><p style="font-size:24px; color:{INK}; text-align:right">{name}</p></div>')
        nb, pb = [], []
        for v, (_, c) in zip(values, series):
            w = round(abs(v) * scale)
            bar = f'<div style="flex:none; width:{max(w, 2)}px; height:{bar_h}px; background:{c}; border-radius:{{r}}"></div>'
            blank = f'<div style="flex:none; height:{bar_h}px"></div>'
            pb.append(bar.replace("{r}", "0 4px 4px 0") if v >= 0 else blank)
            nb.append(bar.replace("{r}", "4px 0 0 4px") if v < 0 else blank)
        negs.append(f'<div style="{rcol}; align-items:flex-end">{"".join(nb)}</div>')
        poss.append(f'<div style="{rcol}; align-items:flex-start">{"".join(pb)}</div>')
        vals.append(f'<div style="{rcol}"><p style="font-size:24px; color:{BODY}">{vtext}</p></div>')
    step = ticks[1][0] * scale
    grid_bg = (f"repeating-linear-gradient(90deg, transparent 0px, transparent {step - 2:g}px, {GRID} {step - 2:g}px, {GRID} {step:g}px)")
    colw = lambda w, extra="": f'display:flex; flex-direction:column; flex:none; width:{w}px{extra}'
    body = (f'<div style="display:flex; gap:16px">'
            f'<div style="{colw(label_w)}">{"".join(labels)}</div>'
            + (f'<div style="{colw(neg_px)}">{"".join(negs)}</div>' if neg_px else "")
            + f'<div style="{colw(pos_px)}; border-left:2px solid {ZERO}; background:{grid_bg}">{"".join(poss)}</div>'
            f'<div style="{colw(value_w)}">{"".join(vals)}</div></div>')
    tick_row = (f'<div style="display:flex"><div style="flex:none; width:{label_w + 16 + neg_px - 10}px"></div>'
                + "".join(f'<p style="flex:none; width:{step:g}px; font-size:24px; color:{MUTED}">{t}</p>' for _, t in ticks) + "</div>")
    spec_attr = html.escape(json.dumps(spec), quote=True)
    return f'<div data-chart="{spec_attr}" style="display:flex; flex-direction:column; gap:6px">{body}{tick_row}</div>'


def legend(series, note=""):
    items = "".join(f'<div style="display:flex; gap:10px; align-items:center"><div style="flex:none; width:28px; height:18px; background:{c}; border-radius:4px"></div>'
                    f'<p style="font-size:24px; color:{BODY}">{n}</p></div>' for n, c in series)
    return f'<div style="display:flex; gap:32px; align-items:center">{items}{p(note, 24, MUTED) if note else ""}</div>'


# Per pretrained model: the homes it was tested on, its mean gain over all of their settings, and its wins.
MODELS = [("uci_b", "UCI B model", ["uci_b"]), ("kasteren_a", "Kasteren A model", ["kasteren_a"]),
          ("kasteren_c", "Kasteren C model", ["kasteren_c"]), ("mural", "MuRAL model", ["mural"]),
          ("casas_test", "CASAS model", CASAS_TEST)]
PRE_EVENTS = {"uci_b": 24_586_445, "kasteren_a": 24_588_475, "kasteren_c": 24_546_463, "mural": 24_582_632, "casas_test": 24_591_111}  # train.log
mstats = {}
for key, name, homes in MODELS:
    gs = [gain(h, t, pct) for h in homes for t, pct in SETTINGS]
    last = RUN3[homes[0]]["pretrain_history"][-1]
    mstats[key] = {"name": name, "mean": sum(gs) / len(gs), "wins": sum(g > 0 for g in gs), "n": len(gs), "loss": last["contrastive"]}
losses = [m["loss"] for m in mstats.values()]

others = lambda k: ", ".join(NICE[x] for x in PAPER_SETS if x != k)
mrows = [(f"{i + 1} · {mstats[k]['name']}", f"77 CASAS homes + {others(k) if k != 'casas_test' else 'all 4 paper datasets'}",
          NICE.get(k, "6 CASAS homes: hh101, 103, 105, 110, 119, 122"), f"{PRE_EVENTS[k] / 1e6:.1f} M", f"{mstats[k]['loss']:.2f}")
         for i, (k, _, _) in enumerate(MODELS)]
add("run3-models", head("DomusFM, the baseline", "Run 3 trained five pretrained models, one per test group", DOM)
    + table(["Pretrained model", "Pretrained on (unlabelled)", "Fine-tuned and tested on", "Events seen", "Loss at end"], mrows,
            [18, 35, 25, 11, 11], hl=[4])
    + row(card(h3("Why five") + p("The paper's rule: a test dataset never appears in its own pretraining, but the other paper datasets do. "
                                  "So homes with cupboard, fridge and flush sensors have similar homes in pretraining (run 2 had one CASAS-only model).", 24)),
          card(h3("Near-twins") + p(f"The 77 CASAS homes are about 95 % of every model's batches, and all five end at the same loss "
                                    f"({min(losses):.2f}–{max(losses):.2f}). Differences in results come from the <b>test homes</b>, not from one model being better trained.", 24)), gap=24),
    notes="Each model: 40,000 steps (20,000 attribute masking + 20,000 event masking), about 68 min on one RTX 4090. "
          "Loss at end = contrastive loss at the last logged step. Model 5 is shared by the six CASAS test homes, which are never pretrained on. "
          "Config: configs/domusfm_corpus_clean.yaml (pretrain_groups).")

ranked = sorted(mstats.values(), key=lambda m: -m["mean"])
brows = [(m["name"], [m["mean"]], f"{sg(m['mean'])} · helped {m['wins']} of {m['n']}") for m in ranked]
bspec = {"type": "bar", "horizontal": True, "categories": [m["name"] for m in ranked],
         "series": [{"name": "Mean gain from pretraining", "values": [round(m["mean"], 4) for m in ranked], "color": C_A}],
         "min": 0, "max": 0.14, "step": 0.02, "fmt": "+0.000"}
best_cards = [
    (OK, "Biggest gain: UCI B model",
     f"Activity {sg(gain('uci_b', 'adl', '5%'), 2)} / {sg(gain('uci_b', 'adl', '30%'), 2)} (5 % / 30 % labels); Kasteren A close behind with the single biggest gain, "
     f"{sg(gain('kasteren_a', 'adl', '30%'), 2)}. <b>Why:</b> the two smallest test sets (4,666 and 2,636 events): from scratch, so few labels give only 0.26–0.28, "
     f"so pretrained features fill the gap."),
    (HOME, "Most consistent: CASAS model",
     f"Helped {mstats['casas_test']['wins']} of {mstats['casas_test']['n']} settings on 6 homes, and the only model that helps next-30 on every home it was tested on "
     f"(+0.01 to +0.07). <b>Why:</b> its test homes look like its pretraining data (CASAS style sensors, about 95 % of batches), so it also learned their event order."),
    (DOM, "Smallest gain: Kasteren C model",
     f"{sg(mstats['kasteren_c']['mean'])} on average. <b>Why:</b> no room to improve. 83 % of its events are “go to bed”, so even the untrained model scores 0.85. "
     f"Its top absolute score (0.88) says little about the model."),
]
add("run3-best", head("DomusFM, the baseline", "Which model helped most: UCI B and Kasteren A gained most, CASAS was steadiest", DOM)
    + p("Mean gain from pretraining per model (pretrained − no pretraining, F1), over both tasks and both label amounts", 24, INK, "; font-weight:600")
    + hbars(brows, [("Mean gain", C_A)], 4400, 0, 616, 44, 26, 250, 330,
            [(0, "0"), (0.02, "0.02"), (0.04, "0.04"), (0.06, "0.06"), (0.08, "0.08"), (0.10, "0.10"), (0.12, "0.12")], bspec)
    + row(*[card(h3(t, c, 28) + p(d, 24)) for c, t, d in best_cards], gap=20),
    notes="Gains are averaged over activity and next-30 at 5 % and 30 % labels (4 settings per home; the CASAS model covers 6 homes, 24 settings). "
          "Pretrained models are near-identical, so 'best model' means where pretraining paid off most. Source: results/2026-09-29/domusfm_corpus_clean/results.json.")

hrows, cats, adl_g, nxt_g = [], [], [], []
for grp, homes in (("4 paper datasets", PAPER_SETS), ("6 CASAS homes", CASAS_TEST)):
    hrows.append(("header", grp))
    for h in homes:
        a = (gain(h, "adl", "5%") + gain(h, "adl", "30%")) / 2
        n = (gain(h, "next30", "5%") + gain(h, "next30", "30%")) / 2
        hrows.append((NICE.get(h, h), [a, n], f"{sg(a, 2)} · {sg(n, 2)}"))
        cats.append(NICE.get(h, h)); adl_g.append(round(a, 2)); nxt_g.append(round(n, 2))
pspec = {"type": "bar", "horizontal": True, "categories": cats, "min": -0.05, "max": 0.35, "step": 0.05, "legend": False, "fmt": "+0.00;-0.00;0.00",
         "series": [{"name": "Activity recognition", "values": adl_g, "color": C_A}, {"name": "Next-30 prediction", "values": nxt_g, "color": C_B}]}
add("run3-perhome", head("DomusFM, the baseline", "Per home: activity gains everywhere; next-30 gains only on CASAS homes", DOM)
    + legend([("Activity recognition", C_A), ("Next-30 prediction", C_B)], "Gain from pretraining, mean of 5 % and 30 % labels")
    + row(hbars(hrows, [("Activity", C_A), ("Next-30", C_B)], 2000, 100, 700, 42, 18, 190, 200,
                [(0, "0"), (0.1, "0.1"), (0.2, "0.2"), (0.3, "0.3")], pspec),
          card(p(f"<b>Activity:</b> every home gains; the small paper datasets gain most (UCI B, Kasteren A, MuRAL {sg(min(adl_g[0], adl_g[1], adl_g[3]), 2)} to {sg(max(adl_g[0], adl_g[1], adl_g[3]), 2)}).", 24)
               + p(f"<b>Next-30:</b> the six CASAS homes gain {sg(min(nxt_g[4:]), 2)} to {sg(max(nxt_g[4:]), 2)}; the four paper datasets stay near zero.", 24)
               + p("<b>Why:</b> predicting the next events needs a home's own event order. The CASAS test homes resemble the model's pretraining data; the paper datasets are small and unlike CASAS homes.", 24),
               flex="1", gap=16), gap=24),
    notes="Values right of each row: activity gain · next-30 gain, each the mean of the 5 % and 30 % label settings. Per-setting tables: results/2026-09-29/domusfm_corpus_clean/comparison.md.")

pap = [("uci_b", "UCI B"), ("kasteren_a", "Kasteren A"), ("kasteren_c", "Kasteren C"), ("mural", "MuRAL")]
PAPER_ADL = {"uci_b": (0.38, 0.60), "kasteren_a": (0.48, 0.68), "kasteren_c": (0.59, 0.81), "mural": (0.60, 0.80)}
prow = []
for h, nm in pap:
    rf = f"{r3(RF, h, 'adl', '5%'):.2f} / {r3(RF, h, 'adl', '30%'):.2f}" if h in RF else "not run"
    prow.append((nm, f"{r3(RUN3, h, 'adl', '5%'):.2f} / {r3(RUN3, h, 'adl', '30%'):.2f}",
                 f"{r3(RUN3, h, 'adl', '5%', 'w/o Pretrain'):.2f} / {r3(RUN3, h, 'adl', '30%', 'w/o Pretrain'):.2f}", rf,
                 f"{PAPER_ADL[h][0]:.2f} / {PAPER_ADL[h][1]:.2f}"))
add("run3-paper", head("DomusFM, the baseline", "Against the paper: pretraining helps, our test is stricter", DOM)
    + p("Activity recognition, weighted F1, 5 % / 30 % labels", 24, INK, "; font-weight:600")
    + table(["Dataset", "Ours, pretrained", "Ours, no pretraining", "Ours, random folds", "Paper (Table 1)"], prow, [18, 21, 22, 20, 19])
    + row(card(p("<b>Pretraining helps at least as much as in the paper:</b> Kasteren A +0.33 at 30 % labels (paper +0.11), MuRAL +0.19 (paper +0.04), UCI B +0.30 (paper +0.24).", 24)),
          card(p("<b>Lower absolute scores come from the test, not the model:</b> with random folds, which let near-copies of test windows into training (likely the paper's protocol), Kasteren A reaches 0.82 and MuRAL 0.79 at 30 % (paper: 0.68 and 0.80).", 24)),
          card(p("<b>Caveats:</b> Kasteren C looks high because 83 % of events are “go to bed” (even untrained: 0.86). Next-30 gains on these small sets are near zero (paper: +0.15 to +0.25).", 24)), gap=20),
    notes="Random-fold diagnostic reuses run 3's pretrained backbones for UCI B, Kasteren A and MuRAL; only the fold split changes. Results in results/2026-09-29/domusfm_corpus_clean/random_folds/.")

act_w = sum(gain(h, "adl", pct) > 0 for h in RUN3 for pct in ("5%", "30%"))
nxt_w = sum(gain(h, "next30", pct) > 0 for h in RUN3 for pct in ("5%", "30%"))
findings = [
    ("1", "Pretraining works for activity recognition",
     f"It helps in {act_w} of 20 activity settings, by +0.10 on average, at 5 % and at 30 % of labels. The paper's main claim holds."),
    ("2", "It helps most where labels are scarce",
     "The small paper datasets gain most: Kasteren A +0.33 and UCI B +0.30 at 30 % labels. The UCI B model has the biggest average gain."),
    ("3", "It does little for next-30 prediction",
     f"It helps in {nxt_w} of 20 next-30 settings, only +0.04 / +0.01 on average, and about zero on the paper datasets. Only CASAS homes gain."),
    ("4", "The five models are equally good",
     "Same data (95 % CASAS batches) and the same final loss. Differences come from the test homes: size, labels and how CASAS-like they are."),
    ("5", "Our test is stricter than the paper's",
     "Time-ordered folds. With the paper's likely random folds, Kasteren A reaches 0.82 and MuRAL 0.79 (paper 0.68 and 0.80)."),
    ("6", "Kasteren C and one-run caveats",
     "Kasteren C is easy (83 % “go to bed”). One run per setting, 3 folds: differences under about 0.03 on one home are noise."),
]
add("run3-findings", head("DomusFM, the baseline", "Run 3 overall findings", DOM)
    + grid([card(f'<p style="font-family:{FH}; font-size:40px; font-weight:700; line-height:1; color:{DOM}">{n}</p>' + h3(t, INK, 28) + p(d, 24), pad=22, gap=8)
            for n, t, d in findings], 3, 20)
    + card(p("<b>What it means for HomeFM:</b> this run, the fixed and cleaned DomusFM, is the baseline to beat. Its gap is predicting what comes next, "
             "which is exactly what HomeFM's game 1 (next event and when) trains.", 24, INK), bg=HOMET, border="#B9CDE8"),
    notes="Counts and averages from results/2026-09-29/domusfm_corpus_clean/results.json; random-fold numbers from random_folds/results.json. "
          "Plain-language write-up: results/2026-09-29/domusfm_corpus_clean/SUMMARY.md.")


# ---------------------------------------------------------------- 3. Limitations
lims = [("L1", "Closed label set", "Names only activities it has labels for", "Match minutes with words: any concept"),
        ("L2", "Binary events only", "Sees fridge ON, not fridge at 180 W", "Numbers enter as numbers"),
        ("L3", "30-event window", "Memory of 2 min to 6 h, never weeks", "1-min steps, hours in context, day tokens"),
        ("L4", "Single occupant", "Cannot tell people or pets apart", "Person cards per resident and pet"),
        ("L5", "One label per window", "Cannot count, or see two things at once", "Start/end head, multi-tags, count rules"),
        ("L6", "No anomaly / health", "Cannot say unusual, or sensor broken", "Surprise score, anomaly and health engines"),
        ("L7", "No timing in forecasts", "Knows what comes next, not when", "Also predicts time until the next event"),
        ("L8", "Not searchable by text", "Cannot be searched with words", "Minutes stored in the same space as text"),
        ("L9", "No audio or vision", "Cannot hear a cry or see a parcel", "On-device sound and camera tags as events"),
        ("L10", "Small, old data", "Never saw locks, cameras or vacuums", "More homes, simulated homes, pilot homes"),
        ("L11", "Weak training game", "Learns sensor quirks, not behaviour", "Next event + hidden chunks (+ sentences)")]
add("limits-map", head("11 limitations", "Every DomusFM limitation has a HomeFM remedy", DOM)
    + table(["#", "Limitation", "In one line", "HomeFM fix"], lims, [6, 22, 35, 37]),
    notes="The next slides take them two at a time. L11, the training game, gets its own section in pretraining.")


def lim_card(code, name, status, simple, example, wrong, fix):
    return card(row(pill(code, DOMT, DOM), h3(name, size=32), gap=14, extra="; align-items:center")
                + p(f"<b>Simple words.</b> {simple}", 24)
                + p(f"<b>Breaks on:</b> <i>{example}</i>", 24, INK)
                + p(f"<b>What goes wrong.</b> {wrong}", 24)
                + p(f"<b>HomeFM fix.</b> {fix}", 24, HOME)
                + p(f"Status: {status}", 24, MUTED), gap=14)


L = {
    "L1": ("Closed label set (R1)", "tagger designed; alignment built per window",
           "It learns patterns alone, but can only <b>name</b> an activity it was shown labelled examples of.",
           "“How many times did someone vacuum today?” No public dataset labels vacuuming.",
           "No output for vacuuming. Adding it needs days of labelled examples a family will not provide.",
           "Minutes are trained to sit near sentences (SigLIP). A new concept is a new sentence, no labels."),
    "L2": ("Binary events only (R2, R6)", "numbers as input built; detectors designed",
           "Meters (power, temperature, humidity) are squashed to on/off and the amount is thrown away.",
           "“Is my fridge OK?” A failing compressor runs 25 min instead of 12 and draws 20 % more.",
           "Both look like fridge ON … OFF. No energy answers, slow faults invisible, hand-tuned thresholds.",
           "Numbers enter as numbers (sent when the value changes), normalised per device."),
    "L3": ("Fixed 30-event window (R3)", "1-minute stream built; day tokens designed",
           "Always the last 30 events: about 2 min in a busy kitchen, 6 h at night, never weeks.",
           "“Cooking in the last 4 hours?” A 40-min session spans many windows; none sees start and end.",
           "Long activities cut up, quiet periods blurred together, slow trends invisible.",
           "Fixed 1-minute slots (4 h = 240 steps), a stream of hours, day tokens for weeks."),
    "L4": ("Single occupant (R4)", "designed",
           "Assumes one resident. A motion sensor fires the same for grandma, her grandson or the dog.",
           "“Did grandma eat lunch?” while her grandson cooks at noon.",
           "No people counts, no per-person care questions, pets cause false alarms at night.",
           "A person card per resident and pet, per-room counts; honest ranges with motion sensors only."),
    "L5": ("One label per window (R5)", "designed",
           "One label per moment; never where an activity starts and ends, and no overlaps.",
           "“How many times did we cook today?” A 2-minute pause splits one session into three.",
           "Counts and durations unreliable; cooking while the baby cries loses one of the two.",
           "Several tags per minute, a start/end head, per-activity rules (merge gaps, minimum duration)."),
    "L6": ("No anomaly or device health (R6)", "surprise score built; engines designed",
           "Describes what happens, but not how unusual it is, or that a device is broken.",
           "“Anything unusual last night?” Front door at 03:12; a PIR stuck ON looks like someone home.",
           "No security alerts, no fault detection, no early warning of changing routines.",
           "Next-event surprise, −log p(event | history), plus anomaly and device-health engines that explain flags."),
    "L7": ("Forecast without timing (R7)", "next-event head with Δt built",
           "Guesses which events are likely soon, but not when, or in what order.",
           "“Will Dad be up soon?” Bedroom events are likely in the next 30: in 5 min or 3 h?",
           "No useful when-forecasts; skipped meals or missed medicine cannot be detected.",
           "The next-event head predicts device, value and time until it happens (log-normal mixture)."),
    "L8": ("Not searchable by text (R10)", "alignment built per window; vector store designed",
           "Its vectors are not connected to language: you cannot type a question and find moments.",
           "“When did someone come home late this week?”",
           "Open-ended and summary questions cannot use the model at all.",
           "Minute vectors are projected into the text space; the agent's semantic search finds them."),
    "L9": ("No audio or vision (R2)", "detectors designed",
           "It cannot hear or see: anything that flips no sensor is invisible.",
           "“How many times did my kid cry?” “How many parcels came yesterday?”",
           "Whole domains (childcare, pets, deliveries) cannot be answered.",
           "On-hub detectors send tags with a confidence as Home Tokens; HomeFM fuses them. Raw media stays home."),
    "L10": ("Small, old pretraining data", "84 public homes in use; simulator scaffold; pilots planned",
            "Learned from a few public datasets: mostly one-person homes with older binary sensors.",
            "A new home with Matter plugs, a smart lock, a doorbell camera and a robot vacuum.",
            "Understanding of new devices rests only on their text description, with no practice data.",
            "More homes, energy datasets, simulated homes with injected faults, pilot and donated homes."),
}
for a, b in [("L1", "L2"), ("L3", "L4"), ("L5", "L6"), ("L7", "L8"), ("L9", "L10")]:
    add(f"lim-{a[1:]}-{b[1:]}", head("11 limitations", f"{a} and {b}", DOM)
        + row(lim_card(a, *L[a]), lim_card(b, *L[b]), gap=28))

# ---------------------------------------------------------------- 4. Architecture
def layer(label, *boxes, bg=CARD, border=LINE):
    return row(f'<p style="width:230px; flex:none; font-size:24px; font-weight:600; color:{HOME}; align-self:center">{label}</p>',
               *[box(b, bg, border) for b in boxes], gap=14)


add("system", head("Architecture", "Four layers: HomeFM sits in layer 2")
    + col(layer("L1 · Perception (on device)", "Binary sensors<br>PIR · contact · leak", "Telemetry<br>power · temp · CO₂", "Audio expert<br>open-vocab tags", "Vision expert<br>detector · people count"),
          arrow_d(),
          layer("Event bus", "Every signal in one common format: the Home Token", bg=HOMET, border="#B9CDE8"),
          arrow_d(),
          layer("L2 · HomeFM", "Streaming encoder → heads: tagger · episodes · occupancy · forecast · anomaly · device health · search vectors", bg=HOMET, border="#B9CDE8"),
          arrow_d(),
          layer("L3 · Knowledge", "Event store", "Episodes · states", "Vector index", "Ontology + capabilities", "Anomaly · health · routines"),
          arrow_d(),
          layer("L4 · Query agent", "LLM planner ⇄ tool API", "Answer + evidence + confidence", bg=CARD), gap=8),
    notes="Raw events go both to the event store and to HomeFM. The agent never reads vectors; it calls tools over the stores.")

paths = [("Fast path", "The concept is stored (cooking, parcel, bark): exact SQL over events, episodes and states.", "Search"),
         ("Open path", "Not stored: the question becomes a vector and is matched against stored minute vectors, with a confidence.", "Lightbulb"),
         ("Analysis path", "Anomaly, device health, comparison with baselines, forecasts.", "Chart"),
         ("Capability check", "No sensor can observe it: an honest “I can't sense that here”.", "Warning")]
add("paths", head("Architecture", "Three answer paths, and an honest refusal")
    + grid([card(f'<x-icon name="{i}" style="color:{HOME}; width:40px; height:40px"></x-icon>' + h3(t) + p(d, 26)) for t, d, i in paths], 2, 24)
    + card(p("<b>Promotion loop.</b> An open-path concept that is asked often is added to the stored concept list, and from then on it answers by the fast path. Because minute vectors are kept, its full history is filled in retroactively.", 26), bg=HOMET, border="#B9CDE8"))

chain = ["Events<br><i>words</i>", "Minutes<br><i>sentences</i>", "Hours<br><i>the story</i>", "Days and weeks<br><i>chapters</i>"]
cc = []
for i, c in enumerate(chain):
    cc.append(box(c, "#1F2E4F", "#2E4270", ON, 32))
    if i < 3:
        cc.append(arrow_r(HOMED))
add("diary", f"""
<p style="font-size:24px; font-weight:600; letter-spacing:2px; text-transform:uppercase; color:{HOMED}">Architecture</p>
<h2 style="font-family:{FH}; font-size:72px; font-weight:600; line-height:1.1; color:{ON}">HomeFM reads a home like a diary</h2>
<div style="flex:1"></div>
{row(*cc, gap=16)}
<div style="flex:1"></div>
{p("Each event becomes a card of four facts, each minute one summary vector, minutes are read in order for hours of context, and daily summaries cover weeks. Side branches zoom back into single events (next event, surprise) and single devices (health).", 30, SOFT)}""", dark=True)

inputs = col(*[box(t, HOMET, "#B9CDE8", size=24) for t in ["What: device sentence", "Value: state + number", "When: time + gaps", "Meta: type + confidence"]], gap=10)
main = row(f'<div style="width:400px; flex:none">{inputs}</div>', arrow_r(),
           box("Attribute fusion<br>→ event vector", flex="1"), arrow_r(),
           box("L1 Moment encoder<br>events → 1 vector per minute", flex="1"), arrow_r(),
           box("L2 Stream transformer<br>live or look-back", flex="1"), arrow_r(),
           box("Heads<br>tagger · start/end · people · search", flex="1"), gap=12, extra="; align-items:center")
branches = row(card(pill("Built", OKT, OK) + h3("Event read-out") + p("A small transformer back over single events, using the previous minute's context: next event and surprise.", 24)),
               card(pill("Designed", DOMT, DOM) + h3("L3 Daily memory") + p("A few summary tokens per day; a small transformer over weeks: routines and drift.", 24)),
               card(pill("Designed", DOMT, DOM) + h3("Entity axis") + p("Each device's own timeline over time: a health vector per device, compared with its past and the fleet.", 24)), gap=24)
add("model", head("Architecture", "The HomeFM model: four blocks and two side branches") + main + branches
    + p("Built: fusion, moment encoder, stream transformer (both modes), event read-out, next-event head. HomeFM is <b>encoder-only</b>; the only decoder is the LLM agent, outside the model.", 24, MUTED),
    notes="Encoder-only: every transformer turns inputs into vectors. No text is produced on the model path.")

tok = [("ts", "float, unix seconds", "Event time; the home's timezone is stored separately"),
       ("home_id", "string", ""),
       ("entity", "Entity", "Points to a device registry row (item, room, type, capability)"),
       ("modality", "enum", "binary · scalar · audio_tag · vision_det · embedding"),
       ("state", "OFF · ON · NA", "For binary signals"),
       ("value", "float or none", "Scalar reading: 1,850 W stays 1,850 W"),
       ("vector", "floats or none", "Optional audio or vision embedding from a detector"),
       ("tag", "row or none", "Planned: detector tag, e.g. “baby crying”, from the tag vocabulary"),
       ("confidence", "0 to 1", "1.0 for physical sensors; the model score for detectors"),
       ("source", "enum", "sensor · expert · homefm"),
       ("person_id", "string or none", "When known: camera, phone presence, wearable")]
add("token", head("Architecture", "The Home Token: one format for every signal")
    + table(["Field", "Type", "Meaning"], tok, [18, 22, 60])
    + mono("raw  2026-09-24 18:40:05  nursery_mic  tag=“baby crying” 0.87   →   entity=row 2 · tag=row 17 · audio_tag · conf 0.87 · expert", 24))

reg = [("M014", "Kitchen Motion · Kitchen · motion", "ceiling in kitchen, motion sensor", "0"),
       ("fridge_power", "Fridge Plug · Kitchen · power", "fridge in kitchen, power sensor", "1"),
       ("nursery_mic", "Nursery Mic · Nursery · audio", "microphone in nursery, audio sensor", "2")]
add("registry", head("Architecture", "Devices are sentences: the device registry")
    + table(["Device code", "Names from the hub", "Description (embedded once, frozen MiniLM)", "Row"], reg, [17, 30, 45, 8])
    + row(card(h3("Code = lookup key") + p("M014 is never turned into numbers. It only looks up its sentence's vector, like a phone showing “Mum” for a number.", 24)),
          card(h3("One model, any home") + p("Model weights are shared by every home; the registry is per home. A new device is a new row: no retraining.", 24)),
          card(h3("Onboarding") + p("Import names from the hub, normalise, guess unclear devices from behaviour (Plug 3: ~150 W cycling every 40 min → fridge), one-tap confirm.", 24)), gap=24)
    + p("150 devices × 384 numbers ≈ 230 KB. The next-event head can only predict devices in this home's registry.", 24, MUTED))

enc = [("What", "Device registry row: MiniLM sentence vector, 384 numbers (frozen) → linear projection to d (learned, ~98k weights)."),
       ("Value", "Learned vector for ON / OFF / NA + a small network on the number, normalised per device (1,850 W on a 300 W stove ≈ +2.1)."),
       ("When", "Cyclic hour and weekday (sin/cos at 4 speeds) + log time since the previous event and since this device last fired."),
       ("Meta", "Signal type (5 learned vectors: binary, scalar, audio tag, vision tag, embedding) + confidence.")]
cmp_rows = [("Device", "3 sentence vectors: item, type, room", "1 sentence “{item} in {room}, {type} sensor” + projection"),
            ("Value", "Status only: OFF / ON / MASK", "ON / OFF / NA + a network on the number"),
            ("Time", "Cyclic weekday + hour + 3,600 seconds-in-hour vectors", "Cyclic weekday + hour + gap network"),
            ("Meta", "None (all events binary)", "Signal type + confidence")]
add("fusion", head("Architecture · step 1", "Attribute fusion: four facts become one event vector")
    + grid([card(h3(t) + p(d, 24), pad=22) for t, d in enc], 4, 20)
    + box("Each fact gets an attribute-type tag → one self-attention layer across the four → average → normalise = event vector (d numbers). “Kitchen motion at 07:00” ≠ “at 23:30”.", HOMET, "#B9CDE8", INK, 24, "none")
    + table(["Fact", "DomusFM", "HomeFM"], cmp_rows, [12, 40, 48]))

att = [("Kitchen motion × 3", "0.5 each", "5 % each, 16 % together"), ("Fridge open", "1.5", "15 %"),
       ("Stove ON, 1,850 W", "3.0", "66 %"), ("“Quiet” option", "0.0", "3 %")]
add("moment", head("Architecture · step 2", "Moment encoder: any number of events → one vector per minute")
    + row(col(p("<b>Why fixed minutes.</b> A cooking minute has 40 events, a night minute none. One vector per minute makes “the last 4 hours” always 240 steps, in busy and quiet homes alike.", 24),
              p("<b>Why not just average.</b> 38 motion events would drown the one stove turning on, the most important thing that happened.", 24),
              p("<b>Four learned reporters</b> (K = 4 latent queries, Perceiver cross-attention, segment softmax for all minutes at once) each learn to focus on different things.", 24),
              p("<b>Quiet key.</b> An empty minute still gets a proper vector: “nobody moved all morning” is information.", 24),
              p("<b>Added:</b> the event count (log scale) and the minute's time of day and weekday.", 24), gap=16, extra="; flex:1"),
          col(p("Minute 18:31, as seen by an “appliances” reporter:", 24, INK, "; font-weight:600"),
              table_w(760, ["Event", "Score", "Share of attention"], att, [42, 18, 40], hl=[2]),
              p("A “movement” reporter weighs the same minute mostly toward the motion events. The four reports are averaged.", 24, MUTED), gap=16, extra="; width:760px; flex:none"), gap=48))

modes = [("Runs", "Every minute", "About once an hour"),
         ("Reads", "The new minute + everything before it", "A block of recent hours at once"),
         ("Each minute looks at", "Only earlier minutes", "Earlier and later minutes in the block"),
         ("Good for", "What is happening now, alarms, surprise, forecasts", "Correct counts and durations for history"),
         ("Weakness", "A pause looks like an end", "Delayed: time has to pass first"),
         ("Code", "encode(batch, causal=True)", "encode(batch, causal=False)")]
add("stream", head("Architecture · step 3", "The stream transformer: one set of weights, two reading modes")
    + table(["", "Live mode (causal)", "Look-back mode (bidirectional)"], modes, [20, 40, 40])
    + p("Only the attention mask changes. Learned positions up to 1,024 minutes (about 17 h). Corpus configuration: d = 256, 6 layers, 8 heads. A KV cache (planned) makes each new minute cost one step on the hub. Look-back never sees the real future: at 20:00 it re-reads 17:00–20:00, which has already happened.", 24))

tl = [("18:30", "stove on"), ("18:52", "stove off (resting)"), ("18:59", "stove on again"), ("19:22", "stove off"), ("19:22–19:25", "washing up"), ("19:26", "kitchen empty")]
add("dinner", head("Architecture · worked example", "One dinner: live mode guesses, look-back corrects")
    + row(*[box(f"<b>{t}</b><br>{e}", CARD, LINE, INK, 24) for t, e in tl], gap=12)
    + p("Truth: <b>one</b> cooking session, 18:30–19:22.", 26, INK)
    + row(card(pill("Live mode, minute by minute", DOMT, DOM) + p("18:57: stove off 5 min → closes <b>cooking 18:30–18:52</b>", 24)
               + p("18:59: stove on → opens a <b>new</b> session", 24) + p("19:26: kitchen empty → <b>cooking 18:59–19:25</b>", 24)
               + p("Result: 2 sessions, wrong, but the best possible guess at each moment.", 24, DOM)),
          card(pill("Look-back at 20:00", OKT, OK) + p("18:53 re-checked with 18:59 known → a <b>pause</b>, not an end", 24)
               + p("18:59 → the <b>same</b> session continuing", 24) + p("19:23–19:25 tap and wiping → cleaning, so cooking ended 19:22", 24)
               + p("Result: 1 session, 18:30–19:22, correct.", 24, OK)), gap=28)
    + p("Questions about <b>now</b> use live mode; questions about <b>the past</b> use the corrected episodes.", 26, INK))

heads = [("Next event with timing", "Which device, what value, and when", "Forecasts, surprise scores", "Built"),
         ("Tagger (activity by sentence)", "Scores each minute against any text", "“Did anyone vacuum?” with no labels", "Designed"),
         ("Start / end", "Where an activity begins and ends", "Counts and durations", "Designed"),
         ("Occupancy / identity", "People per room, and who", "“How many people are in the living room?”", "Designed"),
         ("Anomaly score", "How unusual this is for this home", "“Anything unusual last night?”", "Surprise built"),
         ("Device health", "Is a device's behaviour drifting?", "“Is the fridge OK?”", "Designed"),
         ("Retrieval embedding", "A searchable vector per minute", "The agent's semantic search", "Designed")]
add("heads", head("Architecture", "Heads: small outputs on the same minute vectors")
    + table(["Head", "Job", "Enables", "Status"], heads, [27, 28, 30, 15])
    + p("DomusFM has two outputs: one label from a fixed list, and an unordered bag of the next 30 events with no timing.", 24, MUTED))

toy = [("“someone is cooking food”", "[0.9, 0.1, 0.0]", "0.78  ← highest"), ("“someone is sleeping”", "[0.0, 0.1, 0.9]", "0.06"),
       ("“someone is vacuuming”", "[0.1, 0.9, 0.1]", "0.23")]
add("tagger", head("Architecture", "A sentence instead of learned weights")
    + row(col(pill("DomusFM", DOMT, DOM), mono("score(Cook) = W_Cook · vector + b_Cook", 24),
              p("W_Cook is learned from labelled cooking examples. A new activity needs new weights: labels and training.", 24), gap=12, extra="; flex:1"),
          col(pill("HomeFM tagger", HOMET, HOME), mono("score = σ( a · cos( P(minute), Q(text(“someone is cooking”)) ) + b )", 24),
              p("P, Q, a and b are shared by all activities. The only activity-specific part is the sentence.", 24), gap=12, extra="; flex:1"), gap=40)
    + row(col(p("Toy example: minute 18:31 projected to [0.85, 0.15, 0.05]", 24, INK, "; font-weight:600"),
              table_w(900, ["Sentence", "Sentence vector", "Score"], toy, [46, 27, 27], hl=[0]), gap=12, extra="; width:900px; flex:none"),
          col(p("<b>P</b>: 256 → 256 → 128 (~99k weights). <b>Q</b>: 384 → 384 → 128 (~197k). Both normalised to length 1.", 24),
              p("It only works because game 3 trains minutes and sentences into one space. Bolting a sentence head onto DomusFM would not work.", 24, DOM),
              p("Several activities can be on at once: each sentence is scored separately.", 24), gap=14, extra="; flex:1"), gap=40),
    notes="The same trick is used at both ends: devices are sentences at the input (an ingredient), activities are sentences at the output (a comparison target).")

sizes = [("Teacher", "300M–1B", "Server GPU", "Pretraining at scale, pseudo-labels"),
         ("Student", "20–50M, int8", "Home hub", "Streaming inference, per-home adaptation"),
         ("Scaffold tiny", "~1–3M", "Laptop / CI", "Smoke tests, objective comparisons on synthetic homes")]
cfgs = [("DomusFM reimplementation", "384", "12 × 12", "28,596,480"),
        ("HomeFM corpus (homefm_corpus.yaml)", "256", "6 × 8 stream + 2 read-out", "8,031,004"),
        ("HomeFM size-matched (homefm_corpus_384.yaml)", "384", "12 × 12 stream + 2 read-out", "28,591,516")]
add("sizes", head("Architecture", "Model sizes: teacher, student, and the corpus runs")
    + table(["Variant", "Parameters", "Where", "Purpose"], sizes, [18, 20, 18, 44])
    + table(["Configuration", "d", "Layers × heads", "Learnable parameters"], cfgs, [44, 8, 28, 20], hl=[1])
    + p("d = 256 was <b>chosen, not tuned</b>: HomeFM runs more parts per window and game 2 keeps a second copy of the model. The size-matched run exists so a win or loss against DomusFM cannot be explained by size alone.", 24))

# ---------------------------------------------------------------- 5. Pretraining
def flowrow(items, bg, border, color=INK):
    out = []
    for i, t in enumerate(items):
        out.append(box(t, bg, border, color, 24))
        if i < len(items) - 1:
            out.append(arrow_r())
    return row(*out, gap=10, extra="; align-items:stretch")


add("pretrain", head("Pretraining", "Pretraining makes the model file; the live pipeline runs it")
    + p("Training · offline, on a server, once per release", 26, HOME, "; font-weight:600")
    + flowrow(["Many homes' data<br>public · simulated · pilot", "Home Tokens", "Event vectors → minutes → stream", "Practice games<br>(no labels)", "Adjust weights<br>millions of steps", "Teacher → int8 student"], HOMET, "#B9CDE8")
    + p("Live · on the home hub, all the time", 26, OK, "; font-weight:600")
    + flowrow(["Sensors + sound/camera tags", "Home Tokens", "Student model", "Heads", "Episode builder", "Stores → agent"], OKT, "#BFDCC8")
    + card(p("<b>A common misunderstanding:</b> pretraining is not a step each event passes through. It produces the model; the live pipeline uses it with frozen weights. The games exist only in training.", 26), bg=CARD),
    notes="Self-supervised games need no labels, except captions for game 3. Labels come later, only for fine-tuning.")

add("domus-game", head("Pretraining · DomusFM's game", "How DomusFM pretrains: hide a little, then find your own window", DOM)
    + row(card(pill("Phase 1 · attribute masking", DOMT, DOM) + p("For about 15 % of events, hide <b>one attribute</b>:", 24)
               + mono("18:29 fridge door · contact · kitchen · OPEN<br>18:29 fridge door · contact · kitchen · ???", 24)),
          card(pill("Phase 2 · event masking", DOMT, DOM) + p("For about 15 % of events, hide <b>all attributes</b>; event layers frozen:", 24)
               + mono("18:30 stove · power · kitchen · ON<br>  ???  ·  ???  ·   ???   · ???", 24)), gap=28)
    + row(box("Original window → model → average → vector A", DOMT, "#E9C3A2"), arrow_r(),
          box("Among 64–128 windows in the batch, each A must pick out its own masked copy B (InfoNCE)", DOMT, "#E9C3A2", flex="1.4"), arrow_r(),
          box("Masked window → model → average → vector B", DOMT, "#E9C3A2"), gap=12, extra="; align-items:stretch")
    + card(p("<b>The key weakness: the model is never asked what was hidden.</b> Masking is only damage; the task is to recognise the window despite it. Knowing that “the stove usually follows the fridge” gives no advantage.", 28, INK), bg=DOMT, border="#E9C3A2"),
    notes="Both phases use the same contrastive game. Only the kind of damage differs.")

dis = [("1 · Never recovers the hidden part", "Only: recognise your own window", "Understanding routines gives no advantage"),
       ("2 · Too little is hidden", "About 15 % masked: 28 of 30 events untouched", "The masked copy is nearly identical"),
       ("3 · Timestamps are a fingerprint", "Exact seconds survive the masking", "Windows matched like serial numbers"),
       ("4 · ON/OFF pairs give it away", "Hide the OFF: it follows from the ON", "Learns sensor mechanics, not behaviour"),
       ("5 · Similar routines pushed apart", "Two normal nights are “different”", "False negatives: routines unlearned"),
       ("6 · One check per window", "One averaged vector per 30 events", "Weak signal; no per-minute meaning"),
       ("7 · Bidirectional only", "Always sees the whole window", "No probability of what comes next"),
       ("8 · Timing ignored", "Order counts; durations do not", "3 s and 30 min visits look alike"),
       ("9 · Common sensors dominate", "About 99 % of CASAS events are motion", "Rare doors, leaks, night events unpractised")]
add("why-drop", head("Pretraining · why HomeFM drops it", "Why HomeFM removes attribute masking and event masking", DOM)
    + table(["Disadvantage", "What happens", "Consequence"], dis, [33, 36, 31])
    + p("Problems 1–6 make the game easy to win without learning behaviour. Problems 7–9 mean that even a perfectly played game trains the wrong skills for a home.", 24, INK),
    notes="From DESIGN.md sections 8.0, 8.1 and 8.6, and our reproduction logs.")

uc = [("Anything unusual last night?", "A probability for what just happened", "Contrastive vectors give no likelihood", "Game 1: surprise = −log p"),
      ("Will Dad be up soon?", "When the next event comes", "No timing; order-free bag of events", "Game 1: time until next event"),
      ("Did we skip breakfast?", "A timed expectation, “expected by 9:00”", "Nothing is expected at a time", "Game 1 (+ G3 multi-horizon)"),
      ("How many times did we cook?", "Meaningful minutes, starts and ends", "One averaged vector per window", "Game 2: per-minute targets"),
      ("A sensor dies", "Fill in from the rest of the home", "Hides random bits, never a whole device", "Game 2: device, room, type masks"),
      ("Live on a small hub", "Past-only reading, cached", "Bidirectional window recomputed per event", "Game 1 trains causal mode"),
      ("Is Mum sleeping worse?", "Similar nights seen as similar", "Similar nights pushed apart", "Games 1–2 have no negatives"),
      ("Did anyone vacuum?", "A link from minutes to words", "No language anywhere", "Game 3: SigLIP")]
add("usecases", head("Pretraining · why HomeFM drops it", "Use cases the masking game would struggle with", DOM)
    + table(["Question", "What it needs", "Why DomusFM's game can't give it", "HomeFM game that trains it"], uc, [23, 26, 28, 23]),
    notes="The masking game is not wrong for everything: with fixes it helps activity recognition with labels. It just never trains forecasting, surprise, per-minute meaning, robustness to missing sensors or language.")

ev = [("UCI B", "0.25", "0.22"), ("hh101", "0.49", "0.56"), ("hh103", "0.54", "0.68"), ("hh105", "0.36", "0.41"),
      ("hh110", "0.32", "0.33"), ("hh119", "0.35", "0.41"), ("hh122", "0.38", "0.43"), ("Mean", "0.385", "0.435")]
add("evidence", head("Pretraining · why HomeFM drops it", "What our runs showed", DOM)
    + row(col(p("Activity F1, 5 % labels, paper's game, 77 homes", 24, INK, "; font-weight:600"),
              table_w(720, ["Held-out home", "Pretrained", "No pretraining"], ev, [40, 30, 30], hl=[7]), gap=12, extra="; width:720px; flex:none"),
          col(p("<b>Loss collapse:</b> 2.35 → about 0.0005 within 5,000 of 40,000 steps. Next-30 prediction: no difference either way.", 26),
              p("<b>Patching the game works for activities:</b> with six fixes, pretraining wins 26 of 28 settings (activity F1 at 5 % labels: 0.517 vs 0.441); run 3 on 10 homes: 34 of 40.", 26),
              p("<b>But the fixed game still trains none of:</b> timing, surprise, per-minute meaning, robustness to a missing device, language.", 26, DOM),
              p("<b>So HomeFM replaces the game instead of patching it.</b> Masking is not thrown away: game 2 hides big structured chunks and must recover them; our DomusFM fix's fill-in-the-blanks loss points the same way.", 26, INK), gap=20, extra="; flex:1"), gap=48))

stg = [("Stage 1", "What happens next?", "Next event: which device, what value, when. Past only.", "No labels"),
       ("Stage 2", "Fill in a missing chunk", "Hide big structured pieces; predict their meaning (JEPA).", "No labels"),
       ("Stage 3", "Make things agree", "Sensors ↔ audio/vision, minute ↔ hour ↔ day, minutes ↔ language, across homes.", "Captions"),
       ("Stage 4", "A few labels, then each home", "Supervised heads; keep playing stages 1–2 on each home's own data.", "Labels, feedback")]
sc = []
for i, (s, t, d, need) in enumerate(stg):
    sc.append(card(pill(s, HOMET, HOME) + h3(t) + p(d, 24) + p(need, 24, MUTED), pad=24))
    if i < 3:
        sc.append(arrow_r())
add("recipe", head("Pretraining · the HomeFM recipe", "The four-stage recipe")
    + row(*sc, gap=12, extra="; align-items:stretch")
    + p("Stages 1 and 2 are trained <b>together</b> (variant E). Stage 3's language part is game 3 (variant F). Stage 3's home-invariance uses gradient reversal so the model cannot tell which home data came from.", 26)
    + card(p("<b>Cheap extra games:</b> guess the time of day from a window · predict the unordered bag of the next k events (DomusFM's task) · consistency checks (stove power should mean someone is in the kitchen).", 24), bg=CARD))

g1 = [("Fridge door open", "Kitchen motion, in ~5 s", "Motion after 4 s", "Close"),
      ("Kitchen motion", "Fridge closes, in ~20 s", "Stove on after 51 s", "Wrong"),
      ("Stove on", "Kitchen motion, in ~1 min", "Motion after 75 s", "Close")]
add("game1", head("Pretraining · game 1", "Game 1: predict the next event, and when")
    + mono("L_next = CE(device) + CE(state) + λ·L_value + NLL_LogNormalMixture(Δt)", 26)
    + row(col(p("<b>Which device:</b> scored against every row of <i>this</i> home's registry, so it never predicts a device the home lacks.", 24),
              p("<b>When:</b> a log-normal mixture over the wait, so “in 10 s” and “in 3 h” can both be likely.", 24),
              p("<b>Surprise for free:</b> how wrong a guess was, −log p(event | history), is the anomaly signal.", 24),
              p("<b>Cannot be cheated:</b> the future is unseen, with no copy or timestamp to peek at. Every event is a question: dozens per window.", 24),
              p("<b>Weak spots:</b> past only, and short-sighted (much of it is “motion OFF after motion ON”).", 24, DOM), gap=14, extra="; flex:1"),
          col(p("Example (hh102, Tuesday evening):", 24, INK, "; font-weight:600"),
              table_w(860, ["After seeing", "The model guesses", "What came next", "Result"], g1, [26, 30, 28, 16]), gap=12, extra="; width:860px; flex:none"), gap=40),
    notes="Technically a marked temporal point process. It trains live mode, forecasting (L7) and the surprise score (L6).")

masks = [("Time block", "All events in a run of minutes (3–10 min)", "Behaviour over minutes", "0.4"),
         ("Device span", "All events of 1–3 devices", "Cross-sensor redundancy (stove ↔ kitchen motion)", "0.2"),
         ("Room", "All devices in one room", "Spatial reasoning, occupancy", "0.2"),
         ("Signal type", "All power readings, or all audio", "Coping when a signal is missing", "0.1"),
         ("Rare devices", "Events sampled ∝ 1 / frequency", "Rare but important events", "0.1")]
add("game2", head("Pretraining · game 2", "Game 2: hide a big chunk, predict its meaning (JEPA)")
    + mono("18:28 motion · 18:29 fridge OPEN · [ 18:30–18:36 HIDDEN ] · 18:37 stove OFF · 18:38 dining light ON", 24)
    + table(["Mask family", "What is hidden", "Forces the model to learn", "Share"], masks, [17, 33, 38, 12])
    + row(card(p("<b>How:</b> the context encoder sees the masked input; a predictor fills in each hidden minute's vector; the target comes from a slowly updated copy that saw everything (99.6 % old + 0.4 % new per step). Loss: Smooth-L1 on normalised vectors, hidden minutes only.", 24)),
          card(p("<b>Why meaning, not events:</b> exact seconds and cupboard-vs-fridge order are noise. <b>Why big chunks:</b> small holes are filled by copying the ON/OFF partner, DomusFM's shortcut. <b>Weak spot:</b> alone, no forecasting or surprise.", 24)), gap=24))

need = [("Activity recognition after fine-tuning with labels", "Works without game 3"), ("Forecasting, anomalies, counts, missing sensors", "Works without game 3"),
        ("“Did anyone vacuum today?” with no labelled examples", "Needs game 3"), ("Search history in plain words; tagger by sentence", "Needs game 3")]
add("game3", head("Pretraining · game 3", "Game 3: match minutes with sentences (SigLIP), optional")
    + row(col(p("<b>How:</b> P(minute vector) and Q(sentence vector) go into a shared 128-number space. Every pair is scored <b>separately</b> as match or no-match, so one minute can match “cooking”, “someone in the kitchen” and “dinner prep”, and repeated routines are not pushed apart. MiniLM stays frozen.", 24),
              p("<b>Caption sources:</b> activity labels turned into sentences, MuRAL's free-text description on every event, simulator captions, checked LLM captions.", 24),
              p("<b>Risk:</b> if every caption is one of ~35 label names, game 3 adds little beyond labels.", 24, DOM), gap=16, extra="; flex:1"),
          col(table_w(820, ["Use", "Game 3?"], need, [70, 30], hl=[2, 3]),
              card(p("<b>Decision:</b> build and prove E (games 1 + 2) first. F = E + game 3 is tested on finding held-out concepts from text, and kept only if it helps.", 24), bg=HOMET, border="#B9CDE8"), gap=20, extra="; width:820px; flex:none"), gap=40))

sbs = [("What is hidden", "1 attribute, or a few events (~15 %)", "The future: the next event", "Minutes, a device, a room, a signal type"),
       ("Must it recover it?", "No: only recognise its own window", "Yes: device, value and time", "Yes: each hidden minute's meaning"),
       ("Guesses per window", "1", "One per event (dozens)", "One per hidden minute"),
       ("Cheap shortcuts", "Timestamps, untouched events, pairs", "None: the future is unseen", "Few: neighbours hidden too"),
       ("Similar routines", "Pushed apart", "No negatives", "No negatives"),
       ("Loss in practice", "~0.001 within 1,000 of 40,000 steps", "Expected to fall gradually", "Expected to fall gradually"),
       ("What vectors learn", "Stay stable when bits are removed", "Routines, order, timing, values", "How the parts of a home fit")]
add("side-by-side", head("Pretraining · comparison", "The same window, three games")
    + table(["", "DomusFM masking", "HomeFM game 1", "HomeFM game 2"], sbs, [18, 28, 26, 28])
    + p("Window: 18:28 kitchen motion · 18:29 fridge OPEN · 18:30 stove 1,850 W · 18:31 motion · … · 18:37 stove OFF · 18:38 dining light ON.", 24, MUTED))

both = [("Reading direction", "Past only (live)", "Both sides (look-back)", "Both modes the system uses"),
        ("Time scale", "Seconds to minutes", "5–10 min, rooms, signal types", "Short and long"),
        ("Level", "Single events", "Whole minutes", "Both"),
        ("Forecasting and surprise", "Yes", "No", "Yes"),
        ("Meaningful minute vectors", "Partly", "Yes", "Yes"),
        ("Robust to missing sensors", "No", "Yes", "Yes"),
        ("Needs labels", "No", "No", "No: all training homes usable")]
add("why-both", head("Pretraining · the choice", "Why games 1 and 2 together (variant E)")
    + table(["", "Game 1", "Game 2", "Together"], both, [28, 22, 26, 24])
    + p("<b>Game 1 teaches what happens next. Game 2 teaches what was going on. A home system needs both.</b> Chosen from the jobs the system must do, not from what worked in other fields.", 28, INK))

abl = [("A", "DomusFM-style contrastive (attribute → event masking)"), ("B", "BERT-style random masking + reconstruction"),
       ("C", "Game 2 only (JEPA)"), ("D", "Game 1 only (next event)"), ("E", "D + C, proposed base"),
       ("F", "E + game 3 + home-invariance"), ("G", "F + refinements (not built)")]
smoke = [("A", "0.82", "40 %"), ("C", "0.81", "40 %"), ("D", "0.84", "48 %"), ("E", "0.85", "49 %")]
add("ablation", head("Pretraining · test, don't assume", "The A–G comparison: same model, data and compute; only the game changes")
    + row(table_w(900, ["Variant", "Game"], abl, [16, 84], hl=[4]),
          col(p("Tiny synthetic smoke test (10–25 s of training): a weak hint only", 24, INK, "; font-weight:600"),
              table_w(700, ["Variant", "Activity F1 (probe)", "Exact counts"], smoke, [26, 40, 34], hl=[3]),
              p("<b>Hypothesis:</b> E beats A–D on anomaly, forecasting and counting and matches A on activities; F adds zero-shot; G trains more stably.", 24), gap=14, extra="; flex:1"), gap=40)
    + p("Scored on: activity F1 (probe and fine-tune) · exact episode counts · anomaly AUROC on injected faults · forecast NLL · retrieval of held-out concepts · 1 % and 5 % label adaptation.", 24, MUTED))

gref = [("G1", "Pair-aware masking", "Hide motion ON and its OFF together", "Easy guesses"),
        ("G2", "Adaptive difficulty", "If the loss collapses, hide bigger chunks (1 → 5 → 15 min; device → room)", "Collapse"),
        ("G3", "Multi-horizon forecasting", "Predict the summary of the next 1, 10 and 60 min, past only", "Routines, when"),
        ("G4", "Routine-aware positives", "Same home, same time of day, other days = probably similar", "False negatives"),
        ("G5", "Collapse guard", "Penalise embedding dimensions whose spread falls below a floor", "JEPA stability")]
add("variant-g", head("Pretraining · variant G", "Five refinements aimed at the collapse we measured")
    + table(["#", "Refinement", "In simple words", "Fixes"], gref, [6, 24, 52, 18])
    + mono("L_G = L_next + λf·L_future + λj·L_jepa(pair-aware, adaptive masks) + λv·L_var + alignment(routine-aware)", 24)
    + p("<b>Success means:</b> the loss falls steadily instead of collapsing, embedding spread stays above the floor, and pretraining beats no pretraining on UCI B and hh101.", 24))

run = [("Training homes", "77: same pool as the DomusFM run (7 test homes held out)"),
       ("Stretch", "30 minutes of 60-second minutes, at most 256 events; one starts every 10 min"),
       ("Per round / rounds", "32 stretches, evenly across homes / 20,000 rounds (first 500 warm-up)"),
       ("Learning rate", "0.0003, AdamW"), ("Model", "d = 256, 6 layers, about 8 million learnable numbers"),
       ("Games", "1 + 2 (variant E); game 2 hides 3–10 min, a device, a room or a signal type"),
       ("Slow copy (game 2)", "99.6 % old + 0.4 % new after each round")]
add("corpus-run", head("Pretraining · the first real run", "The HomeFM corpus run: settings and how we will judge it")
    + table(["Setting", "Value (configs/homefm_corpus.yaml)"], run, [26, 74])
    + row(card(h3("Two runs, one question each") + p("8.0M: can a small model with better games beat DomusFM? 28.6M, size-matched: is HomeFM better at the same size?", 24)),
          card(h3("It succeeds if") + p("The loss falls gradually (not to ~0.001), and pretrained HomeFM beats HomeFM without pretraining on held-out homes, especially at 5 % labels.", 24)), gap=24),
    notes="Neither HomeFM corpus run has been started. We are on DomusFM only for now; these run on request, one at a time on the GPU.")

# ---------------------------------------------------------------- 6. Answers
add("episodes", head("From model to answers", "Episodes: turning minute scores into things you can count")
    + flowrow(["Minute vectors", "Tagger (vs concept sentence) + start/end head", "Smoothing + hysteresis<br>on above θ_on, off below θ_off", "Merge gaps under g(concept)<br>drop under min_duration", "Episode store<br>concept · start · end · conf"], HOMET, "#B9CDE8")
    + row(card(h3("What counts as “one time” is per concept") + p("Cooking: breaks under 10 min are the same session; under 3 min is not cooking, so a 1-minute microwave use is not counted.", 24)
               + p("Baby crying: cries under 60 s apart are one episode.", 24) + p("Rules live in the ontology and are tuned per home from feedback.", 24, MUTED)),
          card(h3("The biggest driver of count accuracy") + p("Counts come from episodes, not windows. The episode row is plain text and timestamps:", 24)
               + mono("cooking · 18:30–19:22 · kitchen · 0.94", 24)), gap=24))

add("engines", head("From model to answers", "Anomaly, device health and occupancy")
    + row(card(h3("Anomaly engine") + mono("score = w1·surprise + w2·rarity + w3·drift", 24)
               + p("<b>Surprise:</b> −log p(event | history) from game 1.", 24) + p("<b>Rarity:</b> how far the window vector is from this home's history (nearest neighbours).", 24)
               + p("<b>Drift:</b> episode timing, duration and frequency against a per-person baseline.", 24)
               + p("Every flag is explained: “front door opened at 03:12; normally no door events after 23:00”.", 24, HOME)),
          card(h3("Device-health engine") + p("<b>Sensor faults:</b> silence, stuck-on, battery decay, radio signal drop, event rate that disagrees with neighbouring sensors.", 24)
               + p("<b>Appliance faults:</b> power-signature vectors against the device's own history and the same type in other homes: longer fridge compressor cycles, changed wash cycles, standby creep.", 24)),
          card(h3("Occupancy and identity") + p("People per room, supervised by camera or radar counts where allowed; person cards for who.", 24)
               + p("Motion-only homes get a calibrated range (“1–2 people”), not a falsely confident number.", 24)), gap=24))

add("agent", head("From model to answers", "The query agent: text in, tools, text out")
    + row(card(pill("Bridge 1 · stored facts (main path)", HOMET, HOME) + p("Heads turn vectors into rows with text labels: episodes, states, anomalies, device health. The agent answers with plain SQL, e.g. count_episodes(cooking, 16:00, 20:00).", 24)),
          card(pill("Bridge 2 · search by meaning (open path)", HOMET, HOME) + p("Projected minute vectors in a vector index (1,440 × 128 × fp16 ≈ 370 KB per home per day). semantic_search: text in, JSON out; the tool does the vector maths.", 24)), gap=24)
    + row(card(h3("Rules") + p("The LLM never produces numbers; tools do · time resolved in code, in the home's timezone · every answer has evidence and confidence · unobservable concepts are refused honestly.", 24)),
          card(h3("Example, open path") + p("<i>“Did anyone vacuum today?”</i> Not a stored concept, but a vacuum plug exists → search “someone vacuuming” → 10:05–10:38, living room, score 0.78, plug at 1.2 kW → <b>“Probably yes, ~10:05–10:38 (medium confidence)”</b>.", 24)), gap=24)
    + p("Tools: query_events · query_episodes · get_state · semantic_search · compare_baseline · detect_anomalies · device_health · energy_breakdown · forecast · list_capabilities. Model: a small local 3–8B LLM with function calling; cloud only if the household opts in.", 24, MUTED))

rh = [("Every event", "Sensor reading → Home Token → event vector", "Raw event → event store"),
      ("Every minute", "Minute summary → stream (live) → heads; episode builder opens, extends or closes", "Minute vector → vector store; tags, surprise, people"),
      ("About every hour", "Look-back pass over the last few hours, both directions", "Corrected episode starts and ends"),
      ("Every day", "Day summary; anomaly and device-health checks", "Day vector; flags in anomaly and health tables"),
      ("On a question", "Agent resolves words and time, calls tools, answers", "Nothing new, except feedback saved as labels")]
add("rhythms", head("From model to answers", "The live pipeline runs at five rhythms")
    + table(["Rhythm", "What happens", "What gets stored"], rh, [18, 46, 36])
    + p("Two kinds of minute vector: the <b>minute summary</b> (this minute's events only, input to the stream) and the <b>contextual minute vector</b> (this minute plus the past; projected and stored for search).", 24))

add("e2e", head("From model to answers · end to end", "“How many times did cooking happen in the last 4 hours?” at 20:00")
    + row(box("<b>16:20</b><br>microwave, 1 min", CARD, LINE, INK), box("<b>17:05–17:25</b><br>making tea", CARD, LINE, INK),
          box("<b>18:30–19:25</b><br>dinner; stove off 6 min in the middle", CARD, LINE, INK, flex="1.6"), gap=12)
    + row(card(pill("A · setup, once", HOMET, HOME) + p("HomeFM pretrained offline on other homes. Devices registered as sentences (“stove in kitchen, power sensor”). Concept <b>cooking</b>: “someone is cooking food in the kitchen”, merge gap 10 min, minimum 3 min.", 24)),
          card(pill("B · continuously", HOMET, HOME) + p("18:31:02 stove plug 1,850 W → Home Token → minute vector → p(cooking) = 0.93 → episode builder. The 6-min stove break is merged; the 1-min microwave is dropped.", 24)),
          card(pill("C · at the question", HOMET, HOME) + p("Agent resolves “cooking” (stored) and 16:00–20:00 → query_episodes → 17:05–17:25 (0.78), 18:30–19:22 (0.94).", 24)), gap=24)
    + card(p("<b>Answer:</b> “2 times: 17:05–17:25 and 18:30–19:22”, with the evidence and a confidence for each.", 30, INK), bg=HOMET, border="#B9CDE8"))

# ---------------------------------------------------------------- 7. Plan
src = [("Public datasets", "CASAS, Kasteren, UCI, Orange4Home, ARAS, MuRAL; energy (REFIT, UK-DALE, REDD)"),
       ("Simulated homes", "LLM-written routines → simulator: several residents, babies, pets, injected faults, free captions"),
       ("Pretrained experts", "Audio (CLAP, YAMNet) and open-vocabulary vision detectors, fine-tuned for home events"),
       ("Real homes", "10–30 pilot homes; opt-in Home Assistant data donation: hundreds of modern homes")]
bystage = [("Stages 1–2 (games 1, 2)", "No", "Every home's raw events. The number of homes matters more than labels."),
           ("Stage 3 (alignment)", "Captions", "Labels as sentences, MuRAL descriptions, simulator and checked LLM captions"),
           ("Stage 4 (tuning)", "Yes", "Labelled datasets, injected faults, user corrections"),
           ("Evaluation", "Yes", "Whole homes or datasets held out; SmartHomeQA")]
add("data", head("Plan", "Data, not compute, limits scaling here")
    + grid([card(h3(t) + p(d, 24), pad=22) for t, d in src], 4, 20)
    + table(["Stage", "Labels?", "Data"], bystage, [26, 14, 60])
    + p("In use today (DomusFM runs): 81 labelled CASAS homes + Milan + Aruba + UCI B, and since 28 September Kasteren A, Kasteren C and MuRAL. Orange4Home is by email request.", 24, MUTED))

gapsd = [("Pets in event streams", "No public labelled pet-triggered events", "Simulator + pilot homes with pets"),
         ("Baby crying in context", "Clips exist, never inside a home stream", "Simulator places detector outputs in routines"),
         ("Parcels and guests", "In no activity dataset", "Simulator + pilot doorbell footage"),
         ("Device faults", "Almost no labelled appliance faults", "Faults injected into real signals (REFIT)"),
         ("Elderly decline over weeks", "Few long, labelled datasets", "Simulated drift, longest CASAS homes, pilots"),
         ("Modern devices", "Public data uses older sensors", "Pilot and donated Home Assistant histories")]
add("data-gaps", head("Plan", "Gaps no public dataset covers, and the rules we keep")
    + table(["Gap", "Why", "Source"], gapsd, [26, 36, 38])
    + row(card(p("<b>Hold out whole homes</b>, never random windows: overlapping windows leak and inflate scores.", 24)),
          card(p("<b>Balance datasets</b> in pretraining so CASAS or simulated homes do not dominate.", 24)),
          card(p("<b>Report on real held-out homes only</b>; check every dataset's licence.", 24)), gap=20))

evl = [("Activity recognition", "Weighted F1", "Leave-one-dataset / home-out; 1, 5, 10, 30 % labels"),
       ("Counting / durations", "Exact match, MAE", "Per concept, per question range"),
       ("Open vocabulary", "Retrieval mAP, tagging F1", "Concepts never seen in training"),
       ("Anomaly", "AUROC, false alarms per home-week", "Injected and real faults"),
       ("Device health", "Lead time, precision", "Injected degradation, real faults"),
       ("Occupancy", "Count MAE, identity accuracy", "Camera or radar ground truth"),
       ("Forecasting", "NLL, next-k multiset F1", "Next 5 minutes, next k events"),
       ("End-to-end QA", "Answer accuracy", "SmartHomeQA benchmark"),
       ("Edge", "Latency, RAM, energy", "Hub-class device")]
add("eval", head("Plan", "How HomeFM will be judged")
    + table(["Area", "Metric", "Protocol"], evl, [26, 32, 42])
    + p("Baselines: the DomusFM reproduction (primary), DeepCASAS, Chronos, a GPT-2-style event model, a zero-shot LLM.", 24, MUTED))

add("deploy", head("Plan", "Deployment and privacy: the home hub does the work")
    + flowrow(["Perception experts on the hub", "HomeFM student<br>int8, streaming", "Local stores", "Local agent"], OKT, "#BFDCC8")
    + row(box("Teacher on the training cluster (opt-in data only)", HOMET, "#B9CDE8"), arrow_r(), box("Distil → student release", HOMET, "#B9CDE8"), arrow_r(),
          box("Over-the-air update to every hub", HOMET, "#B9CDE8"), gap=12)
    + grid([card(p(t, 24)) for t in ["Raw audio and video never leave the device; only events and embeddings are stored.",
                                      "Retention limits per data class; history is visible and deletable by the user.",
                                      "Per-home adapters keep learning on the home's own stream; feedback only by opt-in.",
                                      "Targets: student under 500 MB, under 20 ms per minute update, ONNX runtime."]], 2, 20))

rm = [("M1", "Baseline + data pipeline (6 weeks)", "Common token format, converters, DomusFM within ±0.03 F1 of the paper"),
      ("M2", "HomeFM v0: moments + stream (8 weeks)", "Beats the DomusFM reproduction leave-one-dataset-out"),
      ("Ablation", "Games A–F (3 weeks)", "Pretraining recipe fixed with evidence"),
      ("M3", "Continuous signals + device health (6 weeks)", "Faults detected on injected data; no binarisation"),
      ("M4", "Language alignment (6 weeks)", "Zero-shot tagging on held-out concepts above threshold"),
      ("M5", "Long horizon + multi-occupant (8 weeks)", "Routine drift and occupancy heads; multi-resident data"),
      ("M6", "Edge distillation (4 weeks)", "Student under 500 MB, under 20 ms per minute on a hub")]
add("roadmap", head("Plan", "Roadmap: model track from October 2026")
    + table(["Milestone", "Scope", "Exit criterion"], rm, [13, 40, 47])
    + p("System track in parallel: perception experts (8 weeks from 15 Oct), stores + query agent (6 weeks), SmartHomeQA benchmark (10 weeks from 1 Nov). First end-to-end demo in about 8 weeks, with rule-based episodes, swapping HomeFM in as it matures.", 24))

built = ["Home Token schema, batching; converters for CASAS, UCI, Kasteren, MuRAL", "Attribute fusion, moment encoder, stream transformer (live + look-back)",
         "Event read-out and next-event head (surprise)", "Games A–F in code, tested on small data; language alignment per window",
         "DomusFM reimplementation, corpus runs, fixes, cleaning, EDA"]
designed = ["Detector tag vocabulary; audio and vision experts", "KV-cache streaming, day tokens, entity axis, person cards",
            "Tagger, start/end, occupancy, device-health heads; episode builder", "Stores, vector index, semantic search, agent; onboarding",
            "Per-minute alignment, variant G, teacher → student"]
add("status", head("Plan", "Build status today")
    + row(card(pill("Built", OKT, OK) + col(*[p("• " + b, 24, INK) for b in built], gap=10), bg=CARD),
          card(pill("Designed, not built", DOMT, DOM) + col(*[p("• " + d, 24, INK) for d in designed], gap=10), bg=CARD), gap=28)
    + card(p("<b>DomusFM baseline done:</b> run 3 finished 29 Sep (34 of 40 settings helped by pretraining). <b>Not run yet:</b> the HomeFM corpus runs (variant E at 8.0M and size-matched 28.6M).", 26, INK), bg=HOMET, border="#B9CDE8"))

risks = [("Pretraining data too small or uniform", "Simulated homes, more CASAS homes, pilots, distillation"),
         ("“One occurrence” is ambiguous", "Per-concept episode rules in the ontology; feedback tuning"),
         ("Who did it, with several occupants", "Identity-bearing sensors where allowed; calibrated ranges otherwise"),
         ("False alarms (care, security)", "Per-home calibration, alert budgets, a human confirms"),
         ("Simulated vs real gap", "Home-adversarial training; validate on real pilot homes"),
         ("Privacy and trust", "Edge-only default, events-only storage, visible history")]
add("risks", head("Plan", "Risks and open decisions")
    + table(["Risk", "Mitigation"], risks, [40, 60])
    + p("<b>Open decisions:</b> edge-only vs edge + cloud · which hub ecosystem first (Home Assistant, Matter) · 2–3 deep v1 domains · access to pilot homes · free-text messages as tags or text tokens · keep the device registry outside the model file · detector tag field in the Home Token.", 24))

nxt = [("1", "DomusFM run 3: done", "Pretraining helps in 34 of 40 settings on 10 homes. The fixed, cleaned DomusFM is the baseline to beat."),
       ("2", "HomeFM E corpus runs", "Games 1 + 2 at 8.0M, then size-matched 28.6M, same homes and fine-tuning as DomusFM."),
       ("3", "Keep E only if", "Its loss falls gradually and pretrained E beats E without pretraining, especially at 5 % labels."),
       ("4", "Then F and G", "Add game 3 where captions allow; test the five refinements against the measured collapse.")]
add("next", f"""
<p style="font-size:24px; font-weight:600; letter-spacing:2px; text-transform:uppercase; color:{HOMED}">Next steps</p>
<h2 style="font-family:{FH}; font-size:72px; font-weight:600; line-height:1.1; color:{ON}">Prove the games on real homes, one run at a time</h2>
<div style="flex:1"></div>
{grid([card(f'<p style="font-family:{FH}; font-size:64px; font-weight:700; line-height:1; color:{HOMED}">{n}</p>' + h3(t, ON, 32) + p(d, 24, SOFT), bg="#1F2E4F", border="#2E4270") for n, t, d in nxt], 4, 24)}
{p("M2 exit criterion: HomeFM v0 beats the DomusFM reproduction on leave-one-dataset-out.", 28, ON)}""", dark=True)

# ---------------------------------------------------------------- write
for sid, s in slides.items():
    (SL / f"{sid}.html").write_text(s, encoding="utf-8")
sections = {
    "s1": {"description": "What HomeFM is for: the questions, goals and targets", "start": "cover"},
    "s2": {"description": "DomusFM, the baseline, and what our reproduction showed", "start": "domus-overview"},
    "s3": {"description": "DomusFM's 11 limitations and HomeFM's remedy for each", "start": "limits-map"},
    "s4": {"description": "System and model architecture, from Home Tokens to heads", "start": "system"},
    "s5": {"description": "Pretraining: why DomusFM's masking goes, and the games that replace it", "start": "pretrain"},
    "s6": {"description": "From model outputs to answers: episodes, engines, the agent", "start": "episodes"},
    "s7": {"description": "Data, evaluation, deployment, roadmap, status and next steps", "start": "data"},
}
_idx = ROOT / "project" / "deck.json"
_created = json.loads(_idx.read_text(encoding="utf-8")).get("createdOnFiles") if _idx.exists() else None
deck = {"v": 4, "createdOnFiles": _created or {"v": 1, "at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
        "title": "HomeFM: architecture and design", "cover": "cover", "order": order, "sections": sections,
        "faces": {"space-grotesk": {"family": "Space Grotesk", "href": "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400..700&display=swap"},
                  "ibm-plex-sans": {"family": "IBM Plex Sans", "href": "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap"},
                  "jetbrains-mono": {"family": "JetBrains Mono", "href": "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap"}},
        "designSystems": []}
(ROOT / "project" / "deck.json").write_text(json.dumps(deck, indent=1), encoding="utf-8")
print(len(order), "slides:", ", ".join(order))
