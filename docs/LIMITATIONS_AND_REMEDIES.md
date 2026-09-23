# DomusFM limitations and HomeFM remedies, in plain words

Companion to [DESIGN.md](DESIGN.md) §4.1 (the formal list) and §4.2 (the step-by-step technical comparison). This document explains each limitation without jargon. Any technical term that remains is defined in DESIGN.md §1.1.

Each limitation answers the same questions:

1. **The problem:** what DomusFM cannot do, with an everyday comparison.
2. **Why DomusFM does it this way:** usually a sensible choice for the paper's goal, which was winning research benchmarks, not answering household questions.
3. **Why it matters for us:** the user questions that break.
4. **What HomeFM does differently:** the remedy.
5. **Examples:** questions answered by each model, side by side.
6. **The honest catch:** what is still hard, and what is built (✅) or only designed (📐).

## Summary

| # | Limitation | In one line | HomeFM remedy | Status |
|---|---|---|---|---|
| [L1](#l1--it-can-only-name-what-it-was-taught) | Closed label set | Can only name activities it was shown examples of | Link sensor activity to words, so any activity can be described | 📐 (alignment training ✅) |
| [L2](#l2--it-only-understands-on-and-off) | On/off only | Sees "fridge ON", not "fridge draws 180 W" | Keep real numbers as input | ✅ input · 📐 device health |
| [L3](#l3--its-memory-is-30-events-not-a-length-of-time) | 30-event window | Memory is sometimes 2 minutes, sometimes 6 hours, never weeks | 1-minute steps, hours of context, daily summaries | ✅ minutes · 📐 days |
| [L4](#l4--it-assumes-one-person-lives-there) | One person assumed | Cannot tell people apart or count them | Person cards, clues, per-room counts | 📐 |
| [L5](#l5--it-labels-moments-but-cannot-count-occurrences) | No start or end | Cannot count "cooked 3 times" or see two things at once | Several tags at once, start/end detection, counting rules | 📐 |
| [L6](#l6--it-cannot-say-unusual-or-broken) | No anomaly or fault detection | Cannot say "this is unusual" or "this sensor is broken" | Surprise score, anomaly and device-health engines | ✅ surprise · 📐 engines |
| [L7](#l7--it-knows-what-comes-next-but-not-when) | Forecast without timing | Knows what comes next, not when | Predicts each next event with a range of likely times | ✅ head · 📐 overdue checks |
| [L8](#l8--its-knowledge-cannot-be-searched-with-words) | Not searchable by text | Its knowledge cannot be searched with a question | A "map of meanings" shared by minutes and sentences | 📐 |
| [L9](#l9--it-cannot-hear-or-see) | No sound or camera | Cannot hear a baby cry or see a parcel | On-hub sound and camera detectors send short tags | 📐 (event format ✅) |
| [L10](#l10--it-learned-from-a-small-old-set-of-homes) | Small, old training data | Never saw smart locks, cameras or robot vacuums | Many more homes, a simulator, pilot and donated homes | ✅ 84 homes · 📐 rest |
| [L11](#l11--its-practice-game-is-too-easy) | Practice game too easy | Learns sensor tricks instead of behaviour | Three harder, more useful practice games | ✅ A–F · 📐 corpus run, G |

Several limitations share one remedy:

- **L1 and L8** are both fixed by linking sensor activity to words. L1 is about *naming* what is happening now, and L8 is about *searching* the past.
- **L6 and L7** both come from one skill: predicting what happens next and when.
- **L10 and L11** are connected. Our own runs show that more data does not help while the practice game stays too easy.

---

## L1 · It can only name what it was taught

### The problem

DomusFM learns by watching sensor data, so after pretraining it can tell that two stretches of activity *look similar*. It cannot put a **name** on them by itself. To name an activity, someone has to show it labelled examples ("this stretch was cooking, this was sleeping"). It then learns a **fixed list** of names, and anything outside the list cannot be named.

It is like a child who has seen flashcards for "cat", "dog" and "cow". Shown a horse, the child says "cow" (the closest card) or nothing, and cannot say "horse" until someone brings a horse flashcard.

### Why DomusFM does it this way

- **Research datasets come with a fixed list.** Public smart-home datasets have 10–35 activities written down by people ("Cook", "Sleep", "Toilet"). Researchers compare models on how well they recognise that list, and DomusFM was built to win that comparison.
- **Nothing else was available.** Understanding new words needs many examples of "sensor activity + a sentence describing it". The datasets only have short names.
- **It is simple and cheap.** A fixed list is easy to train and runs on a tiny computer.
- **The authors say so:** DomusFM "does not yet operate in a zero-shot fashion" (paper §7.4.3), meaning it cannot recognise something without examples.

### Why it matters for us

Users ask about anything: "How many times did someone **vacuum** today?", "Did **guests** come?", "Did Dad **take his medicine**?". None of these is on a research list. With DomusFM, every new question would need the household to label days of examples first, which nobody will do.

### What HomeFM does differently

HomeFM describes activities **in words** instead of learning them from a fixed list.

- **DomusFM** has a checklist with fixed boxes. A new activity needs a new box, and a new box needs examples.
- **HomeFM** works from a description. A new activity is just a new sentence, such as "someone is vacuuming".

Two things make this work:

1. **HomeFM learns to connect sensor activity with words.** During training it sees sensor minutes paired with sentences ("stove on, kitchen motion" ↔ "someone is cooking") and learns to place each minute close to the words that describe it.
2. **Any sentence can then be looked up.** The sentence is turned into numbers, and HomeFM finds the minutes that sit closest to it. No new training is needed.

**How the comparison works.** A HomeFM minute vector and a sentence vector are made in different ways, so they are not comparable by nature. They become comparable because of the training in step 1: a small "translator" (projection head) on each side is trained on the pairs until matching minutes and sentences land close together. This is the same idea as CLIP, which lets people search photos by typing words. In a toy example with 3 numbers instead of hundreds:

| Sentence | Sentence vector | Score against the 18:31 minute `[0.85, 0.15, 0.05]` |
|---|---|---|
| "someone is cooking food" | `[0.9, 0.1, 0.0]` | **0.78** ← highest |
| "someone is sleeping" | `[0.0, 0.1, 0.9]` | 0.06 |
| "someone is vacuuming" | `[0.1, 0.9, 0.1]` | 0.23 |

### Example: "How many times did someone vacuum today?"

**DomusFM**

1. "Vacuuming" is not on its list.
2. The vacuum minutes get labelled "Other", or the nearest activity it knows.
3. Answer: "I can't tell", or worse, a wrong "0".
4. To fix it, someone labels days of vacuuming and the model is retrained.

**HomeFM**

1. Searches for minutes that match "someone is vacuuming".
2. Finds 10:02–10:40: the vacuum plug is on and motion moves room by room.
3. Answer: "**Probably once**, 10:02–10:40 (medium confidence)."
4. The user confirms, and "vacuuming" becomes a known activity, so later answers are faster and more confident.

With DomusFM, labels are **the entry ticket**: no labels, no answer. With HomeFM, labels are **a bonus**: it answers without them, and they make it more accurate.

### The honest catch

- HomeFM can only link words to activities that are **somewhat similar to what it saw in training**. That is why collecting many sensor-plus-sentence examples matters so much (DESIGN.md §11), and why answers from this path always show a confidence.
- **Status:** the training that links windows and sentences exists (variant F) ✅. Per-minute tagging, the vector store and the search tool are 📐. There is no measured result yet. The planned test hides some activity names during training and checks whether HomeFM finds them from words alone.

---

## L2 · It only understands on and off

### The problem

DomusFM only understands **on and off**. Many devices report **amounts** instead:

| Device | What it reports |
|---|---|
| Smart plug | Power: 5 W, 180 W, 1,850 W |
| Thermostat | Temperature: 21.5 °C |
| Humidity sensor | 45 %, 85 % |
| Water meter | Litres per minute |

To feed these to DomusFM, the amounts must first be turned into on/off with a rule like "above 50 W = ON" (**binarising**). **The number itself is then thrown away.**

It is like being told only whether a tap is **open or closed**, never **how much water is flowing**. A dripping tap and a fully open tap both say "open".

### What gets lost: a fridge

The smart plug reports:

```
08:00    5 W   (resting)
08:10  150 W   (compressor starts)
08:22    5 W   (compressor stops)
```

DomusFM receives `08:10 fridge ON`, `08:22 fridge OFF`. A month later the fridge is failing:

```
08:10  180 W   ← 20 % more power
08:35    5 W   ← ran 25 minutes instead of 12
```

DomusFM still receives `fridge ON … fridge OFF`. The only hint left is a slightly longer gap, which the model was never taught to watch. **The extra power is invisible.**

### Why DomusFM does it this way

- **Most research datasets use on/off sensors:** motion, door contacts, bed pressure. Only a few have meters.
- **One simple format is easier:** if every event is "device + ON/OFF + time", the model stays simple.
- **The paper admits the cost:** continuous readings become "virtual" on/off events, with possible information loss (paper §3.1, §7.4.2).

### Why it matters for us

1. *"Is my fridge OK?"* The failing fridge above looks normal as on/off.
2. *"Which appliance used the most energy today?"* "Kettle ON" and "heater ON" look the same, but the kettle uses 2,000 W for 3 minutes and the heater 1,500 W for 5 hours.
3. *"How many times did someone shower?"* Bathroom humidity needs a threshold per bathroom and season. At 70 %, a humid winter bathroom sitting at 72 % becomes a fake all-day shower, and a well-ventilated bathroom that only reaches 65 % misses real showers. The errors are silent.
4. *"Is the washing machine working properly?"* A wash cycle has a shape (heat water at 2,000 W, drum at 200 W, spin at 500 W). As on/off it is only "washer ON … washer OFF", and the shape is gone.

### What HomeFM does differently

**Numbers go in as numbers.** 1,850 W stays 1,850 W. Every event has a **value slot**, so on/off devices and number devices go through the same model:

| Event | ON/OFF | Value |
|---|---|---|
| Kitchen motion | ON | — |
| Fridge plug | — | **180 W** |
| Bathroom humidity | — | **85 %** |

A reading is sent only **when it changes meaningfully**, not every second. The model sees the **level** (how much), the **rhythm** (how long and how often) and the **trend** (slow change over weeks).

### Example: "Is my fridge OK?"

**DomusFM:** sees "fridge ON … fridge OFF" as always. It has no power information and no notion of a normal fridge, so it cannot answer.

**HomeFM:**

1. Sees each cycle: 180 W for 25 minutes.
2. The device-health engine compares this with the fridge's own history (150 W for 12 minutes last month) and with similar fridges elsewhere.
3. Answer: "**Your fridge is working harder than usual.** Its cooling cycles are about twice as long as last month (25 vs 12 minutes) and use about 20 % more power. This can mean a worn door seal or a compressor problem."

### A note on energy totals

*"How much energy did we use in the last hour?"* does not need the model. The raw meter readings are kept in the event store, and a database sum answers it exactly. The model adds **understanding**: which activity used the energy ("cooking dinner used 1.2 kWh"), whether consumption is unusual ("heating is 40 % higher than a normal Tuesday"), and whether an appliance is faulty.

### The honest catch

- Public data has few power meters, so this skill needs energy datasets, simulated homes and pilot homes (L10, DESIGN.md §11).
- **Status:** numbers as input ✅ (the value slot is in HomeFM's code). The device-health engine is 📐.

---

## L3 · Its memory is 30 events, not a length of time

### The problem

DomusFM always looks at exactly **the last 30 events**. How much time that covers depends on how busy the house is:

| Time | What is happening | How long 30 events covers |
|---|---|---|
| 18:30, busy kitchen | Motion, fridge, stove, cupboards firing constantly | **About 2 minutes** |
| 15:00, someone reading on the sofa | An occasional motion event | **About 40 minutes** |
| 01:00, everyone asleep | Almost nothing fires | **About 6 hours** |

Its memory keeps stretching and shrinking, and never reaches yesterday or last week.

It is like remembering "the last 30 words someone said". In a fast conversation that is 10 seconds, and in a slow one it is an hour.

### Why DomusFM does it this way

- **It is the standard in activity-recognition research.** Earlier models (such as DeepCASAS) used "last N events" too, so results are comparable.
- **Every window holds the same amount of information.** The model never gets an empty window.
- **The paper tested a time-based alternative** (5-minute windows), and it did slightly worse *on their benchmark* (paper §7.2).

### Why it matters for us

Users ask in time, not events: "in the last 4 hours", "last night", "compared with last month". The 30-event window breaks in three ways:

1. **Long activities get chopped up.** A 40-minute dinner in a busy kitchen becomes about 20 fragments of about 2 minutes. No single window sees both its start and its end, and stitching the pieces together is how "cooked once" becomes "cooked 3 times" (L5).
2. **Quiet periods get blurred together.** At night one window mixes "23:00 bed → 02:30 bathroom trip → 05:00 early kitchen visit" into one blob with one label. The 02:30 bathroom trip, which elderly-care questions care about, is buried.
3. **Slow changes are invisible.** "Is Mum sleeping worse than last month?", "Is the fridge degrading?" and "Is Dad getting up more at night lately?" need weeks.

### What HomeFM does differently

1. **Time is cut into 1-minute slots.** Every minute gets one summary, however many events it had. A quiet minute becomes a "nothing happened" summary. "The last 4 hours" is always 240 steps.
2. **It reads hours of minutes in order**, so it sees whole activities from start to end.
3. **Daily summaries cover weeks.** Each day is also squeezed into one day summary, and comparing them shows slow trends.

It is like a diary with **one line per minute**, plus **one line per day**.

### Examples

**"How many times did cooking happen in the last 4 hours?"**
DomusFM sees the 40-minute dinner as about 20 fragments, some labelled "Other" during a pause, and a stitched count is wrong (for example 3). HomeFM reads all 240 minutes, sees the start (stove on at 18:30) and the end (kitchen quiet at 19:22), and counts **one** session.

**"Did Dad get up at night?"**
DomusFM mixes the 02:30 bathroom trip into one big night window. In HomeFM, minutes 02:30–02:34 stand out among the quiet minutes: "**Yes, once**: 02:30–02:34, bathroom."

**"Is Mum sleeping worse than last month?"**
DomusFM has no memory beyond a few hours. HomeFM compares daily summaries over 5 weeks: "**Yes, a little.** She has gone to bed about 40 minutes later on average over the last 2 weeks, and gets up twice a night instead of once." (📐)

### The honest catch

- A busy minute (200 events) becomes one summary, so some detail could be lost. HomeFM keeps a small event-level reader for when single events matter, such as predicting the next one.
- DomusFM's time-based test did worse. HomeFM summarises minutes differently, but it still has to **prove** it does not lose the same way, in the head-to-head comparison (DESIGN.md §12).
- **Status:** 1-minute summaries and reading hours of minutes ✅. Daily summaries for weeks 📐.

---

## L4 · It assumes one person lives there

### The problem

A motion sensor only says "something moved here". It reports the same "kitchen motion ON" for grandma, for her grandson, for four people at once, or for the dog. DomusFM handles this by **assuming one person lives in the home**, or that someone has already said who caused each event.

It is like hearing footsteps in the house without knowing whether they belong to your mother, your son or the dog.

### Why DomusFM does it this way

- **Most research datasets are single-person homes**, often one elderly resident living alone.
- **On/off sensors carry no identity.** Telling people apart from motion alone is an unsolved research problem.
- **The paper says so:** it assumes one occupant or pre-labelled events, and calls the multi-person case an open challenge (paper §7.4.1).

### Why it matters for us

1. **Counting people:** *"How many people are in the living room?"* One person and four people look the same.
2. **Care for one person:** *"Did grandma eat lunch?"* If her grandson cooked at noon while she stayed in her room, DomusFM credits the kitchen activity to "the resident", and the missed meal is hidden.
3. **Pets:** the dog in the hallway at 2 am looks like an elderly person wandering at night. Either there are false alarms every night, or real wandering is learned as normal.
4. **Guests:** *"Did we have guests?"* Guests show up mainly as *more people than usual*, which a one-person model cannot represent.

### What HomeFM does differently

HomeFM works like a detective using clues.

1. **A person card for each resident and pet** (for example Grandma, Aarav, Bruno the dog). Each event is assigned to a card, or to "unknown / guest".
2. **Clues:**

| Clue | Example |
|---|---|
| Where | Motion in grandma's bedroom is probably grandma |
| Habits | Aarav usually gets home at 16:00, grandma naps at 14:00 |
| Movement | The dog moves fast and low, and never opens the fridge |
| Phone presence | Aarav's phone joined the Wi-Fi at 16:02 |
| Smart-lock codes | The door was unlocked with Aarav's code |
| Camera or radar (if allowed) | "2 people in the living room", counted without recognising faces |
| Pet-aware sensors | Pet-immune motion sensors, a collar tag |

3. **A people count per room.**
4. **Honest ranges.** With motion sensors only, it says "probably 1–2 people" instead of a confident wrong number.

### Examples

**"Did grandma eat lunch?"**
DomusFM sees kitchen activity at noon and implies "yes", which is wrong. HomeFM sees that Aarav's phone has been home since 11:30, motion continued in grandma's bedroom from 11:30 to 14:00, and the noon kitchen activity belongs to Aarav's card: "**I see no sign that grandma went to the kitchen between 12:00 and 14:00.** Aarav was cooking at noon, and grandma stayed in her bedroom." (medium confidence)

**The dog at 2 am.**
DomusFM sees "resident moving in the hallway", which may trigger an alarm. HomeFM sees fast, low movement (the pet-immune sensor did not fire) and no doors or lights touched, assigns it to Bruno, and raises no alarm.

**"How many people are in the living room?"**
With a camera or radar: "**3 people** (high confidence)." With motion sensors only: "**Probably 2–3 people.** Two phones are home, and there is steady motion on both sides of the room." (medium confidence)

**"Did we have guests today?"**
The front door opened at 18:00 without a family lock code, the doorbell rang, and the living-room count rose from 2 to 5 until 21:30: "**Yes**, about 3 guests, 18:00–21:30."

### The honest catch

- Motion sensors alone cannot truly tell people apart. Good answers need at least one identity clue (phones, lock codes, camera or radar, pet-aware sensors). Without any, HomeFM can only give ranges and guesses from habits.
- Cameras and tracking who is where are sensitive. Each household chooses what to allow, room by room, and nothing leaves the home (DESIGN.md §13).
- **Status:** 📐.

---

## L5 · It labels moments but cannot count occurrences

### The problem

DomusFM puts **one activity label on each moment** ("Cook", "Sleep", "Other"), but never says where an activity **starts and ends**. Counting "how many times" needs exactly that: "cooking happened 3 times: 07:30–07:50, 12:15–12:40, 18:30–19:22". Each "time" is an **episode** with a start and an end. DomusFM also cannot show **two things at once**, such as cooking while the baby cries.

It is like a photo album with a caption on every photo, but nobody marks where one party ends and the next begins.

### Why DomusFM does it this way

- **The research test is "label each moment".** Datasets are scored by checking each window's label against a human's (paper §6.4.1).
- Turning labels into counted occurrences was outside the paper's scope.
- Single-person datasets usually have one activity at a time.

### Why it matters for us

"How many times" and "how long" are among the most common questions. Counting from a plain list of labels goes wrong in three ways:

1. **Short pauses split one activity into many.** A 6-minute stirring pause in the middle of dinner turns `Cook Cook … Other Other Other … Cook Cook` into 2 counts, and stray "Other" labels make it 3 or 4. The true answer is 1.
2. **Tiny blips get counted.** A 1-minute microwave use to warm milk is labelled "Cook", adding a false cooking session.
3. **Two things at once.** At 18:40 someone is cooking and the baby starts crying. One label must pick one, so either a cry is lost or the cooking gets a hole and is counted twice.

### What HomeFM does differently

1. **Several tags per minute**, each with its own score, so cooking (0.92) and baby crying (0.85) can both be on.
2. **A start/end detector** that looks for "an activity starts here" and "it ends here".
3. **Counting rules per activity**, kept in the home's dictionary:

| Activity | Join pauses shorter than | Ignore if shorter than |
|---|---|---|
| Cooking | 10 minutes | 3 minutes |
| Baby crying | 60 seconds | 5 seconds |
| Sleeping | 30 minutes | 1 hour |
| Shower | 2 minutes | 2 minutes |

4. **"Switch on high, switch off low"** (hysteresis). An episode starts only when the score rises above 0.6 and ends only when it falls below 0.4, so a score wobbling around 0.6 does not create many tiny episodes.
5. **The result is a list of episodes**, such as `cooking 18:30–19:22` and `baby crying 18:40–18:46`. "How many" means counting rows, and "how long" means end minus start.

### Examples

**"How many times did cooking happen today?"**
DomusFM counts stretches of "Cook" and gets 4, because the pause splits dinner and the microwave adds one. HomeFM joins the pause, ignores the microwave, and answers "**2 times**: breakfast 07:30–07:50 and dinner 18:30–19:22."

**"How many times did the baby cry while we were cooking?"**
DomusFM cannot answer, because it had to choose one label at 18:40. HomeFM has overlapping episodes: "**Once**, 18:40–18:46, during dinner cooking."

**"How long did Dad sleep last night?"**
DomusFM has "Sleep" labels broken by a few "Other"s, with no clean start and end. HomeFM keeps the 4-minute bathroom trip inside the sleep episode (shorter than the 30-minute rule): "**7 h 35 min**, 23:10–06:45, with one bathroom trip at 02:30."

**Feedback.** If the user says "the 17:05 one was just making tea", HomeFM raises cooking's minimum length for this home or adds a "tea" activity, and remembers the example. The rules start from sensible defaults and adjust to each home.

### The honest catch

- The rules matter a lot. Too short a join time splits activities, and too long merges separate ones (breakfast and lunch prep becoming one). Good defaults and per-home tuning from feedback are the biggest single factor in count accuracy.
- **Status:** 📐. Counting accuracy is a headline measure in the evaluation plan (DESIGN.md §12). DESIGN.md §10.2 walks through the cooking case end to end.

---

## L6 · It cannot say "unusual" or "broken"

### The problem

DomusFM can describe what is happening ("this looks like sleeping"), but it cannot say **"this is unusual for this home"** (a front door opening at 3 am) or **"this device is broken"** (a stuck sensor, a dying battery, a failing fridge).

It is like a guard who can describe everything they see, but has no idea what is normal for this particular house.

### Why DomusFM does it this way

- **It was not the paper's goal.** Anomaly and behaviour-change detection are listed as future work (paper §7.4.4).
- **Its training only teaches "alike or not alike".** It never learns *how likely* something is, such as "a 3 am door opening happens 1 night in 1,000 here". Without a sense of likely and unlikely, it cannot say "unusual".

### Why it matters for us

1. **Security:** *"Anything unusual last night?"* The front door opened at 03:12.
2. **Broken sensors:** a motion sensor stuck ON after a battery fault looks like someone is always present. Grandma could appear active all night when she has not moved.
3. **Failing appliances:** the fridge running longer and longer (L2) goes unnoticed until food spoils.
4. **Health changes:** *"Is Dad's routine changing?"* Three night-time bathroom trips instead of one, every night for three weeks, is an early health warning.

### What HomeFM does differently

**1. Surprise: "I didn't expect that."** HomeFM is trained to predict what normally happens next in this home (L7), so it knows how expected each real event was:

| Event | What HomeFM expected | Surprise |
|---|---|---|
| 07:05 kitchen motion | Breakfast time, very likely | Low |
| 18:30 stove on | Dinner time, very likely | Low |
| 03:12 front door opens | Everyone asleep, almost never happens | **High** |

**2. The anomaly engine asks three questions together**, and always gives a plain explanation:

| Question | Example |
|---|---|
| Surprise: was this unexpected right now? | Door at 03:12 while everyone sleeps |
| Rarity: has anything like this happened in this home before? | No door events after 23:00 in 6 weeks |
| Routine drift: is a person's pattern slowly shifting? | Dad's night bathroom trips went from 1 to 3 over 3 weeks |

**3. The device-health engine** compares each device with its own past and with the same kind of device in other homes:

| Fault | What it looks like |
|---|---|
| Silent sensor | Hallway motion usually fires 200 times a day, today 0 |
| Stuck sensor | Motion ON every 5 seconds for 10 hours, while phones say nobody is home |
| Dying battery | Weakening signal, irregular reports |
| Disagrees with neighbours | The bedroom door opened, but bedroom motion saw nothing, for days |
| Appliance wearing out | Fridge cycles twice as long as last month (L2) |

### Examples

**"Anything unusual last night?"**
DomusFM can label the 03:12 window but has no sense of how strange it is. HomeFM: "**Yes, one thing:** the front door opened at 03:12 and closed at 03:14. This home normally has no door activity after 23:00, and nobody's phone left or arrived at that time."

**A stuck motion sensor.**
DomusFM treats constant hallway motion as someone active all night. HomeFM notices a perfectly regular 5-second rhythm that people do not make, with no other sensor agreeing: "**The hallway motion sensor looks stuck.** It has reported motion every 5 seconds since 22:00 with nothing else happening. Please check its battery." Its readings are then ignored until it is fixed.

**"Is Dad's routine changing?"**
DomusFM has no memory of past weeks. HomeFM compares with a two-month baseline: "**Yes.** Dad has been getting up about 3 times a night for the bathroom over the last 3 weeks, compared with about once before. It might be worth mentioning to his doctor."

### The honest catch

- "Unusual" needs a few weeks of this home's history. In the first days only general knowledge from other homes is available.
- Too many alerts and people stop listening, and too few and real problems are missed. Thresholds must be tuned, and families can say "that's fine, don't flag it again".
- Surprise is not danger: a late-night snack is surprising but harmless. That is why every flag carries an explanation.
- **Status:** the surprise score ✅ (from the next-event training in HomeFM's code). The anomaly and device-health engines 📐.

---

## L7 · It knows what comes next, but not when

### The problem

DomusFM guesses **which** events are likely soon, but not **when** or **in what order**. Its forecast is literally a bag of the next 30 events:

```
{ bedroom motion ×8, bathroom motion ×6, bathroom door ×4, kitchen motion ×12 }
```

Is the first one in 5 minutes or 3 hours? Does the bathroom come before the kitchen? The bag does not say.

It is like a weather forecast that says "rain is coming" without saying whether that means this afternoon or next week.

### Why DomusFM does it this way

- **It is easier to score.** Guessing the exact order and times of 30 events is very hard, and a bag gives credit for the right events in any order (paper §6.5.1).
- **Order is often noisy.** Whether the fridge or the cupboard comes first is random in a busy kitchen.
- **Its time-based version did slightly worse** on the benchmark (paper §7.2).

### Why it matters for us

1. **"Will Dad be up soon? I want the coffee ready."** "Bedroom and bathroom events are likely among the next 30" is useless without a time.
2. **Spotting what did not happen.** Grandma normally has breakfast by 9:00, and at 11:00 the kitchen is still quiet. That could mean she is unwell or has fallen. Noticing it needs a timed expectation ("kitchen activity should have started by about 9:00"). A bag of future events has no times, so it can never notice something missing. The same goes for missed medicine or a child not home by the usual time.
3. **Planning ahead.** "When will someone be home?" lets the heating warm up in time.

### What HomeFM does differently

HomeFM predicts **the next event, one at a time, with its time**: which device, what value, and **how long until it happens**. It predicts a **range of likely waiting times**, not a single number:

```
Dad's next bedroom motion, predicted at 06:50:

 likely │         ▂▅█▇▅▃▂
        │      ▂▃▅       ▂▁
        └──────────────────────────────
         06:50  07:00  07:10  07:20  07:30
                 most likely ~07:10, usually 07:00–07:30
```

- A regular routine gives a narrow peak ("almost certainly around 07:10"), and an irregular one gives a wide spread.
- It can show two peaks: "either in about 5 minutes (a quick bathroom trip) or in about 2 hours (back to sleep)".

This enables an **"expected by" check**: if breakfast normally starts around 08:15 (usually 07:45–09:00), and there has been no kitchen activity by 09:30, the check fires.

### Examples

**"Will Dad be up soon?"**
DomusFM: "Bedroom and bathroom events are likely next", with no time. HomeFM: "**Probably in about 20 minutes**, most likely between 07:00 and 07:30. On weekdays he is usually up by 07:10."

**Grandma's missed breakfast.**
DomusFM notices nothing, because a quiet morning just looks like "nothing happening". HomeFM: "**Grandma hasn't been to the kitchen this morning.** She usually has breakfast by about 08:15. Her last activity was bedroom motion at 07:40."

**"When will Aarav be home from school?"**
HomeFM: "**Around 16:00** (usually 15:50–16:15 on weekdays)."

The same skill also powers the **surprise score** in L6: if HomeFM expected nothing until 07:00, a door opening at 03:12 is very surprising.

### The honest catch

- Predictions are only as good as the routine is regular. For irregular schedules the honest answer is a wide range.
- "Overdue" checks must know about weekends, holidays and visits ("it's Sunday, breakfast is later"), and families can adjust them.
- **Status:** the "what + when" head ✅ (in HomeFM's code, trained in variants D and E). The "expected by" checks 📐.

---

## L8 · Its knowledge cannot be searched with words

### The problem

DomusFM stores what it learned as lists of numbers that are **not linked to language**. You cannot type a question and find the matching moments in the home's history.

It is like a library where all the books are on the shelves but there is no catalogue: you cannot search "books about cooking".

**How this differs from L1:** L1 is about **naming** what is happening now ("Is someone vacuuming?"). L8 is about **searching** the past ("When did we fry something this week?"). The same remedy fixes both.

### Why DomusFM does it this way

- **The datasets have no descriptive sentences** to link with (the same reason as L1).
- **Its benchmarks never needed search.** They only ask for labels or next events.

### Why it matters for us

Questions nobody planned for have no stored label: *"When did we fry something this week?"*, *"Show me the evening the kitchen was busy for hours"*, *"When did it look like we had a party?"*, *"Was there a day nobody was home?"*. With DomusFM there is nothing to search, so these questions cannot use the model at all.

### What HomeFM does differently

**1. Every minute gets a place on a "map of meanings".** Similar things sit close together, and sentences are placed on the same map:

```
   "someone is sleeping"  ●
                            ● 02:00 Tue    ● 02:00 Wed     (sleeping minutes)
  "frying food in a pan"  ●
                            ● 18:35 Tue    ● 19:10 Fri     (frying minutes)
```

Each minute's position is saved in a **vector store**, a database built to find the nearest points quickly.

**2. Searching means finding the nearest points.** The question's sentence is placed on the map, the closest minutes are found, neighbouring close minutes are joined into episodes (L5), and the agent answers with times, confidence and evidence. It searches by meaning, so "frying", "sautéing" and "pan cooking" lead to the same place.

**3. Two ways to answer:**

| | Stored facts | Search by meaning |
|---|---|---|
| For | Common, planned activities (cooking, sleeping, showering) | Anything unplanned (frying, parties, "busy evening") |
| How | Already stored as episodes, so a quick lookup | Search the map with a sentence |
| Speed and sureness | Fast, high confidence | Slower, medium confidence |

Activities that people ask about often "graduate" from search to stored facts.

**4. Search by example.** No words are needed for "find other nights like last Tuesday at 03:00": HomeFM takes that minute's position and finds its nearest neighbours.

### Examples

**"When did we fry something this week?"**
DomusFM cannot answer. HomeFM finds Tuesday 18:35–18:50 (closeness 0.71) and Friday 19:10–19:25 (0.68): "**Probably twice**: Tuesday around 18:35 and Friday around 19:10 (medium confidence). I don't track frying directly; these are the moments that looked most like it."

**"Show me the evening the kitchen was busy for hours."**
HomeFM: "**Saturday, 16:00–21:30.** The stove, oven and fridge were in use almost continuously, with 3–5 people present. It looks like you had guests."

**"Was there a day nobody was home?"**
HomeFM: "**Yes, last Sunday, 09:00–20:00.** No motion anywhere and no phones at home."

### The honest catch

- Search is only as good as the word-linking training (the same catch as L1), which is why answers on this path show medium confidence.
- Closeness is not certainty: "0.71 close" means "looks like frying". Thresholds are tuned per kind of question.
- Storage is small: one position per minute is about 0.4 MB per day, roughly 130 MB per year.
- Positions from different model versions do not match. After an upgrade, history is re-processed or the old index is kept until it expires, and old and new are never mixed.
- **Status:** 📐. The word-linking training exists per window, but the per-minute map, the vector store and the search tool are not built.

---

## L9 · It cannot hear or see

### The problem

DomusFM only knows what switches and sensors report. Anything that does not flip a sensor is invisible:

| Real event | Does a switch flip? | Visible to DomusFM? |
|---|---|---|
| Baby crying | No, it is a sound | ❌ |
| Dog barking | No, it is a sound | ❌ |
| Parcel left at the door | Maybe the door opens, but "parcel" is visual | ❌ |
| Glass breaking, a non-smart smoke alarm | No, they are sounds | ❌ |
| Someone falls | Maybe some motion, but "a fall" is visual | ❌ |

It is like a guard who only watches the switch panel: they know when the door opens, but cannot hear the baby or see the parcel on the step.

### Why DomusFM does it this way

- **Research datasets have no audio or video.**
- **Privacy:** microphones and cameras in homes are sensitive, and many studies avoid them.
- **Computing cost:** sound and video need much more processing than on/off events.
- **Out of scope:** the paper focuses on binary sensors by design (paper §3.1).

### Why it matters for us

Whole areas of questions disappear: childcare ("How many times did my kid cry?"), pets ("Did the dog bark while we were out?"), deliveries ("How many parcels yesterday?"), safety ("Did anything break?"), elderly care ("Did grandma fall?") and visitors ("Who came to the door?"). A door contact says the door opened, but not *why*.

### What HomeFM does differently

**1. Small "expert" detectors on the home hub**, each good at one job:

| Detector | Input | Sends |
|---|---|---|
| Sound | Nursery microphone | "baby crying" |
| Sound | Living-room microphone | "dog barking", "glass breaking" |
| Camera | Doorbell camera | "parcel at door", "person at door" |
| Radar or camera | Grandma's room | "person fell" |

These are off-the-shelf kinds of models (for example, sound classifiers trained on large sound collections already recognise hundreds of everyday sounds).

**2. Only short tags are sent, never raw sound or video.** Each tag is an ordinary event with a confidence:

```
18:40:05  nursery mic      "baby crying"     confidence 0.87
10:14:30  doorbell camera  "parcel at door"  confidence 0.92
```

Raw audio and video are discarded immediately and never leave the home.

**3. HomeFM combines tags with the other sensors.** A detector knows what it heard, and HomeFM knows what is going on in the house. Together they are much more reliable:

| Detector says | Other sensors say | HomeFM concludes |
|---|---|---|
| "baby crying" in the living room | TV on, a film playing | Probably the TV, low confidence |
| "baby crying" in the nursery | Nursery motion, a parent walks in a minute later | Real crying, high confidence |
| "parcel at door" | Door did not open, courier left within 20 s | Parcel delivered |
| "person at door" | Door opened, living-room count rose from 2 to 4 | Guests arrived |
| "dog barking" | All phones away, front door closed | Barked while everyone was out |

### Examples

**"How many times did my kid cry today?"**
DomusFM has no signal. HomeFM merges nursery "baby crying" tags less than 60 seconds apart into episodes (L5), ignores a "crying" tag in the living room while the TV was on, and answers: "**3 times**: 07:12 (4 min), 13:40 (2 min) and 19:05 (6 min). High confidence."

**"How many parcels did we receive yesterday?"**
DomusFM sees "front door opened" 9 times and cannot tell parcels from people. HomeFM: "**2 parcels**: 10:14 and 15:42. Both were picked up later (the door opened at 12:30 and 17:05)."

**"Did the dog bark while we were out?"**
HomeFM: "**Yes, twice**: at 10:20 for about 4 minutes (the doorbell camera saw a delivery person at 10:19, so probably at them) and briefly at 11:50."

### The honest catch

- Answers are only as good as the detectors, and rooms without a microphone or camera stay blind.
- Microphones and cameras are opt-in, room by room. The system tells families what it can and cannot answer with their choices.
- The hub needs enough power for the detectors. Sound models are cheap, and video needs more.
- **Status:** the event format already carries tags with confidence ✅. The detectors themselves are 📐. They are components to plug in, not part of the foundation model.

---

## L10 · It learned from a small, old set of homes

### The problem

DomusFM learned from a handful of public research datasets that are **small** (a few homes, weeks to months each), **old-fashioned** (simple motion, door and bed sensors) and **mostly one elderly person living alone**. A modern home has things those homes never had:

| Modern device | In DomusFM's training data? |
|---|---|
| Smart plugs with power readings | Hardly any |
| Smart locks with user codes | ❌ |
| Doorbell cameras | ❌ |
| Robot vacuums | ❌ |
| Air purifiers, smart speakers, Matter devices | ❌ |
| 150+ devices in one home | ❌ (research homes had about 20–60) |

It is like a driver who only ever practised in one small, quiet town and is then put on a busy city highway.

### Why DomusFM does it this way

- **That is what was publicly available.** Labelled smart-home data is rare and expensive to collect.
- **The model was sized to match** the small corpus, so it would not simply memorise (paper §6.2.1).
- **Shared data lets papers be compared fairly.**

### Why it matters for us

1. **New devices are guesswork.** For a smart lock, all DomusFM has is the text "smart lock, front door". It has never seen codes, auto-locking or failed attempts.
2. **Families, pets and guests are missing** (L4).
3. **Faults and rare events are almost never in the data:** stuck sensors, dying fridges, falls, break-ins.
4. **Few homes means narrow habits.** If most training homes belong to retired people who wake at 6:00, a family with shift workers and teenagers looks "unusual" all the time.

### More data alone does not help (our own runs)

The obvious fix is "use more homes", so we tested it. We pretrained DomusFM on **77 homes and 27 million events**, far more than the paper's corpus. On the test homes so far, the pretrained model is **no better, and mostly worse, than a model with no pretraining at all** (full table in [L11](#what-our-own-runs-show)). More data does not help while the practice game is too easy. **Data and the training game have to improve together.**

### What HomeFM does differently

**1. Much more, and more varied, data:**

| Source | What it adds |
|---|---|
| 81 labelled CASAS homes + Milan + Aruba (✅ loaded) | Many more homes; unlabelled ones (Milan, Aruba) are fine for pretraining |
| Multi-person datasets (ARAS, MuRAL, MARBLE) | Families, couples, who did what (L4) |
| Energy datasets (REFIT: 20 UK homes, about 2 years, per appliance; UK-DALE, REDD and others) | Real power readings (L2) and normal appliance behaviour |
| Sound and image datasets (AudioSet, COCO and others) | Training material for the detectors (L9) |
| Simulated homes | Anything rare or missing (below) |
| Real pilot and donated homes | Modern devices, real noise, real feedback |

**2. A simulator for what no dataset has.** Pets, babies crying in context, parcels, guests, appliance faults and slow health decline do not exist in public event data. An LLM writes realistic daily routines for several residents, babies and pets ("Mum wakes at 6:30, the baby cries at 7:00, the dog goes out at 7:15, a parcel arrives at 10:14"). The simulator turns them into sensor, sound and camera events, **injects faults on purpose** (stretched fridge cycles, stuck sensors, draining batteries), and provides perfect labels and sentences for free. It is like a driving simulator for snow, night and emergencies.

**3. Real homes, the biggest lever.** 10–30 consenting pilot homes provide modern devices, real noise and feedback. Many Home Assistant users already have months to years of history on their own hub, so an opt-in, anonymised **data-donation** tool could add hundreds of modern homes. Pretraining needs no labels, so this takes the corpus from dozens of homes to hundreds.

**4. A big teacher and a small student.** A large model is trained on a server with all the data and then teaches a small model to copy its answers. The small model runs on the home hub: fast, cheap and private. It is like an expert writing a compact handbook for a trainee.

### Example: a new home with a smart lock and a robot vacuum

**DomusFM** has never seen a lock or a vacuum, and relies only on the words "smart lock, front door" and "robot vacuum, living room". It may treat the vacuum's room-by-room path as a person wandering, or ignore the lock.

**HomeFM**, trained with pilot and donated homes that include locks and vacuums, recognises the vacuum's pattern (plug on, a steady sweep across rooms, no doors or lights) and uses lock codes as identity clues: "unlocked with Aarav's code at 16:02".

### The honest catch

- Public data still lacks modern devices. Pilots and data donation take time, consent and privacy safeguards.
- A simulator that is too tidy teaches wrong habits. Simulated data is mixed with real data at a controlled ratio, and **results are only reported on real homes**.
- Every dataset's licence must be checked (Pecan Street, for example, is restricted).
- **Status:** loading all 84 homes (81 CASAS labelled + Milan, Aruba, UCI B) and corpus pretraining ✅ (used in the run below). Energy and multi-person datasets, the simulator at scale, pilot homes, data donation and the teacher–student step 📐.

---

## L11 · Its practice game is too easy

### What a practice game is

Before seeing any labels, a model **practises on its own** with raw data. This is **pretraining**, and it needs a game with automatic right and wrong answers. A good game can only be won by **understanding** the data. A bad game can be won with **cheap tricks**, and then the model learns the tricks.

### DomusFM's game: "find your own copy"

1. Take a window of 30 events.
2. Make a copy and hide a few bits of it (for example, "motion ON" becomes "motion ???").
3. Mix the copies of many windows together.
4. For each original window, pick out its own damaged copy.

It is like being shown your holiday photo and 64 slightly smudged photos, and asked which smudged one is yours.

### Why it is too easy on home data

1. **Nearly everything stays visible.** Hiding 2 of 30 events leaves 28 unchanged.
2. **Time stamps act as a fingerprint.** Every window happened at a different time, so the remaining timestamps identify the copy without any understanding of the home.
3. **Paired signals reveal hidden parts.** Motion sensors fire ON and then OFF seconds later, and doors go OPEN then CLOSE. The visible partner gives the hidden one away. The model learns facts about sensors, not people.
4. **It punishes real similarity.** Two ordinary nights of sleep in the same round are pushed apart as "different", although they are the same behaviour.

It is like a quiz where the answer is printed next to each question: a 100 % score, and nothing learned.

### Why DomusFM does it this way

- **This kind of game (contrastive learning) worked very well for images and text**, so it was a natural choice.
- **It is simple and needs no labels.**
- **The paper reported gains** on its benchmarks.

Home sensor data is different from photos. Its timestamps, paired ON/OFF signals and endlessly repeating routines make this game far too easy.

### What our own runs show

**1. The game is won almost immediately.** The loss (lower means winning more easily) falls to about 0.001 within the first 1,000 of 40,000 practice steps, and then barely moves. This happened with a handful of homes and again with 77 homes:

```
step     0:  loss 0.03
step  1000:  loss 0.001    ← already "won"
step 19000:  loss 0.0005   ← nothing more to learn for 18,000 steps
```

**2. Practising this game can make the model worse at real tasks.** Results on the held-out homes from the 77-home run (run 1, 2026-09-24; 3 of 7 homes finished, the rest are still running). Higher is better, ± is the spread across 3 folds:

| Test home | Task | Pretrained on 77 homes | No pretraining | Difference |
|---|---|---|---|---|
| UCI B | Activity, 5 % labels | 0.25 ± 0.03 | 0.22 ± 0.06 | +0.04 |
| UCI B | Activity, 30 % labels | 0.28 ± 0.03 | 0.26 ± 0.05 | +0.02 |
| UCI B | Next 30 events, 5 % / 30 % | 0.70 / 0.70 | 0.71 / 0.71 | −0.01 |
| hh101 | Activity, 5 % labels | 0.49 ± 0.02 | **0.56** ± 0.01 | **−0.07** |
| hh101 | Activity, 30 % labels | 0.49 ± 0.01 | **0.55** ± 0.02 | **−0.06** |
| hh101 | Next 30 events, 5 % / 30 % | 0.64 / 0.63 | 0.65 / 0.63 | −0.01 / 0.00 |
| hh103 | Activity, 5 % labels | 0.54 ± 0.02 | **0.68** ± 0.01 | **−0.14** |
| hh103 | Activity, 30 % labels | 0.59 ± 0.01 | **0.73** ± 0.01 | **−0.14** |
| hh103 | Next 30 events, 5 % / 30 % | 0.61 / 0.62 | 0.61 / 0.61 | 0.00 / +0.01 |

In two of three homes, the pretrained model is clearly **worse** at activity recognition than a model that never practised, even with 30 % of labels. Next-event prediction shows no difference. Our best explanation is that the model learns sensor tricks (timestamp fingerprints, ON/OFF pairs) that get in the way when it later has to learn real activities. Details: [DOMUSFM_REPRODUCTION.md](DOMUSFM_REPRODUCTION.md).

### What HomeFM does differently: three harder, more useful games

**Game 1: "What happens next, and when?"** From everything so far, predict the next event and how long until it happens. There is no copy to peek at, because the future is genuinely unseen. To win, the model must learn routines ("after the fridge opens and the stove turns on, the kitchen stays busy for about 30 minutes"). It also yields the surprise score (L6) and "when" predictions (L7). It is like finishing someone's sentence: you have to understand what they are saying, not just recognise their handwriting.

**Game 2: "Hide a big chunk and guess what it meant."** Hide 10 whole minutes, a whole room, all power meters, or rare devices, and predict the **meaning** of the hidden part (a summary), not its exact events. Everything nearby is hidden too, so there is no ON/OFF partner to copy and no fingerprint. The model must reason from context: "before, someone entered the kitchen and opened the fridge; after, the stove turned off and the dining-room light came on; so the gap was probably cooking". It is like guessing what a missing page of a story was about.

**Game 3: "Match minutes with sentences."** Pair minutes with sentences and learn to put them close together (L1, L8). Many right answers are allowed (a cooking minute can match "cooking", "someone in the kitchen" and "dinner prep"), so two nights of sleep both match "someone is sleeping" and end up together, not apart.

**Extra fixes aimed at "too easy" (variant G, DESIGN.md §8.4):**

| Fix | In plain words |
|---|---|
| Hide pairs together | If "motion ON" is hidden, hide its "OFF" too |
| Auto-difficulty | If the game gets too easy, hide bigger chunks automatically, like a video game raising the level |
| Look further ahead | Predict the summary of the next 1, 10 and 60 minutes, not only the next event |
| Same time, different day | Tuesday 7:00 and Wednesday 7:00 are treated as "probably similar" |
| Anti-cheating check | Stops the model from giving the same answer to everything |

### Example: what each game teaches

**DomusFM's game** sees "18:31 motion ???, 18:31 motion OFF" with 28 other events visible. It learns "a hidden motion before an OFF was an ON" and "Tuesday 18:31 events belong together". It learns nothing about people.

**HomeFM's games:**

- **Game 1** sees "fridge open, stove on" and must predict what comes next. It learns that the kitchen stays busy, the stove goes off in about 30 minutes, and the dining-room light follows.
- **Game 2** sees 10 hidden kitchen minutes between "fridge opened" and "dining light on", and learns that the gap was cooking.
- **Game 3** sees a cooking minute next to "someone is cooking", and learns what that pattern is called.

Each game can only be won by understanding what people are doing.

### The honest catch

- **Not proven yet.** HomeFM's games are designed to be harder and more useful, but they have not yet been shown to beat DomusFM on real homes. The next step is the head-to-head run: HomeFM (games 1 + 2, variant E) against DomusFM on the same 77 homes and the same test homes.
- **Success looks like** a loss that falls gradually instead of reaching 0.001 in minutes, and a clear gain from pretraining over no pretraining on the test homes, especially with few labels.
- Harder games cost more computing time, especially game 2, which runs a second copy of the model.
- **Status:** games 1–3 (variants A–F) ✅, implemented and tested on small data. The corpus-scale comparison and variant G 📐.

---

## What DomusFM does well, and HomeFM keeps

- **Describing devices in words** ("motion sensor, kitchen, ceiling") is what lets a model work in a home it has never seen.
- **Combining what, where, state and time** into one vector per event.
- **Training without labels** first, then using a few labels.
- **Small enough to run on a home hub:** about 10 ms per window on a low-power CPU.
- **Testing on homes never seen in training**, which HomeFM adopts as its evaluation method.
