# DomusFM limitations and HomeFM remedies, in plain words

Companion to [DESIGN.md](DESIGN.md) §4.1 (the formal list) and §4.2 (the step-by-step technical comparison). This document explains each limitation without jargon. Any technical term that remains is defined in DESIGN.md §1.1.

Each limitation answers the same six questions:

1. **The problem:** what DomusFM cannot do, with an everyday comparison.
2. **Why DomusFM does it this way:** the limitation is usually a sensible choice for the paper's goal, which was winning research benchmarks, not answering household questions.
3. **Why it matters for us:** the user questions that break.
4. **What HomeFM does differently:** the remedy.
5. **Example:** one question, answered by each model.
6. **The honest catch:** what is still hard, and what is built (✅) or only designed (📐).

## Summary

| # | Limitation | In one line | HomeFM remedy | Status |
|---|---|---|---|---|
| [L1](#l1--it-can-only-name-what-it-was-taught) | Closed label set | Can only name activities it was shown examples of | Link sensor activity to words, so any activity can be described | 📐 (alignment training ✅) |
| [L2](#l2--it-only-understands-on-and-off) | On/off only | Sees "fridge ON", not "fridge draws 180 W" | Keep real numbers as input | ✅ input · 📐 device health |
| [L3](#l3--its-memory-is-30-events-not-a-length-of-time) | 30-event window | Memory is sometimes 2 minutes, sometimes 6 hours, never weeks | 1-minute steps, hours of context, daily summaries | ✅ minutes · 📐 days |
| [L4](#l4--it-assumes-one-person-lives-there) | One person assumed | Cannot tell people apart or count them | Person cards, per-room counts | 📐 |
| [L5](#l5--it-labels-moments-but-cannot-count-occurrences) | No start or end | Cannot count "cooked 3 times" or see two things at once | Several tags at once, start/end detection, counting rules | 📐 |
| [L6](#l6--it-cannot-say-unusual-or-broken) | No anomaly or fault detection | Cannot say "this is unusual" or "this sensor is broken" | Surprise score, anomaly and device-health engines | ✅ surprise · 📐 engines |
| [L7](#l7--it-knows-what-comes-next-but-not-when) | Forecast without timing | Knows what comes next, not when | Also predicts how long until the next event | ✅ |
| [L8](#l8--its-knowledge-cannot-be-searched-with-words) | Not searchable by text | Its knowledge cannot be searched with a question | Minute summaries stored next to words | 📐 |
| [L9](#l9--it-cannot-hear-or-see) | No sound or camera | Cannot hear a baby cry or see a parcel | On-hub sound and camera detectors send tags | 📐 |
| [L10](#l10--it-learned-from-a-small-old-set-of-homes) | Small, old training data | Never saw smart locks, cameras or robot vacuums | Many more homes, simulated homes, pilot homes | ✅ 82 homes · 📐 rest |
| [L11](#l11--its-practice-game-is-too-easy) | Practice game too easy | Learns sensor quirks instead of behaviour | Harder, more useful practice games | ✅ A–F · 📐 G |

Several limitations share one remedy. **L1 and L8** are both fixed by linking sensor activity to words: L1 is about *naming* an activity, and L8 is about *searching* for it. **L10 and L11** are connected: our own runs show that more data does not help while the practice game stays too easy.

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
- **Status:** the training that links minutes and sentences exists (variant F, per window) ✅. Per-minute tagging, the vector store and the search tool are 📐. There is no measured result yet. The planned test hides some activity names during training and checks whether HomeFM finds them from words alone.

---

## L2 · It only understands on and off

### The problem

DomusFM only understands **on and off**. Devices that report amounts, such as power, temperature or humidity, must first be squashed into on/off, and the amount is thrown away.

It is like being told only whether a tap is open or closed, never how much water is flowing.

### Why DomusFM does it this way

- **Most research datasets use on/off sensors:** motion, door contacts, bed pressure. One simple format fits all of them.
- **It keeps the model simple:** every event looks the same.
- **The paper admits the cost:** continuous readings become "virtual" on/off events, with possible information loss (paper §3.1, §7.4.2).

### Why it matters for us

- *"Is my fridge OK?"* A failing compressor runs 25 minutes per cycle instead of 12 and draws 20 % more power. As on/off, both look like "fridge ON … fridge OFF".
- *"Which appliance used the most energy today?"* On/off events carry no watts.
- A humidity sensor turned into "shower ON/OFF" needs a threshold tuned for each bathroom and season. A wrong threshold silently invents or deletes showers.

### What HomeFM does differently

Numbers enter the model **as numbers**. 1,850 W stays 1,850 W. Each reading is sent when it changes, so the model sees the level, the on/off rhythm and the trend.

### Example: "Is my fridge OK?"

**DomusFM:** sees "fridge ON, fridge OFF" as usual. It notices nothing, because the power level was never given to it.

**HomeFM:**

1. Sees each cooling cycle's length and power.
2. The device-health engine compares this month with the fridge's own history: cycles have doubled from 12 to 25 minutes, and power is 20 % higher.
3. Answer: "**The fridge is working harder than usual**: cycles are twice as long as last month. This can mean a door-seal or compressor problem."

### The honest catch

- **Energy totals are simple sums.** "How much energy in the last hour?" is answered directly from the stored meter readings with a database sum, and the model is not needed. The model adds *understanding*: which activity used the energy, and whether consumption is unusual.
- Public data has few power meters, so this skill needs energy datasets and simulated or pilot homes (DESIGN.md §11).
- **Status:** numbers as input ✅. The device-health engine is 📐.

---

## L3 · Its memory is 30 events, not a length of time

### The problem

DomusFM always looks at exactly **the last 30 events**. How much time that covers depends on how busy the house is: about 2 minutes while cooking, about 6 hours at night. It never sees days or weeks.

It is like remembering "the last 30 words someone said". In a fast conversation that is 10 seconds, and in a slow one it is an hour.

### Why DomusFM does it this way

- **It is the standard in activity-recognition research.** Earlier models used event-count windows too, so results are comparable.
- **Every window holds the same amount of information.** Busy and quiet periods are both 30 events.
- **The paper tested a time-based alternative** (5-minute windows), and it did slightly worse *on their benchmark* (paper §7.2).

### Why it matters for us

Users ask in time, not events: "in the last 4 hours", "last night", "compared with last month".

- A 40-minute cooking session in a busy kitchen spans about 20 windows, and none of them sees both its start and its end.
- A quiet night packs sleep, a bathroom trip and an early breakfast into one window.
- Slow changes ("Is Mum sleeping worse than last month?", "Is the fridge degrading?") need weeks.

### What HomeFM does differently

- Time is cut into **1-minute slots**, and each minute is summarised into one vector. "The last 4 hours" is always 240 steps, however busy the home was.
- The model reads **hours of minutes** in order.
- **Daily summaries** let it compare weeks.

### Example: "How many times did cooking happen in the last 4 hours?"

**DomusFM:** the 40-minute dinner is seen in about 20 overlapping fragments of about 2 minutes each. Each fragment gets a label, and something outside the model has to guess where cooking started and ended.

**HomeFM:** reads the 40 cooking minutes together with the minutes before and after them. It sees the start (someone enters the kitchen, the stove turns on) and the end (the stove turns off, the kitchen goes quiet). The result is one clean cooking session.

### The honest catch

- A minute summary could blur detail inside the minute. HomeFM keeps an event-level read-out for when single events matter, such as predicting the next one.
- DomusFM's time-based test did worse. HomeFM has to *show* that its minute design does not lose the same way, and this is tested in the same comparison (DESIGN.md §12).
- **Status:** minute slots and the hours-long stream ✅. Daily summaries for weeks are 📐.

---

## L4 · It assumes one person lives there

### The problem

DomusFM assumes **one person** lives in the home, or that someone has already said who caused each event. A motion sensor fires the same way for grandma, her grandson or the dog.

It is like hearing footsteps in the house without knowing whose they are.

### Why DomusFM does it this way

- **Most research datasets are single-person homes.**
- **On/off sensors carry no identity.** Telling people apart from motion sensors alone is an unsolved research problem, and the paper calls it an open challenge (paper §7.4.1).

### Why it matters for us

- *"How many people are in the living room?"* One person and four people trigger the same motion sensor.
- *"Did grandma eat lunch?"* If her grandson cooked at noon, the kitchen activity is credited to "the resident", and grandma's missed meal is hidden.
- The dog walking through the hallway at 2 am looks like an elderly person wandering at night: either a false alarm, or real wandering is learned as normal.
- *"Did we have guests?"* Guests show up mainly as *more people than usual*, which a one-person model cannot represent.

### What HomeFM does differently

- A **person card** for each resident and pet. Events are assigned to cards.
- A **people count per room**.
- It uses signals that carry identity where the household allows them: phone presence, camera person counts, radar.
- With motion sensors only, it gives an **honest range** ("1–2 people") instead of a confident wrong number.

### Example: "Did grandma eat lunch?"

**DomusFM:** sees kitchen activity at noon and labels it "eating" for "the resident". The implied answer is "yes", which is wrong.

**HomeFM:**

1. Phone presence shows the grandson is home. Grandma's bedroom motion continues from 11:30 to 14:00.
2. The kitchen activity at noon is assigned to the grandson's card.
3. Answer: "**I see no sign that grandma went to the kitchen between 12:00 and 14:00.** Her grandson was cooking at noon. She stayed in her bedroom." (medium confidence)

### The honest catch

- With motion sensors only, telling people apart stays fundamentally hard. Good answers need at least one identity signal, and that raises privacy and consent questions (DESIGN.md §13).
- **Status:** 📐.

---

## L5 · It labels moments but cannot count occurrences

### The problem

DomusFM puts **one activity label on each moment**, but never says where an activity **starts and ends**. Counting "how many times" needs those start and end points. It also cannot show two things happening at once.

It is like a photo album with a caption on every photo, but nobody marks where one party ends and the next begins.

### Why DomusFM does it this way

- **The benchmark task is "label each window".** Datasets are scored that way, so that is what the model was built for (paper §6.4.1).
- Turning labels into counted occurrences was outside the paper's scope.

### Why it matters for us

- *"How many times did cooking happen today?"* A 2-minute pause while stirring can split one cooking session into three, so the count is wrong.
- Short "Other" labels in the middle fragment sessions further.
- Cooking while the baby cries: one label must pick one of the two, so one count is wrong.

### What HomeFM does differently

- **Several tags per minute**, so cooking and crying can both be on.
- A head that marks **starts and ends**.
- **Counting rules per activity**, kept in the home's dictionary, for example "breaks under 10 minutes are still the same cooking", "cries under 60 seconds apart count as one", and "cooking under 3 minutes does not count".
- Counts come from these **episodes**, not from individual moments.

### Example: "How many times did cooking happen today?"

**DomusFM:** the labels read cooking, cooking, other, cooking, other, cooking… Counting stretches of "cooking" gives **3**.

**HomeFM:** the stirring pause (6 minutes) is shorter than the 10-minute rule, so the pieces merge. The answer is **1 time, 18:30–19:22** (DESIGN.md §10.2 walks through this exact case).

### The honest catch

- The rules need sensible defaults per activity and tuning per home from feedback ("that was just tea"). This is the biggest single driver of counting accuracy.
- **Status:** 📐. Counting accuracy is a headline measure in the evaluation plan (DESIGN.md §12).

---

## L6 · It cannot say "unusual" or "broken"

### The problem

DomusFM can describe what is happening, but it cannot say **how unusual** it is for this home, and it cannot tell when a sensor or appliance is **broken**.

It is like a guard who can describe what they see, but has no idea what is normal for this particular house.

### Why DomusFM does it this way

- **It was not the paper's goal.** Anomaly detection and behaviour-change detection are listed as future work (paper §7.4.4).
- **Its training gives similarities, not likelihoods.** It learns "these windows are alike", not "this event had a 1-in-1,000 chance".

### Why it matters for us

- *"Anything unusual last night?"* The front door opened at 03:12.
- A motion sensor stuck ON after a battery fault looks like someone is always present.
- *"Is Dad's routine changing?"* Three night-time bathroom trips instead of one, week after week, is an early health signal.

### What HomeFM does differently

- HomeFM learns to **predict the next event**, so it can measure **surprise**: how unlikely what just happened was, given this home's history.
- The **anomaly engine** combines surprise with rarity ("how often have we seen anything like this here?") and routine drift ("how far has this person's routine moved?"), and **always explains** its flags.
- The **device-health engine** compares each device with its own past and with the same type of device in other homes.

### Example: "Anything unusual last night?"

**DomusFM:** it can label the 03:12 window, but has no measure of how odd it is, so it cannot answer.

**HomeFM:**

1. The door event at 03:12 gets a high surprise score, because this home has had no door events after 23:00 in six weeks.
2. The anomaly engine flags it and writes an explanation.
3. Answer: "**Yes, one thing:** the front door opened at 03:12 and closed at 03:14. This home normally has no door activity after 23:00."

### The honest catch

- "Unusual" needs a few weeks of this home's history before it is reliable.
- Too many alerts and people ignore them. Too few and real problems are missed, so thresholds must be tuned.
- **Status:** the surprise score ✅ (it comes from the next-event game). The anomaly and device-health engines are 📐.

---

## L7 · It knows what comes next, but not when

### The problem

DomusFM can guess **which** events are likely soon, but not **when**, or in what order.

It is like a weather forecast that says "rain is coming" without saying whether that means today or next week.

### Why DomusFM does it this way

- **It predicts a bag of the next 30 events** and deliberately ignores order and timing (paper §6.5.1). This makes the task easier to score.
- Its time-based variant did slightly worse (paper §7.2).

### Why it matters for us

- *"Will Dad be up soon? I want the coffee ready."* The model can say bedroom and bathroom events are likely among the next 30, but not whether that is in 5 minutes or 3 hours.
- **Missed routines:** there is no kitchen activity by 11:00 when breakfast is normally done by 9:00. Spotting this needs a timed expectation ("breakfast should have started by now").

### What HomeFM does differently

The next-event head predicts **which device, what value, and how long until it happens**, as a spread of likely waiting times rather than one guess. "Expected by" deadlines and missed routines can then be checked.

### Example: "Will Dad be up soon?"

**DomusFM:** "Bedroom and bathroom events are likely next." There is no time, so the question is not answered.

**HomeFM:** "**Probably in about 20 minutes** (most likely between 10 and 40). On weekdays he is usually up by 7:10."

### The honest catch

- Predictions are only as good as the routine is regular. Irregular days give wide ranges, which is the honest answer.
- **Status:** the timing head ✅ (trained in variants D and E). The missed-routine checks built on it are 📐.

---

## L8 · Its knowledge cannot be searched with words

### The problem

DomusFM stores what it learned as lists of numbers that are **not linked to language**. You cannot type a question and find matching moments.

It is like a library where the books are on the shelves but there is no catalogue: you cannot search by topic.

### Why DomusFM does it this way

- **The datasets have no descriptive sentences** to link with (the same reason as L1).
- **Its benchmarks never needed search.** They only needed labels.

### Why it matters for us

Questions nobody planned for, such as *"When did we fry something this week?"* or *"Show me the evening the kitchen was busy for hours"*, cannot use the model at all.

### What HomeFM does differently

Each minute's summary is stored in a **vector store**, in the same "space" as sentences. The agent's search tool turns the question into numbers and finds the closest minutes. This is the same idea that fixes L1: L1 uses it to *name* activities, and L8 uses it to *search* history.

### Example: "When did we fry something this week?"

**DomusFM:** no "frying" label and no way to search, so it cannot answer.

**HomeFM:**

1. Searches for minutes like "frying food in a pan".
2. Finds Tuesday 18:35–18:50 and Friday 19:10–19:25.
3. Answer: "**Probably twice**: Tuesday around 18:35 and Friday around 19:10 (medium confidence). I don't track frying directly."

### The honest catch

- Search quality depends on the same sensor-plus-sentence training as L1.
- Stored vectors belong to one model version. After an upgrade, history is re-processed so old and new are never mixed.
- **Status:** 📐.

---

## L9 · It cannot hear or see

### The problem

DomusFM cannot hear or see. Anything that does not flip a sensor switch is invisible to it.

It is like a guard who only reads the switch panel and cannot hear the baby or see the front step.

### Why DomusFM does it this way

- **The research datasets have no audio or video.**
- **Privacy and computing cost:** processing sound and video is heavier and more sensitive, and it was outside the paper's scope.

### Why it matters for us

Whole question areas disappear:

- *"How many times did my kid cry?"* and *"Did the dog bark while we were out?"* Crying and barking flip no switch.
- *"How many parcels did we receive yesterday?"* A door contact shows the door opened. Whether a parcel was left, a guest arrived or someone went out needs a camera.

### What HomeFM does differently

Small **sound and camera detectors run on the home hub**. They send only short tags with a confidence, such as "baby crying (0.87)" or "parcel at front door (0.92)", in the same format as every other event. HomeFM combines them with the sensor stream. **Raw audio and video never leave the home.**

### Example: "How many times did my kid cry today?"

**DomusFM:** there is no signal for crying, so it cannot answer.

**HomeFM:**

1. The nursery sound detector sends "baby crying" tags through the day.
2. Cries less than 60 seconds apart are merged into one episode (the counting rule from L5).
3. Answer: "**3 times**: 07:12, 13:40 and 19:05 (high confidence)."

### The honest catch

- Answers are only as good as the detectors. Rooms without a microphone or camera stay blind.
- Households must opt in, per room.
- **Status:** 📐. The detectors are off-the-shelf components to integrate, not part of the foundation model itself.

---

## L10 · It learned from a small, old set of homes

### The problem

DomusFM learned from a few public datasets: mostly single-person homes with older on/off sensors. It never saw many of the devices in a modern smart home.

It is like a driver who only ever practised in one small town.

### Why DomusFM does it this way

- **That is what was publicly available** with activity labels.
- The model was sized to that small amount of data (paper §6.2.1).

### Why it matters for us

A modern home has smart plugs, a doorbell camera, a smart lock, a robot vacuum and 150 devices. Locks, cameras and vacuums never appeared in training, so the model's only knowledge of them is their text description ("smart lock, front door").

### What HomeFM does differently

- **Many more homes:** 82 labelled CASAS homes plus other public datasets, energy datasets, simulated homes with multiple people and faults added on purpose, and real pilot homes (DESIGN.md §11).
- **A big teacher, a small student:** a large model trained on a server passes its knowledge to a small model that runs on the home hub (DESIGN.md §7.3).

### What our own runs show

We pretrained DomusFM on **77 homes and 27 million events**, far more than the paper's corpus. On the first test home (UCI B), the result is about the same as pretraining on a handful of homes, and only slightly better than no pretraining at all:

| Test (UCI B) | Pretrained on 77 homes | No pretraining |
|---|---|---|
| Activity, 5 % labels | 0.25 | 0.22 |
| Activity, 30 % labels | 0.28 | 0.26 |
| Next 30 events, 5 % labels | 0.70 | 0.71 |

**More data alone did not help, because the practice game is too easy (L11).** Data and the training game have to improve together.

### The honest catch

- Public data still lacks new device types. The simulator must be realistic enough to teach something useful, and pilot homes are needed for real coverage.
- **Status:** the 82-home loader and corpus pretraining ✅. Energy data, the simulator at scale, pilot homes and the teacher–student step are 📐.

---

## L11 · Its practice game is too easy

### The problem

DomusFM's self-training game is: "hide a few events, then recognise that the two versions are the same window". On home data this game is **too easy**. The model wins by learning sensor quirks, not what people do.

It is like a quiz where the answer is printed next to each question. You score 100 % and learn nothing.

### Why DomusFM does it this way

- **This style of game worked very well for images and text**, so it was a natural choice.
- On the paper's benchmarks it reported gains.

### Why it's too easy on home data

- **Paired signals give it away.** A motion sensor fires ON, then OFF a few seconds later. Hide the ON and the OFF next to it reveals it.
- **The two versions are nearly identical.** Hiding 2 of 30 events leaves 28 unchanged, so matching them is trivial.
- **It punishes real similarity.** Two ordinary nights of sleep in the same batch are pushed apart as "different", although they are the same behaviour.

### What our own runs show

- The game's score (the loss) falls to **about 0.001 within the first 1,000 steps**. This happened in both phases, with a few homes and with 77 homes.
- Pretraining then gives small, uncertain gains on activity recognition and none on next-event prediction (see the L10 table). This is our measured evidence, not the paper's.

### What HomeFM does differently

Three harder, more useful games:

1. **Predict the next event and when it will happen.** This teaches routines and timing, and gives the surprise score (L6, L7).
2. **Hide a big chunk and guess what it meant:** 10 whole minutes, a whole room, or all power meters. There is no partner signal to copy from, so the model must reason about what was going on.
3. **Match minutes with sentences.** This links the model to words (L1, L8), and a minute may match several sentences, so repeated routines are not pushed apart.

Variant G (DESIGN.md §8.4) adds further refinements aimed directly at the "too easy" failure. Examples: hide ON/OFF pairs together, and make the game harder automatically when the score gets too low.

### Example: what the model has to figure out

**DomusFM game:** "motion ON (hidden) … motion OFF 5 s later". The answer is obvious from the neighbour, and nothing is learned about behaviour.

**HomeFM game:** "18:30–18:40 in the kitchen is hidden entirely. Before: someone entered the kitchen and opened the fridge. After: the stove turned off and the dining-room light came on." To fill the gap well, the model has to learn "that was cooking".

### The honest catch

- That these games work better is **still unproven**. The comparison of HomeFM (variant E) against DomusFM on the same homes and data is the next run (DESIGN.md §8.3).
- **Status:** variants A–F ✅ (implemented and tested on small data). The corpus-scale comparison and variant G are 📐.

---

## What DomusFM does well, and HomeFM keeps

- **Describing devices in words** ("motion sensor, kitchen, ceiling") is what lets a model work in a home it has never seen.
- **Combining what, where, state and time** into one vector per event.
- **Training without labels** first, then using a few labels.
- **Small enough to run on a home hub:** about 10 ms per window on a low-power CPU.
- **Testing on homes never seen in training**, which HomeFM adopts as its evaluation method.
