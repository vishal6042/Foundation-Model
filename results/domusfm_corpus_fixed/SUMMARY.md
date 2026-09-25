# DomusFM pretraining fix: plain-language summary

A one-page explanation of the 2026-09-25 run. The full numbers are in [results.md](results.md). The technical
details of each fix are in [docs/DOMUSFM_REPRODUCTION.md](../../docs/DOMUSFM_REPRODUCTION.md), under "Why
pretraining did not help, and the fix".

## What we are testing

DomusFM is trained in two steps:

1. **Pretraining.** The model learns from unlabelled smart-home sensor data from 77 homes, 27.3 million
   events in total. Nobody tells it what the person is doing.
2. **Fine-tuning.** The model is given a small number of labelled examples from 7 homes it has never seen,
   and it is scored on two tasks:
   - **ADL:** recognise the activity (cooking, sleeping, and so on).
   - **Next-30:** predict the next 30 sensor events.

   Each task is run with 5 % of the labels and with 30 % of the labels.

Every score is compared against **the same model trained from scratch**, with no pretraining. Pretraining is only
worth having if it beats that baseline. Scores are F1, from 0 to 1, where higher is better.

## The problem last time: the model cheated

The pretraining task is a matching game. The model sees two partly hidden copies of the same stretch of sensor
events and has to pick the right partner out of 128 candidates. To win, it should learn how people behave in a home.

Instead, it found shortcuts that let it win without learning anything useful:

| Shortcut | Why it made the game too easy |
|---|---|
| **The wrong candidates came from other homes** | Every home has different sensor names, so the model could rule out a wrong candidate from the sensor names alone. |
| **Exact timestamps** | Each event records its exact second, and most events stay visible. Matching two copies was like matching serial numbers. |
| **One copy was never hidden** | 85 % of the hidden copy matched the unhidden copy exactly. |
| **No separate head for the game** | The main network was bent toward the matching game itself, and fine-tuning later had to undo that. |
| **Nothing to predict** | Every answer was visible, so there was no need to understand the data. |
| **One learning rate for everything** | During fine-tuning, the pretrained part and the new output layer moved at the same speed, which could wipe out what pretraining had learned. |

**The sign of cheating:** the matching loss fell from 2.35 to about **0.0005 within 5,000 steps**, which means
near-perfect scores almost immediately. The model "passed the test" without learning.

**The result:** the pretrained model did **worse** than the model trained from scratch. It lost on 6 of 7 homes
for activity recognition.

## The fixes

Each fix removes one shortcut. All six are combined in `configs/domusfm_corpus_fixed.yaml`. They are switched off
by default, so the paper reproduction is unchanged.

| # | Fix | What it does | Config setting |
|---|---|---|---|
| 1 | **Same-home candidates** | Each batch holds 4 homes × 32 windows, so the wrong candidates come from the same home and sensor names no longer give the answer away. | `homes_per_batch: 4` |
| 2 | **Clock jitter** | Each copy's clock is shifted by up to ±15 minutes. The gaps between events and the time of day are kept, but exact-second matching no longer works. | `time_jitter_s: 900` |
| 3 | **Hide both copies** | Both copies are partly hidden (40 % masking), and the game is made less strict. | `mask_both_views: true`, `attr_mask_p` / `event_mask_p: 0.4`, `temperature: 0.2` |
| 4 | **A separate head for the game** | A small extra layer plays the matching game and is thrown away after pretraining, so the main network is not distorted. | `projector: true` |
| 5 | **Fill in the blanks** | The model must also predict hidden details (which sensor, what type, which room, ON/OFF). It cannot do that without understanding the data. | `mlm_weight: 1.0` |
| 6 | **Gentler fine-tuning** | The pretrained part learns slowly (5e-5), the new output layer learns fast (1e-3), and there is a short warm-up. This is applied to the from-scratch baseline too, to keep the comparison fair. | `backbone_lr`, `head_lr`, `warmup_frac` |

## How much it improved

### 1. The model stopped cheating

| | Paper setup (last run) | Fixed setup (this run) |
|---|---|---|
| Matching loss | Fell to ~0.0005 by step 5,000 | Stayed at **0.70–0.82** for all 40,000 steps |
| Fill-in-the-blanks loss | Not used | Fell steadily: 3.60 → 0.16 (phase 1), 2.28 → 0.58 (phase 2) |

A loss that stays at a healthy level means the game is still hard, so the model keeps learning.

### 2. Pretraining now helps instead of hurting

The gain from pretraining is the pretrained score minus the from-scratch score, averaged over the 7 new homes. A
positive number means pretraining helped.

| Task | Labels | Last run | **This run** | Homes where pretraining wins |
|---|---|---|---|---|
| Activity recognition | 5 % | −0.050 | **+0.076** | 1/7 → **7/7** |
| Activity recognition | 30 % | −0.068 | **+0.059** | 1/7 → **6/7** |
| Next-30 prediction | 5 % | −0.009 | **+0.053** | 1/7 → **7/7** |
| Next-30 prediction | 30 % | +0.002 | **+0.030** | 3/7 → **6/7** |

**Overall, pretraining now wins in 26 of 28 comparisons** (7 homes × 2 tasks × 2 label amounts). The two losses
are about 0.01 each, which is within noise.

### 3. Absolute scores

| Task | Labels | Pretrained, last run | **Pretrained, this run** | From scratch, this run |
|---|---|---|---|---|
| Activity recognition | 5 % | 0.385 | **0.517** | 0.441 |
| Activity recognition | 30 % | 0.400 | **0.529** | 0.470 |
| Next-30 prediction | 5 % | 0.587 | **0.649** | 0.597 |
| Next-30 prediction | 30 % | 0.591 | **0.613** | 0.583 |

The pretrained model improved by about **0.13 on activity recognition**. The from-scratch baseline barely moved
(0.435 → 0.441 at 5 %), so the improvement comes from better pretraining, not from the fine-tuning change.

## Caveats

- **One home inflates the average.** UCI B, the smallest and most unusual home, gains +0.20 to +0.28 on activity
  recognition. Without it, the activity-recognition gain is +0.055 (5 %) and +0.022 (30 %). That is still
  positive, but more modest.
- **Small differences are noise.** A difference under about 0.03 on a single home is within the spread between
  folds.
- **The help shrinks as labels grow.** The gain is larger with 5 % labels than with 30 %. That is expected:
  pretraining matters most when labelled data is scarce.
- **One health signal looks odd.** `z_std`, the spread of the embeddings, rose to 0.088 by step 1,000 and then
  stayed exactly there. That is far from 0, so the embeddings did not collapse, but a perfectly flat value should
  be checked before relying on it.
- **This is still far from solved.** Activity recognition is around 0.5 F1.

## Next steps

- Run strong-masking DomusFM and HomeFM (8.0M and size-matched 28.6M) on the same setup, and compare them against
  these numbers.
- Run an ablation: turn the six fixes off one at a time to see which ones matter.

Run details: one RTX 4090, about 6.5 hours in total. Pretraining took about 2 hours because the GPU was shared;
on an idle GPU it takes about 70 minutes.
