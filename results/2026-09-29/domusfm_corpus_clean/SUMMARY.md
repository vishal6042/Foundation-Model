# DomusFM run 3: plain-language summary

The third corpus-scale DomusFM run, finished 2026-09-29 after 10.1 hours on one RTX 4090. Config:
[configs/domusfm_corpus_clean.yaml](../../../configs/domusfm_corpus_clean.yaml). Full tables:
[comparison.md](comparison.md). Raw numbers: [results.json](results.json), log: [train.log](train.log).

## What changed from run 2

Run 2 ([results/domusfm_corpus_fixed](../../domusfm_corpus_fixed/SUMMARY.md)) fixed DomusFM's pretraining game and
made pretraining help. Run 3 keeps that model, game and fine-tuning, and changes three things, each taken from the
paper:

| # | Change | Why |
|---|---|---|
| 1 | **Clean the data before cutting windows.** A sensor's events must alternate ON/OFF; a repeated ON (or OFF) is dropped. Undocumented numeric codes on motion sensors are dropped too. 10.1 % of events removed. | The paper's Appendix A does this; our runs 1–2 did not. |
| 2 | **Test on the paper's own datasets too.** Kasteren A, Kasteren C and MuRAL are added, so 10 test homes in total. | Three of the paper's seven datasets. Orange4Home is available only by email request. |
| 3 | **Pretrain the way the paper does.** Each paper dataset is left out of its own pretraining, but the other paper datasets are included. That makes 5 pretraining runs instead of 1. | Run 2 pretrained on CASAS homes only, so homes with cupboard, fridge and flush sensors had nothing like themselves in pretraining. |

Scores are F1 from 0 to 1, higher is better, averaged over 3 time-ordered folds. "Gain" is the pretrained score minus the
same model trained from scratch.

## The headline

**Pretraining helps in 34 of 40 settings** (10 homes × 2 tasks × 2 label amounts), and in **all 10 homes** for
activity recognition with 5 % of the labels.

| Task | Labels | Mean gain, all 10 homes | 6 CASAS homes | 4 paper datasets | Homes where it helps |
|---|---|---|---|---|---|
| Activity recognition | 5 % | **+0.097** | +0.068 | +0.140 | 10 / 10 |
| Activity recognition | 30 % | **+0.099** | +0.027 | +0.208 | 9 / 10 |
| Next-30 prediction | 5 % | +0.038 | +0.055 | +0.014 | 9 / 10 |
| Next-30 prediction | 30 % | +0.009 | +0.027 | −0.017 | 6 / 10 |

Pretraining helps most where labels are scarce and where the home is unlike the CASAS homes. The biggest single gains
are on the paper's small datasets: Kasteren A +0.33 and UCI B +0.30 in activity recognition with 30 % of labels.

## Compared with run 2, on the 7 homes both runs tested

| Task | Labels | Pretrained, run 3 | Pretrained, run 2 | Gain, run 3 | Gain, run 2 |
|---|---|---|---|---|---|
| Activity recognition | 5 % | 0.530 | 0.517 | **+0.087** | +0.076 |
| Activity recognition | 30 % | 0.530 | 0.529 | **+0.066** | +0.059 |
| Next-30 prediction | 5 % | 0.630 | 0.649 | +0.047 | +0.053 |
| Next-30 prediction | 30 % | 0.592 | 0.613 | +0.021 | +0.030 |

- **Activity recognition:** slightly better (+0.013 at 5 % labels), and pretraining's gain is slightly larger. Most
  per-home differences are under 0.03, which is within the spread between folds, so this is "at least as good", not a
  clear jump.
- **Next-30 is about 0.02 lower, but this is not a like-for-like comparison.** Cleaning removes repeated ON/OFF events,
  so "the next 30 events" is a different, slightly harder target than in run 2. The scratch model drops by the same
  amount (0.597 → 0.582), so this is the task changing, not the model getting worse.

## Against the paper

Only these four datasets are shared with the paper. Activity recognition, 5 % / 30 % of labels:

| Dataset | Ours, pretrained | Ours, from scratch | Ours, random folds | Paper |
|---|---|---|---|---|
| UCI B | 0.48 / 0.56 | 0.28 / 0.26 | 0.54 / 0.68 | 0.38 / 0.60 |
| Kasteren A | 0.33 / 0.60 | 0.19 / 0.27 | 0.45 / 0.82 | 0.48 / 0.68 |
| Kasteren C | 0.88 / 0.87 | 0.85 / 0.86 | not run | 0.59 / 0.81 |
| MuRAL | 0.27 / 0.33 | 0.08 / 0.14 | 0.52 / 0.79 | 0.60 / 0.80 |

What this shows:

1. **The paper's main claim holds for activity recognition: pretraining helps, and more with fewer labels.** Our gains
   are as large as the paper's or larger (Kasteren A +0.33 vs the paper's +0.11; MuRAL +0.19 vs +0.04, both at 30 %).
2. **Our lower scores on Kasteren A and MuRAL come from a stricter test, not a weaker model.** We test on a later stretch
   of time than we train on. The paper most likely shuffles windows at random, which puts near-copies of each test
   window (29 of its 30 events) into training. A diagnostic with random folds, reusing the same pretrained models, lifts
   MuRAL from 0.33 to **0.79** (paper 0.80) and Kasteren A from 0.60 to **0.82** (paper 0.68). Our headline numbers stay
   on the stricter, time-ordered test.
3. **Kasteren C looks better than the paper, but it is an easy test.** 83 % of its events are "go to bed", because the
   bed pressure mat fires every few seconds during sleep, so even the untrained model scores 0.85. It says little.
4. **The paper's next-30 claim is not reproduced on these small datasets.** The paper reports +0.15 to +0.25 from
   pretraining. We see about zero on UCI B, Kasteren A and Kasteren C, and +0.05 / −0.04 on MuRAL, even with random
   folds (+0.01 to +0.03). On the larger CASAS homes pretraining does help next-30 (+0.01 to +0.07).

## Caveats

- **One run per setting, 3 folds.** Differences under about 0.03 on one home are within the spread between folds.
- **Random folds are a diagnostic only.** They inflate scores through leakage; they show where the gap to the paper
  comes from, not how good the model is.
- **Unofficial Kasteren copies.** The original download is gone; House A and C come from GitHub copies in the original
  format (no licence stated). House A's ids were named by matching timestamps against a second copy.
- **MuRAL choices are ours.** It records only times of day, so each session is placed on its own day of a dummy
  calendar. Labels are per event and per resident, so with 2–4 people the label of the last event depends on who
  triggered it, which makes the task hard with few labels.
- **Orange4Home is still missing**, so this covers 4 of the paper's 7 datasets.

## What it means

- **The baseline HomeFM must beat is now this run:** the fixed and cleaned DomusFM, pretrained leave-one-dataset-out,
  tested on the 10 homes with time-ordered folds.
- **DomusFM's pretraining helps activity recognition but does little for predicting what comes next.** That is the gap
  HomeFM's first pretraining game (predict the next event and when) is designed for, and the next comparison will test it.
- **Protocol matters as much as the model.** Every future comparison should report time-ordered folds, and random
  folds only when matching a paper.

## Files

| File | What it is |
|---|---|
| [comparison.md](comparison.md) | All tables: per home with fold spread, gains, run 2 comparison, paper comparison, random-fold diagnostic, timing |
| [results.md](results.md), [results.json](results.json) | The run's own tables and full per-fold numbers |
| [cleaning_report.json](cleaning_report.json) | Events removed per dataset by each cleaning rule |
| [train.log](train.log) | Training log: 5 pretraining runs (loss never collapsed, 0.7–0.8), fine-tuning results |
| [random_folds/](random_folds/) | The random-fold diagnostic on UCI B, Kasteren A and MuRAL |
