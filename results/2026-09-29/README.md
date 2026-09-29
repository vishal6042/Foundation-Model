# 2026-09-29

## DomusFM run 3 results (`domusfm_corpus_clean/`)

Start with [domusfm_corpus_clean/SUMMARY.md](domusfm_corpus_clean/SUMMARY.md). This run used cleaned data, added the paper's own Kasteren A, Kasteren C and MuRAL test sets (10 test homes in total), and pretrained leave-one-dataset-out. **Pretraining helps in 34 of 40 settings**, and in all 10 homes for activity recognition with 5 % of the labels. The pretraining gains on the paper's datasets match or beat the paper's. Our absolute scores are lower because our test is stricter (time-ordered folds); a random-fold diagnostic brings MuRAL and Kasteren A level with or above the paper.

| Path | What it is |
|---|---|
| `domusfm_corpus_clean/SUMMARY.md` | Plain-language summary: what changed, headline, run 2 and paper comparisons, caveats |
| `domusfm_corpus_clean/comparison.md` | All tables: per home with fold spread, gains, run 2, paper, random-fold diagnostic, timing |
| `domusfm_corpus_clean/results.json`, `results.md` | The run's raw results and tables |
| `domusfm_corpus_clean/cleaning_report.json` | Events removed per dataset by each cleaning rule |
| `domusfm_corpus_clean/train.log` | Training log |
| `domusfm_corpus_clean/random_folds/` | Random-fold diagnostic on UCI B, Kasteren A and MuRAL (same pretrained backbones) |

## HomeFM design deck (`homefm_slides/`)

A 65-slide deck on HomeFM: goals and use cases, DomusFM as the baseline (slides 11–13 in plain words: why run 1 failed and our six fixes, the matching and fill-in-the-blanks training losses as charts, how we differ from the paper; slides 14–19 on run 3: headline, the five pretrained models, which model helped most and why, per-home gains, the paper comparison and the overall findings), its 11 limitations with HomeFM's remedy for each, the system and model architecture, pretraining (why DomusFM's attribute and event masking are dropped, which use cases they would struggle with, and HomeFM's three games), the path from model outputs to answers, and the data, evaluation, deployment and roadmap plan. Content comes from [docs/DESIGN.md](../../docs/DESIGN.md), [docs/LIMITATIONS_AND_REMEDIES.md](../../docs/LIMITATIONS_AND_REMEDIES.md), [docs/DOMUSFM_REPRODUCTION.md](../../docs/DOMUSFM_REPRODUCTION.md) and the run 3 results above.

| Path | What it is |
|---|---|
| `homefm_slides/project/deck.json` | Deck index: title, slide order, the 7 sections, fonts |
| `homefm_slides/project/slides/*.html` | One file per slide (1920×1080, inline styles, the Claude Slides artifact format) |
| `homefm_slides/build.py` | Generator that writes both; the run 3 slides read `domusfm_corpus_clean/results.json`: `python results/2026-09-29/homefm_slides/build.py` |
| `homefm_slides/HomeFM_architecture_and_design.pptx` | The same 65 slides as a PowerPoint file, with editable text and shapes, native charts and the speaker notes. Opens offline in PowerPoint, Keynote, Google Slides or LibreOffice |
| `homefm_slides/build_pptx.js` | Regenerates the PPTX from `project/` (lays each slide out in headless Chromium, then writes native shapes): `npm install pptxgenjs playwright`, then `node results/2026-09-29/homefm_slides/build_pptx.js`. Run it after `build.py` |

To present from a local copy of the repo, open `homefm_slides/HomeFM_architecture_and_design.pptx`. The online version is a private Claude Slides artifact (https://claude.ai/artifact/CPBCKTGsYnNGiCrKMBGBQ5); share it from its Share menu, or export PDF / PowerPoint from there. The HTML files here are its source and use the artifact's elements (`x-shape`, `x-icon`), so they are not meant to be opened directly in a browser.

## Code, notebook and docs (committed earlier today, elsewhere in the repo)

| Change | Where |
|---|---|
| Converters for the paper's Kasteren A/C and MuRAL test sets | `src/homefm/data/converters/kasteren.py`, `mural.py` |
| Data cleaning before segmentation (paper Appendix A: repeated states removed; plus EDA fixes), opt-in | `src/homefm/baselines/domusfm/clean.py`, `datasets.clean` |
| Leave-one-dataset-out pretraining groups (`pretrain_mode: groups`) | `src/homefm/baselines/domusfm/run.py` |
| DomusFM run 3 config | `configs/domusfm_corpus_clean.yaml` |
| EDA of all 87 datasets, with charts and tables | `notebooks/domusfm_eda.ipynb` (and `.py`), `notebooks/figures/`, `notebooks/tables/` |
| Docs | `docs/DATA.md` (§1, §2.1, §2.5, §3.6), `docs/DOMUSFM_REPRODUCTION.md` (run 3 setup and results) |
