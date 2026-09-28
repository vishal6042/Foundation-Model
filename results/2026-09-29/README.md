# 2026-09-29

## HomeFM design deck (`homefm_slides/`)

A 56-slide deck on HomeFM: goals and use cases, DomusFM as the baseline and its 11 limitations with HomeFM's remedy for each, the system and model architecture, pretraining (why DomusFM's attribute and event masking are dropped, which use cases they would struggle with, and HomeFM's three games), the path from model outputs to answers, and the data, evaluation, deployment and roadmap plan. All content comes from [docs/DESIGN.md](../../docs/DESIGN.md), [docs/LIMITATIONS_AND_REMEDIES.md](../../docs/LIMITATIONS_AND_REMEDIES.md) and [docs/DOMUSFM_REPRODUCTION.md](../../docs/DOMUSFM_REPRODUCTION.md).

| Path | What it is |
|---|---|
| `homefm_slides/project/deck.json` | Deck index: title, slide order, the 7 sections, fonts |
| `homefm_slides/project/slides/*.html` | One file per slide (1920×1080, inline styles, the Claude Slides artifact format) |
| `homefm_slides/build.py` | Generator that writes both from the content in the script: `python results/2026-09-29/homefm_slides/build.py` |

The presentable version is a private Claude Slides artifact (https://claude.ai/artifact/CPBCKTGsYnNGiCrKMBGBQ5); share it from its Share menu, or export PDF / PowerPoint from there. The HTML files here are its source and use the artifact's elements (`x-shape`, `x-icon`), so they are not meant to be opened directly in a browser.

## Also in this commit (elsewhere in the repo)

| Change | Where |
|---|---|
| Converters for the paper's Kasteren A/C and MuRAL test sets | `src/homefm/data/converters/kasteren.py`, `mural.py` |
| Data cleaning before segmentation (paper Appendix A: repeated states removed; plus EDA fixes), opt-in | `src/homefm/baselines/domusfm/clean.py`, `datasets.clean` |
| Leave-one-dataset-out pretraining groups (`pretrain_mode: groups`) | `src/homefm/baselines/domusfm/run.py` |
| DomusFM run 3 config: cleaned data, 10 targets, paper-style pretraining pools | `configs/domusfm_corpus_clean.yaml` |
| EDA of all 87 datasets, with charts and tables | `notebooks/domusfm_eda.ipynb` (and `.py`), `notebooks/figures/`, `notebooks/tables/` |
| Docs | `docs/DATA.md` (§1, §2.1, §2.5, §3.6), `docs/DOMUSFM_REPRODUCTION.md` (run 3) |

DomusFM run 3 (`configs/domusfm_corpus_clean.yaml`) was still training when this was committed; its results are not included yet. First finished target, UCI Home B: activity F1 at 5 % labels 0.482 pretrained vs 0.280 without (run 2: 0.452 vs 0.252).
