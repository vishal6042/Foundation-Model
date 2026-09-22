# DomusFM reproduction

Reproduction of *DomusFM: A Foundation Model for Event-Based Behavioral Monitoring in Smart-Homes*
(Fiori, Civitarese, Salim, Bettini; arXiv 2602.01910v2). The authors' code is not public yet, so this is
an independent implementation from the paper. It is the primary baseline for HomeFM (DESIGN.md §4, §12).

Code: `src/homefm/baselines/domusfm/` · Config: `configs/domusfm.yaml`

```bash
.venv/Scripts/python -m homefm.baselines.domusfm.run                                  # full protocol
.venv/Scripts/python -m homefm.baselines.domusfm.run --set "held_out=[milan]" finetune.folds=2   # one dataset
```

## What follows the paper

| Component | Paper | Implementation |
|---|---|---|
| Input | Binary ON/OFF events; continuous sensors → virtual ON/OFF events (§3.1) | `binarize_stream`: threshold at midpoint of 10th/90th percentile, emit on state change |
| Sensor semantics | House item, sensor type, room as text, embedded by all-MiniLM-L6-v2 (§4.1.1, §6.2.1) | `TextTable` + `SentenceTransformerEncoder`, frozen |
| Status | Learned embedding for ON / OFF / MASK (§4.1.2) | `status_emb` (3 entries) |
| Time | Day-of-week and hour: cyclic sin/cos harmonics + projection; seconds-in-hour: learned embedding (§4.1.3) | `Cyclic(7)`, `Cyclic(24)`, `Embedding(3600)` summed into one temporal attribute |
| Attribute fusion | Self-attention across attributes → one event vector (§4.1.4) | Transformer layer over the 5 attribute vectors |
| Context | Transformer encoder, 12 layers, 12 heads, d = 384 (§4.2, §6.2.1) | `nn.TransformerEncoder`, post-norm (Fig. 2b), no positional encoding (§7.3) |
| Windows | 30 events, overlap 29 (§6.2.2) | `DomusWindows(window=30, stride=1)` |
| Pretraining | Phase 1: attribute masking; Phase 2: whole-event masking with event extractor frozen; InfoNCE, in-batch negatives (§4.3) | `pretrain()` |
| Dataset balance | Random oversampling at dataset level (§6.1.2) | `WeightedRandomSampler` with weight 1/len(dataset) |
| Protocol | Leave-one-dataset-out over 7 datasets (§6.1) | `run.py` |
| "Other" class | Kept everywhere (§6.1.1) | Class 0 |
| Fine-tuning | Full fine-tuning, 10 epochs, no early stopping, 5/10/15/30 % labels, 5-fold CV (§6.1.3) | `finetune_and_eval()` |
| ADL head | Linear layer on contextualised embeddings (§6.4.3); weighted F1 | `ADLHead` |
| Next-k head | Two linear layers: event-type distribution + counts; multiset F1; k = 10, 30 (§6.5) | `NextKHead` |
| Ablation | Without pretraining (§6.7.2) | `ablation_no_pretrain: true` |

## Assumptions (not specified in the paper)

| Item | Choice | Why |
|---|---|---|
| Feed-forward width | 2048 | PyTorch default. Gives 28.6M parameters vs 36.1M reported; the gap is unexplained (wider FF, extra projections, or counting differences) |
| Attribute attention | 1 layer, 4 heads, attribute-type embeddings, mean-pooled | Paper gives no sizes; without type embeddings attributes are indistinguishable to attention |
| Temporal attribute | dow + hour + seconds embeddings summed into one vector | Fig. 2a shows a single "Temporal Embeddings" input |
| Harmonics | 4 per cyclic feature | Not reported |
| Window embedding (InfoNCE) | Mean over contextualised events | Not reported |
| Masking probabilities | 0.15 (attribute and event) | "predefined probability", value not reported |
| Temperature | 0.07 | Not reported |
| Pretraining length / batch / LR | 3000 + 3000 steps, batch 128, AdamW 1e-4 | Not reported |
| ADL head input | Last event's contextualised embedding | Task is "activity at the last event" (§6.4.1) |
| Next-k head input and decoding | Mean-pooled window; type present if p > 0.5, count = round(pred) ≥ 1 | Not reported |
| Fine-tuning batch / LR | 64, AdamW 1e-4 | Not reported |
| CV folds | **Time-contiguous** folds | With 29-event overlap, random folds leak test windows into training and inflate scores |
| Overlapping activity labels | Shortest covering episode wins | Datasets differ; single-label task needs one |
| Test set size | Up to 20 000 windows per fold (subsampled evenly) | Speed; set `max_test_windows: null` for all |

## Not implemented yet

- Unsupervised clustering task (HDBSCAN + purity, §6.6) — needs scikit-learn.
- Baselines DeepCASAS (BiLSTM), GPT-2 event model, Chronos (§6.3).
- "w/o Context" ablation (§6.7.1).
- Converters for Kasteren, UCI, Orange4Home, MuRAL (only CASAS and synthetic exist).

## Datasets

| Dataset (paper) | Source | Access | Status |
|---|---|---|---|
| CASAS Milan, Aruba | Zenodo record 15708568, `data.zip` (2.7 GB; only `milan.csv`, `aruba.csv` extracted via HTTP range requests) | CC BY 4.0 | ✅ downloaded — **but see below** |
| UCI ADL Binary, Home B | UCI ML Repository, dataset 271 (32 KB) | Direct | ✅ downloaded, `uci_adl.py` |
| Kasteren A, C | Tim van Kasteren's dataset page (Matlab files) | Direct | ⬜ not downloaded |
| MuRAL | mural.imag.fr | Check terms | ⬜ not downloaded |
| Orange4Home | Orange Labs | **By email request** | ⬜ not requested |
| *(substitute)* CASAS labelled homes hh101–hh130, … | Zenodo record 15708568, `labeled_data.zip` (236 MB, 83 homes) | CC BY 4.0 | ✅ downloaded, `casas_zenodo.py` |

**The 2025 CASAS release differs from the version the paper used:**

- Milan and Aruba have **no activity labels** (4 columns only) and the original per-sensor ids (M001…, D001…, T001…) are **merged into location names** (Kitchen, Bedroom, LoungeChair…): Milan has 9 sensor names, Aruba 10. The original annotated versions are no longer on the CASAS site.
- Labelled data exists only for newer homes (2011+), with compositional names like `KitchenAStove`, `BathroomBToilet`. `casas_zenodo.py` parses these into (item, room) automatically, so no hand-written sensor maps are needed. They have 5–13 sensor names and 28–38 activity classes.

**Consequence for the protocol** (`configs/domusfm_real.yaml`): 7 held-out targets like the paper — UCI Home B plus six labelled CASAS homes (hh101, hh103, hh105, hh110, hh119, hh122) standing in for Kasteren A/C, Orange4Home and MuRAL. Milan and Aruba are used for pretraining only. Numbers are therefore **not directly comparable** to the paper's tables, except UCI B.

## Results

### Smoke run: synthetic homes, hashing text encoder (not faithful)

Held-out `sim_000`; 200 + 200 pretraining steps; 2 folds; 2 epochs. Only proves the pipeline runs.

| Task | 5 % DomusFM | 5 % w/o Pretrain | 30 % DomusFM | 30 % w/o Pretrain |
|---|---|---|---|---|
| ADL (weighted F1) | 0.56 | 0.53 | 0.68 | 0.66 |
| Next-10 (multiset F1) | 0.32 | 0.29 | 0.53 | 0.53 |
| Next-30 (multiset F1) | 0.28 | 0.22 | 0.57 | 0.53 |

Observation: the contrastive loss drops from 2.6 to ~0.07 within 100 steps, so with 15 % masking the two views are nearly identical and the task becomes easy. This matches the concern in DESIGN.md §8.1 (#1, #3) and is worth checking on real data.

### Real datasets: run 1 (UCI Home B)

Setup: MiniLM text encoder; pretraining on the other 6 labelled homes + Milan + Aruba (3000 + 3000 steps);
3 time-contiguous folds; 10 epochs; batch 64; LR 1e-4. Cells are **DomusFM / w/o Pretrain**; paper values
(5-fold) in brackets.

| Task | 5 % | 10 % | 15 % | 30 % |
|---|---|---|---|---|
| ADL, weighted F1 | 0.25 / 0.23 [0.38 / 0.30] | 0.28 / 0.28 [0.46 / 0.32] | 0.27 / 0.28 [0.51 / 0.33] | 0.29 / 0.28 [0.60 / 0.36] |
| Next-10, multiset F1 | 0.58 / 0.60 [0.62 / —] | 0.61 / 0.63 [0.68 / —] | 0.61 / 0.61 [0.70 / —] | 0.60 / 0.62 [0.76 / —] |
| Next-30, multiset F1 | 0.70 / 0.69 [0.76 / 0.53] | 0.71 / 0.71 [0.82 / 0.56] | 0.71 / 0.72 [0.86 / 0.58] | 0.71 / 0.71 [0.90 / 0.65] |

Majority-class weighted F1 on UCI B is 0.19.

**Finding 1: the fold protocol explains part of the gap.** Same model without pretraining, 30 % labels, 3 folds:

| CV folds | ADL weighted F1 | Next-30 multiset F1 |
|---|---|---|
| Time-contiguous (ours) | 0.280 | 0.718 |
| Random | 0.395 | 0.781 |

Consecutive windows share 29 of 30 events, so random folds put near-duplicates of test windows in the training set. With random folds our *non-pretrained* ADL score (0.40) matches the paper's non-pretrained score (0.36). We keep contiguous folds as the honest protocol (`finetune.fold_mode: contiguous`); `random` is available for like-for-like comparison with the paper.

**Finding 2: pretraining gives no benefit here, unlike the paper** (+0.22 ADL and +0.25 next-30 at 30 % in the paper's ablation). The contrastive loss falls to ~0.002 within 500 steps in both phases, i.e. the model separates each window from its 15 %-masked copy almost trivially, so pretraining carries little signal. Possible causes, in order of likelihood:

1. Masking too weak / temperature too low (both unreported) — the "easy positives" problem in DESIGN.md §8.1.
2. Pretraining corpus differs: room-level merged sensors in the 2025 CASAS release, and 5–13 sensor names per home.
3. Unreported fine-tuning hyperparameters. A sweep on UCI B (LR 1e-4/3e-4, batch 16/64, 10/30 epochs) did not raise ADL above 0.27, so this is not the main cause.

### Real datasets: run 2 (corpus-scale, in progress)

Pretraining once on **77 homes / 27.3M events** (all labelled CASAS homes except the 7 targets, plus Milan and
Aruba), then fine-tuning on the 7 targets (ADL and next-30; 5 % and 30 %; 3 contiguous folds).

| Run | Config | Log |
|---|---|---|
| DomusFM, paper masking (0.15, τ = 0.07) | `configs/domusfm_corpus.yaml` | `runs/domusfm_corpus.log` |
| DomusFM, strong masking (0.5, τ = 0.2) | same + `--set` overrides in `scripts/run_domusfm_corpus.sh` | `runs/domusfm_corpus_strongmask.log` |
| HomeFM objective E (8.0M params), same protocol | `configs/homefm_corpus.yaml` | `runs/homefm_corpus.log` |

Fast loading: `casas_fast.py` parses a home into numpy arrays in ~0.1 s and caches it as `.npz`
(identical output to `casas_zenodo.py`, tested). `BalancedSampler` replaces `WeightedRandomSampler`,
which cannot sample from more than 2^24 windows.

Fine-tuning hyperparameter sweep (w/o pretrain, 30 %, contiguous folds): LR 1e-4 bs 64 ep 10 → 0.27; LR 1e-4 bs 16 → 0.23; LR 3e-4 bs 16 → 0.20; LR 1e-4 bs 16 ep 30 → 0.25.
