### Activity recognition (weighted F1)

Mean ± std over 3 time-contiguous folds. **Bold** = the better of pretrained / no pretraining.

| Home | 5 % pretrained | 5 % no pretraining | Δ | 30 % pretrained | 30 % no pretraining | Δ |
|---|---|---|---|---|---|---|
| uci_b | **0.482 ± 0.012** | 0.280 ± 0.044 | +0.202 | **0.561 ± 0.022** | 0.263 ± 0.052 | +0.298 |
| kasteren_a | **0.327 ± 0.020** | 0.187 ± 0.014 | +0.140 | **0.603 ± 0.014** | 0.272 ± 0.021 | +0.331 |
| kasteren_c | **0.881 ± 0.014** | 0.849 ± 0.013 | +0.032 | **0.872 ± 0.011** | 0.860 ± 0.014 | +0.012 |
| mural | **0.272 ± 0.021** | 0.084 ± 0.031 | +0.187 | **0.327 ± 0.024** | 0.137 ± 0.006 | +0.189 |
| hh101 | **0.591 ± 0.025** | 0.577 ± 0.008 | +0.013 | 0.570 ± 0.012 | **0.571 ± 0.009** | -0.001 |
| hh103 | **0.754 ± 0.011** | 0.666 ± 0.012 | +0.087 | **0.758 ± 0.022** | 0.715 ± 0.006 | +0.042 |
| hh105 | **0.461 ± 0.037** | 0.407 ± 0.021 | +0.053 | **0.440 ± 0.013** | 0.426 ± 0.035 | +0.013 |
| hh110 | **0.449 ± 0.011** | 0.313 ± 0.023 | +0.136 | **0.421 ± 0.009** | 0.370 ± 0.006 | +0.051 |
| hh119 | **0.465 ± 0.064** | 0.409 ± 0.052 | +0.056 | **0.455 ± 0.050** | 0.441 ± 0.050 | +0.014 |
| hh122 | **0.509 ± 0.018** | 0.448 ± 0.021 | +0.060 | **0.507 ± 0.003** | 0.463 ± 0.015 | +0.044 |

### Next-30 prediction (multiset F1)

Mean ± std over 3 time-contiguous folds. **Bold** = the better of pretrained / no pretraining.

| Home | 5 % pretrained | 5 % no pretraining | Δ | 30 % pretrained | 30 % no pretraining | Δ |
|---|---|---|---|---|---|---|
| uci_b | **0.705 ± 0.015** | 0.703 ± 0.011 | +0.002 | 0.701 ± 0.005 | **0.717 ± 0.010** | -0.016 |
| kasteren_a | 0.526 ± 0.010 | **0.531 ± 0.024** | -0.005 | 0.533 ± 0.014 | **0.544 ± 0.025** | -0.011 |
| kasteren_c | **0.859 ± 0.048** | 0.857 ± 0.023 | +0.002 | 0.847 ± 0.029 | **0.850 ± 0.032** | -0.003 |
| mural | **0.615 ± 0.016** | 0.560 ± 0.032 | +0.055 | 0.567 ± 0.009 | **0.604 ± 0.031** | -0.037 |
| hh101 | **0.684 ± 0.013** | 0.619 ± 0.018 | +0.066 | **0.626 ± 0.019** | 0.605 ± 0.014 | +0.021 |
| hh103 | **0.672 ± 0.012** | 0.620 ± 0.014 | +0.052 | **0.686 ± 0.009** | 0.615 ± 0.004 | +0.071 |
| hh105 | **0.547 ± 0.005** | 0.512 ± 0.002 | +0.035 | **0.496 ± 0.008** | 0.475 ± 0.010 | +0.020 |
| hh110 | **0.615 ± 0.012** | 0.550 ± 0.005 | +0.065 | **0.552 ± 0.021** | 0.529 ± 0.011 | +0.023 |
| hh119 | **0.562 ± 0.011** | 0.509 ± 0.016 | +0.054 | **0.512 ± 0.013** | 0.495 ± 0.009 | +0.017 |
| hh122 | **0.621 ± 0.007** | 0.564 ± 0.011 | +0.057 | **0.569 ± 0.002** | 0.559 ± 0.008 | +0.010 |

### Mean gain from pretraining (pretrained − no pretraining)

| Setting | All 10 | 6 CASAS test homes | 4 paper datasets | Homes where pretraining wins | Run 2 (7 homes) |
|---|---|---|---|---|---|
| adl 5% | +0.097 | +0.068 | +0.140 | 10 / 10 | +0.076 |
| adl 30% | +0.099 | +0.027 | +0.208 | 9 / 10 | +0.059 |
| next30 5% | +0.038 | +0.055 | +0.014 | 9 / 10 | +0.053 |
| next30 30% | +0.009 | +0.027 | -0.017 | 6 / 10 | +0.030 |

Pretraining wins 34 of 40 settings.

### The 7 homes shared with run 2 (UCI B + 6 CASAS test homes)

Mean over the 7 homes. Run 2 = `results/domusfm_corpus_fixed` (same model, game and fine-tuning; no cleaning, CASAS-only pretraining).

| Setting | Run 3 pretrained | Run 2 pretrained | Change | Run 3 no pretraining | Run 2 no pretraining | Gain run 3 | Gain run 2 |
|---|---|---|---|---|---|---|---|
| adl 5% | 0.530 | 0.517 | +0.013 | 0.443 | 0.441 | +0.087 | +0.076 |
| adl 30% | 0.530 | 0.529 | +0.001 | 0.464 | 0.470 | +0.066 | +0.059 |
| next30 5% | 0.630 | 0.649 | -0.020 | 0.582 | 0.597 | +0.047 | +0.053 |
| next30 30% | 0.592 | 0.613 | -0.021 | 0.571 | 0.583 | +0.021 | +0.030 |

Per home, pretrained, run 3 − run 2:

| Home | adl 5% | adl 30% | next30 5% | next30 30% |
|---|---|---|---|---|
| uci_b | +0.030 | -0.007 | -0.001 | +0.005 |
| hh101 | +0.009 | +0.015 | -0.029 | -0.023 |
| hh103 | +0.010 | -0.001 | +0.002 | +0.005 |
| hh105 | +0.008 | -0.003 | -0.034 | -0.035 |
| hh110 | +0.016 | -0.006 | -0.035 | -0.055 |
| hh119 | -0.006 | -0.008 | -0.011 | -0.015 |
| hh122 | +0.024 | +0.015 | -0.032 | -0.032 |

### Against the paper (the 4 datasets we share with it)

Paper values: Table 1 (activity) and Table 2 (next-30) for DomusFM; Tables 8–9 for the pretraining ablation. The paper uses 5 folds that are most likely random (overlapping windows leak between train and test); we use 3 time-contiguous folds.

| Dataset | Task | Labels | Ours pretrained | Paper pretrained | Ours no pretraining | Paper no pretraining | Our gain | Paper gain |
|---|---|---|---|---|---|---|---|---|
| uci_b | adl | 5% | 0.48 | 0.38 | 0.28 | 0.30 | +0.20 | +0.08 |
| uci_b | adl | 30% | 0.56 | 0.60 | 0.26 | 0.36 | +0.30 | +0.24 |
| uci_b | next30 | 5% | 0.70 | 0.76 | 0.70 | 0.53 | +0.00 | +0.23 |
| uci_b | next30 | 30% | 0.70 | 0.90 | 0.72 | 0.65 | -0.02 | +0.25 |
| kasteren_a | adl | 5% | 0.33 | 0.48 | 0.19 | 0.44 | +0.14 | +0.04 |
| kasteren_a | adl | 30% | 0.60 | 0.68 | 0.27 | 0.57 | +0.33 | +0.11 |
| kasteren_a | next30 | 5% | 0.53 | 0.66 | 0.53 | 0.51 | -0.01 | +0.15 |
| kasteren_a | next30 | 30% | 0.53 | 0.87 | 0.54 | 0.67 | -0.01 | +0.20 |
| kasteren_c | adl | 5% | 0.88 | 0.59 | 0.85 | 0.44 | +0.03 | +0.15 |
| kasteren_c | adl | 30% | 0.87 | 0.81 | 0.86 | 0.75 | +0.01 | +0.06 |
| kasteren_c | next30 | 5% | 0.86 | 0.59 | 0.86 | 0.41 | +0.00 | +0.18 |
| kasteren_c | next30 | 30% | 0.85 | 0.86 | 0.85 | 0.66 | -0.00 | +0.20 |
| mural | adl | 5% | 0.27 | 0.60 | 0.08 | 0.41 | +0.19 | +0.19 |
| mural | adl | 30% | 0.33 | 0.80 | 0.14 | 0.76 | +0.19 | +0.04 |
| mural | next30 | 5% | 0.61 | 0.69 | 0.56 | 0.50 | +0.05 | +0.19 |
| mural | next30 | 30% | 0.57 | 0.84 | 0.60 | 0.60 | -0.04 | +0.24 |

### Timing

Minutes per target (first target of each pretraining group includes its ~68-min pretraining; the first run was slowed by another GPU program): uci_b 90, kasteren_a 67, kasteren_c 80, mural 68, hh101 128, hh103 39, hh105 41, hh110 28, hh119 28, hh122 38. Total 10.1 h.

### Diagnostic: random folds (likely the paper's protocol)

Same pretrained backbones as the main run (reused, not retrained), same fine-tuning; only `finetune.fold_mode: random`. Random folds let near-copies of test windows (29 of 30 events shared) into the training set, so they are an upper bound, not an honest score.

| Dataset | Task | Labels | Time-ordered: pretrained / none | Random: pretrained / none | Paper: pretrained |
|---|---|---|---|---|---|
| uci_b | adl | 5% | 0.48 / 0.28 | 0.54 / 0.30 | 0.38 |
| uci_b | adl | 30% | 0.56 / 0.26 | 0.68 / 0.43 | 0.60 |
| uci_b | next30 | 5% | 0.70 / 0.70 | 0.73 / 0.72 | 0.76 |
| uci_b | next30 | 30% | 0.70 / 0.72 | 0.82 / 0.79 | 0.90 |
| kasteren_a | adl | 5% | 0.33 / 0.19 | 0.45 / 0.28 | 0.48 |
| kasteren_a | adl | 30% | 0.60 / 0.27 | 0.82 / 0.52 | 0.68 |
| kasteren_a | next30 | 5% | 0.53 / 0.53 | 0.55 / 0.54 | 0.66 |
| kasteren_a | next30 | 30% | 0.53 / 0.54 | 0.70 / 0.68 | 0.87 |
| mural | adl | 5% | 0.27 / 0.08 | 0.52 / 0.29 | 0.60 |
| mural | adl | 30% | 0.33 / 0.14 | 0.79 / 0.56 | 0.80 |
| mural | next30 | 5% | 0.61 / 0.56 | 0.65 / 0.62 | 0.69 |
| mural | next30 | 30% | 0.57 / 0.60 | 0.73 / 0.71 | 0.84 |
