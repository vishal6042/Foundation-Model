# DomusFM with the collapse fixes, 77-home pretraining (run finished 2026-09-25)

`python -m homefm.baselines.domusfm.run --overlay configs/domusfm_corpus_fixed.yaml`, launched with
`scripts/domusfm_train.py start` on one RTX 4090. Same data, held-out homes, model (28.6M params) and fine-tuning
protocol as the paper-masking run in [`results/domusfm_corpus`](../domusfm_corpus/results.md); only the pretraining
game and the fine-tuning learning rates change (the six fixes in
[docs/DOMUSFM_REPRODUCTION.md](../../docs/DOMUSFM_REPRODUCTION.md), "Why pretraining did not help, and the fix").

## Summary

Plain-language version (what went wrong, the fixes, the improvement): [SUMMARY.md](SUMMARY.md).


**The fixes work: pretraining now helps.** With the paper's game, the pretrained model lost to the same model
trained from random weights on almost every home. With the fixes, it wins 26 of 28 settings (7 homes × 2 tasks ×
2 label fractions), and the two losses are 0.01.

Mean over the 7 held-out homes, pretrained minus no pretraining (positive = pretraining helps):

| Setting | This run (fixed) | Last run (paper masking) | Homes where pretraining wins |
|---|---|---|---|
| ADL, 5 % labels | **+0.076** | −0.050 | 7 / 7 (last run 1 / 7) |
| ADL, 30 % labels | **+0.059** | −0.068 | 6 / 7 (last run 1 / 7) |
| Next-30, 5 % labels | **+0.053** | −0.009 | 7 / 7 (last run 1 / 7) |
| Next-30, 30 % labels | **+0.030** | +0.002 | 6 / 7 (last run 3 / 7) |

Absolute scores (mean over the 7 homes):

| Setting | Pretrained, fixed | Pretrained, paper | No pretraining, fixed | No pretraining, paper |
|---|---|---|---|---|
| ADL 5 % | **0.517** | 0.385 | 0.441 | 0.435 |
| ADL 30 % | **0.529** | 0.400 | 0.470 | 0.468 |
| Next-30 5 % | **0.649** | 0.587 | 0.597 | 0.596 |
| Next-30 30 % | **0.613** | 0.591 | 0.583 | 0.590 |

The no-pretraining baseline barely moved between runs (fix 6, the split learning rate with warmup, is applied to
it too), while the pretrained model gained 0.13 on ADL. So the improvement comes from the pretraining itself, not
from the fine-tuning changes.

## Findings

1. **Pretraining no longer collapses.** The contrastive loss stayed at about 0.70–0.82 for all 40,000 steps
   (chance = ln 128 = 4.85). The paper-masking run fell to 0.0005 by step 5,000. The masked-attribute loss (`mlm`)
   kept falling: 3.60 → 0.16 in phase 1, 2.28 → 0.58 in phase 2.
2. **Pretraining helps activity recognition** on every home with 5 % labels. The gain is largest on UCI B
   (+0.20 / +0.28), the smallest and most different home. Without UCI B the mean gain is +0.055 (5 %) and +0.022
   (30 %).
3. **Pretraining now helps next-30 prediction too:** +0.053 with 5 % labels, +0.030 with 30 %. Last run it made no
   difference. Without UCI B: +0.061 and +0.037.
4. **The gain shrinks as labels grow** (ADL +0.076 → +0.059, Next-30 +0.053 → +0.030). This is the expected shape
   when pretraining is useful: it matters most when labelled data is scarce.
5. **The two losses are within noise:** hh101 ADL 30 % (0.555 vs 0.567) and UCI B Next-30 30 % (0.695 vs 0.708).
   Some small wins are also within the fold spread, e.g. hh119 ADL 30 % (+0.02, std 0.065 / 0.037). Differences
   under about 0.03 on one home should not be read as real.

## Results per home

Mean ± std over 3 contiguous folds. **Bold** = the better of pretrained / no pretraining.

### ADL (weighted F1)

| Home | 5 % pretrained | 5 % no PT | diff | 30 % pretrained | 30 % no PT | diff |
|---|---|---|---|---|---|---|
| uci_b | **0.452 ± 0.021** | 0.252 ± 0.041 | +0.200 | **0.568 ± 0.030** | 0.285 ± 0.036 | +0.283 |
| hh101 | **0.581 ± 0.020** | 0.550 ± 0.023 | +0.031 | 0.555 ± 0.025 | **0.567 ± 0.001** | −0.012 |
| hh103 | **0.744 ± 0.023** | 0.683 ± 0.006 | +0.061 | **0.758 ± 0.020** | 0.729 ± 0.013 | +0.029 |
| hh105 | **0.453 ± 0.027** | 0.397 ± 0.021 | +0.056 | **0.443 ± 0.032** | 0.433 ± 0.028 | +0.010 |
| hh110 | **0.433 ± 0.006** | 0.341 ± 0.027 | +0.092 | **0.427 ± 0.009** | 0.369 ± 0.027 | +0.058 |
| hh119 | **0.472 ± 0.047** | 0.411 ± 0.048 | +0.061 | **0.463 ± 0.065** | 0.441 ± 0.037 | +0.022 |
| hh122 | **0.484 ± 0.009** | 0.453 ± 0.024 | +0.031 | **0.491 ± 0.020** | 0.465 ± 0.013 | +0.026 |
| **Mean** | **0.517** | 0.441 | **+0.076** | **0.529** | 0.470 | **+0.059** |

### Next-30 (multiset F1)

| Home | 5 % pretrained | 5 % no PT | diff | 30 % pretrained | 30 % no PT | diff |
|---|---|---|---|---|---|---|
| uci_b | **0.705 ± 0.022** | 0.703 ± 0.008 | +0.002 | 0.695 ± 0.002 | **0.708 ± 0.012** | −0.013 |
| hh101 | **0.714 ± 0.015** | 0.635 ± 0.014 | +0.079 | **0.649 ± 0.016** | 0.616 ± 0.021 | +0.033 |
| hh103 | **0.670 ± 0.005** | 0.610 ± 0.009 | +0.060 | **0.680 ± 0.007** | 0.621 ± 0.003 | +0.059 |
| hh105 | **0.581 ± 0.005** | 0.534 ± 0.003 | +0.047 | **0.530 ± 0.003** | 0.498 ± 0.004 | +0.032 |
| hh110 | **0.650 ± 0.015** | 0.576 ± 0.010 | +0.074 | **0.607 ± 0.019** | 0.554 ± 0.016 | +0.053 |
| hh119 | **0.573 ± 0.026** | 0.509 ± 0.022 | +0.064 | **0.527 ± 0.016** | 0.499 ± 0.022 | +0.028 |
| hh122 | **0.653 ± 0.003** | 0.608 ± 0.007 | +0.045 | **0.600 ± 0.008** | 0.585 ± 0.006 | +0.015 |
| **Mean** | **0.649** | 0.597 | **+0.053** | **0.613** | 0.583 | **+0.030** |

## Pretraining

77 homes, 27.3M events, 20,000 steps per phase, batch 128 (4 homes × 32 windows).

| Phase | Step | Contrastive | MLM | z_std |
|---|---|---|---|---|
| 1 (attribute) | 0 | 4.111 | 3.601 | 0.039 |
| | 1,000 | 0.859 | 0.394 | 0.088 |
| | 5,000 | 0.736 | 0.181 | 0.088 |
| | 10,000 | 0.723 | 0.200 | 0.088 |
| | 19,000 | 0.698 | 0.162 | 0.088 |
| 2 (event) | 0 | 1.666 | 2.280 | 0.088 |
| | 1,000 | 0.849 | 0.621 | 0.088 |
| | 5,000 | 0.803 | 0.609 | 0.088 |
| | 10,000 | 0.784 | 0.507 | 0.088 |
| | 19,000 | 0.817 | 0.577 | 0.088 |

For comparison, the paper-masking run: contrastive 2.35 at step 0, about 0.0005 by step 5,000, in both phases.

z_std (the spread of the embeddings) rose from 0.039 to 0.088 by step 1,000 and stayed exactly there for the rest
of the run. It is far from 0, so the embeddings did not collapse. The flat value is worth a look before relying on
z_std as a health signal.

## Time

| Stage | Time |
|---|---|
| Loading 77 homes | about 1 min |
| Pretraining, phase 1 | 70 min (a game shared the GPU from about step 8,000: 3.5 steps/s instead of 9) |
| Pretraining, phase 2 | 47 min (about 4.6 steps/s with the game, 10 steps/s without it) |
| Fine-tuning + test, 56 settings | 4 h 36 min (uci_b 2 min, hh101 77 min, the others 32–49 min each) |
| **Total** | **about 6 h 35 min** (23:36 → 06:10) |

On an idle GPU, pretraining takes about 70 minutes.

## Files

| File | Contents |
|---|---|
| `SUMMARY.md` | Plain-language summary: the problem, the fixes, the improvement |
| `results.md` | This write-up |
| `results.json` | Every result per held-out home, task, label fraction and fold, plus the pretraining loss history |
| `train.log` | The full console log of the run (UTF-8) |

Not included: the pretrained model (`runs/domusfm_corpus_fixed/pretrained_shared.pt`, 115 MB), which can be
recreated by rerunning the command at the top.

## Next

- The strong-masking DomusFM run and the HomeFM runs (`configs/homefm_corpus.yaml` at 8.0M params,
  `configs/homefm_corpus_384.yaml` size-matched at 28.6M), on this same protocol, to compare against these numbers.
- An ablation of the six fixes, to see which ones matter.
