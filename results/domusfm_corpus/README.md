# DomusFM, paper masking, 77-home pretraining (run finished 2026-09-24)

Raw outputs of `python -m homefm.baselines.domusfm.run --overlay configs/domusfm_corpus.yaml`, copied from the git-ignored `runs/` folder.

| File | Contents |
|---|---|
| `results.json` | Every result per held-out home, task, label fraction and fold, plus the pretraining loss history |
| `results.md` | The summary tables the script generates |
| `train.log` | The full console log of the run (converted to UTF-8) |

Not included: the pretrained model (`runs/domusfm_corpus/pretrained_shared.pt`, 115 MB), which can be recreated by rerunning the command above.

Summary and findings: [docs/DOMUSFM_REPRODUCTION.md](../../docs/DOMUSFM_REPRODUCTION.md) (run 2, paper masking).
