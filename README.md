# HomeFM

A foundation model for smart-home event streams: activity, security, pets, childcare, elderly care,
device health, energy and anomaly questions answered from one open-vocabulary model.

**Start with the design:** [docs/DESIGN.md](docs/DESIGN.md), which covers requirements, DomusFM gaps with
scenarios, architecture diagrams, the pretraining recipe, evaluation and the roadmap.

## Setup

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -e ".[dev]"      # add ,text for sentence-transformers
```

## Common commands

```bash
.venv/Scripts/python -m pytest -q                                              # tests
.venv/Scripts/python -m homefm.train.pretrain --objective configs/objectives/E_next_event_latent.yaml
.venv/Scripts/python scripts/run_ablation.py --set train.steps=2000           # A–F objective ablation
```

Runs write `runs/<name>/model.pt` and `results.json`; the ablation writes `runs/ablation.json`.

## Layout

| Path | What |
|---|---|
| `src/homefm/schema` | Home Token, Entity, Episode, ontology + capability registry |
| `src/homefm/data` | synthetic simulator, CASAS converter, text encoders, time-based windowing |
| `src/homefm/model` | event embedder → Perceiver moment encoder → stream model → event read-out, heads |
| `src/homefm/objectives` | masking strategies and objectives A–F (DESIGN.md §8.3) |
| `src/homefm/eval` | metrics, linear probe, episode counting, anomaly AUROC |
| `configs/` | `base.yaml` + one file per objective variant |

## Status

Implemented: data schema, synthetic homes with injected faults, CASAS converter, HomeFM backbone
(moments, causal/bidirectional stream, event read-out), objectives A–F, linear-probe / counting /
anomaly evaluation, ablation runner.

Not yet: other dataset converters (see `src/homefm/data/converters/README.md`), long-horizon daily
memory, entity-axis device-health head, occupant slots, cross-modal and cross-scale alignment,
inverse-frequency loss weighting, perception experts, stores and query agent.
