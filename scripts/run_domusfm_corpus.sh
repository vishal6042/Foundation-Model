#!/usr/bin/env bash
# Corpus-scale DomusFM: (1) paper-style masking, (2) stronger masking + softer temperature.
# Logs: runs/domusfm_corpus.log, runs/domusfm_corpus_strongmask.log
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONWARNINGS=ignore
mkdir -p runs
$PY -u -m homefm.baselines.domusfm.run --overlay configs/domusfm_corpus.yaml > runs/domusfm_corpus.log 2>&1
$PY -u -m homefm.baselines.domusfm.run --overlay configs/domusfm_corpus.yaml --set \
    out_dir=runs/domusfm_corpus_strongmask ablation_no_pretrain=false \
    pretrain.attr_mask_p=0.5 pretrain.event_mask_p=0.5 pretrain.temperature=0.2 \
    > runs/domusfm_corpus_strongmask.log 2>&1
