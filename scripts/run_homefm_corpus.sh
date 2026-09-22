#!/usr/bin/env bash
# HomeFM objective E on the CASAS corpus with the DomusFM evaluation protocol. Log: runs/homefm_corpus.log
set -e
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONWARNINGS=ignore
mkdir -p runs
.venv/Scripts/python -u -m homefm.experiments.corpus_compare "$@" > runs/homefm_corpus.log 2>&1
