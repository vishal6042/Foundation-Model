"""DomusFM (Fiori et al., arXiv 2602.01910) reproduction. See docs/DOMUSFM_REPRODUCTION.md."""

from .data import DomusDataset, DomusWindows, TextTable, binarize_stream, build_domus_dataset, build_domus_from_arrays
from .model import ADLHead, DomusConfig, DomusFM, NextKHead
