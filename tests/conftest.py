import pytest
import torch

from homefm.data import HashingTextEncoder, HomeFMData, WindowConfig, WindowDataset, simulate_homes
from homefm.model import HomeFM, ModelConfig
from homefm.schema.ontology import DEFAULT_CONCEPTS


@pytest.fixture(scope="session")
def data():
    streams = simulate_homes(3, days=3, seed=1)
    return HomeFMData(streams, HashingTextEncoder(64), list(DEFAULT_CONCEPTS),
                      WindowConfig(span_s=1800, moment_s=60, stride_s=1800, max_events=512))


@pytest.fixture(scope="session")
def batch(data):
    ds = WindowDataset(data)
    items = [ds[i] for i in range(0, len(ds), max(1, len(ds) // 6))][:6]
    return ds.collate(items)


@pytest.fixture
def model(data):
    torch.manual_seed(0)
    cfg = ModelConfig(text_dim=64, d_model=32, n_heads=4, n_latents=2, n_stream_layers=1, n_readout_layers=1,
                      max_moments=data.cfg.n_moments, dropout=0.0)
    return HomeFM(cfg, data.entity_table)
