import torch

from homefm.baselines.domusfm import DomusConfig, DomusFM, DomusWindows, TextTable, build_domus_dataset
from homefm.baselines.domusfm.model import ADLHead, NextKHead
from homefm.baselines.domusfm.train import attribute_mask, event_mask, info_nce, multiset_f1_counts
from homefm.data import HashingTextEncoder, simulate_home
from homefm.data.synthetic import SimConfig

SMALL = DomusConfig(d=32, n_layers=1, n_heads=4, ff=64, attr_heads=4)


def _setup(k=0):
    d = build_domus_dataset("h", [simulate_home("h", SimConfig(days=2, inject_faults=False), seed=0)])
    text = TextTable([d], HashingTextEncoder(32))
    return d, text, DomusWindows(d, text, window=30, stride=50, k=k)


def test_dataset_is_binary_with_other_class():
    d, _, w = _setup()
    assert set(d.status.tolist()) <= {0, 1}
    assert d.activities[0] == "Other"
    item = w[0]
    assert item["status"].shape == (30,) and item["sec"].max() < 3600


def test_pretraining_losses_and_heads():
    d, text, w = _setup(k=10)
    batch = torch.utils.data.default_collate([w[i] for i in range(8)])
    model = DomusFM(SMALL, text.table)
    h = model(batch)
    assert h.shape == (8, 30, 32)
    for m in (attribute_mask(8, 30, 0.3, "cpu"), event_mask(8, 30, 0.3, "cpu")):
        loss = info_nce(model.window_embedding(h), model.window_embedding(model(batch, attr_mask=m)), 0.07)
        assert torch.isfinite(loss)
    adl = ADLHead(32, len(d.activities))
    assert torch.isfinite(ADLHead.loss(adl(h), batch))
    nk = NextKHead(32, d.n_event_types)
    out = nk(h)
    assert torch.isfinite(NextKHead.loss(out, batch))
    assert batch["next_counts"].sum(-1).eq(10).all()
    assert multiset_f1_counts(batch["next_counts"], batch["next_counts"]).eq(1).all()
