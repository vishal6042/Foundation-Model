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


def test_time_jitter_keeps_gaps_and_ranges():
    from homefm.baselines.domusfm.train import time_jitter

    _, _, w = _setup()
    x = torch.utils.data.default_collate([w[i] for i in range(8)])
    y = time_jitter(x, 900)
    t = lambda b: b["dow"] * 86400 + b["hour"] * 3600 + b["sec"]
    gaps_x, gaps_y = t(x).diff(dim=1) % 604800, t(y).diff(dim=1) % 604800
    assert torch.equal(gaps_x, gaps_y)
    assert y["sec"].max() < 3600 and y["hour"].max() < 24 and y["dow"].max() < 7
    assert time_jitter(x, 0) is x


def test_home_batch_sampler_groups_homes():
    from homefm.baselines.domusfm.train import HomeBatchSampler

    lengths = [100, 50, 70, 30]
    offsets = [0, 100, 150, 220]
    s = HomeBatchSampler(lengths, n_batches=5, batch_size=8, homes_per_batch=2)
    batches = list(s)
    assert len(batches) == 5
    home = lambda i: max(h for h, o in enumerate(offsets) if i >= o)
    for b in batches:
        assert len(b) == 8 and max(b) < sum(lengths)
        assert len({home(i) for i in b}) <= 2


def test_fixed_pretraining_runs_and_mlm_uses_hidden_answers():
    from homefm.baselines.domusfm.model import PretrainHeads
    from homefm.baselines.domusfm.train import pretrain, pretrain_loader

    d, text, w = _setup()
    loader = pretrain_loader([w, w], batch_size=8, n_samples=8 * 4, homes_per_batch=2)
    model = DomusFM(SMALL, text.table)
    cfg = dict(steps_phase1=2, steps_phase2=2, lr=1e-4, weight_decay=0.0, attr_mask_p=0.4, event_mask_p=0.4,
               temperature=0.2, log_every=1, mask_both_views=True, time_jitter_s=900, projector=True,
               mlm_weight=1.0)
    hist = pretrain(model, loader, cfg, "cpu")
    assert len(hist) == 4 and all(torch.isfinite(torch.tensor(h["mlm"])) and h["mlm"] > 0 for h in hist)

    batch = torch.utils.data.default_collate([w[i] for i in range(8)])
    heads = PretrainHeads(32, text.table)
    m = event_mask(8, 30, 0.5, "cpu")
    assert torch.isfinite(heads.mlm_loss(model(batch, attr_mask=m), batch, m))
