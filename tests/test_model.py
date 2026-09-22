import torch


def test_encode_shapes(model, batch):
    enc = model.encode(batch, causal=False, with_readout=True)
    B, N = batch.valid.shape
    assert enc.stream.shape == (B, batch.n_moments, model.cfg.d_model)
    assert enc.readout.shape == (B, N, model.cfg.d_model)
    assert torch.isfinite(enc.stream).all()


def test_empty_moments_are_finite(model, batch):
    drop = batch.valid.clone()  # drop every event: all moments empty
    enc = model.encode(batch, causal=False, drop_mask=drop)
    assert torch.isfinite(enc.stream).all()


def test_causal_stream_does_not_see_future(model, batch):
    model.eval()
    M = batch.n_moments
    cut = M // 2
    base = model.encode(batch, causal=True).stream
    drop = batch.valid & (batch.moment_idx >= cut)  # remove all future events
    fut = model.encode(batch, causal=True, drop_mask=drop).stream
    assert torch.allclose(base[:, :cut], fut[:, :cut], atol=1e-5)


def test_causal_readout_does_not_see_future_events(model, batch):
    model.eval()
    b = int(batch.valid.sum(1).argmax())
    n = int(batch.valid[b].sum())
    i = n // 2
    base = model.encode(batch, causal=True, with_readout=True).readout[b, :i + 1]
    batch2 = batch.to("cpu")
    batch2.value = batch.value.clone()
    batch2.state = batch.state.clone()
    batch2.state[b, i + 1:] = 1 - batch2.state[b, i + 1:].clamp(max=1)  # change future events
    out = model.encode(batch2, causal=True, with_readout=True).readout[b, :i + 1]
    assert torch.allclose(base, out, atol=1e-5)
