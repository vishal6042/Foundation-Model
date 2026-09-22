from homefm.train.pretrain import ROOT, load_config, train


def test_end_to_end_tiny_run(tmp_path):
    cfg = load_config(str(ROOT / "configs" / "objectives" / "E_next_event_latent.yaml"), [
        "data.synthetic.n_homes=3", "data.synthetic.days=3", "data.synthetic.held_out_homes=1",
        "data.span_s=1800", "data.stride_s=1800", "text_encoder.dim=64",
        "model.d_model=32", "model.n_stream_layers=1", "model.n_readout_layers=1",
        "train.steps=4", "train.batch_size=4", "train.warmup_steps=1", "train.log_every=2",
        "train.device=cpu", f"train.out_dir={tmp_path.as_posix()}",
    ])
    res = train(cfg, run_name="smoke", verbose=False)
    assert 0.0 <= res["probe_weighted_f1"] <= 1.0
    assert "anomaly_auroc_surprise" in res
    assert (tmp_path / "smoke" / "model.pt").exists()
