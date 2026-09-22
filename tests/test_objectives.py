import pytest
import torch
import yaml

from homefm.objectives import build_objective
from homefm.train.pretrain import ROOT

VARIANTS = ["A_contrastive", "B_masked_recon", "C_latent_mask", "D_next_event", "E_next_event_latent", "F_full"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_objective_trains_one_step(variant, model, batch, data):
    cfg = yaml.safe_load((ROOT / "configs" / "objectives" / f"{variant}.yaml").read_text())["objective"]
    obj = build_objective(cfg, model.cfg.d_model, model.cfg.text_dim, len(data.home_ids), data.entity_freq)
    params = list(model.parameters()) + list(obj.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    losses = []
    for step in range(3):
        obj.set_progress(step / 3 if variant != "A_contrastive" else 0.9 * step / 2, model)
        loss, logs = obj(model, batch)
        assert torch.isfinite(loss), logs
        opt.zero_grad()
        loss.backward()
        opt.step()
        obj.after_step(model)
        losses.append(loss.item())
    for p in model.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()
