"""Pretraining CLI.

    python -m homefm.train.pretrain --objective configs/objectives/D_next_event.yaml
    python -m homefm.train.pretrain --objective configs/objectives/E_next_event_latent.yaml --set train.steps=5000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from homefm.data import HomeFMData, WindowConfig, WindowDataset, build_text_encoder, simulate_homes
from homefm.data.converters.casas import load_casas
from homefm.eval import evaluate
from homefm.model import HomeFM, ModelConfig
from homefm.objectives import build_objective
from homefm.schema.ontology import DEFAULT_CONCEPTS

ROOT = Path(__file__).resolve().parents[3]


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(objective_path: str | None, overrides: list[str] = (), base: str | None = None) -> dict:
    cfg = yaml.safe_load(Path(base or ROOT / "configs" / "base.yaml").read_text())
    if objective_path:
        cfg = deep_merge(cfg, yaml.safe_load(Path(objective_path).read_text()))
    for ov in overrides:
        key, val = ov.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_data(cfg: dict):
    dcfg = cfg["data"]
    if dcfg["source"] == "synthetic":
        s = dcfg["synthetic"]
        streams = simulate_homes(s["n_homes"], s["days"], seed=cfg["seed"])
        held_out = [st.home_id for st in streams[-s["held_out_homes"]:]]
    elif dcfg["source"] == "casas":
        streams = [load_casas(h["data"], h["sensor_map"], h["home_id"], h.get("label_map"))
                   for h in dcfg["casas"]["homes"]]
        held_out = dcfg["casas"]["held_out"]
    else:
        raise ValueError(dcfg["source"])
    concepts = dcfg.get("concepts") or list(DEFAULT_CONCEPTS)
    wcfg = WindowConfig(dcfg["span_s"], dcfg["moment_s"], dcfg["stride_s"], dcfg["max_events"])
    data = HomeFMData(streams, build_text_encoder(cfg["text_encoder"]), concepts, wcfg)
    train_homes = [h for h in data.home_ids if h not in held_out]
    return data, train_homes, held_out


def make_loaders(data: HomeFMData, train_homes, test_homes, batch_size: int, workers: int = 0):
    train_ds = WindowDataset(data, train_homes)
    eval_train = WindowDataset(data, train_homes, stride_s=data.cfg.span_s)
    eval_test = WindowDataset(data, test_homes, stride_s=data.cfg.span_s)
    mk = lambda ds, shuffle: DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=ds.collate,
                                        num_workers=workers, drop_last=shuffle)
    return mk(train_ds, True), mk(eval_train, False), mk(eval_test, False)


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def train(cfg: dict, run_name: str | None = None, verbose: bool = True) -> dict:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])
    data, train_homes, test_homes = build_data(cfg)
    tr = cfg["train"]
    train_loader, eval_train, eval_test = make_loaders(data, train_homes, test_homes, tr["batch_size"],
                                                       tr.get("num_workers", 0))

    mcfg = ModelConfig(text_dim=data.entity_table.shape[1], max_moments=data.cfg.n_moments, **cfg["model"])
    model = HomeFM(mcfg, data.entity_table).to(device)
    objective = build_objective(cfg["objective"], mcfg.d_model, mcfg.text_dim, len(data.home_ids),
                                data.entity_freq).to(device)
    params = [p for p in list(model.parameters()) + list(objective.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tr["lr"], weight_decay=tr["weight_decay"])
    name = run_name or cfg["objective"].get("run_name", cfg["objective"]["name"])
    out_dir = ROOT / tr["out_dir"] / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[{name}] device={device} params={model.n_parameters():,} train_windows={len(train_loader.dataset)} "
              f"train_homes={len(train_homes)} test_homes={len(test_homes)}")

    step, t0, history = 0, time.time(), []
    model.train()
    objective.train()
    while step < tr["steps"]:
        for batch in train_loader:
            if step >= tr["steps"]:
                break
            batch = batch.to(device)
            objective.set_progress(step / tr["steps"], model)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, tr["steps"], tr["warmup_steps"], tr["lr"])
            loss, logs = objective(model, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tr.get("grad_clip", 1.0))
            opt.step()
            objective.after_step(model)
            if step % tr["log_every"] == 0 or step == tr["steps"] - 1:
                history.append({"step": step, **logs})
                if verbose:
                    msg = " ".join(f"{k}={v:.4f}" for k, v in logs.items())
                    print(f"[{name}] step {step:5d} {time.time() - t0:6.0f}s {msg}")
            step += 1

    torch.save({"model": model.state_dict(), "config": cfg, "entity_texts": data.entity_texts},
               out_dir / "model.pt")
    model.eval()
    objective.eval()
    results = evaluate(model, objective, eval_train, eval_test, data.concepts, device)
    results["train_seconds"] = time.time() - t0
    (out_dir / "results.json").write_text(json.dumps({"results": results, "history": history}, indent=2))
    if verbose:
        print(f"[{name}] eval: " + " ".join(f"{k}={v:.3f}" for k, v in results.items() if not k.startswith("f1/")))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", required=True, help="objective config yaml (configs/objectives/*.yaml)")
    ap.add_argument("--base", default=None, help="base config (default configs/base.yaml)")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.steps=500 model.d_model=64")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    train(load_config(args.objective, args.set, args.base), args.name)


if __name__ == "__main__":
    main()
