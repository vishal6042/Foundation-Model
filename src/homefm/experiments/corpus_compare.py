"""HomeFM on the CASAS corpus, evaluated with exactly the DomusFM protocol (DESIGN.md §8.3, §12).

Same pretraining pool (every labelled CASAS home except the targets, plus Milan and Aruba), same targets,
same windows' end events and labels, same folds / label percentages / epochs, same ADL and next-k heads.
The only differences are the backbone and its pretraining objective. HomeFM sees the raw event stream
(scalar sensors are not binarised) in a time window ending at the same event DomusFM's window ends at.

    python -m homefm.experiments.corpus_compare            # uses configs/homefm_corpus.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from homefm.baselines.domusfm.data import build_domus_dataset, build_domus_from_arrays
from homefm.baselines.domusfm.model import ADLHead, NextKHead
from homefm.baselines.domusfm.train import (BalancedSampler, contiguous_folds, multiset_f1_counts,
                                            weighted_f1_multiclass)
from homefm.data import HomeFMData, WindowConfig, WindowDataset, build_text_encoder
from homefm.data.converters.casas_fast import load_casas_fast
from homefm.data.converters.uci_adl import load_uci_adl
from homefm.model import HomeFM, ModelConfig
from homefm.objectives import build_objective
from homefm.train.pretrain import ROOT, load_config, lr_at, resolve_device


def load_homes(cfg: dict):
    """→ homes (HomeStream | HomeArrays), pretrain-only ids, DomusFM-style datasets for targets."""
    homes, pretrain_only, domus = [], set(), {}
    cache = ROOT / cfg["datasets"].get("cache_dir", "data/cache/casas")
    for spec in cfg["datasets"]["list"]:
        kind = spec["kind"]
        if kind == "uci_adl":
            h = load_uci_adl(ROOT / spec["path"], spec.get("home", "B"))
            h.home_id = spec.get("id", h.home_id)
            for t in h.tokens:
                t.home_id = h.home_id
            items = [(h, lambda h=h: build_domus_dataset(h.home_id, [h]))]
        else:
            files = [ROOT / spec["path"]] if kind == "casas_fast" else sorted(ROOT.glob(spec["path"]))
            items = []
            for f in files:
                hid = spec.get("id", f.stem) if kind == "casas_fast" else f.stem
                if any(x.home_id == hid for x in homes) or hid in spec.get("exclude", []):
                    continue
                a = load_casas_fast(f, hid, cache_dir=cache)
                items.append((a, lambda a=a: build_domus_from_arrays(a)))
        for h, mk in items:
            homes.append(h)
            if spec.get("pretrain_only"):
                pretrain_only.add(h.home_id)
            elif h.home_id in cfg["held_out"]:
                domus[h.home_id] = mk()
    return homes, pretrain_only, domus


# ---- pretraining ------------------------------------------------------------------------------

def pretrain_homefm(cfg, data: HomeFMData, pool: list[str], device, ckpt: Path) -> dict:
    if cfg.get("reuse_pretrained", True) and ckpt.exists():
        print(f"[homefm] reusing {ckpt.name}", flush=True)
        return torch.load(ckpt, map_location="cpu")["state"]
    pt = cfg["pretrain"]
    ds = WindowDataset(data, pool, stride_s=pt["stride_s"])
    counts = Counter(h for h, _ in ds.index)  # WindowDataset.index is grouped by home, in pool order
    lengths = [counts[hid] for hid in pool]
    loader = DataLoader(ds, batch_size=pt["batch_size"], collate_fn=ds.collate, drop_last=True,
                        sampler=BalancedSampler(lengths, pt["batch_size"] * pt["steps"]), num_workers=pt.get("num_workers", 0),
                        persistent_workers=pt.get("num_workers", 0) > 0)
    model = make_model(cfg, data).to(device)
    objective = build_objective(cfg["objective"], model.cfg.d_model, model.cfg.text_dim, len(data.home_ids),
                                data.entity_freq).to(device)
    params = [p for p in list(model.parameters()) + list(objective.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=pt["lr"], weight_decay=pt["weight_decay"])
    print(f"[homefm] pretraining on {len(pool)} homes, {len(ds):,} windows, params={model.n_parameters():,}", flush=True)
    t0 = time.time()
    model.train()
    objective.train()
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        objective.set_progress(step / pt["steps"], model)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, pt["steps"], pt["warmup_steps"], pt["lr"])
        loss, logs = objective(model, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        objective.after_step(model)
        if step % pt["log_every"] == 0:
            print(f"  pretrain step {step:6d} ({time.time() - t0:.0f}s) " +
                  " ".join(f"{k}={v:.4f}" for k, v in logs.items()), flush=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state": state, "pool": pool}, ckpt)
    return state


def make_model(cfg, data: HomeFMData) -> HomeFM:
    mcfg = ModelConfig(text_dim=data.entity_table.shape[1], max_moments=data.cfg.n_moments, **cfg["model"])
    return HomeFM(mcfg, data.entity_table)


# ---- downstream (DomusFM protocol) -------------------------------------------------------------

class EndWindows(Dataset):
    """HomeFM time windows ending at the same events as DomusFM's 30-event windows."""

    def __init__(self, data: HomeFMData, home_id: str, dom, k: int = 0, window: int = 30):
        self.data, self.home_id, self.dom, self.k = data, home_id, dom, k
        self.ends = np.arange(window - 1, dom.n_events - k)
        self._collator = WindowDataset(data, home_ids=[])

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, i):
        e = int(self.ends[i])
        item = self.data.window(self.home_id, self.dom.ts[e] - self.data.cfg.span_s + 1e-3, keep="last")
        item["y"] = int(self.dom.label[e])
        if self.k:
            nxt = slice(e + 1, e + 1 + self.k)
            types = self.dom.sensor[nxt] * 2 + self.dom.status[nxt]
            item["next_counts"] = np.bincount(types, minlength=self.dom.n_event_types).astype(np.float32)
        return item

    def collate(self, items):
        batch = self._collator.collate(items)
        extra = {"label": torch.tensor([it["y"] for it in items])}
        if self.k:
            extra["next_counts"] = torch.from_numpy(np.stack([it["next_counts"] for it in items]))
        return batch, extra


class Downstream(nn.Module):
    def __init__(self, backbone: HomeFM, head: nn.Module):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, batch):
        enc = self.backbone.encode(batch, causal=True, with_readout=True)
        last = (batch.valid.sum(1) - 1).clamp(min=0)
        h = enc.readout[torch.arange(len(last), device=last.device), last]
        return self.head(h.unsqueeze(1))


def finetune_eval(cfg, data, state, windows: EndWindows, task: str, pct: float, device, seed=0) -> dict:
    ft = cfg["finetune"]
    rng = np.random.default_rng(seed)
    folds = (np.array_split(rng.permutation(len(windows)), ft["folds"]) if ft.get("fold_mode") == "random"
             else contiguous_folds(len(windows), ft["folds"]))
    n_classes = len(windows.dom.activities)
    scores = []
    for f, test_idx in enumerate(folds):
        pool = np.concatenate([folds[j] for j in range(len(folds)) if j != f])
        n_train = max(ft["batch_size"], int(len(pool) * pct))
        train_idx = rng.choice(pool, size=min(n_train, len(pool)), replace=False)
        backbone = make_model(cfg, data)
        if state is not None:
            backbone.load_state_dict(state)
        head = ADLHead(backbone.cfg.d_model, n_classes) if task == "adl" else \
            NextKHead(backbone.cfg.d_model, windows.dom.n_event_types)
        model = Downstream(backbone, head).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=ft["lr"], weight_decay=ft["weight_decay"])
        mk = lambda idx, shuffle, bs: DataLoader(Subset(windows, idx), batch_size=bs, shuffle=shuffle,
                                                 collate_fn=windows.collate, num_workers=ft.get("num_workers", 0))
        if ft.get("max_test_windows"):
            test_idx = test_idx[:: max(1, len(test_idx) // ft["max_test_windows"])]
        for _ in range(ft["epochs"]):
            model.train()
            for batch, y in mk(train_idx, True, ft["batch_size"]):
                batch, y = batch.to(device), {k: v.to(device) for k, v in y.items()}
                out = model(batch)
                loss = ADLHead.loss(out, y) if task == "adl" else NextKHead.loss(out, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        model.eval()
        preds, trues, f1s = [], [], []
        with torch.no_grad():
            for batch, y in mk(test_idx, False, ft["eval_batch_size"]):
                out = model(batch.to(device))
                if task == "adl":
                    preds.append(out.argmax(-1).cpu())
                    trues.append(y["label"])
                else:
                    f1s.append(multiset_f1_counts(NextKHead.predict(out).cpu(), y["next_counts"]))
        scores.append(weighted_f1_multiclass(torch.cat(preds).numpy(), torch.cat(trues).numpy(), n_classes)
                      if task == "adl" else torch.cat(f1s).mean().item())
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "folds": [float(s) for s in scores]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "homefm_corpus.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(None, args.set, base=args.config)
    torch.manual_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    out_dir = ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    homes, pretrain_only, domus = load_homes(cfg)
    w = cfg["window"]
    data = HomeFMData(homes, build_text_encoder(cfg["text_encoder"]), [],
                      WindowConfig(w["span_s"], w["moment_s"], cfg["pretrain"]["stride_s"], w["max_events"]))
    print(f"[homefm] {len(homes)} homes loaded in {time.time() - t0:.0f}s; targets={list(domus)}", flush=True)

    pool = [h for h in data.home_ids if h in pretrain_only]
    state = pretrain_homefm(cfg, data, pool, device, out_dir / "pretrained_shared.pt")
    states = {"HomeFM-E": state}
    if cfg.get("ablation_no_pretrain"):
        states["HomeFM w/o Pretrain"] = None

    results = {}
    for hid, dom in domus.items():
        print(f"\n=== held-out: {hid} ===", flush=True)
        res = {}
        for task in cfg["finetune"]["tasks"]:
            k = 0 if task == "adl" else int(task.removeprefix("next"))
            windows = EndWindows(data, hid, dom, k)
            for pct in cfg["finetune"]["train_pcts"]:
                for name, st in states.items():
                    r = finetune_eval(cfg, data, st, windows, "adl" if task == "adl" else "nextk", pct, device,
                                      seed=cfg["seed"])
                    res[f"{task}|{int(pct * 100)}%|{name}"] = r
                    print(f"  {task:6s} {int(pct * 100):3d}% {name:20s} {r['mean']:.3f} ± {r['std']:.3f}", flush=True)
        results[hid] = res
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[homefm] saved {out_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
