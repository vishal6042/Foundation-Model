"""Leave-one-dataset-out DomusFM reproduction (paper §6).

    python -m homefm.baselines.domusfm.run                                   # synthetic homes
    python -m homefm.baselines.domusfm.run --overlay configs/domusfm_real.yaml  # real datasets
    python -m homefm.baselines.domusfm.run --config configs/domusfm.yaml --set finetune.folds=2 held_out=[sim_0]

For each held-out dataset: pretrain on all other datasets, then fine-tune + test on the held-out one for
every task × train-% (and optionally the same without pretraining, the §6.7.2 ablation).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from homefm.data import build_text_encoder, simulate_homes
from homefm.data.converters.casas import load_casas
from homefm.data.converters.casas_fast import load_casas_fast
from homefm.data.converters.casas_zenodo import load_casas_zenodo
from homefm.data.converters.uci_adl import load_uci_adl
from homefm.train.pretrain import ROOT, load_config, resolve_device

from .data import DomusWindows, TextTable, build_domus_dataset, build_domus_from_arrays
from .model import DomusConfig, DomusFM
from .train import finetune_and_eval, fresh_copy, pretrain, pretrain_loader


def load_datasets(cfg: dict):
    """Returns (datasets, pretrain_only_names). Pretrain-only datasets (e.g. unlabelled homes) are never held out."""
    src = cfg["datasets"]["source"]
    if src == "synthetic":
        s = cfg["datasets"]["synthetic"]
        homes = simulate_homes(s["n"], s["days"], seed=cfg["seed"], inject_faults=False)
        return [build_domus_dataset(h.home_id, [h]) for h in homes], set()
    if src == "list":
        out, pretrain_only = [], set()
        for spec in cfg["datasets"]["list"]:
            kind, path = spec["kind"], ROOT / spec["path"]
            if kind == "casas_zenodo":
                stream = load_casas_zenodo(path, spec.get("id"))
            elif kind == "uci_adl":
                stream = load_uci_adl(path, spec.get("home", "B"))
            elif kind == "casas":
                stream = load_casas(path, ROOT / spec["sensor_map"], spec["id"], spec.get("label_map"))
            elif kind in ("casas_fast", "casas_fast_glob"):
                files = [path] if kind == "casas_fast" else sorted(ROOT.glob(spec["path"]))
                taken = {d.name for d in out}
                for f in files:
                    hid = spec.get("id", f.stem) if kind == "casas_fast" else f.stem
                    if hid in taken or hid in spec.get("exclude", []):
                        continue
                    h = load_casas_fast(f, hid, cache_dir=ROOT / cfg["datasets"].get("cache_dir", "data/cache/casas"))
                    out.append(build_domus_from_arrays(h))
                    if spec.get("pretrain_only"):
                        pretrain_only.add(hid)
                continue
            else:
                raise ValueError(kind)
            out.append(build_domus_dataset(spec.get("id", stream.home_id), [stream]))
            if spec.get("pretrain_only"):
                pretrain_only.add(out[-1].name)
        return out, pretrain_only
    raise ValueError(src)


def text_encoder(cfg: dict):
    try:
        return build_text_encoder(cfg["text_encoder"])
    except ImportError:
        print("[domusfm] sentence-transformers not installed; falling back to the hashing encoder "
              "(NOT faithful to the paper — install with: pip install -e .[text])")
        return build_text_encoder({"kind": "hashing", "dim": cfg["model"]["d"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "domusfm.yaml"))
    ap.add_argument("--overlay", default=None, help="yaml merged on top of --config (e.g. configs/domusfm_real.yaml)")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.overlay, args.set, base=args.config)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    out_dir = ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets, pretrain_only = load_datasets(cfg)
    text = TextTable(datasets, text_encoder(cfg))
    mcfg = DomusConfig(**cfg["model"])
    ctor = lambda: DomusFM(mcfg, text.table)
    print(f"[domusfm] device={device} params={ctor().n_parameters():,} datasets="
          + ", ".join(f"{d.name}({d.n_events} ev, {len(d.activities)} cls)" for d in datasets))

    W, ft = cfg["window"], cfg["finetune"]
    held_out = cfg.get("held_out") or [d.name for d in datasets if d.name not in pretrain_only]
    results_path = out_dir / "results.json"
    # resume: keep every result already saved and skip it (a restart loses at most the setting in progress)
    results = json.loads(results_path.read_text()) if cfg.get("resume", True) and results_path.exists() else {}
    if results:
        done = sum(len(r["results"]) for r in results.values())
        print(f"[domusfm] resuming: {done} results already in {results_path.name}", flush=True)
    targets = [d for d in datasets if d.name in held_out]
    shared = cfg.get("pretrain_mode", "per_target") == "shared"
    shared_state, shared_hist = None, []
    if shared:  # one pretraining run on every dataset that is not a target (stricter than leave-one-out)
        shared_state, shared_hist = get_pretrained(cfg, ctor, text, [d for d in datasets if d.name not in held_out],
                                                   out_dir / "pretrained_shared.pt", device)
    for target in targets:
        t0 = time.time()
        print(f"\n=== held-out: {target.name} ===", flush=True)
        if shared:
            state, history = shared_state, shared_hist
        else:
            others = [d for d in datasets if d is not target]
            state, history = get_pretrained(cfg, ctor, text, others, out_dir / f"pretrained_{target.name}.pt", device)
        states = {"DomusFM": state}
        if cfg.get("ablation_no_pretrain"):
            states["w/o Pretrain"] = None

        entry = results.setdefault(target.name, {"results": {}, "pretrain_history": history, "seconds": 0.0})
        res = entry["results"]
        for task in ft["tasks"]:
            k = 0 if task == "adl" else int(task.removeprefix("next"))
            windows = None
            for pct in ft["train_pcts"]:
                for name, st in states.items():
                    key = f"{task}|{int(pct * 100)}%|{name}"
                    if key in res:
                        continue
                    windows = windows or DomusWindows(target, text, W, 1, k=k)
                    r = finetune_and_eval(st, ctor, windows, "adl" if task == "adl" else "nextk", pct, ft, device,
                                          seed=cfg["seed"])
                    res[key] = r
                    entry["seconds"] += time.time() - t0
                    t0 = time.time()
                    results_path.write_text(json.dumps(results, indent=2))  # save after every setting
                    print(f"  {task:6s} {int(pct * 100):3d}% {name:13s} {r['mean']:.3f} ± {r['std']:.3f}", flush=True)

    write_table(results, ft, out_dir / "results.md")
    print(f"\n[domusfm] saved {out_dir / 'results.json'} and results.md")


def get_pretrained(cfg, ctor, text, pool, ckpt: Path, device):
    """Load a saved pretrained backbone if present (and `reuse_pretrained`), else pretrain and save it."""
    pt, W = cfg["pretrain"], cfg["window"]
    if cfg.get("reuse_pretrained", True) and ckpt.exists():
        saved = torch.load(ckpt, map_location="cpu")
        print(f"[domusfm] reusing {ckpt.name}", flush=True)
        return saved["state"], saved["history"]
    n_windows = sum(max(0, d.n_events - W + 1) for d in pool)
    print(f"[domusfm] pretraining on {len(pool)} datasets, {sum(d.n_events for d in pool):,} events, "
          f"{n_windows:,} windows", flush=True)
    loader = pretrain_loader([DomusWindows(d, text, W, pt["stride"]) for d in pool], pt["batch_size"],
                             pt["batch_size"] * (pt["steps_phase1"] + pt["steps_phase2"]), pt.get("num_workers", 0))
    model = ctor().to(device)
    progress = ckpt.with_name(ckpt.stem + "_progress.pt")  # mid-pretraining checkpoint, removed when done
    history = pretrain(model, loader, pt, device, log=lambda m: print(m, flush=True), ckpt_path=progress)
    state = fresh_copy(model)
    torch.save({"state": state, "history": history, "pool": [d.name for d in pool]}, ckpt)
    progress.unlink(missing_ok=True)
    return state, history


def write_table(results: dict, ft: dict, path: Path):
    lines = []
    for task in ft["tasks"]:
        metric = "weighted F1" if task == "adl" else "multiset F1"
        names = sorted({k.split("|")[2] for r in results.values() for k in r["results"]})
        cols = [f"{int(p * 100)}% {n}" for p in ft["train_pcts"] for n in names]
        lines += [f"### {task} ({metric})", "", "| Dataset | " + " | ".join(cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for ds, r in results.items():
            vals = [r["results"].get(f"{task}|{int(p * 100)}%|{n}", {}).get("mean") for p in ft["train_pcts"]
                    for n in names]
            lines.append(f"| {ds} | " + " | ".join("—" if v is None else f"{v:.2f}" for v in vals) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
