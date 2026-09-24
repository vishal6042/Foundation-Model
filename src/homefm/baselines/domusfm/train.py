"""DomusFM pretraining (dual contrastive, §4.3) and downstream fine-tuning / evaluation (§6.1–6.6)."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Sampler, Subset

from .data import DomusWindows
from .model import N_ATTR, ADLHead, DomusFM, NextKHead, PretrainHeads


def _to(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(len(z1), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def attribute_mask(B: int, L: int, p: float, device) -> torch.Tensor:
    """Phase 1: each event is selected with prob p; one random attribute of it is masked."""
    sel = torch.rand(B, L, device=device) < p
    which = torch.randint(0, N_ATTR, (B, L), device=device)
    return F.one_hot(which, N_ATTR).bool() & sel.unsqueeze(-1)


def event_mask(B: int, L: int, p: float, device) -> torch.Tensor:
    """Phase 2: each event is selected with prob p; all of its attributes are masked."""
    return (torch.rand(B, L, device=device) < p).unsqueeze(-1).expand(B, L, N_ATTR)


def time_jitter(x: dict, max_s: int) -> dict:
    """Shift each window's clock by one random offset in [-max_s, max_s] s (not in the paper).

    Keeps the gaps between events but breaks the exact seconds-within-hour pattern, which otherwise
    fingerprints a window and lets the contrastive game be solved without learning behaviour.
    """
    if not max_s:
        return x
    t = x["dow"] * 86400 + x["hour"] * 3600 + x["sec"]
    off = torch.randint(-max_s, max_s + 1, (t.shape[0], 1), device=t.device)
    t = (t + off) % (7 * 86400)
    return {**x, "dow": t // 86400, "hour": (t % 86400) // 3600, "sec": t % 3600}


class BalancedSampler(Sampler):
    """Dataset-level oversampling (§6.1.2): pick a dataset uniformly, then a window uniformly within it.

    Equivalent to WeightedRandomSampler with weights 1/len(dataset), but works beyond torch.multinomial's
    2^24-category limit (the 82-home corpus has >20M windows).
    """

    def __init__(self, lengths: list[int], n_samples: int, seed: int = 0):
        self.lengths = np.array(lengths)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])
        self.n_samples, self.seed, self.epoch = n_samples, seed, 0

    def __len__(self) -> int:
        return self.n_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        ds = rng.integers(0, len(self.lengths), self.n_samples)
        idx = (rng.random(self.n_samples) * self.lengths[ds]).astype(np.int64)
        return iter((self.offsets[ds] + idx).tolist())


class HomeBatchSampler(Sampler):
    """Each batch holds `homes_per_batch` homes (uniform), with batch_size // homes_per_batch windows each.

    With mixed-home batches almost every in-batch negative comes from another home, and the frozen text
    embeddings alone tell them apart. Same-home negatives force the model to separate windows by behaviour.
    """

    def __init__(self, lengths: list[int], n_batches: int, batch_size: int, homes_per_batch: int, seed: int = 0):
        assert batch_size % homes_per_batch == 0, "batch_size must be divisible by homes_per_batch"
        self.lengths = np.array(lengths)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])
        self.n_batches, self.per_home, self.homes = n_batches, batch_size // homes_per_batch, homes_per_batch
        self.seed, self.epoch = seed, 0

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.n_batches):
            ds = np.repeat(rng.choice(len(self.lengths), self.homes, replace=self.homes > len(self.lengths)),
                           self.per_home)
            idx = (rng.random(len(ds)) * self.lengths[ds]).astype(np.int64)
            yield (self.offsets[ds] + idx).tolist()


def pretrain_loader(windows: list[DomusWindows], batch_size: int, n_samples: int, workers: int = 0,
                    homes_per_batch: int | None = None) -> DataLoader:
    """Dataset-level random oversampling so small datasets contribute proportionally (§6.1.2).

    homes_per_batch: None = mix homes freely (paper); k = every batch comes from k homes (hard negatives).
    """
    cat = ConcatDataset(windows)
    kw = dict(num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
    lengths = [len(w) for w in windows]
    if homes_per_batch:
        return DataLoader(cat, batch_sampler=HomeBatchSampler(lengths, n_samples // batch_size, batch_size,
                                                              homes_per_batch), **kw)
    return DataLoader(cat, batch_size=batch_size, sampler=BalancedSampler(lengths, n_samples), drop_last=True, **kw)


def pretrain(model: DomusFM, loader: DataLoader, cfg: dict, device, log=print, ckpt_path=None) -> list[dict]:
    """Dual contrastive pretraining. Phase 2 freezes the event-level feature extractor (§4.3).

    If `ckpt_path` is given, saves model + optimiser + position every `ckpt_every` steps and resumes from it,
    so an interruption loses at most `ckpt_every` steps.

    Optional fixes for the collapse seen on real data (all off by default = paper; see
    docs/DOMUSFM_REPRODUCTION.md, "Why pretraining did not help"): `mask_both_views`, `time_jitter_s`,
    `projector`, `mlm_weight` (and `homes_per_batch` in `pretrain_loader`).
    """
    history, start_phase, start_step, opt_state = [], "attribute", 0, None
    every = cfg.get("ckpt_every", 2000)
    both, jitter, mlm_w = cfg.get("mask_both_views", False), cfg.get("time_jitter_s", 0), cfg.get("mlm_weight", 0.0)
    heads = None
    if cfg.get("projector", False) or mlm_w:
        heads = PretrainHeads(model.cfg.d, model.event.text_table).to(device)
    if ckpt_path is not None and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"])
        if heads is not None and ck.get("heads") is not None:
            heads.load_state_dict(ck["heads"])
        history, start_phase, start_step, opt_state = ck["history"], ck["phase"], ck["step"], ck["opt"]
        log(f"  resuming pretraining from {start_phase} step {start_step}")
    phases = (("attribute", cfg["steps_phase1"]), ("event", cfg["steps_phase2"]))
    for phase, steps in phases:
        if start_phase == "event" and phase == "attribute":
            continue
        if phase == "event":
            model.event.requires_grad_(False)
        params = [p for p in model.parameters() if p.requires_grad]
        if heads is not None:
            params += list(heads.parameters())
        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        step = 0
        if phase == start_phase and start_step:
            step = start_step
            if opt_state is not None:
                opt.load_state_dict(opt_state)
        t0 = time.time()
        model.train()
        if heads is not None:
            heads.train()
        while step < steps:
            for batch in loader:
                if step >= steps:
                    break
                if ckpt_path is not None and step and step % every == 0:
                    tmp = ckpt_path.with_suffix(".tmp")
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "phase": phase,
                                "step": step, "history": history,
                                "heads": heads.state_dict() if heads is not None else None}, tmp)
                    tmp.replace(ckpt_path)  # atomic: a crash mid-save never corrupts the last checkpoint
                x = _to(batch, device)
                B, L = x["status"].shape
                mask_fn = (lambda: attribute_mask(B, L, cfg["attr_mask_p"], device)) if phase == "attribute" else \
                    (lambda: event_mask(B, L, cfg["event_mask_p"], device))
                m1, m2 = (mask_fn() if both else None), mask_fn()
                h1, h2 = model(time_jitter(x, jitter), attr_mask=m1), model(time_jitter(x, jitter), attr_mask=m2)
                z1, z2 = model.window_embedding(h1), model.window_embedding(h2)
                if heads is not None and cfg.get("projector", False):
                    z1, z2 = heads.projector(z1), heads.projector(z2)
                loss_c = info_nce(z1, z2, cfg["temperature"])
                loss_m = torch.zeros((), device=device)
                if mlm_w:
                    loss_m = heads.mlm_loss(h2, x, m2)
                    if m1 is not None:
                        loss_m = 0.5 * (loss_m + heads.mlm_loss(h1, x, m1))
                loss = loss_c + mlm_w * loss_m
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                if step % cfg["log_every"] == 0:
                    # z_std: spread of the normalised embeddings across the batch; near 0 = collapse
                    z_std = F.normalize(z2.detach(), dim=-1).std(0).mean().item()
                    history.append({"phase": phase, "step": step, "loss": loss.item(), "contrastive": loss_c.item(),
                                    "mlm": loss_m.item(), "z_std": z_std})
                    log(f"  pretrain[{phase}] step {step:5d} loss {loss.item():.4f} (contrastive {loss_c.item():.4f}"
                        f" mlm {loss_m.item():.4f} z_std {z_std:.3f}) ({time.time() - t0:.0f}s)")
                step += 1
    model.event.requires_grad_(True)
    return history  # `heads` are pretraining-only and are dropped here


# ---- downstream -----------------------------------------------------------------------------

def weighted_f1_multiclass(pred: np.ndarray, true: np.ndarray, n_classes: int) -> float:
    f1s, support = [], []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (true == c))
        fp, fn = np.sum((pred == c) & (true != c)), np.sum((pred != c) & (true == c))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
        support.append(np.sum(true == c))
    support = np.array(support, float)
    return float(np.dot(f1s, support / support.sum()))


def multiset_f1_counts(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Per-window bag-of-events F1 from count vectors (§6.5.4)."""
    inter = torch.minimum(pred, true).sum(-1)
    p = inter / pred.sum(-1).clamp(min=1e-9)
    r = inter / true.sum(-1).clamp(min=1e-9)
    f1 = 2 * p * r / (p + r).clamp(min=1e-9)
    return torch.where(pred.sum(-1) == 0, torch.zeros_like(f1), f1)


def contiguous_folds(n: int, k: int) -> list[np.ndarray]:
    """ASSUMPTION: time-contiguous folds (random folds would leak through 29-event window overlap)."""
    return np.array_split(np.arange(n), k)


@torch.no_grad()
def _evaluate(backbone, head, loader, task, device):
    backbone.eval()
    head.eval()
    preds, trues, f1s = [], [], []
    for batch in loader:
        x = _to(batch, device)
        out = head(backbone(x))
        if task == "adl":
            preds.append(out.argmax(-1).cpu())
            trues.append(x["label"].cpu())
        else:
            f1s.append(multiset_f1_counts(NextKHead.predict(out), x["next_counts"]).cpu())
    if task == "adl":
        return torch.cat(preds).numpy(), torch.cat(trues).numpy()
    return torch.cat(f1s).mean().item()


def finetune_and_eval(state: dict | None, model_ctor, windows: DomusWindows, task: str, train_pct: float,
                      cfg: dict, device, seed: int = 0) -> dict:
    """k-fold CV on the held-out dataset; fine-tune backbone + head on `train_pct` of the training folds (§6.1.3)."""
    rng = np.random.default_rng(seed)
    if cfg.get("fold_mode", "contiguous") == "random":  # likely the paper's protocol; overlapping windows leak
        folds = np.array_split(rng.permutation(len(windows)), cfg["folds"])
    else:
        folds = contiguous_folds(len(windows), cfg["folds"])
    n_classes = len(windows.d.activities)
    scores = []
    for f, test_idx in enumerate(folds):
        pool = np.concatenate([folds[j] for j in range(len(folds)) if j != f])
        n_train = max(cfg["batch_size"], int(len(pool) * train_pct))
        train_idx = rng.choice(pool, size=min(n_train, len(pool)), replace=False)

        backbone = model_ctor().to(device)
        if state is not None:
            backbone.load_state_dict(state)
        head = (ADLHead(backbone.cfg.d, n_classes) if task == "adl"
                else NextKHead(backbone.cfg.d, windows.d.n_event_types)).to(device)
        params = list(backbone.parameters()) + list(head.parameters())  # full fine-tuning (§6.1.3)
        # backbone_lr / head_lr / warmup_frac are not in the paper; defaults (lr, lr, 0) reproduce it
        opt = torch.optim.AdamW([{"params": backbone.parameters(), "lr": cfg.get("backbone_lr", cfg["lr"])},
                                 {"params": head.parameters(), "lr": cfg.get("head_lr", cfg["lr"])}],
                                weight_decay=cfg["weight_decay"])
        train_loader = DataLoader(Subset(windows, train_idx), batch_size=cfg["batch_size"], shuffle=True,
                                  drop_last=len(train_idx) > cfg["batch_size"])
        warmup = int(cfg.get("warmup_frac", 0.0) * cfg["epochs"] * len(train_loader))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / warmup) if warmup else 1.0)
        test_sub = test_idx[:: max(1, len(test_idx) // cfg["max_test_windows"])] if cfg.get("max_test_windows") else test_idx
        test_loader = DataLoader(Subset(windows, test_sub), batch_size=cfg["eval_batch_size"])

        for _ in range(cfg["epochs"]):  # fixed epochs, no early stopping (§6.1.3)
            backbone.train()
            head.train()
            for batch in train_loader:
                x = _to(batch, device)
                out = head(backbone(x))
                loss = ADLHead.loss(out, x) if task == "adl" else NextKHead.loss(out, x)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
        res = _evaluate(backbone, head, test_loader, task, device)
        scores.append(weighted_f1_multiclass(*res, n_classes) if task == "adl" else res)
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "folds": [float(s) for s in scores]}


def fresh_copy(model: DomusFM) -> dict:
    return copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
