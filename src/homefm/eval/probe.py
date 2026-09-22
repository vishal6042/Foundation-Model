"""Downstream evaluation of a pretrained backbone (DESIGN.md §12).

- Linear probe on frozen moment embeddings → activity/event tagging F1 and episode-count accuracy.
- Anomaly AUROC from embedding rarity (kNN distance, any variant) and next-event surprise (D/E/F).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from homefm.objectives import NextEvent, find_part

from .metrics import auroc, count_accuracy, extract_episodes, weighted_f1


@torch.no_grad()
def extract(model, loader: DataLoader, device, surprise_obj: NextEvent | None = None) -> dict[str, np.ndarray]:
    model.eval()
    embs, labels, anomaly, surprise = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        embs.append(model.moment_embeddings(batch).float().cpu())
        labels.append(batch.labels.cpu())
        anomaly.append(batch.anomaly.cpu())
        if surprise_obj is not None:
            ev = surprise_obj.surprise(model, batch)                                   # [B, N]
            B, M = batch.anomaly.shape
            seg = torch.where(batch.valid, batch.moment_idx, torch.full_like(batch.moment_idx, M))
            tot = torch.zeros(B, M + 1, device=device).scatter_add(1, seg, ev * batch.valid)
            cnt = torch.zeros(B, M + 1, device=device).scatter_add(1, seg, batch.valid.float())
            surprise.append((tot / cnt.clamp(min=1))[:, :M].cpu())
    out = dict(emb=torch.cat(embs).numpy(), labels=torch.cat(labels).numpy(), anomaly=torch.cat(anomaly).numpy())
    if surprise:
        out["surprise"] = torch.cat(surprise).numpy()
    return out


def linear_probe(train: dict, test: dict, concepts: list[str], epochs: int = 300, lr: float = 1e-2,
                 device: str = "cpu", merge_gap: int = 2) -> dict[str, float]:
    d, C = train["emb"].shape[-1], len(concepts)
    xtr = torch.tensor(train["emb"].reshape(-1, d), device=device)
    ytr = torch.tensor(train["labels"].reshape(-1, C), device=device)
    mu, sd = xtr.mean(0), xtr.std(0).clamp(min=1e-6)
    clf = torch.nn.Linear(d, C).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-4)
    pos = ytr.mean(0).clamp(1e-4, 1 - 1e-4)
    pos_weight = ((1 - pos) / pos).clamp(max=20)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(clf((xtr - mu) / sd), ytr, pos_weight=pos_weight)
        loss.backward()
        opt.step()
    with torch.no_grad():
        xte = torch.tensor(test["emb"], device=device)
        prob = torch.sigmoid(clf((xte - mu) / sd)).cpu().numpy()                     # [W, M, C]
    true = test["labels"] > 0.5
    wf1, per = weighted_f1(prob > 0.5, true)

    pred_counts = np.zeros(true.shape[0::2], dtype=int)
    true_counts = np.zeros_like(pred_counts)
    for w in range(true.shape[0]):
        for c in range(C):
            pred_counts[w, c] = len(extract_episodes(prob[w, :, c], on=0.6, off=0.4, merge_gap=merge_gap))
            true_counts[w, c] = len(extract_episodes(true[w, :, c].astype(float)))
    res = {"probe_weighted_f1": wf1, **count_accuracy(pred_counts, true_counts)}
    res.update({f"f1/{c}": f for c, f in zip(concepts, per)})
    return res


def knn_rarity(train_emb: np.ndarray, test_emb: np.ndarray, k: int = 5, max_ref: int = 20000,
               device: str = "cpu") -> np.ndarray:
    d = train_emb.shape[-1]
    ref = train_emb.reshape(-1, d)
    if len(ref) > max_ref:
        ref = ref[np.random.default_rng(0).choice(len(ref), max_ref, replace=False)]
    ref_t = torch.tensor(ref, device=device)
    q = torch.tensor(test_emb.reshape(-1, d), device=device)
    scores = []
    for chunk in q.split(4096):
        dist = torch.cdist(chunk, ref_t)
        scores.append(dist.topk(k, largest=False).values.mean(-1).cpu())
    return torch.cat(scores).numpy().reshape(test_emb.shape[:-1])


def evaluate(model, objective, train_loader: DataLoader, test_loader: DataLoader, concepts: list[str],
             device) -> dict[str, float]:
    ne = find_part(objective, NextEvent)
    tr = extract(model, train_loader, device)
    te = extract(model, test_loader, device, surprise_obj=ne)
    res = linear_probe(tr, te, concepts, device=str(device))
    res["anomaly_auroc_rarity"] = auroc(knn_rarity(tr["emb"], te["emb"], device=str(device)).ravel(),
                                        te["anomaly"].ravel())
    if "surprise" in te:
        res["anomaly_auroc_surprise"] = auroc(te["surprise"].ravel(), te["anomaly"].ravel())
    return res
