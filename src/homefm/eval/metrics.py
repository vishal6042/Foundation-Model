"""Metrics (DESIGN.md §12)."""

from __future__ import annotations

from collections import Counter

import numpy as np


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC with average ranks for ties. NaN if only one class is present."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_f1(pred: np.ndarray, true: np.ndarray) -> float:
    pred, true = np.asarray(pred, bool), np.asarray(true, bool)
    tp = (pred & true).sum()
    denom = pred.sum() + true.sum()
    return float(2 * tp / denom) if denom else 1.0


def weighted_f1(pred: np.ndarray, true: np.ndarray) -> tuple[float, list[float]]:
    """Multi-label [.., C] → support-weighted F1 and per-class F1."""
    pred, true = pred.reshape(-1, pred.shape[-1]), true.reshape(-1, true.shape[-1])
    per = [binary_f1(pred[:, c], true[:, c]) for c in range(true.shape[1])]
    support = true.sum(0)
    w = support / support.sum() if support.sum() else np.ones_like(support) / len(support)
    return float((np.array(per) * w).sum()), per


def extract_episodes(prob: np.ndarray, on: float = 0.5, off: float | None = None, merge_gap: int = 0,
                     min_len: int = 1) -> list[tuple[int, int]]:
    """Per-moment probabilities → episodes [(start, end_inclusive)] with hysteresis, gap merging, min length."""
    off = on if off is None else off
    eps, active, s = [], False, 0
    for i, p in enumerate(prob):
        if not active and p >= on:
            active, s = True, i
        elif active and p < off:
            eps.append((s, i - 1))
            active = False
    if active:
        eps.append((s, len(prob) - 1))
    merged: list[tuple[int, int]] = []
    for e in eps:
        if merged and e[0] - merged[-1][1] - 1 <= merge_gap:
            merged[-1] = (merged[-1][0], e[1])
        else:
            merged.append(e)
    return [e for e in merged if e[1] - e[0] + 1 >= min_len]


def count_accuracy(pred_counts: np.ndarray, true_counts: np.ndarray) -> dict[str, float]:
    """Exact-match rate and MAE over (window, concept) pairs where either side is non-zero."""
    pred_counts, true_counts = np.asarray(pred_counts), np.asarray(true_counts)
    sel = (pred_counts > 0) | (true_counts > 0)
    if not sel.any():
        return {"count_exact": float("nan"), "count_mae": float("nan")}
    return {"count_exact": float((pred_counts[sel] == true_counts[sel]).mean()),
            "count_mae": float(np.abs(pred_counts[sel] - true_counts[sel]).mean())}


def multiset_f1(pred: list, true: list) -> float:
    """Bag-of-events F1 used by DomusFM for next-k prediction."""
    p, t = Counter(pred), Counter(true)
    inter = sum((p & t).values())
    if not p or not t:
        return float(p == t)
    prec, rec = inter / sum(p.values()), inter / sum(t.values())
    return 0.0 if inter == 0 else 2 * prec * rec / (prec + rec)


def purity(clusters: np.ndarray, labels: np.ndarray) -> float:
    clusters, labels = np.asarray(clusters), np.asarray(labels)
    total = 0
    for c in np.unique(clusters):
        total += Counter(labels[clusters == c]).most_common(1)[0][1]
    return total / len(labels)
