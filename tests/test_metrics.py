import numpy as np

from homefm.eval import auroc, count_accuracy, extract_episodes, multiset_f1, purity


def test_auroc():
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert auroc(np.array([0.5, 0.5]), np.array([0, 1])) == 0.5


def test_extract_episodes_merge_and_hysteresis():
    p = np.array([0, .9, .9, .45, .9, 0, 0, 0, .9, 0])
    assert extract_episodes(p, on=0.6, off=0.4) == [(1, 4), (8, 8)]
    assert extract_episodes(p, on=0.6, off=0.5, merge_gap=0) == [(1, 2), (4, 4), (8, 8)]
    assert extract_episodes(p, on=0.6, off=0.5, merge_gap=1) == [(1, 4), (8, 8)]
    assert extract_episodes(p, on=0.6, off=0.4, min_len=2) == [(1, 4)]


def test_counts_and_multiset():
    r = count_accuracy(np.array([[1, 0], [2, 0]]), np.array([[1, 0], [1, 0]]))
    assert r["count_exact"] == 0.5 and r["count_mae"] == 0.5
    assert multiset_f1(["a", "a", "b"], ["a", "b", "b"]) == 2 / 3
    assert purity(np.array([0, 0, 1, 1]), np.array(["x", "x", "y", "x"])) == 0.75
