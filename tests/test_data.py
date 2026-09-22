import numpy as np

from homefm.data import simulate_home
from homefm.data.synthetic import SimConfig


def test_simulator_is_deterministic_and_labelled():
    a = simulate_home("h", SimConfig(days=2), seed=3)
    b = simulate_home("h", SimConfig(days=2), seed=3)
    assert [t.ts for t in a.tokens] == [t.ts for t in b.tokens]
    assert np.all(np.diff([t.ts for t in a.tokens]) >= 0)
    concepts = {e.concept for e in a.episodes}
    assert {"cooking", "sleeping", "bathroom visit"} <= concepts
    assert all(e.caption for e in a.episodes)


def test_faults_are_injected():
    s = simulate_home("h", SimConfig(days=5, inject_faults=True), seed=0)
    kinds = {a.kind for a in s.anomalies}
    assert "device_degradation" in kinds and len(kinds) >= 3


def test_batch_shapes(data, batch):
    B, N = batch.valid.shape
    M = data.cfg.n_moments
    assert batch.labels.shape == (B, M, len(data.concepts))
    assert batch.moment_idx.max() < M
    assert batch.home_entity_mask.shape == (B, data.n_entities)
    # every valid event's entity exists in its own home
    for b in range(B):
        ents = batch.entity_idx[b][batch.valid[b]]
        assert batch.home_entity_mask[b, ents].all()
